/**
 * Indexing worker — everything expensive happens here.
 *
 * The transformer forward pass is 99.58% of indexing cost, so running it on the
 * main thread would freeze the interface for the entire job. Extraction and
 * chunking come along for the ride: they are cheap, but keeping them beside the
 * embedder avoids shipping file bytes back and forth.
 *
 * Backend selection is not a preference. Measured on this stack:
 *   WebGPU + fp16 → 171.1 chunks/s   (45.3 MB)
 *   WASM   + int8 →  40.6 chunks/s   (23.0 MB)
 *   WebGPU + int8 → produces noise (cosine 0.0236 against reference)
 * so int8 is never used on WebGPU, and fp16 is never downloaded without it.
 */

import { pipeline, env, AutoTokenizer, AutoModelForSequenceClassification } from
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js';

import { chunkText, batchesByLength } from './src/chunk.js';
import { extractText, isProbablyScanned } from './src/extract.js';
import { quantizeToInt8, DIMENSIONS } from './src/search.js';
import { isBoilerplate } from './src/quality.js';
import { clampText, LIMITS } from './src/validate.js';
import { withTimeout } from './src/timeout.js';
import { CONTROL, controlPasses } from './src/rerank.js';

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const RERANKER_ID = 'Xenova/ms-marco-MiniLM-L-6-v2';
const BATCH_SIZE = 64;

// Weights come from the HuggingFace CDN rather than our own origin. Verified to
// work under COOP/COEP: the page stays crossOriginIsolated and throughput is
// unchanged, while the host serves ~100 KB instead of tens of megabytes.
env.allowRemoteModels = true;
env.allowLocalModels = false;

let extractor = null;
let backend = null;
let reranker = null;

/** Rolling measurement of this device's real throughput, in chunks per second. */
const calibration = { chunks: 0, seconds: 0 };

const post = (message, transfer = []) => self.postMessage(message, transfer);

/** Files the user has asked to abandon; checked between units of work. */
const cancelled = new Set();

class Cancelled extends Error {}

/**
 * Choose a device and precision this machine can actually run.
 *
 * `navigator.gpu` existing is not the same as fp16 being supported — some
 * adapters expose WebGPU without the `shader-f16` feature, and asking for fp16
 * there throws. There is no third option: §7 measured int8 on WebGPU returning
 * noise (cosine 0.0236), so a GPU without fp16 must fall back to the CPU path
 * rather than quietly producing garbage embeddings.
 */
async function pickBackend() {
  const cpu = { device: 'wasm', dtype: 'q8' };
  if (!navigator.gpu) return cpu;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter?.features?.has('shader-f16')
      ? { device: 'webgpu', dtype: 'fp16' }
      : cpu;
  } catch {
    return cpu;
  }
}

async function loadModel() {
  if (extractor) return extractor;

  backend = await pickBackend();
  const useWebGpu = backend.device === 'webgpu';

  if (!useWebGpu && env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads = self.crossOriginIsolated
      ? Math.min(navigator.hardwareConcurrency || 4, 8)
      : 1;
  }

  const started = performance.now();
  try {
    extractor = await pipeline('feature-extraction', MODEL_ID, backend);
  } catch (error) {
    // A GPU that advertised fp16 and then failed is still no reason to give up;
    // the CPU path works everywhere. Better slow than broken.
    if (!useWebGpu) throw error;
    backend = { device: 'wasm', dtype: 'q8' };
    extractor = await pipeline('feature-extraction', MODEL_ID, backend);
  }

  post({
    type: 'ready',
    backend: backend.device,
    dtype: backend.dtype,
    threads: env.backends?.onnx?.wasm?.numThreads ?? null,
    crossOriginIsolated: self.crossOriginIsolated,
    loadSeconds: (performance.now() - started) / 1000,
  });
  return extractor;
}

/** Embed one batch of chunk texts, returning a float32 matrix. */
async function embedBatch(texts) {
  const output = await extractor(texts, { pooling: 'mean', normalize: true });
  return output.data instanceof Float32Array
    ? output.data
    : Float32Array.from(output.data);
}

/**
 * Extract, chunk, embed and quantize a single file.
 *
 * Progress is reported per batch so the UI can show real movement rather than an
 * indeterminate spinner, and the estimate is emitted as soon as the chunk count
 * is known — before any embedding is paid for.
 */
async function indexFile({ fileId, name, kind, buffer }) {
  await loadModel();

  post({ type: 'phase', fileId, phase: 'extracting' });
  const extraction = await withTimeout(
    extractText(buffer, kind, (done, total) => {
      post({ type: 'progress', fileId, phase: 'extracting', done, total });
    }),
    LIMITS.extractTimeoutMs, 'Reading the document');

  if (isProbablyScanned(kind, extraction.text)) {
    post({ type: 'skipped', fileId, name, reason: 'no-text' });
    return;
  }

  post({ type: 'phase', fileId, phase: 'chunking' });

  // Enforce the text ceiling here, where the text actually exists. A limit that
  // is only declared is not a limit: a 31 MB document previously sailed past a
  // stated 20 MB cap because nothing called this.
  const { text: bounded, clamped } = clampText(extraction.text);

  const result = await chunkText(bounded, extractor.tokenizer);
  let chunks = result.chunks;
  if (chunks.length === 0) {
    post({ type: 'skipped', fileId, name, reason: 'no-text' });
    return;
  }
  // Bounds worker memory: the vector buffer alone is chunks x 384 floats.
  const capped = chunks.length > LIMITS.maxChunksPerFile;
  if (capped) chunks = chunks.slice(0, LIMITS.maxChunksPerFile);

  post({
    type: 'estimate',
    fileId,
    chunks: chunks.length,
    chunksPerSecond: calibration.seconds > 0
      ? calibration.chunks / calibration.seconds
      : null,
  });

  // Sorting by length makes each batch nearly uniform in width. Batches pad to
  // their longest member, which is 31.6% of compute on a real corpus; this
  // recovers roughly 23.7% of it with no change to the output.
  const batches = batchesByLength(chunks, BATCH_SIZE);
  const vectors = new Float32Array(chunks.length * DIMENSIONS);
  let embedded = 0;

  post({ type: 'phase', fileId, phase: 'embedding' });
  for (const batch of batches) {
    // Messages are delivered between awaits, so a cancel lands here.
    if (cancelled.has(fileId)) throw new Cancelled();

    const started = performance.now();
    const output = await withTimeout(
      embedBatch(batch.map((index) => chunks[index].text)),
      LIMITS.batchTimeoutMs, 'Indexing a batch');

    // Scatter each row back to its original position so chunk order is preserved.
    batch.forEach((chunkIndex, row) => {
      vectors.set(
        output.subarray(row * DIMENSIONS, (row + 1) * DIMENSIONS),
        chunkIndex * DIMENSIONS,
      );
    });

    calibration.seconds += (performance.now() - started) / 1000;
    calibration.chunks += batch.length;
    embedded += batch.length;

    post({
      type: 'progress',
      fileId,
      phase: 'embedding',
      done: embedded,
      total: chunks.length,
      chunksPerSecond: calibration.chunks / calibration.seconds,
    });
  }

  const { scale, data } = quantizeToInt8(vectors);
  post({
    type: 'indexed',
    fileId,
    name,
    truncated: extraction.truncated || clamped || capped,
    pages: extraction.pages ?? null,
    chunks: chunks.map(({ text, start, end }) => ({
      text, start, end, boiler: isBoilerplate(text),
    })),
    scale,
    data,
  }, [data.buffer]);
}

/**
 * Load the cross-encoder on first use, and refuse to use it unless it works.
 *
 * 23 MB is only fetched when the reader actually turns reranking on, so nobody
 * pays for a feature they left off. The control is not optional: §7 caught a
 * backend that ran to completion, reported a plausible speed and returned noise,
 * and a reranker doing that would actively destroy ranking rather than merely
 * fail to improve it.
 */
async function loadReranker() {
  if (reranker) return reranker;

  const tokenizer = await AutoTokenizer.from_pretrained(RERANKER_ID);
  const model = await AutoModelForSequenceClassification.from_pretrained(
    RERANKER_ID, backend ?? { device: 'wasm', dtype: 'q8' });
  const candidate = { tokenizer, model };

  const [relevant, irrelevant] = await scoreWith(
    candidate, CONTROL.query, [CONTROL.relevant, CONTROL.irrelevant]);
  if (!controlPasses(relevant, irrelevant)) {
    throw new Error(
      `Reranker failed its self-check on this device (margin ${(relevant - irrelevant).toFixed(1)}); `
      + 'leaving it off rather than reordering your results with noise');
  }

  reranker = candidate;
  return reranker;
}

/** Score one query against many passages. Returns one logit per passage. */
async function scoreWith({ tokenizer, model }, query, passages) {
  const inputs = tokenizer(new Array(passages.length).fill(query), {
    text_pair: passages, padding: true, truncation: true, max_length: 256,
  });
  const { logits } = await model(inputs);
  return Array.from(logits.data, Number);
}

async function rerankPassages({ token, query, passages }) {
  const scorer = await loadReranker();
  const logits = await withTimeout(
    scoreWith(scorer, query, passages), LIMITS.rerankTimeoutMs, 'Reranking');
  post({ type: 'reranked', token, logits });
}

async function embedQuery(text) {
  await loadModel();
  const vector = await embedBatch([text]);
  const copy = new Float32Array(vector.subarray(0, DIMENSIONS));
  post({ type: 'queryEmbedding', vector: copy }, [copy.buffer]);
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    switch (message.type) {
      case 'init':
        await loadModel();
        break;
      case 'index':
        await indexFile(message);
        break;
      case 'query':
        await embedQuery(message.text);
        break;
      case 'rerank':
        await rerankPassages(message);
        break;
      case 'cancel':
        cancelled.add(message.fileId);
        break;
      default:
        throw new Error(`Unknown message: ${message.type}`);
    }
  } catch (error) {
    if (error instanceof Cancelled) {
      post({ type: 'cancelled', fileId: message.fileId });
    } else if (message.type === 'rerank') {
      post({ type: 'rerankFailed', token: message.token,
             message: error?.message ?? String(error) });
    } else {
      post({
        type: 'error',
        fileId: message.fileId ?? null,
        message: error?.message ?? String(error),
      });
    }
  } finally {
    if (message.fileId) cancelled.delete(message.fileId);
  }
};
