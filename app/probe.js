/**
 * Diagnostic: run the app's real search pipeline over the documents you have
 * already indexed, printing every number along the way.
 *
 * This exists because reranked results render as 0.00 on one machine and
 * correctly on another. It reads your existing index from this browser's
 * storage — nothing is indexed again, nothing is uploaded, and the output is
 * printed on this page only.
 *
 * Type a query, press Run.
 */
import { VectorIndex, diversify, DIMENSIONS } from './src/search.js';
import { applyRerank, RERANK_DEPTH, sigmoid } from './src/rerank.js';
import * as store from './src/store.js';

const out = document.getElementById('out');
const log = (t) => { out.textContent += t + '\n'; };
const report = {};

const worker = new Worker('./worker.js', { type: 'module' });
const ask = (message, wanted) => new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error(`${message.type} timed out`)), 240000);
  worker.onmessage = ({ data }) => {
    if (data.type === 'ready') {
      report.ready = data;
      log(`worker ready          : ${data.backend}/${data.dtype}, threads ${data.threads}`);
    }
    if (data.type === wanted) { clearTimeout(timer); resolve(data); }
    if (data.type === 'error' || data.type === 'rerankFailed') {
      clearTimeout(timer); reject(new Error(data.message));
    }
  };
  worker.postMessage(message);
});

const index = new VectorIndex();
const names = new Map();

async function load() {
  report.webgpu = Boolean(navigator.gpu);
  report.crossOriginIsolated = self.crossOriginIsolated;
  log(`navigator.gpu present : ${report.webgpu}`);
  log(`crossOriginIsolated   : ${report.crossOriginIsolated}`);

  const [saved, vectors] = await Promise.all([store.listFiles(), store.getAllVectors()]);
  for (const record of saved) names.set(record.id, record);
  let chunks = 0;
  for (const row of vectors) {
    if (row.data?.length !== (names.get(row.fileId)?.chunkCount ?? 0) * DIMENSIONS) continue;
    const flags = await store.getChunksFor(row.fileId);
    index.add(row.fileId, row.scale, row.data, flags.map((c) => c.boiler));
    chunks += flags.length;
  }
  report.indexedFiles = saved.length;
  report.indexedChunks = chunks;
  log(`your index            : ${saved.length} file(s), ${chunks} chunks`);
  if (!chunks) log('\nNothing indexed in this browser yet — open the app and add a document first.');
}

async function run(query) {
  out.textContent = '';
  log(`query: ${query}\n`);
  report.query = query;

  const hits = index.search(await embed(query), RERANK_DEPTH);
  report.retrievalTop = hits.slice(0, 5).map((h) => +h.score.toFixed(3));
  log(`retrieval top-5 cosine: ${report.retrievalTop.join(', ')}`);

  const texts = await Promise.all(
    hits.map((h) => store.getChunk(`${h.fileId}:${h.ordinal}`)));
  const keep = hits.filter((_, i) => texts[i]);
  const passages = texts.filter(Boolean).map((c) => c.text);

  const { logits } = await ask(
    { type: 'rerank', token: 1, query, passages }, 'reranked');

  // The raw wire value matters: a typed array reinterpreted wrongly shows here.
  report.logitsType = Object.prototype.toString.call(logits);
  report.logitsLength = logits.length;
  report.passagesSent = passages.length;
  report.firstLogits = Array.from(logits).slice(0, 5).map(Number);
  log(`\nlogits type           : ${report.logitsType}  length ${report.logitsLength}`
    + `  (sent ${passages.length} passages)`);
  log(`first 5 raw logits    : ${report.firstLogits.join(', ')}`);
  log(`  -> sigmoid          : ${report.firstLogits.map((v) => sigmoid(v).toFixed(6)).join(', ')}`);

  const shown = diversify(applyRerank(keep, logits), { maxPerFile: 3, limit: 5 });
  report.printed = shown.map((h) => h.score.toFixed(2));
  report.rawScores = shown.map((h) => h.score);
  log('\nWHAT THE RESULT CARDS SHOW:');
  for (const h of shown) {
    const chunk = texts[hits.indexOf(h)] ?? await store.getChunk(`${h.fileId}:${h.ordinal}`);
    log(`  "${h.score.toFixed(2)}"   ${(chunk?.text ?? '').slice(0, 54)}…`);
  }
  log('\n--- copy everything below this line ---');
  log(JSON.stringify(report, null, 1));
}

async function embed(text) {
  const { vector } = await ask({ type: 'query', text }, 'queryEmbedding');
  return vector;
}

document.getElementById('go').addEventListener('click', () => {
  const q = document.getElementById('q').value.trim();
  if (q) run(q).catch((e) => log(`\nFAILED: ${e?.message ?? e}\n` + JSON.stringify(report, null, 1)));
});

await load();
