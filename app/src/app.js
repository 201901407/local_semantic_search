/**
 * Application shell — wiring, not logic.
 *
 * Everything user-supplied reaches the DOM through textContent. There is no
 * innerHTML anywhere in this file, and no template that interpolates document
 * text into markup. That is the structural defence against a document whose
 * contents contain script tags: they become visible characters, never nodes.
 */

import { inspectFile, scanExtractedText, sanitizeFilename, LIMITS, SUPPORTED_EXTENSIONS }
  from './validate.js';
import { VectorIndex, diversify, DIMENSIONS } from './search.js';
import { selectExcerpt, queryTerms } from './excerpt.js';
import { applyRerank, RERANK_DEPTH } from './rerank.js';
import * as store from './store.js';

const RESULT_LIMIT_KEY = 'result-limit';
const RERANK_KEY = 'rerank-enabled';
const DEFAULT_RESULT_LIMIT = 5;
const SEARCH_DEBOUNCE_MS = 160;

const $ = (id) => document.getElementById(id);
const dom = {
  drop: $('drop'), picker: $('picker'), backend: $('backend'), stats: $('stats'),
  clear: $('clear'), origin: $('origin'), resultbar: $('resultbar'), cancel: $('cancel'),
  resultsCount: $('results-count'), limit: $('limit'), limitValue: $('limit-value'),
  rerank: $('rerank'), rerankRow: $('rerank-row'), rerankNote: $('rerank-note'),
  job: $('job'), jobTitle: $('job-title'), jobDetail: $('job-detail'), jobBar: $('job-bar'),
  library: $('library'), files: $('files'), libraryNote: $('library-note'),
  query: $('query'), results: $('results'),
  resultsEmpty: $('results-empty'), resultsList: $('results-list'),
};

const index = new VectorIndex();
const files = new Map();            // fileId -> { id, name, chunkCount, status, warnings }
const queue = [];                   // files awaiting indexing
let busy = false;
let deviceRate = null;              // measured chunks/s, once known
let pendingQuery = null;
let pendingText = '';               // the raw query, which the cross-encoder needs
let activeFileId = null;      // the file the worker is currently indexing
let lastQuery = null;               // { vector, terms } — kept so the result
                                    // count can change without re-embedding
let searchToken = 0;                // guards against a stale rerank landing after
                                    // the reader has moved on to another query
let pendingRerank = null;           // { token, hits } awaiting cross-encoder scores

const worker = new Worker(new URL('../worker.js', import.meta.url), { type: 'module' });

// --------------------------------------------------------------------------
// DOM helpers — the only place elements are created
// --------------------------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;   // never innerHTML
  return node;
}

const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`;

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return '';
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(Math.round(seconds % 60)).padStart(2, '0')}s`;
}

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

function renderStats() {
  const indexed = [...files.values()].filter((file) => file.status === 'indexed');
  const chunks = indexed.reduce((total, file) => total + file.chunkCount, 0);
  dom.stats.textContent = indexed.length
    ? `${plural(indexed.length, 'document')} · ${plural(chunks, 'passage')}`
    : '';
}

function renderLibrary() {
  dom.files.replaceChildren();
  dom.library.hidden = files.size === 0;
  if (files.size === 0) return;

  for (const file of files.values()) {
    const row = el('div', 'row');
    row.append(el('span', 'name', file.name));

    if (file.status === 'indexed') {
      row.append(el('span', 'meta', plural(file.chunkCount, 'passage')));
      if (file.warnings?.length) {
        const pill = el('span', 'pill warn', 'markup');
        pill.title = file.warnings.join(' ');
        row.append(pill);
      }
    } else {
      const label = file.status === 'queued' ? 'queued'
        : file.status === 'no-text' ? 'no text layer'
        : file.status;
      row.append(el('span', `pill ${file.status === 'indexed' ? 'ok' : file.error ? 'bad' : ''}`, label));
      if (file.error) {
        const why = el('span', 'meta', file.error);
        why.title = file.error;
        row.append(why);
      }
    }

    const remove = el('button', 'icon', '×');
    remove.title = `Remove ${file.name}`;
    remove.setAttribute('aria-label', `Remove ${file.name}`);
    remove.addEventListener('click', () => removeFile(file.id));
    row.append(remove);

    dom.files.append(row);
  }

  const scanned = [...files.values()].filter((file) => file.status === 'no-text');
  dom.libraryNote.textContent = scanned.length
    ? `${plural(scanned.length, 'file')} had no text layer — likely a scan. Reading those needs OCR, which this version does not do.`
    : 'Indexed once and stored in this browser. Re-adding the same file is skipped.';
}

/** Render an excerpt with query terms marked, using text nodes only. */
function renderExcerpt(container, text, terms) {
  const excerpt = selectExcerpt(text, terms);

  if (!terms.length) {
    container.append(document.createTextNode(excerpt));
    return;
  }

  const lower = excerpt.toLowerCase();
  let cursor = 0;
  while (cursor < excerpt.length) {
    let at = -1;
    let term = '';
    for (const candidate of terms) {
      const found = lower.indexOf(candidate, cursor);
      if (found !== -1 && (at === -1 || found < at)) { at = found; term = candidate; }
    }
    if (at === -1) break;
    container.append(document.createTextNode(excerpt.slice(cursor, at)));
    container.append(el('mark', null, excerpt.slice(at, at + term.length)));
    cursor = at + term.length;
  }
  container.append(document.createTextNode(excerpt.slice(cursor)));
}

/** How many results the reader has asked for. */
const resultLimit = () => Number(dom.limit.value);

/** Is the reader asking for cross-encoder reranking? */
const rerankOn = () => dom.rerank.checked && !dom.rerank.disabled;

/**
 * Re-rank and redraw from the last query, without embedding it again.
 *
 * Retrieval results are painted immediately even when reranking is on. The
 * cross-encoder takes about a second on the CPU path, and showing real results
 * now and refining them shortly after is far better than a second of blank
 * space — the reader is usually already reading the top hit by then.
 */
function refreshResults() {
  if (!lastQuery || index.isEmpty) return;
  const token = ++searchToken;
  pendingRerank = null;

  // Reranking is only as good as the pool it is given, and depth 30 is measured;
  // without it, retrieval alone needs only a few times the requested count.
  const depth = rerankOn() ? RERANK_DEPTH : resultLimit() * 4;
  const hits = index.search(lastQuery.vector, depth);

  renderResults(diversify(hits, { maxPerFile: 3, limit: resultLimit() }), lastQuery.terms);
  if (rerankOn() && hits.length) requestRerank(token, hits);
}

/** Send the retrieved passages to the worker to be rescored against the query. */
async function requestRerank(token, hits) {
  const chunks = await Promise.all(
    hits.map((hit) => store.getChunk(`${hit.fileId}:${hit.ordinal}`)),
  );
  if (token !== searchToken) return;               // the reader has moved on

  const scored = hits.filter((_, position) => chunks[position]);
  if (!scored.length) return;

  pendingRerank = { token, hits: scored };
  dom.rerankRow.dataset.state = 'working';
  worker.postMessage({
    type: 'rerank',
    token,
    query: lastQuery.text,
    passages: chunks.filter(Boolean).map((chunk) => chunk.text),
  });
}

async function renderResults(hits, terms) {
  dom.resultsList.replaceChildren();
  dom.resultbar.hidden = !lastQuery || index.isEmpty;
  dom.resultsCount.textContent = hits.length
    ? `${plural(hits.length, 'passage')} from ${plural(
        new Set(hits.map((hit) => hit.fileId)).size, 'document')}`
    : '';

  if (!hits.length) {
    dom.resultsEmpty.hidden = false;
    dom.resultsEmpty.textContent = index.isEmpty
      ? 'Add a document to start searching.'
      : 'No passages matched. Try describing the idea differently.';
    return;
  }
  dom.resultsEmpty.hidden = true;

  const chunks = await Promise.all(
    hits.map((hit) => store.getChunk(`${hit.fileId}:${hit.ordinal}`)),
  );

  hits.forEach((hit, position) => {
    const chunk = chunks[position];
    if (!chunk) return;

    const card = el('div', 'result');
    const head = el('div', 'head');
    head.append(el('span', 'src', files.get(hit.fileId)?.name ?? 'Unknown file'));
    const score = el('span', 'score', hit.score.toFixed(2));
    score.title = hit.reranked
      ? `Cross-encoder relevance ${hit.score.toFixed(4)} (logit ${Number(hit.logit).toFixed(3)})`
      : `Cosine similarity ${hit.score.toFixed(4)}`;
    head.append(score);
    card.append(head);

    const body = el('div', 'text');
    renderExcerpt(body, chunk.text, terms);
    card.append(body);

    dom.resultsList.append(card);
  });
}

function showJob(title, detail, ratio) {
  dom.job.hidden = false;
  dom.cancel.disabled = false;
  dom.cancel.textContent = 'Stop';
  dom.jobTitle.textContent = title;
  dom.jobDetail.textContent = detail;
  dom.jobBar.style.width = `${Math.round((ratio ?? 0) * 100)}%`;
}

const hideJob = () => { dom.job.hidden = true; };

// --------------------------------------------------------------------------
// Indexing
// --------------------------------------------------------------------------

async function addFiles(fileList) {
  // Bound a single drop. Without this a dropped folder of ten thousand files
  // queues ten thousand jobs, and the only way out is closing the tab.
  let batch = [...fileList];
  const rejected = [];
  if (batch.length > LIMITS.maxFilesPerDrop) {
    rejected.push(`${batch.length - LIMITS.maxFilesPerDrop} file(s) beyond the `
      + `${LIMITS.maxFilesPerDrop}-per-drop limit were ignored`);
    batch = batch.slice(0, LIMITS.maxFilesPerDrop);
  }
  let running = 0;
  batch = batch.filter((file) => {
    running += file.size;
    return running <= LIMITS.maxBytesPerDrop;
  });
  if (rejected.length) console.warn(rejected.join('; '));

  for (const file of batch) {
    const verdict = await inspectFile(file);
    const id = store.fileIdentity(file);
    // Never display a raw filename: bidi overrides can make "report<U+202E>fdp.txt"
    // render as "reporttxt.pdf", so what the reader sees would be a lie.
    const name = sanitizeFilename(file.name);

    if (!verdict.ok) {
      files.set(id, { id, name, status: 'rejected', error: verdict.reason, chunkCount: 0 });
      continue;
    }
    // Byte-identical file already indexed — nothing to do.
    if (files.has(id) && files.get(id).status === 'indexed') continue;

    // Same filename but a different fingerprint means an edited or newer copy.
    // Drop the stale version rather than leaving two entries with one name.
    for (const existing of [...files.values()]) {
      if (existing.name === name && existing.id !== id) {
        await removeFile(existing.id);
      }
    }

    files.set(id, { id, name, status: 'queued', chunkCount: 0 });
    queue.push({ id, file, name, kind: verdict.kind });
  }
  renderLibrary();
  drainQueue();
}

async function drainQueue() {
  if (busy || queue.length === 0) return;
  busy = true;

  const { id, file, name, kind } = queue.shift();
  activeFileId = id;
  const buffer = await file.arrayBuffer();
  worker.postMessage({ type: 'index', fileId: id, name, kind, buffer }, [buffer]);
}

async function clearEverything() {
  const count = [...files.values()].filter((f) => f.status === 'indexed').length;
  if (count > 0 && !confirm(
    `Remove all ${count} indexed document${count === 1 ? '' : 's'}? `
    + 'They will need to be indexed again.')) return;

  for (const fileId of [...files.keys()]) index.remove(fileId);
  files.clear();
  await store.clearAll();
  renderLibrary();
  renderStats();
  runSearch();
}

async function removeFile(fileId) {
  index.remove(fileId);
  files.delete(fileId);
  await store.deleteFile(fileId);
  renderLibrary();
  renderStats();
  runSearch();
}

// --------------------------------------------------------------------------
// Worker messages
// --------------------------------------------------------------------------

worker.onmessage = async ({ data: message }) => {
  switch (message.type) {
    case 'ready': {
      // Thread count is a WASM concept. On the GPU path the forward pass does not
      // run on WASM threads at all, so reporting one there is noise that hides the
      // thing that does matter — the precision the weights were downloaded at.
      const gpu = message.backend === 'webgpu';
      dom.backend.textContent = gpu
        ? `${message.backend} · ${message.dtype}`
        : `${message.backend} · ${message.dtype}, ${message.threads ?? 1} threads`;
      dom.backend.title = gpu
        ? 'Running on the GPU — about 4x faster than the CPU path, and unaffected '
          + 'by the COOP/COEP headers, which only govern multi-threaded WASM.'
        : message.crossOriginIsolated
          ? 'Multi-threaded — cross-origin isolation is active.'
          : 'Single-threaded: the required COOP/COEP headers are missing, so indexing is slower.';
      break;
    }

    case 'phase':
      showJob(
        message.phase === 'extracting' ? 'Reading document'
          : message.phase === 'chunking' ? 'Splitting into passages'
          : 'Indexing',
        files.get(message.fileId)?.name ?? '',
        0,
      );
      break;

    case 'estimate': {
      const rate = message.chunksPerSecond ?? deviceRate;
      const eta = rate ? ` · about ${formatDuration(message.chunks / rate)}` : '';
      showJob('Indexing', `${plural(message.chunks, 'passage')}${eta}`, 0);
      break;
    }

    case 'progress': {
      if (message.chunksPerSecond) deviceRate = message.chunksPerSecond;
      const name = files.get(message.fileId)?.name ?? '';
      if (message.phase === 'extracting') {
        showJob('Reading document', `${name} · page ${message.done} of ${message.total}`,
                message.done / message.total);
      } else {
        const left = deviceRate ? (message.total - message.done) / deviceRate : null;
        showJob('Indexing', `${name} · ${message.done.toLocaleString()} of ${message.total.toLocaleString()} passages`
          + (left ? ` · ${formatDuration(left)} left` : ''), message.done / message.total);
      }
      break;
    }

    case 'indexed': {
      const record = files.get(message.fileId) ?? { id: message.fileId, name: message.name };
      record.status = 'indexed';
      record.chunkCount = message.chunks.length;
      record.warnings = scanExtractedText(message.chunks.map((c) => c.text).join(' '));
      files.set(message.fileId, record);

      index.add(message.fileId, message.scale, message.data,
                message.chunks.map((chunk) => chunk.boiler));
      await store.putIndexedFile(message.fileId, message.chunks,
        { scale: message.scale, data: message.data });
      await store.putFile({
        id: record.id, name: record.name, chunkCount: record.chunkCount,
        status: 'indexed', warnings: record.warnings,
      });

      renderLibrary(); renderStats();
      busy = false;
      queue.length ? drainQueue() : hideJob();
      runSearch();
      break;
    }

    case 'cancelled': {
      const record = files.get(message.fileId);
      if (record) { record.status = 'stopped'; renderLibrary(); }
      activeFileId = null;
      busy = false;
      queue.length ? drainQueue() : hideJob();
      break;
    }

    case 'skipped': {
      const record = files.get(message.fileId) ?? { id: message.fileId, name: message.name };
      record.status = message.reason === 'no-text' ? 'no-text' : 'skipped';
      record.chunkCount = 0;
      files.set(message.fileId, record);
      await store.putFile({ id: record.id, name: record.name, chunkCount: 0, status: record.status });
      renderLibrary();
      busy = false;
      queue.length ? drainQueue() : hideJob();
      break;
    }

    case 'queryEmbedding': {
      if (!pendingQuery) break;
      const { terms } = pendingQuery;
      pendingQuery = null;
      lastQuery = { vector: message.vector, terms, text: pendingText };
      refreshResults();
      break;
    }

    case 'reranked': {
      // A rerank that lands after the reader has typed again is discarded, not
      // rendered: it describes a question they are no longer asking.
      if (!pendingRerank || message.token !== searchToken) break;
      const { hits } = pendingRerank;
      pendingRerank = null;
      dom.rerankRow.dataset.state = 'on';
      renderResults(
        diversify(applyRerank(hits, message.logits), { maxPerFile: 3, limit: resultLimit() }),
        lastQuery.terms,
      );
      break;
    }

    case 'rerankFailed': {
      // Turn it off rather than leave a switch on that quietly does nothing.
      pendingRerank = null;
      dom.rerank.checked = false;
      dom.rerank.disabled = true;
      dom.rerankRow.dataset.state = 'failed';
      dom.rerankNote.textContent = message.message;
      dom.rerankNote.hidden = false;
      try { localStorage.removeItem(RERANK_KEY); } catch { /* private mode */ }
      break;
    }

    case 'error': {
      if (message.fileId && files.has(message.fileId)) {
        const record = files.get(message.fileId);
        record.status = 'failed';
        record.error = message.message;
        renderLibrary();
      } else {
        dom.backend.textContent = 'error';
        dom.backend.title = message.message;
      }
      busy = false;
      queue.length ? drainQueue() : hideJob();
      break;
    }
  }
};

// --------------------------------------------------------------------------
// Search
// --------------------------------------------------------------------------

function runSearch() {
  const text = dom.query.value.trim();
  if (!text || index.isEmpty) {
    pendingQuery = null;
    lastQuery = null;
    renderResults([], []);
    return;
  }
  pendingQuery = { terms: queryTerms(text) };
  pendingText = text;
  worker.postMessage({ type: 'query', text });
}

let searchTimer = null;
dom.query.addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(runSearch, SEARCH_DEBOUNCE_MS);
});

document.addEventListener('keydown', (event) => {
  if (event.key === '/' && document.activeElement !== dom.query) {
    event.preventDefault();
    dom.query.focus();
  } else if (event.key === 'Escape' && document.activeElement === dom.query) {
    dom.query.value = '';
    runSearch();
    dom.query.blur();
  }
});

// --------------------------------------------------------------------------
// File input
// --------------------------------------------------------------------------

dom.drop.addEventListener('click', () => dom.picker.click());
dom.drop.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); dom.picker.click(); }
});
dom.picker.addEventListener('change', () => {
  addFiles([...dom.picker.files]);
  dom.picker.value = '';
});

for (const type of ['dragenter', 'dragover']) {
  dom.drop.addEventListener(type, (event) => {
    event.preventDefault();
    dom.drop.classList.add('over');
  });
}
for (const type of ['dragleave', 'drop']) {
  dom.drop.addEventListener(type, () => dom.drop.classList.remove('over'));
}
dom.drop.addEventListener('drop', (event) => {
  event.preventDefault();
  addFiles([...event.dataTransfer.files]);
});
// A file dropped outside the zone would otherwise navigate away and lose state.
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => event.preventDefault());

// --------------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------------

dom.cancel.addEventListener('click', () => {
  if (activeFileId) worker.postMessage({ type: 'cancel', fileId: activeFileId });
  queue.length = 0;               // abandon anything still waiting
  dom.cancel.disabled = true;
  dom.cancel.textContent = 'Stopping…';
});

dom.clear.addEventListener('click', clearEverything);

// Changing the count re-ranks the cached query — no re-embedding, so it is
// instant and costs nothing. The choice is remembered per browser.
dom.limit.addEventListener('input', () => {
  dom.limitValue.textContent = dom.limit.value;
  try { localStorage.setItem(RESULT_LIMIT_KEY, dom.limit.value); } catch { /* private mode */ }
  refreshResults();
});

// Reranking is a real trade — about a second per search on the CPU path, and
// one query in thirteen comes out worse — so it is the reader's choice, and the
// choice is remembered.
dom.rerank.addEventListener('change', () => {
  dom.rerankRow.dataset.state = dom.rerank.checked ? 'on' : 'off';
  try { localStorage.setItem(RERANK_KEY, dom.rerank.checked ? '1' : '0'); } catch { /* private mode */ }
  refreshResults();
});

(async function boot() {
  // Without this, stored data is "best effort" and the browser may evict it
  // under disk pressure — silently discarding an index that took minutes to
  // build. Granting is at the browser's discretion; we simply ask.
  if (navigator.storage?.persist) {
    const durable = await navigator.storage.persisted?.() || await navigator.storage.persist();
    dom.origin.title = durable
      ? 'Storage is persistent — the browser will not evict this index.'
      : 'Storage is best-effort; the browser may evict it if disk space runs low.';
  }
  dom.origin.textContent = location.origin.replace(/^https?:\/\//, '');

  let stored = null;
  try { stored = localStorage.getItem(RESULT_LIMIT_KEY); } catch { /* private mode */ }
  const limit = Number(stored) || DEFAULT_RESULT_LIMIT;
  dom.limit.value = String(Math.min(10, Math.max(1, limit)));
  dom.limitValue.textContent = dom.limit.value;

  // Default off: the model is 23 MB and reranking costs a second per search, so
  // it is opted into, never sprung on someone.
  let rerankStored = null;
  try { rerankStored = localStorage.getItem(RERANK_KEY); } catch { /* private mode */ }
  dom.rerank.checked = rerankStored === '1';
  dom.rerankRow.dataset.state = dom.rerank.checked ? 'on' : 'off';

  dom.drop.querySelector('p').textContent =
    `${SUPPORTED_EXTENSIONS.map((extension) => extension.toUpperCase()).join(', ')}`
    + ' · nothing is uploaded — indexing happens in this browser';

  const [saved, vectors] = await Promise.all([store.listFiles(), store.getAllVectors()]);
  for (const record of saved) files.set(record.id, record);
  for (const row of vectors) {
    if (row.data?.length !== (files.get(row.fileId)?.chunkCount ?? 0) * DIMENSIONS) continue;
    // Flags live on the chunk records, so restore them alongside the vectors.
    const saved = await store.getChunksFor(row.fileId);
    index.add(row.fileId, row.scale, row.data, saved.map((chunk) => chunk.boiler));
  }
  renderLibrary();
  renderStats();
  worker.postMessage({ type: 'init' });
})();
