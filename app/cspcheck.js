/**
 * Does the app's real Content-Security-Policy admit everything both models need?
 *
 * selftest.html cannot answer this: it runs its suite from an inline module,
 * which the app's own `script-src 'self'` correctly forbids, so it has to run
 * with no policy at all. This page carries index.html's exact policy and drives
 * the worker through the two things that reach the network — indexing (embedding
 * model) and reranking (cross-encoder) — so a CSP that is too narrow fails here
 * instead of in front of a user.
 */

const out = document.getElementById('out');
const lines = [];
const log = (text) => { lines.push(text); out.textContent = lines.join('\n'); };

const finish = async (ok, detail) => {
  log(ok ? '\nPASS — both models loaded under the app policy'
         : `\nFAIL — ${detail}`);
  await fetch('/result', {
    method: 'POST',
    body: JSON.stringify(
      ok ? { cspcheck: 'pass', log: lines }
         : { cspcheck: 'fail', detail, log: lines }, null, 2),
  });
};

const ask = (worker, message, transfer, wanted) => new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error(`${message.type} timed out`)), 180000);
  worker.onmessage = ({ data }) => {
    if (data.type === wanted) { clearTimeout(timer); resolve(data); }
    if (data.type === 'error' || data.type === 'rerankFailed') {
      clearTimeout(timer); reject(new Error(data.message));
    }
  };
  worker.postMessage(message, transfer);
});

try {
  const worker = new Worker('./worker.js', { type: 'module' });

  log('1/2  embedding model — indexing a short document…');
  const buffer = new TextEncoder().encode(
    'The refund window is thirty days from delivery. '
    + 'Replication keeps a copy of the same data on several machines.').buffer;
  const indexed = await ask(worker,
    { type: 'index', fileId: 'c1', name: 'note.txt', kind: 'txt', buffer }, [buffer], 'indexed');
  log(`     ok — ${indexed.chunks.length} chunk(s) embedded`);

  log('2/2  cross-encoder — reranking two passages…');
  const { logits } = await ask(worker, {
    type: 'rerank', token: 1,
    query: 'how does a database keep two copies of the data in sync',
    passages: ['Knead the dough and bake for thirty minutes until golden.',
               'Replication keeps a copy of the same data on several machines.'],
  }, [], 'reranked');
  log(`     ok — logits ${logits.map((l) => l.toFixed(2)).join(', ')}`);

  worker.terminate();
  await finish(logits.length === 2 && logits[1] > logits[0],
               'reranker loaded but scored the wrong passage higher');
} catch (error) {
  await finish(false, error?.message ?? String(error));
}
