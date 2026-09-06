/**
 * Cross-encoder reranking.
 *
 * Retrieval scores the query and a passage *independently* — the query never
 * sees the passage. A cross-encoder runs the pair through the transformer
 * together, so it can weigh how the two actually relate. Its output is a
 * permutation of passages retrieval already found, so it cannot invent text:
 * this is the one quality lever compatible with the rule that a result must be
 * real document text.
 *
 * Measured in docs/hypothesis-testing.md §9–§10, paired bootstrap, 10,000
 * resamples, against the same top-30 pool:
 *
 *     SciFact, 300 labelled queries   nDCG@10  0.6506 -> 0.7143   p < 0.001
 *     DDIA, 91 queries on a real book nDCG@10  0.5768 -> 0.7755   p < 0.001
 *
 * It is the only change measured in that document that improves retrieval
 * quality significantly. It is also not free: it runs once per *passage* rather
 * than once per query, so it is paid on every search (~0.9 s on the CPU path).
 * That, plus the 1-in-13 queries it makes worse, is why it is a toggle.
 */

/**
 * Candidates to rescore.
 *
 * Depth 30 because N=10 is significantly too shallow on both corpora (p = 0.018
 * and p = 0.001) while 20 through 100 are statistically indistinguishable. Depth
 * is therefore a latency decision, not a quality one, and 30 is where you stop
 * paying for nothing.
 */
export const RERANK_DEPTH = 30;

/** The cross-encoder emits one unbounded logit; readers see a 0–1 score. */
export const sigmoid = (logit) => 1 / (1 + Math.exp(-logit));

/**
 * A pair whose correct ordering is not in doubt, checked once at load.
 *
 * §7 caught the WebGPU int8 embedder running to completion, reporting a
 * plausible speed, and returning noise — a failure invisible to everything
 * except a control. A reranker that silently returns noise would quietly
 * *destroy* ranking (measured: −0.39 nDCG for a random reorder), so it has to
 * prove itself before it is trusted.
 */
export const CONTROL = {
  query: 'how does a database keep two copies of the data in sync',
  relevant:
    'Replication means keeping a copy of the same data on several machines '
    + 'connected via a network. The leader sends every write to its followers, '
    + 'which apply the changes in the same order.',
  irrelevant:
    'Preheat the oven to 220 degrees. Knead the dough for ten minutes, let it '
    + 'rise until doubled in size, then bake for thirty minutes until golden.',
};

/**
 * Minimum logit margin the control must clear.
 *
 * Measured margin for a working int8 model is 10.27. Noise produces a margin
 * near zero. 4.0 sits far above the noise floor with 2.5x headroom below a real
 * signal, so it neither passes a broken model nor fails a working one.
 */
export const CONTROL_MIN_MARGIN = 4.0;

/** Did the reranker order the control pair correctly, and decisively? */
export const controlPasses = (relevantLogit, irrelevantLogit) =>
  Number.isFinite(relevantLogit) && Number.isFinite(irrelevantLogit)
  && relevantLogit - irrelevantLogit >= CONTROL_MIN_MARGIN;

/**
 * Replace retrieval scores with cross-encoder scores.
 *
 * Returns a new array — the caller still owns the retrieval ordering, which is
 * what is shown while reranking is in flight. Hits without a score keep their
 * retrieval score, which cannot happen in normal operation but must not throw.
 *
 * @param {Array<{score: number}>} hits    retrieval hits, best first
 * @param {ArrayLike<number>} logits       one cross-encoder logit per hit
 * @returns {Array} the same hits, rescored and re-sorted
 */
export function applyRerank(hits, logits) {
  return hits
    .map((hit, index) => (index < logits.length
      ? { ...hit, score: sigmoid(logits[index]), reranked: true }
      : { ...hit }))
    .sort((a, b) => b.score - a.score);
}
