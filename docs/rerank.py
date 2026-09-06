"""Experiment: does a cross-encoder rerank stage improve retrieval enough to ship?

Pipeline under test:
  bi-encoder retrieves top-N chunks  ->  cross-encoder rescores those N pairs
  -> reranked chunks occupy the head of the ranking, the tail keeps bi-encoder
     order -> per-doc max-pool -> nDCG@10.

Cross-encoder scores are only comparable within the reranked set, so the head is
ordered by CE rank and lifted above every un-reranked score rather than mixing
two incompatible scales.
"""
import json, time, numpy as np, torch, evallib as E
from sentence_transformers import CrossEncoder

MAX_N = 100
DEPTHS = [10, 20, 30, 50, 100]

chunk_texts = json.load(open("chunk_texts.json"))
qtexts = json.load(open("qtexts.json"))
D = np.load("doc_fp32.npy"); Q = np.load("qry_fp32.npy")
S = Q @ D.T                                        # (300, 12815) bi-encoder scores

pool = np.argsort(-S, axis=1)[:, :MAX_N]           # candidate chunk ids per query

# Recall ceiling: reranking can only reorder what retrieval already found.
cdoc, docs = E.cdoc, E.docs
for n in DEPTHS:
    hit = tot = 0
    for i, qid in enumerate(qtexts):
        g = E.rel.get(E.qids[i], {})
        found = {docs[cdoc[c]] for c in pool[i, :n]}
        hit += sum(1 for d in g if d in found); tot += len(g)
    print(f"candidate pool N={n:3d} chunks -> relevant-doc recall {hit/tot:.4f}", flush=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device, max_length=256)

pairs = [[qtexts[i], chunk_texts[c]] for i in range(len(qtexts)) for c in pool[i]]
t0 = time.time()
ce_scores = np.asarray(ce.predict(pairs, batch_size=64, show_progress_bar=False), dtype=np.float32)
elapsed = time.time() - t0
ce_scores = ce_scores.reshape(len(qtexts), MAX_N)
np.save("ce_scores.npy", ce_scores); np.save("pool.npy", pool)
print(f"\ncross-encoder: {len(pairs):,} pairs in {elapsed:.1f}s "
      f"= {len(pairs)/elapsed:.0f} pairs/s on {device}", flush=True)

def reranked_out(n):
    """Bi-encoder scores with the top-n chunks re-ordered by cross-encoder rank."""
    F = S.copy()
    order = np.argsort(-ce_scores[:, :n], axis=1)
    for i in range(len(qtexts)):
        # 2.0 is above any cosine, so the reranked head always outranks the tail.
        F[i, pool[i, :n][order[i]]] = 2.0 + np.arange(n, 0, -1) / n
    return E.maxpool(F)

base_out = E.maxpool(S)
base = E.per_query_ndcg(base_out)
print("\n%-26s %-7s %-7s %-7s %9s %22s %7s" %
      ("variant", "nDCG", "Recall", "MRR", "delta", "95% CI", "p"))
print("-" * 92)
print("%-26s %.4f  %.4f  %.4f" % (("bi-encoder only",) + E.summary(base_out)))
for n in DEPTHS:
    out = reranked_out(n)
    v = E.per_query_ndcg(out); d = v - base
    lo, hi, p = E.bootstrap(d)
    nd, rc, mrr = E.summary(out)
    print("%-26s %.4f  %.4f  %.4f  %+8.4f  [%+.4f, %+.4f] %7.3f" %
          (f"+ rerank top-{n}", nd, rc, mrr, d.mean(), lo, hi, p))
