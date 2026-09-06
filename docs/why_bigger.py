"""Why is the rerank gain 3x larger on DDIA than on SciFact?

Hypothesis: a single book is topically self-similar, so the bi-encoder's top
candidates are bunched together in score and it cannot separate them. Where the
bi-encoder is already confident (a wide gap between rank 1 and rank 10), there
is little for a reranker to fix.

Measured as the score spread across each query's top-30 candidate chunks.
"""
import json, numpy as np

def spread(S, pool, n=30):
    top = np.take_along_axis(S, pool[:, :n], axis=1)
    return top[:, 0] - top[:, -1], top[:, 0] - top[:, 1]

sci_S = np.load("qry_fp32.npy") @ np.load("doc_fp32.npy").T
sci_pool = np.load("pool.npy")
ddia_S, ddia_pool = np.load("ddia_S.npy"), np.load("ddia_pool.npy")

print(f"{'corpus':10s} {'chunks':>8s} {'rank1-rank30 gap':>18s} {'rank1-rank2 gap':>17s}")
print("-" * 58)
for name, S, pool, n in [("SciFact", sci_S, sci_pool, 12815), ("DDIA", ddia_S, ddia_pool, 1875)]:
    wide, near = spread(S, pool)
    print(f"{name:10s} {n:8,d} {wide.mean():18.4f} {near.mean():17.4f}")

# Does a narrow gap actually predict a large rerank gain, query by query?
ce = np.load("ddia_ce.npy"); chunk_block = np.array(json.load(open("ddia_meta.json"))["chunk_block"])
ev = json.load(open("ddia_eval_set.json")); gold = np.array(ev["blocks"])
nb = int(chunk_block.max()) + 1
def ndcg(F):
    out = np.full((F.shape[0], nb), -1e9, np.float32); np.maximum.at(out.T, chunk_block, F.T)
    top = np.argsort(-out, axis=1)[:, :10]; v = np.zeros(len(gold))
    for i, g in enumerate(gold):
        w = np.where(top[i] == g)[0]
        if len(w): v[i] = 1 / np.log2(int(w[0]) + 2)
    return v
F = ddia_S.copy(); o = np.argsort(-ce[:, :30], axis=1)
for i in range(len(gold)): F[i, ddia_pool[i, :30][o[i]]] = 2.0 + np.arange(30, 0, -1) / 30
gain = ndcg(F) - ndcg(ddia_S)
wide, _ = spread(ddia_S, ddia_pool)
print(f"\nDDIA, per query: correlation(candidate score spread, rerank gain) = "
      f"{np.corrcoef(wide, gain)[0,1]:+.3f}  (n={len(gain)})")
lo, hi = np.percentile(wide, [33, 67])
for label, m in [("narrow spread (bunched)", wide <= lo), ("mid", (wide > lo) & (wide < hi)),
                 ("wide spread (confident)", wide >= hi)]:
    print(f"  {label:26s} n={m.sum():3d}   mean rerank gain {gain[m].mean():+.4f}")
