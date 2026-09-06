"""Is the apparent peak at N=20-30 real, or is the depth curve flat?

Also: how often does reranking make an individual query worse? 'Only reorders
passages that exist' bounds the failure mode but does not make it zero.
"""
import json, numpy as np, evallib as E
qtexts = json.load(open("qtexts.json"))
S = np.load("qry_fp32.npy") @ np.load("doc_fp32.npy").T
pool = np.load("pool.npy"); ce = np.load("ce_scores.npy")
base = E.per_query_ndcg(E.maxpool(S))

def nd(n):
    F = S.copy(); order = np.argsort(-ce[:, :n], axis=1)
    for i in range(len(qtexts)):
        F[i, pool[i, :n][order[i]]] = 2.0 + np.arange(n, 0, -1) / n
    return E.per_query_ndcg(E.maxpool(F))

v = {n: nd(n) for n in (10, 20, 30, 50, 100)}
print("Pairwise depth comparisons (paired bootstrap, 10,000 resamples)")
print("-" * 62)
for a, b in [(30, 10), (30, 20), (30, 50), (30, 100), (20, 100)]:
    d = v[a] - v[b]; lo, hi, p = E.bootstrap(d)
    print(f"N={a:3d} vs N={b:3d}   {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}")

d30 = v[30] - base
print(f"\nPer-query effect at N=30 (n={len(d30)} queries)")
print(f"  improved   {(d30 > 1e-9).sum():3d}  mean gain {d30[d30 > 1e-9].mean():+.4f}")
print(f"  unchanged  {(np.abs(d30) <= 1e-9).sum():3d}")
print(f"  worsened   {(d30 < -1e-9).sum():3d}  mean loss {d30[d30 < -1e-9].mean():+.4f}")
print(f"  worst single query {d30.min():+.4f}   best {d30.max():+.4f}")
