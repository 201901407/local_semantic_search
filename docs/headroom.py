"""How much of the AVAILABLE gain does the cross-encoder capture on each corpus?

An oracle rerank orders the same top-30 pool perfectly. That is the ceiling of a
rerank stage -- it cannot retrieve anything the bi-encoder missed. Expressing the
real gain as a fraction of (oracle - baseline) makes the two corpora comparable
even though their baselines differ.
"""
import json, numpy as np, evallib as E

# ---- SciFact -----------------------------------------------------------------
S = np.load("qry_fp32.npy") @ np.load("doc_fp32.npy").T
pool, ce = np.load("pool.npy"), np.load("ce_scores.npy")
N = 30
def sci_out(scores):
    F = S.copy(); o = np.argsort(-scores[:, :N], axis=1)
    for i in range(S.shape[0]): F[i, pool[i, :N][o[i]]] = 2.0 + np.arange(N, 0, -1) / N
    return E.maxpool(F)
# Oracle: score each candidate chunk by its document's relevance grade.
oracle = np.zeros((S.shape[0], N), dtype=np.float32)
for i, qid in enumerate(E.qids):
    g = E.rel.get(qid, {})
    oracle[i] = [g.get(E.docs[E.cdoc[c]], 0) for c in pool[i, :N]]
sci_base = E.per_query_ndcg(E.maxpool(S)).mean()
sci_ce = E.per_query_ndcg(sci_out(ce)).mean()
sci_or = E.per_query_ndcg(sci_out(oracle)).mean()

# ---- DDIA --------------------------------------------------------------------
cb = np.array(json.load(open("ddia_meta.json"))["chunk_block"])
ev = json.load(open("ddia_eval_set.json")); gold = np.array(ev["blocks"])
dS, dpool, dce = np.load("ddia_S.npy"), np.load("ddia_pool.npy"), np.load("ddia_ce.npy")
nb = int(cb.max()) + 1
def dnd(F):
    out = np.full((F.shape[0], nb), -1e9, np.float32); np.maximum.at(out.T, cb, F.T)
    top = np.argsort(-out, axis=1)[:, :10]; v = np.zeros(len(gold))
    for i, g in enumerate(gold):
        w = np.where(top[i] == g)[0]
        if len(w): v[i] = 1 / np.log2(int(w[0]) + 2)
    return v.mean()
def d_out(scores):
    F = dS.copy(); o = np.argsort(-scores[:, :N], axis=1)
    for i in range(len(gold)): F[i, dpool[i, :N][o[i]]] = 2.0 + np.arange(N, 0, -1) / N
    return F
d_oracle = np.array([[1.0 if cb[c] == gold[i] else 0.0 for c in dpool[i, :N]]
                     for i in range(len(gold))], dtype=np.float32)
d_base, d_ce, d_or = dnd(dS), dnd(d_out(dce)), dnd(d_out(d_oracle))

print(f"{'corpus':10s} {'baseline':>9s} {'reranked':>9s} {'oracle':>8s} {'headroom':>9s} {'captured':>9s}")
print("-" * 60)
for name, b, c, o in [("SciFact", sci_base, sci_ce, sci_or), ("DDIA", d_base, d_ce, d_or)]:
    print(f"{name:10s} {b:9.4f} {c:9.4f} {o:8.4f} {o-b:9.4f} {100*(c-b)/(o-b):8.1f}%")
