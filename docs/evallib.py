"""Shared SciFact scoring: chunk scores -> per-doc max-pool -> nDCG/Recall/MRR@10."""
import json, numpy as np

K = 10
_paths = json.load(open("scifact_paths.json"))
_qrels = [json.loads(l) for l in open(_paths["qrels/test.jsonl"])]
rel = {}
for r in _qrels:
    rel.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
qids = json.load(open("qids.json"))
chunk_doc = json.load(open("chunks_meta.json"))["chunk_doc"]
docs = sorted(set(chunk_doc))
_didx = {d: i for i, d in enumerate(docs)}
cdoc = np.array([_didx[d] for d in chunk_doc])

def maxpool(S):
    """(nq, nchunks) chunk scores -> (nq, ndocs), 'best chunk wins' per document."""
    out = np.full((S.shape[0], len(docs)), -1e9, dtype=np.float32)
    np.maximum.at(out.T, cdoc, S.T)
    return out

def per_query_ndcg(out):
    top = np.argsort(-out, axis=1)[:, :K]
    vals = []
    for i, qid in enumerate(qids):
        g = rel.get(qid, {})
        gains = [g.get(docs[j], 0) for j in top[i]]
        dcg = sum((2**gn - 1) / np.log2(r + 2) for r, gn in enumerate(gains))
        ideal = sorted(g.values(), reverse=True)[:K]
        idcg = sum((2**gn - 1) / np.log2(r + 2) for r, gn in enumerate(ideal))
        vals.append(dcg / idcg if idcg else 0.0)
    return np.array(vals)

def summary(out):
    top = np.argsort(-out, axis=1)[:, :K]
    rc = mrr = 0.0
    for i, qid in enumerate(qids):
        g = rel.get(qid, {})
        if not g: continue
        ranked = [docs[j] for j in top[i]]
        rc += sum(1 for d in ranked if g.get(d, 0) > 0) / len(g)
        for r, d in enumerate(ranked):
            if g.get(d, 0) > 0:
                mrr += 1 / (r + 1); break
    n = len(qids)
    return per_query_ndcg(out).mean(), rc / n, mrr / n

def int8(M):
    s = np.abs(M).max() / 127.0
    R = np.round(M / s).astype(np.int8).astype(np.float32) * s
    return R / np.clip(np.linalg.norm(R, axis=1, keepdims=True), 1e-12, None)

def bootstrap(delta, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), (n, len(delta)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return lo, hi, p
