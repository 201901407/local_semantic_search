"""Does the cross-encoder gain from Experiment E survive on a real book?

Same pipeline as SciFact: bi-encoder retrieves top-N chunks, cross-encoder
rescores them, the reranked head sits above the un-reranked tail, scores are
max-pooled to the 1,000-character reference block, one relevant block per query.
"""
import json, time, numpy as np, torch
from sentence_transformers import SentenceTransformer, CrossEncoder

K, DEPTHS, MAX_N = 10, [10, 20, 30, 50, 100], 100
chunks = json.load(open("ddia_chunks.json"))
chunk_block = np.array(json.load(open("ddia_meta.json"))["chunk_block"])
ev = json.load(open("ddia_eval_set.json"))
gold, qtexts = np.array(ev["blocks"]), ev["queries"]
nblocks = int(chunk_block.max()) + 1

st = SentenceTransformer("all-MiniLM-L6-v2")
D = st.encode(chunks, normalize_embeddings=True, batch_size=64).astype(np.float32)
Q = st.encode(qtexts, normalize_embeddings=True).astype(np.float32)
S = Q @ D.T
pool = np.argsort(-S, axis=1)[:, :MAX_N]

def maxpool(F):
    out = np.full((F.shape[0], nblocks), -1e9, dtype=np.float32)
    np.maximum.at(out.T, chunk_block, F.T)
    return out

def metrics(out):
    """One relevant block per query, so nDCG@10 reduces to a log-discounted MRR."""
    top = np.argsort(-out, axis=1)[:, :K]
    hit1 = np.zeros(len(gold)); rr = np.zeros(len(gold)); nd = np.zeros(len(gold))
    for i, g in enumerate(gold):
        where = np.where(top[i] == g)[0]
        if len(where):
            r = int(where[0])
            hit1[i] = r == 0; rr[i] = 1 / (r + 1); nd[i] = 1 / np.log2(r + 2)
    return hit1, rr, nd

def rerank(scores, n):
    F = S.copy(); order = np.argsort(-scores[:, :n], axis=1)
    for i in range(len(qtexts)):
        F[i, pool[i, :n][order[i]]] = 2.0 + np.arange(n, 0, -1) / n
    return maxpool(F)

def boot(d, N=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (N, len(d)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return lo, hi, 2 * min((b <= 0).mean(), (b >= 0).mean())

print(f"queries {len(qtexts)}  chunks {len(chunks):,}  blocks {nblocks:,}")
for n in DEPTHS:
    found = sum(1 for i in range(len(gold)) if gold[i] in chunk_block[pool[i, :n]])
    print(f"  pool N={n:3d} -> gold block present for {found}/{len(gold)} queries ({found/len(gold):.3f})")

device = "mps" if torch.backends.mps.is_available() else "cpu"
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device, max_length=256)
pairs = [[qtexts[i], chunks[c]] for i in range(len(qtexts)) for c in pool[i]]
t0 = time.time()
ce_scores = np.asarray(ce.predict(pairs, batch_size=64), dtype=np.float32).reshape(len(qtexts), MAX_N)
print(f"\ncross-encoder {len(pairs):,} pairs in {time.time()-t0:.1f}s\n")
np.save("ddia_ce.npy", ce_scores); np.save("ddia_pool.npy", pool); np.save("ddia_S.npy", S)

base_out = maxpool(S); bh, br, bn = metrics(base_out)
print("%-24s %-7s %-8s %-8s %10s %24s %7s" % ("variant","Hit@1","MRR@10","nDCG@10","delta nDCG","95% CI","p"))
print("-" * 92)
print("%-24s %.4f  %.4f   %.4f" % ("bi-encoder only", bh.mean(), br.mean(), bn.mean()))
for n in DEPTHS:
    h, r, nd = metrics(rerank(ce_scores, n))
    lo, hi, p = boot(nd - bn)
    print("%-24s %.4f  %.4f   %.4f  %+10.4f  [%+.4f, %+.4f] %7.3f" %
          (f"+ rerank top-{n}", h.mean(), r.mean(), nd.mean(), (nd-bn).mean(), lo, hi, p))

# Controls, same as SciFact.
own = np.take_along_axis(S, pool, axis=1)
h, r, nd = metrics(rerank(own, 30)); lo, hi, p = boot(nd - bn)
print("\n%-24s %.4f  %.4f   %.4f  %+10.4f   (identity control)" % ("identity rerank", h.mean(), r.mean(), nd.mean(), (nd-bn).mean()))
rng = np.random.default_rng(0)
h, r, nd = metrics(rerank(rng.random(ce_scores.shape).astype(np.float32), 30))
print("%-24s %.4f  %.4f   %.4f  %+10.4f   (random control)" % ("random rerank", h.mean(), r.mean(), nd.mean(), (nd-bn).mean()))

_, _, nd30 = metrics(rerank(ce_scores, 30)); d = nd30 - bn
print(f"\nper-query at N=30: improved {(d>1e-9).sum()}  unchanged {(np.abs(d)<=1e-9).sum()}  worsened {(d<-1e-9).sum()}")
