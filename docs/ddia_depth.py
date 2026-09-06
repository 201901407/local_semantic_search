"""Pairwise depth tests on DDIA, plus the int8 model that would actually ship."""
import json, time, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

K = 10
chunks = json.load(open("ddia_chunks.json"))
chunk_block = np.array(json.load(open("ddia_meta.json"))["chunk_block"])
ev = json.load(open("ddia_eval_set.json")); gold = np.array(ev["blocks"]); qtexts = ev["queries"]
nblocks = int(chunk_block.max()) + 1
S = np.load("ddia_S.npy"); pool = np.load("ddia_pool.npy"); ce = np.load("ddia_ce.npy")

def maxpool(F):
    out = np.full((F.shape[0], nblocks), -1e9, dtype=np.float32)
    np.maximum.at(out.T, chunk_block, F.T); return out

def ndcg(out):
    top = np.argsort(-out, axis=1)[:, :K]; v = np.zeros(len(gold))
    for i, g in enumerate(gold):
        w = np.where(top[i] == g)[0]
        if len(w): v[i] = 1 / np.log2(int(w[0]) + 2)
    return v

def rerank(sc, n):
    F = S.copy(); o = np.argsort(-sc[:, :n], axis=1)
    for i in range(len(qtexts)): F[i, pool[i, :n][o[i]]] = 2.0 + np.arange(n, 0, -1) / n
    return maxpool(F)

def boot(d, N=10000, seed=0):
    rng = np.random.default_rng(seed); b = d[rng.integers(0, len(d), (N, len(d)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5]); return lo, hi, 2 * min((b <= 0).mean(), (b >= 0).mean())

v = {n: ndcg(rerank(ce, n)) for n in (10, 20, 30, 50, 100)}
print("Pairwise depth comparisons on DDIA (91 queries)")
print("-" * 62)
for a, b in [(30, 10), (30, 20), (30, 50), (30, 100), (20, 100)]:
    d = v[a] - v[b]; lo, hi, p = boot(d)
    print(f"N={a:3d} vs N={b:3d}   {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}")

REPO = "Xenova/ms-marco-MiniLM-L-6-v2"
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
tok.enable_truncation(max_length=256); tok.enable_padding()
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(hf_hub_download(REPO, "onnx/model_quantized.onnx"), so,
                            providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}
N = 30
pairs = [(qtexts[i], chunks[c]) for i in range(len(qtexts)) for c in pool[i][:N]]
out, t0 = [], time.time()
for s in range(0, len(pairs), 64):
    e = tok.encode_batch(list(pairs[s:s+64]))
    feed = {"input_ids": np.array([x.ids for x in e], dtype=np.int64),
            "attention_mask": np.array([x.attention_mask for x in e], dtype=np.int64),
            "token_type_ids": np.array([x.type_ids for x in e], dtype=np.int64)}
    out.append(sess.run(None, {k: x for k, x in feed.items() if k in names})[0].reshape(-1))
el = time.time() - t0
ce8 = np.concatenate(out).astype(np.float32).reshape(len(qtexts), N)
base = ndcg(maxpool(S))
print(f"\nint8 native CPU: {len(pairs):,} pairs in {el:.1f}s = {len(pairs)/el:.0f} pairs/s")
for label, sc in [("fp32", ce), ("int8", ce8)]:
    d = ndcg(rerank(sc, N)) - base; lo, hi, p = boot(d)
    print(f"  {label} top-{N}: nDCG {(d+base).mean():.4f}  {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}")
print(f"  fp32 vs int8 per-query score correlation: "
      f"{np.mean([np.corrcoef(ce[i,:N], ce8[i])[0,1] for i in range(len(qtexts))]):.4f}")
