"""Controls for the rerank result, and the int8 model that would actually ship.

1. Identity rerank: reorder the head using the BI-ENCODER's own scores. If the
   'lift the head above the tail' transformation is itself doing work, this
   diverges from the baseline and part of the measured gain is an artifact.
2. Random rerank: shuffle the head. Should hurt; if it helps, the metric is wrong.
3. int8 cross-encoder: the 23.0 MB file the browser would download.
"""
import json, time, numpy as np, evallib as E

chunk_texts = json.load(open("chunk_texts.json")); qtexts = json.load(open("qtexts.json"))
D = np.load("doc_fp32.npy"); Q = np.load("qry_fp32.npy")
S = Q @ D.T
pool = np.load("pool.npy"); ce_fp32 = np.load("ce_scores.npy")
base_out = E.maxpool(S); base = E.per_query_ndcg(base_out)
N = 30

def rerank_out(scores, n=N):
    F = S.copy()
    order = np.argsort(-scores[:, :n], axis=1)
    for i in range(len(qtexts)):
        F[i, pool[i, :n][order[i]]] = 2.0 + np.arange(n, 0, -1) / n
    return E.maxpool(F)

def row(label, out):
    v = E.per_query_ndcg(out); d = v - base
    lo, hi, p = E.bootstrap(d)
    print("%-34s %.4f  %+8.4f  [%+.4f, %+.4f] %7.3f" % ((label,) + (v.mean(), d.mean(), lo, hi, p)))

print("%-34s %-7s %9s %22s %7s" % ("control", "nDCG", "delta", "95% CI", "p"))
print("-" * 84)
print("%-34s %.4f" % ("bi-encoder only (baseline)", base.mean()))
own = np.take_along_axis(S, pool, axis=1)                 # bi-encoder's own order
row("identity rerank (must be +0.0000)", rerank_out(own))
rng = np.random.default_rng(0)
row("random rerank (must be worse)", rerank_out(rng.random(ce_fp32.shape).astype(np.float32)))

# --- the model that would ship -------------------------------------------------
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
REPO = "Xenova/ms-marco-MiniLM-L-6-v2"
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
tok.enable_truncation(max_length=256); tok.enable_padding()
path = hf_hub_download(REPO, "onnx/model_quantized.onnx")
print(f"\nint8 cross-encoder on disk: {__import__('os').path.getsize(path)/1e6:.1f} MB")
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}

pairs = [(qtexts[i], chunk_texts[c]) for i in range(len(qtexts)) for c in pool[i][:N]]
out, t0 = [], time.time()
for s in range(0, len(pairs), 64):
    encs = tok.encode_batch([(a, b) for a, b in pairs[s:s+64]])
    feed = {"input_ids": np.array([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64)}
    out.append(sess.run(None, {k: v for k, v in feed.items() if k in names})[0].reshape(-1))
elapsed = time.time() - t0
ce_int8 = np.concatenate(out).astype(np.float32).reshape(len(qtexts), N)
print(f"int8 CPU throughput: {len(pairs):,} pairs in {elapsed:.1f}s = {len(pairs)/elapsed:.0f} pairs/s")
print(f"per-query rerank of {N} chunks: {N/(len(pairs)/elapsed)*1000:.0f} ms native CPU\n")

print("%-34s %-7s %9s %22s %7s" % ("shipping variant", "nDCG", "delta", "95% CI", "p"))
print("-" * 84)
row(f"fp32 cross-encoder, top-{N}", rerank_out(ce_fp32))
row(f"int8 cross-encoder, top-{N}", rerank_out(ce_int8))
r = np.mean([np.corrcoef(ce_fp32[i, :N], ce_int8[i])[0, 1] for i in range(len(qtexts))])
print(f"\nfp32 vs int8 per-query score correlation: {r:.4f}")
np.save("ce_int8.npy", ce_int8)
