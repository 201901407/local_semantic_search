"""Is the embedding forward pass linear or quadratic in sequence length?

Everyone "knows" transformers are O(L^2). For THIS model that intuition may be
wrong at the lengths we use. Per layer, roughly:
    attention   ~ L^2 * d          feed-forward ~ L * d^2
With d=384 and L<=256 the ratio is L/d = 0.67 at most, so the linear term should
dominate -- meaning cost tracks TOTAL TOKENS, not how they are cut into chunks.

Measured with uniform-length batches so batch padding cannot contaminate it.
"""
import json, time, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

REPO = "Xenova/all-MiniLM-L6-v2"
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json")); tok.no_truncation()
so = ort.SessionOptions(); so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(hf_hub_download(REPO, "onnx/model_quantized.onnx"), so,
                            providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}
text = " ".join(open("ddia_chunks.json").read().split())
ids_all = tok.encode(text, add_special_tokens=False).ids

BATCH, REPS = 32, 7
print(f"ONNX int8, 4 threads, batch={BATCH}, uniform lengths (no padding waste)\n")
print(f"{'L':>5s} {'ms/batch':>10s} {'ms/chunk':>10s} {'us/token':>10s} {'chunks/s':>10s}")
print("-" * 50)
rows = []
for L in [16, 32, 64, 96, 128, 160, 192, 224, 256]:
    ids = np.array([ids_all[i*L:(i+1)*L] for i in range(BATCH)], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": np.ones_like(ids),
            "token_type_ids": np.zeros_like(ids)}
    feed = {k: v for k, v in feed.items() if k in names}
    sess.run(None, feed)                                  # warm up
    ts = [ (lambda t0: (sess.run(None, feed), time.perf_counter()-t0)[1])(time.perf_counter())
           for _ in range(REPS) ]
    ms = np.median(ts) * 1000
    rows.append((L, ms))
    print(f"{L:5d} {ms:10.1f} {ms/BATCH:10.3f} {ms*1000/(BATCH*L):10.2f} {BATCH/(ms/1000):10.1f}")

L = np.array([r[0] for r in rows], float); T = np.array([r[1] for r in rows], float)
lin = np.polyfit(L, T, 1); quad = np.polyfit(L, T, 2)
r2 = lambda p: 1 - ((T - np.polyval(p, L))**2).sum() / ((T - T.mean())**2).sum()
print(f"\nlinear fit    T = {lin[0]:.4f}*L + {lin[1]:.3f}      R^2 = {r2(lin):.5f}")
print(f"quadratic fit T = {quad[0]:.6f}*L^2 + {quad[1]:.4f}*L + {quad[2]:.3f}   R^2 = {r2(quad):.5f}")
share = quad[0]*256**2 / (quad[0]*256**2 + quad[1]*256) * 100
print(f"quadratic term's share of cost at L=256: {share:.1f}%")
print(f"us/token at L=16 vs L=256: {T[0]*1000/(BATCH*16):.2f} -> {T[-1]*1000/(BATCH*256):.2f}")
