"""Thread scaling — does cross-origin isolation earn the COOP/COEP complexity?

The app sets numThreads = min(hardwareConcurrency, 8), but ONLY when the page is
crossOriginIsolated; otherwise WASM silently drops to one thread. That header
work is only worth doing if threads actually buy throughput.
"""
import json, time, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

REPO = "Xenova/all-MiniLM-L6-v2"
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
tok.no_truncation(); tok.no_padding()
chunks = json.load(open("ddia_chunks.json"))
encs = [tok.encode(c, add_special_tokens=True).ids for c in chunks]
path = hf_hub_download(REPO, "onnx/model_quantized.onnx")

def run(threads, B=64):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    t0 = time.perf_counter()
    for s in range(0, len(encs), B):
        b = encs[s:s+B]; m = max(len(e) for e in b)
        ids = np.zeros((len(b), m), dtype=np.int64); mask = np.zeros_like(ids)
        for i, e in enumerate(b): ids[i, :len(e)] = e; mask[i, :len(e)] = 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        sess.run(None, {k: v for k, v in feed.items() if k in names})
    return time.perf_counter() - t0

print(f"DDIA, {len(encs):,} chunks, ONNX int8\n")
print(f"{'threads':>8s} {'seconds':>9s} {'chunks/s':>10s} {'speedup':>9s} {'efficiency':>11s}")
print("-" * 52)
base = None
for t in [1, 2, 4, 6, 8, 10]:
    el = run(t)
    if base is None: base = el
    print(f"{t:8d} {el:9.1f} {len(encs)/el:10.1f} {base/el:8.2f}x {100*base/el/t:10.0f}%")
