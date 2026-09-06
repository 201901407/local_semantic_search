"""Prediction, then measurement: does chunk size change INDEXING time?

From idx_scaling.py, cost per batch of 32 at uniform length L:
    T(L) = 0.001580*L^2 + 0.8704*L - 0.855   ms
Per-token cost therefore RISES with L (27.5 us at L=16 -> 39.7 us at L=256), so
smaller chunks are cheaper per token -- but overlap makes you process each token
A = C/(C-V) times, and every chunk pays 2 extra tokens for [CLS]/[SEP].

Predict each config from that model, then measure wall clock and compare.
"""
import json, time, numpy as np, onnxruntime as ort
import os, sys
sys.path.insert(0, os.environ.get("SEMANTIC_SEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import semantic_search as ss
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

REPO = "Xenova/all-MiniLM-L6-v2"
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json")); tok.no_truncation()
so = ort.SessionOptions(); so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(hf_hub_download(REPO, "onnx/model_quantized.onnx"), so,
                            providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}
BATCH = 32
Tms = lambda L: 0.001580 * L * L + 0.8704 * L - 0.855      # per batch of 32

class Ad:
    def __call__(self, t, add_special_tokens=False, return_offsets_mapping=True):
        e = tok.encode(t, add_special_tokens=add_special_tokens)
        return {"input_ids": e.ids, "offset_mapping": e.offsets}

text = " ".join(open("/dev/stdin").read().split()) if False else None
blocks = json.load(open("ddia_blocks.json")); text = "".join(blocks)
total_tokens = len(tok.encode(text, add_special_tokens=False).ids)
print(f"DDIA: {len(text):,} chars, {total_tokens:,} tokens\n")

CONFIGS = [(64, 0), (128, 0), (128, 30), (200, 30), (256, 0), (256, 50)]
print("%-10s %7s %9s %10s %11s %10s %9s" %
      ("chunk/ovl", "chunks", "tokens", "amplif", "predicted s", "actual s", "err"))
print("-" * 74)
results = []
for C, V in CONFIGS:
    chunks = ss.chunk_text(text, Ad(), max_tokens=C, overlap_tokens=V)
    encs = [tok.encode(c, add_special_tokens=True).ids[:C + 2] for c in chunks]
    fed = sum(len(e) for e in encs)
    # Predict: batches keep source order, each pads to its longest member.
    pred = sum(Tms(max(len(e) for e in encs[s:s+BATCH])) for s in range(0, len(encs), BATCH)) / 1000

    t0 = time.perf_counter()
    for s in range(0, len(encs), BATCH):
        batch = encs[s:s+BATCH]; m = max(len(e) for e in batch)
        ids = np.zeros((len(batch), m), dtype=np.int64); mask = np.zeros_like(ids)
        for i, e in enumerate(batch):
            ids[i, :len(e)] = e; mask[i, :len(e)] = 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        sess.run(None, {k: v for k, v in feed.items() if k in names})
    actual = time.perf_counter() - t0
    amp = fed / total_tokens
    results.append((C, V, len(chunks), fed, amp, pred, actual))
    print("%-10s %7d %9s %10.3f %11.1f %10.1f %8.1f%%" %
          (f"{C}/{V}", len(chunks), f"{fed:,}", amp, pred, actual, 100*(pred-actual)/actual))

best = min(results, key=lambda r: r[6])
worst = max(results, key=lambda r: r[6])
print(f"\nfastest {best[0]}/{best[1]} at {best[6]:.1f}s;  slowest {worst[0]}/{worst[1]} "
      f"at {worst[6]:.1f}s  ({worst[6]/best[6]:.2f}x)")
json.dump([[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in results], open("idx_cost.json", "w"))
