"""Wall clock for length-sorted batching, tokenizer padding disabled.

Token waste and TIME waste are different numbers: cost is superlinear in
sequence length, so padding a short row up to a long batch maximum costs more
than the token count suggests.
"""
import json, time, numpy as np, onnxruntime as ort
import os, sys
sys.path.insert(0, os.environ.get("SEMANTIC_SEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import semantic_search as ss
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

REPO = "Xenova/all-MiniLM-L6-v2"
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
tok.no_truncation(); tok.no_padding()
so = ort.SessionOptions(); so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(hf_hub_download(REPO, "onnx/model_quantized.onnx"), so,
                            providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}

class Ad:
    def __call__(self, t, add_special_tokens=False, return_offsets_mapping=True):
        e = tok.encode(t, add_special_tokens=add_special_tokens)
        return {"input_ids": e.ids, "offset_mapping": e.offsets}

def run(encs, B=64):
    t0 = time.perf_counter()
    for s in range(0, len(encs), B):
        b = encs[s:s+B]; m = max(len(e) for e in b)
        ids = np.zeros((len(b), m), dtype=np.int64); mask = np.zeros_like(ids)
        for i, e in enumerate(b): ids[i, :len(e)] = e; mask[i, :len(e)] = 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        sess.run(None, {k: v for k, v in feed.items() if k in names})
    return time.perf_counter() - t0

corpus = [json.loads(l) for l in open(json.load(open("scifact_paths.json"))["corpus.jsonl"])]
docs = [(d.get("title","") + ". " + d.get("text","")).strip() for d in corpus]
sets = {"SciFact 200/30 (5,183 short docs)": [c for d in docs for c in ss.chunk_text(d, Ad())],
        "DDIA 200/30 (one 613-page book)":   json.load(open("ddia_chunks.json"))}

print(f"{'corpus':36s} {'tok waste':>10s} {'natural s':>10s} {'sorted s':>9s} {'time saved':>11s}")
print("-" * 82)
for label, chunks in sets.items():
    encs = [tok.encode(c, add_special_tokens=True).ids for c in chunks]
    real = sum(len(e) for e in encs)
    pad = sum(max(len(e) for e in encs[s:s+64]) * len(encs[s:s+64]) for s in range(0, len(encs), 64))
    srt = sorted(encs, key=len)
    tn = run(encs); tsv = run(srt)
    print(f"{label:36s} {100*(pad-real)/real:9.1f}% {tn:10.1f} {tsv:9.1f} {100*(tn-tsv)/tn:10.1f}%")
