"""Shared ONNX / torch embedding helpers replicating sentence-transformers'
all-MiniLM-L6-v2 pipeline: tokenize -> transformer -> mean-pool over the
attention mask -> L2 normalize."""
import json, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer

MODEL_PATHS = json.load(open("model_paths.json"))
_tok = None

def tokenizer():
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(MODEL_PATHS["tokenizer.json"])
        _tok.enable_truncation(max_length=256)
    return _tok

def make_session(variant):
    p = MODEL_PATHS[f"onnx/{variant}.onnx"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

def onnx_encode(sess, texts, batch_size=64):
    tok = tokenizer()
    in_names = {i.name for i in sess.get_inputs()}
    fp16 = any(i.type == "tensor(float16)" for i in sess.get_inputs())
    out = []
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s + batch_size]
        encs = tok.encode_batch(batch)
        maxlen = max(len(e.ids) for e in encs)
        ids  = np.zeros((len(encs), maxlen), dtype=np.int64)
        mask = np.zeros((len(encs), maxlen), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.attention_mask)] = e.attention_mask
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in in_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in in_names}
        hidden = sess.run(None, feed)[0].astype(np.float32)
        m = mask[..., None].astype(np.float32)
        emb = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        out.append(emb.astype(np.float32))
    return np.vstack(out)
