import json, time, numpy as np, onnxruntime as ort, embed_lib as E
from tokenizers import Tokenizer

chunk_texts = json.load(open("chunk_texts.json"))
paths = json.load(open("scifact_paths.json"))
queries = [json.loads(l) for l in open(paths["queries.jsonl"])]
qrels   = [json.loads(l) for l in open(paths["qrels/test.jsonl"])]
test_qids = {r["query-id"] for r in qrels}
qs = [q for q in queries if q["_id"] in test_qids]
qtexts = [q["text"] for q in qs]
json.dump([q["_id"] for q in qs], open("qids.json","w"))
print(f"chunks={len(chunk_texts)}  test_queries={len(qs)}", flush=True)

# torch fp32 reference
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("all-MiniLM-L6-v2")
t0=time.time(); D = st.encode(chunk_texts, normalize_embeddings=True, batch_size=64,
                              show_progress_bar=False).astype(np.float32)
dt=time.time()-t0
Q = st.encode(qtexts, normalize_embeddings=True).astype(np.float32)
np.save("doc_torch_fp32.npy", D); np.save("qry_torch_fp32.npy", Q)
print(f"torch_fp32       docs {dt:7.1f}s  {len(chunk_texts)/dt:7.1f} chunks/s", flush=True)

E.tokenizer().enable_truncation(max_length=256)
for v in ["model","model_fp16","model_quantized","model_uint8","model_q4"]:
    try:
        try:
            s = E.make_session(v)
        except Exception:  # fp16 hits an ORT fusion bug at ORT_ENABLE_ALL
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            so.intra_op_num_threads = 4
            s = ort.InferenceSession(E.MODEL_PATHS[f"onnx/{v}.onnx"], so,
                                     providers=["CPUExecutionProvider"])
            print(f"  ({v}: fell back to ORT_ENABLE_BASIC)", flush=True)
        t0=time.time(); d = E.onnx_encode(s, chunk_texts); dt=time.time()-t0
        q = E.onnx_encode(s, qtexts)
        np.save(f"doc_{v}.npy", d); np.save(f"qry_{v}.npy", q)
        print(f"{v:16s} docs {dt:7.1f}s  {len(chunk_texts)/dt:7.1f} chunks/s", flush=True)
    except Exception as ex:
        print(f"{v:16s} ERR {type(ex).__name__}: {str(ex)[:120]}", flush=True)
print("SWEEP_DONE", flush=True)
