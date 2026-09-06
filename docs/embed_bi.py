"""Bi-encoder embeddings: 12,815 SciFact chunks + 300 test queries."""
import json, time, numpy as np
from sentence_transformers import SentenceTransformer

chunk_texts = json.load(open("chunk_texts.json"))
paths = json.load(open("scifact_paths.json"))
queries = [json.loads(l) for l in open(paths["queries.jsonl"])]
qrels = [json.loads(l) for l in open(paths["qrels/test.jsonl"])]
test_qids = {r["query-id"] for r in qrels}
qs = [q for q in queries if q["_id"] in test_qids]
json.dump([q["_id"] for q in qs], open("qids.json", "w"))
json.dump([q["text"] for q in qs], open("qtexts.json", "w"))

st = SentenceTransformer("all-MiniLM-L6-v2")
t0 = time.time()
D = st.encode(chunk_texts, normalize_embeddings=True, batch_size=64).astype(np.float32)
print(f"docs {time.time()-t0:.1f}s  {len(chunk_texts)/(time.time()-t0):.1f} chunks/s", flush=True)
Q = st.encode([q["text"] for q in qs], normalize_embeddings=True).astype(np.float32)
np.save("doc_fp32.npy", D); np.save("qry_fp32.npy", Q)
print("EMBED_DONE", D.shape, Q.shape, flush=True)
