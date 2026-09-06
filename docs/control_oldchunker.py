"""Control: does the chunk_text offset fix move retrieval at all?

The rebuild reproduced nDCG 0.6506 where the archived run recorded 0.6485. The
only pipeline change since is chunk_text, so reconstruct the OLD decode-based
chunks and score them the same way. Same boundaries, different stored strings.
"""
import json, numpy as np
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import evallib as E

tok = Tokenizer.from_file(hf_hub_download("sentence-transformers/all-MiniLM-L6-v2", "tokenizer.json"))
tok.no_truncation()
corpus = [json.loads(l) for l in open(json.load(open("scifact_paths.json"))["corpus.jsonl"])]

old_texts, old_doc = [], []
MAX, OV = 200, 30
for d in corpus:
    text = " ".join(((d.get("title","") + ". " + d.get("text","")).strip()).split())
    ids = tok.encode(text, add_special_tokens=False).ids
    start, step = 0, MAX - OV
    while start < len(ids):
        old_texts.append(tok.decode(ids[start:start + MAX]))   # the old behaviour
        old_doc.append(d["_id"]); start += step

new_texts = json.load(open("chunk_texts.json"))
assert old_doc == E.chunk_doc, f"boundaries differ: {len(old_doc)} vs {len(E.chunk_doc)}"
identical = sum(a == b for a, b in zip(old_texts, new_texts))
upper_old = sum(c.isupper() for t in old_texts for c in t)
upper_new = sum(c.isupper() for t in new_texts for c in t)
print(f"chunks {len(old_texts):,}  byte-identical {identical:,} ({100*identical/len(old_texts):.1f}%)")
print(f"uppercase chars   old {upper_old:,}   new {upper_new:,}")

st = SentenceTransformer("all-MiniLM-L6-v2")
Dold = st.encode(old_texts, normalize_embeddings=True, batch_size=64).astype(np.float32)
np.save("doc_fp32_oldchunker.npy", Dold)
Q = np.load("qry_fp32.npy"); Dnew = np.load("doc_fp32.npy")

old_out, new_out = E.maxpool(Q @ Dold.T), E.maxpool(Q @ Dnew.T)
print("%-22s nDCG %.4f  Recall %.4f  MRR %.4f" % (("old (decode)",) + E.summary(old_out)))
print("%-22s nDCG %.4f  Recall %.4f  MRR %.4f" % (("new (offsets)",) + E.summary(new_out)))
d = E.per_query_ndcg(new_out) - E.per_query_ndcg(old_out)
lo, hi, p = E.bootstrap(d)
print(f"delta nDCG@10 {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}")
