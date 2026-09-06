"""Chunk the SciFact corpus with the project's own chunk_text() (200/30)."""
import json, statistics
import os, sys
sys.path.insert(0, os.environ.get("SEMANTIC_SEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import semantic_search as ss
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

tok = Tokenizer.from_file(hf_hub_download("sentence-transformers/all-MiniLM-L6-v2", "tokenizer.json"))
tok.no_truncation()

class TokAdapter:
    """chunk_text() now slices by character offsets, so the adapter must expose
    __call__ returning an offset_mapping -- the old encode/decode pair is stale."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        e = tok.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": e.ids, "offset_mapping": e.offsets}
    def encode(self, text, add_special_tokens=False):
        return tok.encode(text, add_special_tokens=add_special_tokens).ids

ad = TokAdapter()
corpus = [json.loads(l) for l in open(json.load(open("scifact_paths.json"))["corpus.jsonl"])]

chunk_texts, chunk_doc, per_doc = [], [], []
for d in corpus:
    text = (d.get("title","") + ". " + d.get("text","")).strip()
    cs = ss.chunk_text(text, ad)
    per_doc.append(len(cs))
    for c in cs:
        chunk_texts.append(c); chunk_doc.append(d["_id"])

json.dump({"chunk_doc": chunk_doc}, open("chunks_meta.json","w"))
json.dump(chunk_texts, open("chunk_texts.json","w"))
print(f"docs {len(corpus):,}  chunks {len(chunk_texts):,}  "
      f"chunks/doc mean {statistics.mean(per_doc):.2f} max {max(per_doc)}")
