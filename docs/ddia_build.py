"""Rebuild the DDIA corpus: extract, chunk at 200/30, and cut the fixed
1,000-character reference blocks that ground truth is scored against.

Blocks are independent of chunking on purpose. Each retrieved chunk maps to
exactly one block -- the one containing its midpoint -- so a larger chunk cannot
score better merely by covering more text.
"""
import json, statistics
import os, sys
sys.path.insert(0, os.environ.get("SEMANTIC_SEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
import semantic_search as ss
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

BLOCK = 1000
tok = Tokenizer.from_file(hf_hub_download("sentence-transformers/all-MiniLM-L6-v2", "tokenizer.json"))
tok.no_truncation()

class TokAdapter:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        e = tok.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": e.ids, "offset_mapping": e.offsets}

raw = ss.extract_text(Path(os.environ["DDIA_PDF"]))   # path to your own copy
text = " ".join(raw.split())
chunks = ss.chunk_text(text, TokAdapter())

# Locate every chunk in the normalized text so midpoints map to blocks.
cursor, spans = 0, []
for c in chunks:
    i = text.find(c, cursor)
    if i < 0: i = text.find(c)
    spans.append((i, i + len(c)))
    cursor = i + 1

blocks = [text[i:i + BLOCK] for i in range(0, len(text), BLOCK)]
chunk_block = [((a + b) // 2) // BLOCK for a, b in spans]

json.dump({"text_chars": len(text), "chunk_block": chunk_block,
           "spans": spans}, open("ddia_meta.json", "w"))
json.dump(chunks, open("ddia_chunks.json", "w"))
json.dump(blocks, open("ddia_blocks.json", "w"))
print(f"raw chars {len(raw):,} -> normalized {len(text):,} ({len(text)/1e6:.2f} MB)")
print(f"chunks {len(chunks):,}   (archived run recorded 1,875 at 200/30)")
print(f"blocks {len(blocks):,}   mean chunk len {statistics.mean(len(c) for c in chunks):.0f} chars")
print(f"unlocated chunks: {sum(1 for a,_ in spans if a < 0)}")
