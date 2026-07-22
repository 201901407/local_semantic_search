# Semantic Search CLI

Search a local folder of documents (`.txt`, `.md`, `.pdf`, `.docx`) by *meaning*,
not just keyword match — powered by sentence embeddings + a local vector database.

## How it works

1. **Index**: walks a folder, extracts text from each file, splits it into
   overlapping chunks, embeds each chunk with a small sentence-transformer
   model, and stores the vectors in a local Chroma database (`.semantic_index/`,
   created next to the script — nothing leaves your machine).
2. **Search**: embeds your query with the same model and finds the closest
   chunks by cosine similarity.

Re-running `index` on the same folder only embeds files that are new or have
changed (detected via content hash) — safe to run repeatedly.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run will download the embedding model (~80MB, one-time, then cached).

## Usage

```bash
# Index a folder (recurses into subfolders by default)
python semantic_search.py index ./my_docs

# Search
python semantic_search.py search "how do I configure retries"
python semantic_search.py search "refund policy" --k 3

# See what's indexed
python semantic_search.py stats

# Wipe the index and start over
python semantic_search.py clear
```

### Make it a real CLI command (optional)

```bash
chmod +x semantic_search.py
# add an alias in your shell profile:
alias semsearch="python /path/to/semantic_search.py"
```

Then just run `semsearch search "..."` from anywhere.

## Testing & Evaluation

There are two different questions worth testing separately:

**"Does the code work correctly?"** — `tests/`
```bash
pytest tests/test_chunk.py -v      # fast, pure logic, no model download needed
pytest tests/test_integration.py -v   # slower, loads the real model + real Chroma index
pytest tests/ -v                       # run everything
```
`test_chunk.py` uses a fake tokenizer (just splits on whitespace) so it can verify
chunking/overlap/hashing logic in milliseconds without touching the network or GPU/CPU
inference. `test_integration.py` runs the real embedding model end-to-end and checks
that a query with none of the source document's exact words still retrieves the right
file — that's the actual proof semantic (not keyword) search is working.

**"Is search quality actually good?"** — `eval/`
```bash
python eval/eval_retrieval.py
python eval/eval_retrieval.py --k 3
```
This indexes a small hand-labeled corpus (`eval/sample_docs/`) against a set of
queries with known correct answers (`eval/eval_queries.json`) and reports:
- **Hit@k** — did the right file show up anywhere in the top-k results?
- **MRR** (Mean Reciprocal Rank) — was it ranked *first*, or buried at position 4?

This is the same style of eval real retrieval/RAG systems use in production. To
evaluate against *your own* docs instead of the sample corpus: copy a handful of your
real files into a folder, write 5-10 queries you'd realistically search for along with
which file should answer each one, and point `eval_retrieval.py --docs-dir` /
`--queries-file` at them. Re-run this eval whenever you change `CHUNK_TOKENS`,
`CHUNK_OVERLAP_TOKENS`, or the embedding model — it'll tell you objectively whether
the change helped or hurt, instead of guessing from a handful of manual searches.


- **Different file types**: add `.pptx`, `.html`, `.eml` extraction functions.
- **Better chunking**: split on sentence/paragraph boundaries instead of raw
  character counts (e.g. using `langchain`'s text splitters or a simple regex
  on `\n\n`).
- **Hybrid search**: combine this semantic score with a classic keyword/BM25
  score (e.g. via `rank_bm25`) for queries with exact terms (names, IDs)
  that embeddings alone sometimes miss.
- **Bigger/better embedding models**: swap `all-MiniLM-L6-v2` for something
  like `bge-base-en-v1.5` for higher quality at the cost of speed.
- **File watching**: use `watchdog` to auto re-index on file changes instead
  of running `index` manually.
- **Answer generation**: pipe the top-k chunks into an LLM call to get a
  synthesized answer instead of raw excerpts (this turns it into a RAG app).