"""
Retrieval evaluation for the semantic search CLI.

Unlike the unit/integration tests, this doesn't check "does the code run" —
it checks "does the search actually return the right document," using a
small hand-labeled set of (query, expected_file) pairs. This is the ML
equivalent of a test suite: correctness of retrieval quality, not code logic.

Metrics:
  - Hit@k:  fraction of queries where the expected file appears anywhere
            in the top-k results. Answers "did we find it at all."
  - MRR:    Mean Reciprocal Rank — averages 1/rank of the expected file
            across all queries (1.0 if always first result, 0 if never
            found). Rewards ranking the right answer *higher*, not just
            finding it somewhere in the top-k.

Usage:
    python eval/eval_retrieval.py
    python eval/eval_retrieval.py --k 3
"""

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
import semantic_search as ss

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    k: int = typer.Option(5, help="Top-k results to consider for Hit@k"),
    docs_dir: Path = typer.Option(Path(__file__).parent , help="Folder of eval documents"),
    queries_file: Path = typer.Option(Path(__file__).parent / "eval_queries.json", help="Labeled query set"),
):
    """Index the eval corpus into a throwaway index and score retrieval quality."""
    queries = json.loads(queries_file.read_text())

    console.print(f"Loading model and indexing {docs_dir}...")
    embedder = ss.get_embedder()

    # Use a throwaway on-disk location so this never touches your real index.
    import chromadb
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=tmp_dir)
    collection = client.get_or_create_collection(name="eval", metadata={"hnsw:space": "cosine"})

    for f in docs_dir.glob("*.txt"):
        text = ss.extract_text(f)
        chunks = ss.chunk_text(text, embedder.tokenizer)
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"{f.name}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": f.name} for _ in chunks]
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    # Run every labeled query, record where (if at all) the expected file lands.
    table = Table(show_lines=True)
    table.add_column("Query", style="cyan", max_width=40)
    table.add_column("Expected", style="magenta")
    table.add_column("Top result", style="green")
    table.add_column("Rank", justify="right")

    reciprocal_ranks = []
    hits = 0

    for item in queries:
        query, expected = item["query"], item["expected_file"]
        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=k)
        sources = [m["source"] for m in results["metadatas"][0]]

        # de-duplicate consecutive chunks from the same file, preserving order
        seen = []
        for s in sources:
            if s not in seen:
                seen.append(s)

        if expected in seen:
            rank = seen.index(expected) + 1
            reciprocal_ranks.append(1 / rank)
            hits += 1
        else:
            rank = None
            reciprocal_ranks.append(0)

        top_result = seen[0] if seen else "(none)"
        rank_display = str(rank) if rank else f"not in top-{k}"
        row_style = "" if rank == 1 else ("yellow" if rank else "red")
        table.add_row(query, expected, top_result, rank_display, style=row_style or None)

    console.print(table)

    hit_at_k = hits / len(queries)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    console.print(f"\n[bold]Hit@{k}:[/bold] {hit_at_k:.2%}  ({hits}/{len(queries)} queries found expected file in top-{k})")
    console.print(f"[bold]MRR:[/bold]    {mrr:.3f}  (1.0 = always ranked first, 0.0 = never found)")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    app()