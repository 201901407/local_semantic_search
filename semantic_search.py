"""
Semantic Search CLI
--------------------
Index a local folder of documents (.txt, .md, .pdf, .docx) and search them
by *meaning*, not just keyword match.

Usage:
    python semantic_search.py index ./my_docs
    python semantic_search.py search "how do I configure retries"
    python semantic_search.py search "refund policy" --k 3
    python semantic_search.py stats
    python semantic_search.py clear
"""

import hashlib
from pathlib import Path
from typing import List, Tuple

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(add_completion=False, help="Semantic search over local documents.")
console = Console()

# Where the persistent vector DB lives (next to this script)
DB_DIR = Path(__file__).parent / ".semantic_index"
COLLECTION_NAME = "local_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, CPU-friendly

CHUNK_TOKENS = 200     # tokens per chunk (model max is 256; stay under it with margin)
CHUNK_OVERLAP_TOKENS = 30  # overlap between consecutive chunks, in tokens
SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------

def extract_text(path: Path) -> str:
    """Extract raw text from a supported file type."""
    ext = path.suffix.lower()
    try:
        if ext in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="ignore")

        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        if ext == ".docx":
            import docx
            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs)

    except Exception as e:
        console.print(f"[yellow]  ! Skipped (couldn't read): {path.name} ({e})[/yellow]")
        return ""

    return ""


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """
    Split text into overlapping chunks measured in actual model tokens,
    not characters. This guarantees each chunk fits under the embedding
    model's max sequence length (256 for all-MiniLM-L6-v2) instead of
    relying on a character-count approximation that can silently overflow
    on token-dense text (code, jargon, non-English text, etc).
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    # Tokenize once; encode/decode lets us cut precisely on token boundaries.
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks = []
    start = 0
    step = max(max_tokens - overlap_tokens, 1)  # guard against overlap >= max_tokens
    while start < len(token_ids):
        window = token_ids[start:start + max_tokens]
        chunks.append(tokenizer.decode(window))
        start += step
    return chunks


def file_hash(path: Path) -> str:
    """Return a stable SHA-256 hash of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> str:
    """
    Cheap way to detect whether a file has changed, WITHOUT reading its
    content. Uses size + modification time from stat() (an O(1) metadata
    call, regardless of file size) instead of hashing the file's bytes.

    Why this matters: hashing full content means reading every file
    completely on every `index` run just to check if it changed -- for a
    folder with large PDFs, that's a lot of wasted I/O and memory for
    files that didn't change at all. stat() answers "did this file change"
    in constant time without touching the file's actual content.

    Trade-off: this can theoretically miss a change if a file is edited
    in a way that preserves both its exact byte size AND its mtime down to
    the same tick (essentially never happens from normal editing/saving),
    and it may occasionally flag an untouched file as "changed" if some
    process rewrites it with identical content (e.g. a re-sync tool) --
    that just costs one harmless re-embed, not an incorrect result. This
    is the same size+mtime approach tools like rsync use by default.
    """
    stat = path.stat()
    return f"{stat.st_size}-{stat.st_mtime_ns}"


# --------------------------------------------------------------------------
# Lazy-loaded heavy dependencies (so `--help` stays fast)
# --------------------------------------------------------------------------

def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

@app.command()
def index(
    folder: Path = typer.Argument(..., help="Folder containing documents to index"),
    recursive: bool = typer.Option(True, help="Recurse into subfolders"),
):
    """Index (or re-index changed files in) a folder of documents."""
    if not folder.exists():
        console.print(f"[red]Folder not found: {folder}[/red]")
        raise typer.Exit(1)

    pattern = "**/*" if recursive else "*"
    files = [f for f in folder.glob(pattern) if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]

    if not files:
        console.print(f"[yellow]No supported files ({', '.join(SUPPORTED_EXTS)}) found in {folder}[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found [bold]{len(files)}[/bold] file(s). Loading embedding model...")
    embedder = get_embedder()
    collection = get_collection()

    # Figure out which files are new/changed vs. already indexed
    existing = collection.get(include=["metadatas"])
    existing_hashes = {m["fingerprint"] for m in existing["metadatas"]} if existing["metadatas"] else set()

    added_files = 0
    added_chunks = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Indexing...", total=len(files))
        for f in files:
            progress.update(task, description=f"Reading {f.name}")
            fhash = file_fingerprint(f)

            if fhash in existing_hashes:
                progress.advance(task)
                continue  # unchanged, skip re-embedding

            text = extract_text(f)
            chunks = chunk_text(text, embedder.tokenizer)
            if not chunks:
                progress.advance(task)
                continue

            embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
            ids = [f"{f.as_posix()}::{i}" for i in range(len(chunks))]
            metadatas = [
                {"source": str(f), "fingerprint": fhash, "chunk_index": i}
                for i in range(len(chunks))
            ]

            collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
            added_files += 1
            added_chunks += len(chunks)
            progress.advance(task)

    console.print(f"[green]Done.[/green] Indexed {added_files} new/changed file(s) -> {added_chunks} chunks.")
    if added_files == 0:
        console.print("[dim](Everything was already up to date.)[/dim]")


@app.command()
def search(
    query: str = typer.Argument(..., help="What you're looking for"),
    k: int = typer.Option(5, "--k", "-k", help="Number of results to return"),
):
    """Search indexed documents by meaning."""
    collection = get_collection()
    if collection.count() == 0:
        console.print("[yellow]Index is empty. Run `index <folder>` first.[/yellow]")
        raise typer.Exit(0)

    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection.count()),
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    table = Table(show_lines=True)
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Source", style="magenta")
    table.add_column("Excerpt")

    for doc, meta, dist in zip(docs, metas, dists):
        similarity = 1 - dist  # cosine distance -> similarity
        excerpt = doc[:220] + ("..." if len(doc) > 220 else "")
        table.add_row(f"{similarity:.2f}", Path(meta["source"]).name, excerpt)

    console.print(table)


@app.command()
def stats():
    """Show index statistics."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        console.print("[yellow]Index is empty.[/yellow]")
        return

    metas = collection.get(include=["metadatas"])["metadatas"]
    sources = {m["source"] for m in metas}
    console.print(f"[bold]{len(sources)}[/bold] file(s) indexed, [bold]{count}[/bold] chunks total.")
    console.print(f"Index location: {DB_DIR}")


@app.command()
def clear():
    """Delete the entire index."""
    import shutil
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        console.print("[green]Index cleared.[/green]")
    else:
        console.print("[dim]Nothing to clear.[/dim]")


if __name__ == "__main__":
    app()