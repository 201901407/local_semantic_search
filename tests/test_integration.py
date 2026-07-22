"""
Integration test — exercises the real embedding model and a real (temporary)
Chroma index, end to end. Slower than the unit tests (downloads/loads the
model on first run) and needs network access the first time. Skipped
automatically if the dependencies aren't installed, so it won't break CI
for people who only care about the pure-logic tests.

Run with:
    pytest tests/test_integration.py -v -s
"""

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("chromadb")

import semantic_search as ss


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    """Point the module's DB_DIR at a throwaway temp dir so tests never
    touch your real, populated index."""
    monkeypatch.setattr(ss, "DB_DIR", tmp_path / ".test_index")
    yield tmp_path


def test_end_to_end_search_finds_semantically_relevant_chunk(isolated_index, tmp_path):
    """
    This is the real test that matters: index a couple of files with
    clearly distinct topics, search with a query that never appears
    verbatim in the text, and confirm the semantically-relevant file
    is the top hit. This is what proves "semantic" search is actually
    working, not just returning things in file order.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "cooking.txt").write_text(
        "Preheat the oven to 375 degrees. Whisk the eggs and sugar until "
        "pale and fluffy, then fold in the flour gently to keep the batter light."
    )
    (docs_dir / "finance.txt").write_text(
        "The quarterly earnings report showed revenue growth of 12 percent, "
        "driven by strong performance in the enterprise software division."
    )

    embedder = ss.get_embedder()
    collection = ss.get_collection()

    for f in docs_dir.glob("*.txt"):
        text = ss.extract_text(f)
        chunks = ss.chunk_text(text, embedder.tokenizer)
        embeddings = embedder.encode(chunks).tolist()
        ids = [f"{f.as_posix()}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(f), "file_hash": ss.file_hash(f), "chunk_index": i}
                     for i in range(len(chunks))]
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    # Query uses none of the exact words in cooking.txt, but is clearly
    # about baking — a keyword search would fail this; semantic search shouldn't.
    query = "baking a cake in the kitchen"
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=1)

    top_source = results["metadatas"][0][0]["source"]
    assert "cooking.txt" in top_source


def test_unchanged_file_is_skipped_on_reindex(isolated_index, tmp_path):
    """Re-indexing the same unmodified file should not create duplicate chunks."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    f = docs_dir / "note.txt"
    f.write_text("a short note about nothing in particular")

    embedder = ss.get_embedder()
    collection = ss.get_collection()

    def index_once():
        text = ss.extract_text(f)
        chunks = ss.chunk_text(text, embedder.tokenizer)
        fhash = ss.file_hash(f)
        existing = collection.get(include=["metadatas"])
        existing_hashes = {m["file_hash"] for m in existing["metadatas"]} if existing["metadatas"] else set()
        if fhash in existing_hashes:
            return  # mirrors the skip logic in the `index` command
        embeddings = embedder.encode(chunks).tolist()
        ids = [f"{f.as_posix()}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(f), "file_hash": fhash, "chunk_index": i} for i in range(len(chunks))]
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    index_once()
    count_after_first = collection.count()
    index_once()  # same file, unchanged content
    count_after_second = collection.count()

    assert count_after_first == count_after_second