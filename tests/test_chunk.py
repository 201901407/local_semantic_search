"""
Unit tests for the pure logic in semantic_search.py — chunking, hashing,
and text extraction. These do NOT load the real embedding model, so they
run in well under a second and don't need network access.

Run with:
    pytest tests/test_chunking.py -v
"""

import pytest
from semantic_search import chunk_text, file_hash, extract_text


class FakeTokenizer:
    """
    A minimal stand-in for a HuggingFace *fast* tokenizer, so we can test the
    *chunking logic* (windowing, overlap, boundaries) without downloading
    a real model. Treats each whitespace-separated word as one token —
    good enough to verify chunk_text's behavior in isolation.

    Like a real fast tokenizer, calling it returns an offset mapping: the
    (start, end) character span of each token in the input. chunk_text uses
    those spans to slice verbatim source text rather than decoding tokens
    back into a lossily-normalized string.
    """

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets, pos = [], 0
        for word in text.split():
            start = text.index(word, pos)
            offsets.append((start, start + len(word)))
            pos = start + len(word)
        return {"offset_mapping": offsets}

    def encode(self, text, add_special_tokens=False):
        return text.split()  # "tokens" are just words here


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


# --------------------------------------------------------------------------
# chunk_text
# --------------------------------------------------------------------------

def test_chunk_text_empty_string_returns_no_chunks(fake_tokenizer):
    assert chunk_text("", fake_tokenizer) == []


def test_chunk_text_whitespace_only_returns_no_chunks(fake_tokenizer):
    assert chunk_text("   \n\t  ", fake_tokenizer) == []


def test_chunk_text_short_text_fits_in_one_chunk(fake_tokenizer):
    text = "the quick brown fox jumps over the lazy dog"
    chunks = chunk_text(text, fake_tokenizer, max_tokens=200, overlap_tokens=30)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_respects_max_tokens(fake_tokenizer):
    # 500 "words" (fake tokens), max 200 per chunk -> every chunk must be <= 200 tokens
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, fake_tokenizer, max_tokens=200, overlap_tokens=30)
    for c in chunks:
        token_count = len(fake_tokenizer.encode(c))
        assert token_count <= 200, f"Chunk exceeded max_tokens: {token_count}"


def test_chunk_text_covers_entire_input_no_gaps(fake_tokenizer):
    # Every word in the original text must appear in at least one chunk —
    # overlapping windows should never *skip* content.
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, fake_tokenizer, max_tokens=200, overlap_tokens=30)

    covered = set()
    for c in chunks:
        covered.update(c.split())
    assert covered == set(words), "Some words were dropped by chunking"


def test_chunk_text_produces_overlap_between_consecutive_chunks(fake_tokenizer):
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, fake_tokenizer, max_tokens=200, overlap_tokens=30)

    assert len(chunks) >= 2
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    overlap = set(first_chunk_words) & set(second_chunk_words)
    assert len(overlap) == 30, f"Expected 30-token overlap, got {len(overlap)}"


def test_chunk_text_overlap_larger_than_max_does_not_infinite_loop(fake_tokenizer):
    # Guard against a bad config (overlap >= max_tokens) hanging the indexer forever.
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, fake_tokenizer, max_tokens=10, overlap_tokens=10)
    assert len(chunks) > 0  # should terminate, not hang


# --------------------------------------------------------------------------
# file_hash
# --------------------------------------------------------------------------

def test_file_hash_same_content_same_hash(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("identical content")
    f2.write_text("identical content")
    assert file_hash(f1) == file_hash(f2)


def test_file_hash_different_content_different_hash(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("content A")
    f2.write_text("content B")
    assert file_hash(f1) != file_hash(f2)


def test_file_hash_changes_when_file_is_edited(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("version 1")
    h1 = file_hash(f)
    f.write_text("version 2")
    h2 = file_hash(f)
    assert h1 != h2


# --------------------------------------------------------------------------
# extract_text
# --------------------------------------------------------------------------

def test_extract_text_from_txt(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world")
    assert extract_text(f) == "hello world"


def test_extract_text_from_md(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Heading\n\nSome content")
    assert "Heading" in extract_text(f)
    assert "Some content" in extract_text(f)


def test_extract_text_unsupported_extension_returns_empty(tmp_path):
    f = tmp_path / "data.xyz"
    f.write_text("irrelevant")
    # extract_text only special-cases txt/md/pdf/docx; anything else -> ""
    assert extract_text(f) == ""


def test_extract_text_unreadable_file_does_not_crash(tmp_path):
    # A .pdf file that isn't actually valid PDF binary should be handled
    # gracefully (logged + skipped) rather than raising and killing indexing.
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"not a real pdf")
    result = extract_text(f)  # should not raise
    assert isinstance(result, str)