#!/usr/bin/env python3
"""
Measure whether a real corpus fits the browser's indexing budget.

The browser ceiling is set by *extracted text*, not file size, and that ratio
is wildly document-dependent: a diagram-heavy reference book may be 5% text,
a dense technical text 40%. There is no constant to design around, so measure.

Usage:
    python bench/measure_corpus.py ~/Books/some-book.pdf
    python bench/measure_corpus.py ~/Documents/notes/        # folder, recursive
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import semantic_search as ss

# measured on a Mac, int8 weights, 512-chunk steady state (docs/hypothesis-testing.md §7)
WASM_CPS, WEBGPU_CPS = 40.6, 171.1
BYTES_PER_CHUNK_STORED = 1080          # chunk text + int8 vector


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:,.1f} {u}"
        n /= 1024


def mmss(sec):
    return f"{int(sec // 60)}m {int(sec % 60):02d}s" if sec >= 60 else f"{sec:.1f}s"


def main(paths):
    files = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            files += [f for f in p.rglob("*")
                      if f.is_file() and f.suffix.lower() in ss.SUPPORTED_EXTS]
        elif p.is_file():
            files.append(p)
    if not files:
        print("No supported files found."); return

    print(f"Loading tokenizer for {len(files)} file(s)...\n")
    tokenizer = ss.get_embedder().tokenizer

    tot_bytes = tot_text = tot_tokens = tot_chunks = 0
    print(f"{'file':<44}{'on disk':>11}{'text':>11}{'text %':>8}{'chunks':>9}")
    print("-" * 83)
    for f in files:
        size = f.stat().st_size
        t0 = time.time()
        text = ss.extract_text(f)
        chunks = ss.chunk_text(text, tokenizer) if text.strip() else []
        ntok = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text.strip() else 0
        tb = len(text.encode("utf-8"))
        tot_bytes += size; tot_text += tb; tot_tokens += ntok; tot_chunks += len(chunks)
        name = f.name if len(f.name) <= 42 else f.name[:39] + "..."
        print(f"{name:<44}{human(size):>11}{human(tb):>11}"
              f"{100*tb/size if size else 0:>7.1f}%{len(chunks):>9,}")

    print("-" * 83)
    print(f"{'TOTAL':<44}{human(tot_bytes):>11}{human(tot_text):>11}"
          f"{100*tot_text/tot_bytes if tot_bytes else 0:>7.1f}%{tot_chunks:>9,}")

    print(f"\n{'=' * 62}\nWHAT THIS COSTS IN A BROWSER\n{'=' * 62}")
    print(f"  extracted text        {human(tot_text)}")
    print(f"  tokens                {tot_tokens:,}")
    print(f"  chunks                {tot_chunks:,}")
    print(f"  index size (int8)     {human(tot_chunks * BYTES_PER_CHUNK_STORED)}")
    print(f"\n  index time, WASM      {mmss(tot_chunks / WASM_CPS)}"
          f"        ({WASM_CPS} chunks/s)")
    print(f"  index time, WebGPU    {mmss(tot_chunks / WEBGPU_CPS)}"
          f"        ({WEBGPU_CPS} chunks/s)")
    print(f"  search latency        ~{2 * tot_chunks * 384 / 1.5e9 * 1000:.0f} ms per query")
    print("\n  (throughput measured on a Mac; a phone will be slower)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
