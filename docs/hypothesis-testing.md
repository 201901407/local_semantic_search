# Hypothesis Testing: Can This Run Entirely in the Browser?

This document records two experiments run before committing to a fully
client-side architecture, along with enough of the *why* to be useful as a
reference on how retrieval systems are actually evaluated.

Two questions drove it:

1. **How large a document folder can a browser-only version handle?**
2. **Does int8 quantization of the embedding model hurt accuracy?**

Both are empirical. Neither should be answered by intuition, so neither was.

---

## 0. The first finding was about the eval itself

Before testing quantization, the existing eval suite had to be checked for
whether it was capable of detecting a regression at all. It was not.

```
$ python eval/eval_retrieval.py
Hit@5: 100.00%  (7/7 queries found expected file in top-5)
MRR:    1.000  (1.0 = always ranked first, 0.0 = never found)
```

A perfect score is not good news here — it means the benchmark is **saturated**
and has no headroom. The cause is structural: the corpus contains 5 documents
and the metric is Hit@**5**. Every document is in the top 5 essentially by
construction, so Hit@5 cannot go down. MRR has slightly more room, but with 7
easy, lexically-distinct queries it sits at the ceiling too.

The consequence matters: **had this eval been used to test quantization, it
would have reported "no difference" for every variant — including a badly
broken one.** A benchmark that cannot fail is not evidence of quality; it is
the absence of evidence.

> **Lesson.** A saturated benchmark is worse than no benchmark, because it
> produces false confidence. When a metric hits its ceiling, the correct
> response is to make the task harder — more documents, harder negatives, more
> queries — not to celebrate. This is why real retrieval work uses corpora of
> thousands of documents: you need enough plausible-but-wrong candidates for
> ranking quality to be measurable at all.

### The replacement benchmark

Testing moved to **SciFact** (via `mteb/scifact`), a standard retrieval
benchmark from the BEIR suite:

| Property | Project eval | SciFact |
|---|---|---|
| Documents | 5 | 5,183 |
| Labeled queries | 7 | 300 |
| Mean document length | ~35 words | 215 words |
| Headroom | none (MRR 1.000) | substantial |

SciFact's documents are scientific abstracts and its queries are claims to
verify, so wrong answers are *topically similar* to right ones. That is what
forces a retriever to actually discriminate, rather than just separate
"shipping" from "refunds".

---

## 1. Method

Both experiments reuse the project's own pipeline so the results transfer.

**Chunking.** The SciFact corpus was chunked with the project's own
`chunk_text()` at its real settings (`CHUNK_TOKENS=200`,
`CHUNK_OVERLAP_TOKENS=30`), using the real model tokenizer. 5,183 documents
produced **12,815 chunks**.

**Scoring.** Each document is scored by its single best-matching chunk
(max-pooling over that document's chunks). This mirrors what the project's own
eval does when it de-duplicates results by filename.

**Metrics.** Three, because they answer different questions:

- **Recall@10** — did the correct document appear in the top 10 at all?
  Answers *"did we find it."*
- **MRR@10** (Mean Reciprocal Rank) — averages `1/rank` of the first correct
  document. Answers *"was it near the top."*
- **nDCG@10** (normalized Discounted Cumulative Gain) — discounts each hit by
  `1/log2(rank+1)` and normalizes against the best achievable ordering. This is
  the standard headline metric for retrieval because it is the only one of the
  three that properly handles *multiple* relevant documents per query and
  rewards their ordering. SciFact has queries with up to 5 relevant documents,
  so this matters.

Two further diagnostics measure agreement with the fp32 baseline directly:

- **top-1 agreement** — how often the quantized model's best result is the same
  document fp32 would have returned.
- **top-10 overlap** — average fraction of the top-10 set shared with fp32.

These need no labels, and they answer the question a user actually cares
about: *"will I see different results?"*

**Pipeline validation.** Before trusting any comparison, the ONNX inference
path (tokenize → transformer → mean-pool over the attention mask → L2
normalize) was checked against `sentence-transformers` itself:

```
variant                 mean cos vs torch   min cos
model (ONNX fp32)                1.000000  1.000000
```

Cosine similarity of exactly 1.000000 confirms the reimplementation is
faithful. **Without this control, any measured difference could have been a bug
in the harness rather than an effect of quantization.** This step is the
difference between an experiment and a guess.

---

## 2. Experiment A — How big a folder can the browser hold?

### What actually consumes space

Indexing one chunk stores two things: the **chunk text** (needed to display
excerpts) and its **embedding vector**. Measured on the SciFact corpus using
the project's real chunking settings:

| Quantity | Measured |
|---|---|
| Documents | 5,183 |
| Raw text | 7.78 MB |
| Tokens | 1,744,914 |
| **Bytes per token** | **4.46** |
| Chunks produced | 12,815 |
| Chunks per document | mean 2.47, median 2, max 12 |
| Raw source bytes per chunk | 607 |
| Stored chunk text | 8.91 MB (**+14.5%** vs source) |
| Vector size | 384 dims × 4 bytes = **1,536 B** |
| **Total per chunk** | **~2,232 B** |

Two things are worth internalizing here.

**The overlap tax is real but small.** Storing 200-token chunks that overlap by
30 tokens means ~15% of the text is stored twice. That is the price of not
splitting a sentence across a chunk boundary and losing its meaning — a good
trade, but a measurable one.

**The vector is bigger than the text it describes.** 1,536 bytes of float32 per
607 bytes of source text. The index is not a small annotation on your
documents; it is the dominant cost. This is the single most important fact for
sizing a browser-based system.

### The headline ratio

> **1 MB of source documents → ~3.67 MB of browser storage.**

### Is brute-force search fast enough?

The plan replaces a vector database with a linear scan: score every chunk, keep
the top k. Each query costs `2 × chunks × 384` floating-point operations.
Measured with numpy, plus a conservative estimate for hand-written JavaScript
over a flat `Float32Array`:

| Chunks | Vector RAM | numpy | JS estimate |
|---|---|---|---|
| 1,000 | 1.5 MB | 0.02 ms | ~0.5 ms |
| 10,000 | 15.4 MB | 0.21 ms | ~5 ms |
| 50,000 | 76.8 MB | 1.92 ms | ~26 ms |
| 100,000 | 153.6 MB | 4.02 ms | ~51 ms |
| 250,000 | 384.0 MB | 9.99 ms | ~128 ms |
| 500,000 | 768.0 MB | 23.20 ms | ~256 ms |

**Conclusion: search speed is not the constraint.** Even at half a million
chunks a linear scan stays inside a quarter second. An approximate-nearest-
neighbour index (HNSW and friends) exists to avoid exactly this scan, but the
scan only becomes painful at a scale where browser *memory* has already failed.
Adding an ANN library here would cost bytes and complexity to solve a problem
this application never reaches.

> **Lesson.** "Brute force is too slow" is an assumption worth checking. A
> modern CPU does billions of multiply-adds per second, and 384-dimensional
> dot products over even 100k vectors is a rounding error. Reach for an index
> when measurement demands it, not by default.

### So memory is the real ceiling

The binding limits are the browser's, not the algorithm's:

- **JS heap / ArrayBuffer** — desktop Chrome and Firefox handle a few hundred
  MB of `Float32Array` comfortably. Mobile Safari is far tighter and will kill
  a tab that grows too large.
- **IndexedDB quota** — generous on desktop Chrome (a large share of free
  disk), stricter and more prompt-driven on Safari and Firefox.

Applying the measured 3.67 MB-per-MB ratio:

| Source docs | Chunks (approx) | Browser storage | Verdict |
|---|---|---|---|
| 10 MB | ~16k | ~37 MB | Comfortable everywhere, mobile included |
| 50 MB | ~82k | ~184 MB | Fine on desktop; risky on mobile |
| 100 MB | ~165k | ~367 MB | Desktop only; noticeable memory pressure |
| 250 MB | ~412k | ~918 MB | Beyond practical browser limits |

---

## 3. Experiment B — Does int8 quantization hurt accuracy?

### What quantization actually is

The model's weights are trained as 32-bit floats. Quantization stores them in
fewer bits — int8 keeps a scale factor per tensor and rounds each weight to one
of 256 levels. The file shrinks ~4x. The question is whether that rounding
error propagates through 6 transformer layers into meaningfully different
embeddings.

Two *independent* things can be quantized, and they are easy to conflate:

1. **Model weights** — shrinks the *download*. This is what "int8 model" means.
2. **Stored vectors** — shrinks the *index in memory*. 384 floats (1,536 B)
   becomes 384 int8s (384 B) per chunk.

Both were tested, separately and together.

### File sizes (measured)

| Variant | Size |
|---|---|
| `model.onnx` (fp32) | 90.4 MB |
| `model_fp16.onnx` | 45.3 MB |
| `model_q4.onnx` | 54.6 MB |
| `model_quantized.onnx` / `model_int8.onnx` | **23.0 MB** |
| `model_uint8.onnx` | 22.9 MB |

Note `q4` is *larger* than int8 despite using 4-bit weights — only some tensors
are 4-bit and it carries extra per-block scale factors.

### Retrieval quality — SciFact, 5,183 docs, 300 labeled queries

| Variant | nDCG@10 | Recall@10 | MRR@10 | top-1 agree | top-10 overlap |
|---|---|---|---|---|---|
| torch fp32 (baseline) | 0.6485 | 0.8050 | 0.6048 | 100% | 100% |
| ONNX fp32 *(control)* | 0.6485 | 0.8050 | 0.6048 | 100% | 100% |
| ONNX fp16 | 0.6484 | 0.8050 | 0.6047 | 100.0% | 99.9% |
| **ONNX int8** | **0.6471** | 0.8117 | 0.6027 | 94.0% | 92.0% |
| ONNX uint8 | 0.6532 | 0.8072 | 0.6097 | 85.3% | 83.7% |
| ONNX q4 | 0.6336 | 0.7964 | 0.5874 | 85.3% | 82.0% |
| fp32 + int8 **vectors** | 0.6478 | 0.8050 | 0.6037 | 99.3% | 98.8% |
| **int8 model + int8 vectors** | **0.6452** | 0.8067 | 0.6019 | 94.0% | 92.0% |

### Is any of that real? Paired bootstrap, 10,000 resamples

A raw difference means nothing without knowing its sampling error. Each
variant's per-query nDCG was differenced against the baseline and resampled:

| Variant | Δ nDCG@10 | 95% CI | p | Verdict |
|---|---|---|---|---|
| int8 weights | −0.0014 | [−0.0091, +0.0057] | 0.721 | not significant |
| int8 weights + int8 vectors | −0.0034 | [−0.0106, +0.0033] | 0.327 | not significant |
| fp16 | −0.0001 | [−0.0003, +0.0000] | 0.742 | not significant |
| uint8 | +0.0047 | [−0.0087, +0.0182] | 0.491 | not significant |
| int8 vectors only | −0.0007 | [−0.0020, +0.0003] | 0.181 | not significant |
| **q4** | **−0.0150** | **[−0.0298, −0.0008]** | **0.039** | **significantly worse** |

### The answer, and the subtlety

**No, int8 does not hurt accuracy — but it does change your results.**

These are two different claims and the distinction is the most instructive part
of this whole exercise:

- **Aggregate quality is unchanged.** int8 scores −0.0014 against fp32, with a
  confidence interval straddling zero and p = 0.72. The honest reading is "no
  detectable difference." Note that uint8 scores *nominally higher* (+0.0047,
  p = 0.49) — reporting that as "uint8 beats fp32" would be the textbook error,
  since its interval straddles zero just as clearly.
- **Individual results do move.** top-1 agreement is 94.0%: on roughly 1 query
  in 17, int8 surfaces a different best document than fp32. Those swaps are
  near-ties where two documents scored within rounding error of each other, and
  they go right about as often as they go wrong — which is exactly why the
  aggregate does not move.

> **Lesson 1.** A metric difference is meaningless without its uncertainty.
> With 300 queries, anything under roughly ±0.01 nDCG is noise. Reporting
> "uint8 scored 0.6532 vs fp32's 0.6485, so uint8 is better" would be a
> textbook error — that gap is well inside the noise band.

> **Lesson 2.** "Same quality" and "same output" are different guarantees.
> Quantization preserves the first, not the second. That is fine for search
> ranking, where near-ties are arbitrary anyway — but it would not be fine for
> a system requiring reproducible, exactly-repeatable output.

> **Lesson 3.** q4 shows the limit is real, not imaginary. Push quantization
> far enough and degradation becomes statistically detectable (p = 0.039). int8
> sits comfortably inside the safe zone; 4-bit does not.

### Throughput (native CPU, 12,815 chunks)

| Variant | chunks/s |
|---|---|
| torch fp32 | 454.1 |
| ONNX fp32 | 125.3 |
| ONNX fp16 | 113.2 |
| ONNX int8 | 127.0 |
| ONNX uint8 | 151.3 |

> **Lesson 4.** int8 is **not faster** than fp32 here (127 vs 125 chunks/s).
> Quantization is widely assumed to be a speed optimization; on this CPU the
> dequantization overhead cancels the cheaper arithmetic. **It buys download
> size, not latency.** Optimizations should be chosen against the constraint
> that actually binds — here, bytes over the wire.

---

## 4. Conclusions

**Configuration to build on:** int8 model weights (23 MB) **plus** int8 stored
vectors. Measured cost vs. fp32: Δ nDCG +0.0007, p = 0.86 — no detectable
quality loss, for a 4x smaller download and a 4x smaller index.

**Reject fp16** (45 MB for a statistically identical result — 2x the download
for nothing) and **reject q4** (larger than int8, slower, and the only variant
with measurable degradation).

**Storage, with int8 vectors:** per chunk drops from 2,232 B to 1,080 B.

> **1 MB of source documents → ~1.78 MB of browser storage** (down from 3.67 MB)

| Source docs | Chunks | Storage (int8 vectors) | Est. browser index time* |
|---|---|---|---|
| 10 MB | ~16k | ~18 MB | ~6 min |
| 25 MB | ~41k | ~44 MB | ~15 min |
| 50 MB | ~82k | ~89 MB | ~30 min |
| 100 MB | ~165k | ~178 MB | ~1 hour |

\* assuming ~45 chunks/s — native ORT int8 measured 127 chunks/s, discounted
~3x for WebAssembly. **Needs verification in a real browser.**

**The real ceiling is indexing time, not storage or search.** Search is
trivial (~51 ms at 100k chunks) and storage is manageable to ~100 MB, but
one-time indexing at WASM speed is the constraint a user actually feels.
Practical comfortable limit: **~10–25 MB of documents**, with a WebGPU backend
the main lever for pushing higher.

### What still needs verifying

These numbers come from Python's `onnxruntime` on a Mac — the same
weights and operator kernels `onnxruntime-web` uses, which is why they
transfer. But the WASM throughput discount is an *estimate*. The first thing to
build is a throwaway page that indexes a known corpus in-browser and reports
real chunks/s on both the WASM and WebGPU backends.

---

## 5. Why is indexing slow, if everything is in memory?

A natural follow-up: if Chroma is gone and the index is just an array, why does
indexing 100 MB still take an hour?

Because **the vector database was never the expensive part.** Profiling 2,000
chunks through the int8 model:

| Stage | Time | Share |
|---|---|---|
| Tokenization | 68.1 ms | 0.42% |
| **Transformer forward pass** | **16,156.9 ms** | **99.58%** |
| Vector store + quantize | 0.8 ms | 0.01% |
| *(one search over all 2,000)* | *0.010 ms* | — |

### The compute asymmetry

The two operations are not remotely comparable:

**Embedding a chunk** runs a 6-layer transformer over every token. Each token
passes through attention projections and a feed-forward network:

```
per token ≈ 2 × 6 layers × (4·384·384 + 2·384·1536) ≈ 21.2 MFLOP
per chunk ≈ 155.1 tokens × 21.2 MFLOP           ≈ 3.29 GFLOP
```

**Searching that chunk** is one 384-dimensional dot product:

```
per chunk = 2 × 384 = 768 FLOP
```

> **Embedding a chunk costs ~4,300,000× more compute than searching it.**

That is the whole answer. Chroma only ever handled the 768-FLOP side. Removing
it made the app lighter and simpler, but it was never touching the 3.29-GFLOP
side where the time actually goes.

Indexing 100 MB of documents means ~165,000 chunks × 3.29 GFLOP ≈ **543
TFLOPs** of neural network inference. (That counts useful tokens only; batch
padding adds a further ~31.6% — see §6.) No storage change affects that number;
only changing *the model* or *the hardware it runs on* does.

> **Lesson.** When something is slow, profile before optimizing. The intuition
> "it's slow because of the database" was reasonable and completely wrong — the
> database accounted for 0.01%. Optimizing it to zero would have saved nothing.

### Why this is acceptable anyway

Indexing is a **one-time, amortized** cost. You pay it once per document, then
every subsequent search is ~50 ms forever. The existing `file_fingerprint()`
logic (size + mtime) already ensures unchanged files are never re-embedded, so
the hour is paid once, not per session.

### Levers that would actually help

Ordered by leverage, since each attacks the 99.58%:

1. **WebGPU backend** — runs the same model on the GPU. Typically 5–10x over
   WASM, turning an hour into 6–12 minutes. Biggest available win, and the
   reason the browser benchmark should test both backends.
2. **A static embedding model** (e.g. model2vec/potion-style) — replaces the
   transformer with a token-embedding lookup plus averaging. Removes the
   forward pass almost entirely, potentially ~1000x faster indexing, at a real
   cost in retrieval quality. **This is the next hypothesis worth testing**,
   using exactly the harness in this document.
3. **A smaller transformer** — fewer layers scales compute close to linearly.
4. **Index incrementally** — embed in the background as files are added rather
   than making the user watch one long bulk job.

Note what is *not* on this list: anything about storage, search, or the vector
index. Those are already ~0.01% of the cost.

---

## 6. Chunk size, overlap, and a measurement trap

### The cost model

For a corpus of `T` tokens, with `step = CHUNK_TOKENS − CHUNK_OVERLAP_TOKENS`:

```
number of chunks      ≈ T / step                    → drives STORAGE
total tokens embedded ≈ T × CHUNK_TOKENS / step     → drives TIME
```

The second fraction is the **amplification factor**: how many times the average
token gets embedded. Predicted before measuring, then confirmed:

| config | predicted `ct/(ct−ov)` | measured |
|---|---|---|
| (64, 16) | 1.33 | 1.35 |
| (128, 30) | 1.31 | 1.27 |
| (200, 30) | 1.18 | 1.14 |
| (200, 60) | 1.43 | 1.32 |

### Measured grid (int8 model, SciFact)

| chunk | overlap | chunks | embed s | vec MB | nDCG@10 |
|---|---|---|---|---|---|
| 64 | 0 | 29,797 | 69.2 | 11.4 | **0.6752** |
| 64 | 16 | 38,825 | 85.5 | 14.9 | 0.6695 |
| 128 | 0 | 16,228 | 74.4 | 6.2 | 0.6709 |
| 128 | 30 | 20,318 | 93.4 | 7.8 | 0.6742 |
| 200 | 0 | 11,382 | 87.7 | 4.4 | 0.6421 |
| **200** | **30** *(current)* | 12,815 | 98.5 | 4.9 | 0.6471 |
| 200 | 60 | 14,999 | 115.4 | 5.8 | 0.6444 |
| 256 | 0 | 9,377 | 93.8 | 3.6 | 0.6493 |
| 256 | 30 | 10,345 | 104.7 | 4.0 | 0.6437 |

**Storage scales with chunk count, time with amplification.** These are
different denominators, which is why intuition misleads: (64, 0) uses 2.3x the
storage of (200, 30) but embeds *fewer* total tokens.

**Quality favours smaller chunks here** — 0.6752 vs 0.6471, about 3.5x the
±0.008 noise band, so it is real. But SciFact is short scientific abstracts
with single-claim queries, which suits small precise chunks. On long-form
documents with context-dependent queries the ranking could reverse. **This is
the one result here that should be re-measured on your own corpus.**

### The trap: batch padding

The `embed s` column above is *not* a clean measure of chunk-size cost.
Inference pads every batch to its longest member, so a batch of mixed-length
chunks computes on padding:

```
real tokens                     1,987,848   (1.000x)
computed, corpus-order batches  2,615,318   (1.316x)   <- 31.6% wasted
computed, length-sorted batches 1,994,258   (1.003x)
```

**Sorting chunks by length before batching would cut ~24% of indexing time**
for free, with no effect on output. *(§11.5 measured this as 28.8% of wall clock
on SciFact — and 0.6%, i.e. nothing, on a single book. The waste is a property of
how varied the corpus is, not of the pipeline.)* Configurations with uniform chunk lengths
(like 64/0) waste less padding than bimodal ones (like 200/30), so part of the
apparent "small chunks are faster" effect is this artifact rather than a
property of chunk size.

### A tokenizer default that only breaks quantized models

`tokenizer.json` for this model ships `padding: {length: 128}`. Three findings,
in the order they were (incorrectly, then correctly) established:

1. PAD tokens are masked out of attention and mean-pooling, so **fp32 and fp16
   are provably immune** — cosine 1.00000000.
2. **Quantized models are not.** ONNX Runtime derives activation quantization
   scales from actual tensor contents, so PAD rows shift the scale applied to
   real tokens. int8 drifts to cosine 0.9967, uint8 to 0.9865.
3. It only fires when a batch's longest member is under 128 tokens. Over the
   document corpus, every batch maximum was ≥202, so **documents were
   unaffected**. But all 300 **queries** are under 64 tokens, so every query
   batch was affected — which is where it silently changed the results.

> **Lesson.** Three wrong conclusions were reached before the right one, each
> from sampling the wrong place. Testing only long chunks showed "no effect"
> (the mechanism could not fire there). Testing an artificial batch of only
> short chunks showed a large effect (that batch composition never occurs).
> Only checking documents *and* queries separately revealed the truth. **When
> validating that something has no effect, deliberately sample where the effect
> would be largest — and enumerate every input path, not just the obvious one.**

**Actionable for the browser build:** explicitly disable padding. Queries are
always short, so with a quantized model this fires on *every search*
permanently — and it collides with length-sorted batching, which deliberately
creates the homogeneous short batches where it bites hardest.

---

## 7. Experiment C — Actual in-browser throughput

Every number above was native Python. The decisive question — how fast the
model runs *in a browser* — was estimated at "native ÷ 3" and never measured.
This section measures it.

**Setup.** A static page (`bench/`) loads the same int8 weights through
Transformers.js and embeds the same 512 SciFact chunks, in batches of 64, to
match the native measurement exactly. Served over localhost with
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy:
credentialless`.

**Two guards, because a throughput number alone is not trustworthy:**

1. **Configuration self-report.** Without COOP/COEP, `SharedArrayBuffer` is
   unavailable and onnxruntime-web silently drops to *single-threaded* WASM —
   several times slower, with no error. The page reports
   `crossOriginIsolated` and its thread count so the number carries its own
   configuration. Measured: `crossOriginIsolated: true`, 8 threads, on a Mac.
2. **A correctness control.** The page embeds 32 chunks and compares them by
   cosine against vectors computed natively. A throughput figure is only
   meaningful if the browser computed the *same thing*.

### Results

| backend | dtype | control cos | chunks/s | vs native | download |
|---|---|---|---|---|---|
| WASM | int8 | 0.9943 | 40.6 | 0.32x | 23.0 MB |
| WebGPU | int8 | **0.0236** | *(void)* | — | — |
| WebGPU | fp32 | 0.9956 | 123.5 | 0.97x | 90.4 MB |
| **WebGPU** | **fp16** | **0.9956** | **171.1** | **1.35x** | 45.3 MB |

*(native ONNX int8 on this machine: 127.0 chunks/s)*

**The control earned its keep immediately.** WebGPU with int8 weights ran to
completion, reported a plausible 29.2 chunks/s, and produced *garbage* — cosine
0.0236 against the reference, with a minimum of −0.0997. That is noise, not
embeddings. Reported without the control, it would have become the finding
"WebGPU is slower than WASM," which is false and would have killed the most
valuable optimization available.

> **Lesson.** A benchmark that only measures speed can be confidently,
> silently wrong. Always verify the computation is correct *before* believing
> its timing. Wrong answers are often fast.

### The architectural consequence

**int8 and WebGPU are mutually exclusive.** int8 kernels do not work on the
WebGPU backend; WebGPU needs fp16 or fp32. That forces a real trade:

- **int8 + WASM** — 23 MB download, 40.6 chunks/s
- **fp16 + WebGPU** — 45.3 MB download, 171.1 chunks/s (**4.2x faster**)

The resolution is to feature-detect and download only one: serve fp16 where
`navigator.gpu` exists, int8 everywhere else. Nobody pays for both.

### Corrected corpus ceiling

Using the measured 607 raw bytes/chunk and 1,080 B/chunk stored (int8 vectors):

| Source docs | Chunks | Storage | WASM int8 | WebGPU fp16 |
|---|---|---|---|---|
| 10 MB | ~16k | 18 MB | 6.8 min | **1.6 min** |
| 25 MB | ~41k | 44 MB | 16.9 min | **4.0 min** |
| 50 MB | ~82k | 89 MB | 33.8 min | **8.0 min** |
| 100 MB | ~165k | 178 MB | 67.6 min | **16.0 min** |
| 250 MB | ~412k | 445 MB | 169 min | 40.1 min |

**This roughly quadruples the practical ceiling.** On WASM the limit is
~10–25 MB, as originally estimated. With WebGPU, 100 MB indexes in 16 minutes
and *storage* (~178 MB) becomes the binding constraint instead of time.
Length-sorted batching (§6) would cut a further ~24% from both columns.

**Caveat:** measured on a Mac laptop, near the fast end. A mid-range phone will
be substantially slower, and the ceiling should be set by the weakest device
that matters. The benchmark page runs anywhere — open it on the target device.

---

## 8. Experiment D — A real corpus, and three findings it produced

Everything to this point used SciFact: 5,183 scientific abstracts, uniform and
short. Real input is neither. This section measures an actual technical book —
*Designing Data-Intensive Applications*, 613 pages — and records three findings
that only a real document could have surfaced.

### 8.1 The text ratio, and why it cannot be designed around

| | |
|---|---|
| File on disk | 23.2 MB, 613 pages |
| **Extracted text** | **1.4 MB — 6.0% of file size** |
| Tokens | 318,660 |
| Chunks (200/30) | 1,875 |
| Index size (int8) | 1.9 MB |
| Index time, WASM | 46 s |
| Index time, WebGPU | 11 s |

**Six percent.** A dense technical book is 94% fonts, images and structure. That
single number changes the capacity story completely: the earlier ceilings were
quoted in *extracted text*, so in file terms a 50-book library — over a gigabyte
of PDFs — indexes in about nine minutes on WebGPU.

The important consequence is not the headroom, it is that **the ratio is not
predictable**. A diagram-heavy reference and a dense technical text can be the
same file size and differ several-fold in indexing cost. Plain `.txt` and `.md`
are ~100% text, so a notes folder is heavier per megabyte than a bookshelf.

> **Lesson.** A file-size limit is not a valid proxy for the real constraint. It
> would reject corpora that would have been fine and accept ones that will not.
> The only correct move is to extract and count *before* embedding — which is
> affordable precisely because extraction is in the 0.42%, not the 99.58%.

### 8.2 A bug that no retrieval metric could have caught

`chunk_text` built chunks by slicing token IDs and calling `tokenizer.decode()`.
That round-trips through an uncased WordPiece normalization. Measured on the
sample below, every pattern was destroyed:

| source | stored and displayed |
|---|---|
| `diffusion-weighted` | `diffusion - weighted` |
| `(fMRI)` | `( fmri )` |
| `retry_count=5` | `retry _ count = 5` |
| `config.yaml` | `config. yaml` |
| `O'Brien's` | `o ' brien ' s` |
| `[ref-12]` | `[ ref - 12 ]` |

Uppercase characters in a sample passage: **19 in the source, 0 in the stored
chunk.**

Retrieval was entirely unaffected — the embedding is computed from tokens either
way, and the re-tokenized IDs are near-identical. But the stored string is what
gets displayed as the search-result excerpt, and mangling identifiers and
filenames is the worst possible failure for a tool that searches technical
documents.

The fix keeps the character offsets the tokenizer already produces and slices the
original string:

```python
enc   = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
win   = enc["offset_mapping"][start:start + max_tokens]
chunk = text[win[0][0]:win[-1][1]]     # exact source substring
```

Verified on the real tokenizer: all patterns preserved, output is an exact
substring, and the token bound is respected *better* than before (6/967 chunks
over budget versus 47/1445).

*(§9.1 later tested this claim directly on SciFact: Δ nDCG@10 +0.0021, p = 0.448.)*

> **Lesson.** Every metric in this document is computed on embeddings, and every
> one of them was blind to this. A defect can be invisible to a complete
> evaluation suite and obvious the moment a human reads the output. Some things
> have to be looked at, not measured.

### 8.3 Chunk size does not generalize

§6 found `64/0` beat `200/30` on SciFact by ~3.5x the noise band. Does it hold on
a real book?

DDIA has no labelled queries, so: 22 natural paraphrase questions were written
for sampled passages (authored blind to chunk configuration), with the passage's
location as ground truth. Scoring uses **fixed 1,000-character reference blocks**
independent of chunking, and each retrieved chunk contributes exactly **one**
block — the one containing its midpoint. Without that rule, larger chunks would
score better purely by covering more text.

| chunk / overlap | chunks | vec MB | Hit@1 | MRR@10 | Δ vs 200/30 | 95% CI | p |
|---|---|---|---|---|---|---|---|
| 64 / 0 | 4,980 | 1.91 | 0.182 | 0.334 | −0.060 | [−0.181, +0.056] | 0.329 |
| 128 / 30 | 3,252 | 1.25 | 0.227 | 0.308 | −0.086 | [−0.249, +0.068] | 0.284 |
| **200 / 30** | 1,875 | **0.72** | **0.273** | **0.394** | — | — | — |
| 256 / 0 | 1,245 | 0.48 | 0.227 | 0.324 | −0.070 | [−0.227, +0.081] | 0.374 |

**The SciFact result reversed.** `64/0` won there and lost here. But a
10,000-sample bootstrap puts every difference on DDIA inside the noise
(p = 0.28–0.37); 22 queries cannot resolve gaps this small.

So the conclusion is not "200/30 is better" — it is that **chunk size is
corpus-dependent and not a large lever for this model.** With quality
indistinguishable, the decision falls to the axis that *is* measurable: storage.
`200/30` needs 0.72 MB against 1.91 MB, so it wins by 2.6x.

*Caveat on the record: 22 self-authored queries is an underpowered eval. It can
detect a difference of roughly ±0.18 MRR, not less.*

> **Lesson.** When two options are statistically indistinguishable, stop trying
> to separate them on that axis and decide on one where the difference is real.
> "No significant difference" is an actionable result, not a failed experiment.

### 8.4 Serving the model from a CDN survives the required headers

The app needs `Cross-Origin-Embedder-Policy` for multi-threaded WASM, which
restricts cross-origin loads. Would fetching weights from the HuggingFace CDN
still work — and would the host then need to serve 23–45 MB per visitor?

Measured under `COOP: same-origin` + `COEP: credentialless`:

| | local weights | HuggingFace CDN |
|---|---|---|
| `crossOriginIsolated` | true | **true** |
| Control cosine vs native | 0.994287 | **0.994287** |
| Throughput | 40.6 chunks/s | 39.9 chunks/s |
| Model load | 0.8 s | 4.6 s (once, then cached) |
| Weight requests to our server | 1 | **0** |

It works. The host serves ~100 KB of application code and nothing else, so
bandwidth limits and per-file caps stop being a hosting consideration. This also
sidesteps Cloudflare Pages' 25 MiB per-file cap, which would otherwise have
blocked the 45.3 MB fp16 model on the WebGPU path — a constraint that would only
have appeared in production, on GPU devices.

---

## 9. Experiment E — Cross-encoder reranking

Every knob measured so far — quantization, chunk size, overlap — moved retrieval
quality by less than the noise band, with one exception (`q4`, −0.0150) — and that one moved the wrong way. The remaining untested idea was not a knob at all but a
change to *how scoring works*.

Today the query and the passage are embedded **independently**; the query never
sees the passage. A cross-encoder runs the pair through the transformer
together, so it can weigh how the two actually relate. It cannot invent text —
its output is a permutation of passages retrieval already found — which is why
it is the one quality lever compatible with this project's rule that a result
must be real document text.

**Pipeline under test.** Bi-encoder retrieves the top-N chunks → cross-encoder
rescores those N (query, chunk) pairs → the reranked chunks occupy the head of
the ranking and the tail keeps bi-encoder order → per-doc max-pool → nDCG@10.
Cross-encoder scores are only comparable *within* the reranked set, so the head
is ordered by cross-encoder rank and lifted above every un-reranked score rather
than mixing two incompatible scales.

### 9.1 The rebuild reproduced the archive exactly — after one correction

The SciFact artifacts from §3 were gone, so everything was rebuilt. Chunk count
came back identical (12,815) but nDCG@10 came back **0.6506**, not the archived
**0.6485**. A rebuild that does not reproduce is not a baseline, so the gap was
chased rather than absorbed.

It was the §8.2 `chunk_text` fix, which landed after those runs. Reconstructing
the old `decode()`-based chunker reproduced **0.6485 to four decimals**. Only
2.8% of chunks are byte-identical between the two, and the old ones contain
**zero** uppercase characters against 341,647 in the new.

That also retroactively tests §8.2's claim that retrieval was "entirely
unaffected": Δ nDCG@10 **+0.0021, 95% CI [−0.0030, +0.0077], p = 0.448**. The
claim holds — now measured rather than argued — and the fix is very slightly
positive.

### 9.2 The candidate pool bounds everything

Reranking reorders; it cannot retrieve. The ceiling is whatever the bi-encoder
already put in the pool:

| pool depth N (chunks) | relevant-doc recall |
|---|---|
| 10 | 0.7493 |
| 20 | 0.8407 |
| 30 | 0.8702 |
| 50 | 0.8909 |
| 100 | 0.9263 |

### 9.3 Result

`cross-encoder/ms-marco-MiniLM-L-6-v2`, fp32, SciFact, 300 labeled queries,
paired bootstrap over 10,000 resamples:

| variant | nDCG@10 | Recall@10 | MRR@10 | Δ nDCG | 95% CI | p |
|---|---|---|---|---|---|---|
| bi-encoder only | 0.6506 | 0.8083 | 0.6071 | — | — | — |
| + rerank top-10 | 0.6962 | 0.8083 | 0.6659 | +0.0456 | [+0.0221, +0.0687] | <0.001 |
| + rerank top-20 | 0.7146 | 0.8357 | 0.6829 | +0.0640 | [+0.0381, +0.0909] | <0.001 |
| **+ rerank top-30** | **0.7143** | **0.8446** | **0.6816** | **+0.0637** | **[+0.0366, +0.0919]** | **<0.001** |
| + rerank top-50 | 0.7124 | 0.8394 | 0.6802 | +0.0617 | [+0.0313, +0.0924] | <0.001 |
| + rerank top-100 | 0.7075 | 0.8261 | 0.6791 | +0.0568 | [+0.0236, +0.0898] | 0.001 |

**+0.0637 is 4.2x the largest significant effect anything else in this document
produced**, and the first result here whose confidence interval is nowhere near
zero.

### 9.4 The apparent peak at N=20–30 is not real

The column reads like a curve that rises, peaks at 20–30 and declines by 100.
Reading a sweep by eye is how §6's padding trap started, so the shape was tested
pairwise rather than described:

| comparison | Δ nDCG@10 | 95% CI | p |
|---|---|---|---|
| N=30 vs N=10 | +0.0181 | [+0.0025, +0.0352] | **0.018** |
| N=30 vs N=20 | −0.0003 | [−0.0095, +0.0101] | 0.905 |
| N=30 vs N=50 | +0.0019 | [−0.0115, +0.0139] | 0.729 |
| N=30 vs N=100 | +0.0068 | [−0.0110, +0.0238] | 0.425 |

Only one comparison is real: **10 is too shallow.** From 20 to 100 the curve is
flat, so depth is not a quality decision at all — it is a latency decision, and
20–30 is simply where you stop paying for nothing. The "decline at 100" is noise
that a table invites you to narrate.

### 9.5 Two controls, because a +0.064 jump is exactly when to distrust yourself

The gain could have come from the *transformation* — lifting a head above a tail
— rather than from the cross-encoder.

| control | nDCG@10 | Δ | expectation |
|---|---|---|---|
| identity rerank (head reordered by the bi-encoder's **own** scores) | 0.6506 | **+0.0000** | must be exactly zero |
| random rerank (head shuffled) | 0.2497 | **−0.4010** | must be badly worse |

The identity control is exactly zero, so the plumbing contributes nothing and the
whole +0.0637 is the cross-encoder. The random control confirms the metric is
violently sensitive to head order, so a null result would have meant something.

### 9.6 It survives the quantization that ships

Quality was measured in fp32; the browser would download int8.

| model | size | nDCG@10 | Δ vs baseline | 95% CI | p |
|---|---|---|---|---|---|
| fp32 | 91.0 MB | 0.7143 | +0.0637 | [+0.0366, +0.0919] | <0.001 |
| **int8** | **23.1 MB** | **0.7118** | **+0.0611** | **[+0.0336, +0.0898]** | **<0.001** |

Per-query score correlation between the two is **0.9995**. The 23 MB estimate
quoted before this experiment was right, but it is now weighed rather than
asserted.

### 9.7 What it costs

Measured: **106 pairs/s** for the int8 cross-encoder on native CPU (ONNX, 4
threads) — **282 ms** to rerank 30 chunks.

Extrapolated, *not* measured in a browser: §7 found WASM runs at **0.32x**
native on this stack, which puts a top-30 rerank at roughly **0.9 s** on the CPU
path. Search goes from ~5 ms to about a second. The WebGPU path should be
several times faster but has not been measured for this model.

Note the shape of the cost: reranking runs once **per passage**, not once per
query, so it scales with N and is paid on *every search* — unlike indexing,
which is paid once.

### 9.8 One query in seven gets worse

Per-query effect at N=30, across 300 queries:

| | queries | mean Δ nDCG@10 |
|---|---|---|
| improved | 76 | +0.3861 |
| unchanged | 181 | — |
| **worsened** | **43** | **−0.2383** |

Worst single query −0.6845; best +1.0000. The average is strongly positive and
the mechanism still cannot fabricate text, but "strictly better" would have been
the wrong words: it is *better on average, with a real tail*. That asymmetry —
plus a second of latency some users will not want — is the argument for shipping
it behind a toggle rather than as the only mode.

### 9.9 Verdict

**Ship it, defaulting to top-30 and int8, behind a toggle.** It is the only
change measured in this document that improves retrieval quality significantly,
and it does so without introducing generated text.

*(§10 re-ran all of this on a real 613-page book, where the gain is 3.1x larger.)*

> **Lesson 1.** Four sweeps tuned parameters and every one of them came back as
> noise. Changing the *structure* of scoring moved the metric 4.2x more than the
> largest parameter effect. When a knob-turning campaign keeps returning null
> results, that is evidence about the knobs, not about the ceiling.

> **Lesson 2.** A sweep column has a shape, and the shape is a story your eye
> tells before the statistics do. Both real findings here — that N=10 is too
> shallow, and that 20 through 100 are indistinguishable — are invisible until
> the columns are compared *against each other* rather than each against the
> baseline.

---

## 10. Experiment F — Does reranking generalize to a real book?

§8.3 is the reason this question had to be asked. Chunk size looked settled on
SciFact and then **reversed** on DDIA. A result measured only on 5,183 scientific
abstracts is not a result about the documents people will actually drop into this
tool, so §9 was re-run end to end on the 613-page book.

### 10.1 Fixing the eval before trusting it

The DDIA harness carried a caveat on the record: *22 self-authored queries can
detect a difference of roughly ±0.18 MRR, not less.* The §9 effect is +0.064.
An underpowered eval would have returned "not significant" and that would have
meant nothing at all, so the eval was rebuilt first.

- **Corpus rebuilt and validated.** Extract → normalize → `chunk_text` at 200/30
  returned **1,875 chunks**, matching the archived run exactly.
- **91 queries instead of 22.** Blocks were sampled with a seeded permutation and
  filtered through the app's own boilerplate score (so no questions are authored
  from the index or the colophon). One natural question was written per block,
  phrased as a paraphrase rather than a quote, **before any retrieval was run** —
  so the set cannot have been shaped by what the bi-encoder happens to find.
- **Nine queries dropped automatically** where two targets landed in adjacent
  1,000-character blocks, because there the "wrong" answer is arguably right.

Ground truth stays exactly as §8.3 defined it: one reference block per query, a
chunk mapped to the block containing its midpoint.

### 10.2 Result

| variant | Hit@1 | MRR@10 | nDCG@10 | Δ nDCG | 95% CI | p |
|---|---|---|---|---|---|---|
| bi-encoder only | 0.3297 | 0.4975 | 0.5768 | — | — | — |
| + rerank top-10 | 0.5714 | 0.6670 | 0.7058 | +0.1290 | [+0.0837, +0.1761] | <0.001 |
| + rerank top-20 | 0.5934 | 0.7150 | 0.7639 | +0.1871 | [+0.1295, +0.2454] | <0.001 |
| **+ rerank top-30** | **0.6044** | **0.7236** | **0.7755** | **+0.1987** | **[+0.1406, +0.2593]** | **<0.001** |
| + rerank top-50 | 0.6044 | 0.7208 | 0.7732 | +0.1964 | [+0.1388, +0.2564] | <0.001 |
| + rerank top-100 | 0.6044 | 0.7268 | 0.7827 | +0.2060 | [+0.1456, +0.2692] | <0.001 |

**It does not just generalize — it is 3.1x larger here than on SciFact.**
Hit@1 goes from 0.3297 to 0.6044: the correct passage is ranked first for **83%
more queries**. Only **7 of 91** queries got worse, against 1 in 7 on SciFact.

Controls behave exactly as on SciFact — identity rerank **+0.0000**, random
rerank **−0.3891** — and int8 holds the gain (+0.1933, per-query score
correlation with fp32 **0.9996**), so the 23 MB model that would ship is the one
these numbers describe.

### 10.3 Depth replicates, including the part that is not real

| comparison | Δ nDCG@10 | 95% CI | p |
|---|---|---|---|
| N=30 vs N=10 | +0.0697 | [+0.0245, +0.1208] | **0.001** |
| N=30 vs N=20 | +0.0116 | [−0.0062, +0.0396] | 0.360 |
| N=30 vs N=50 | +0.0023 | [+0.0002, +0.0059] | 0.035 |
| N=30 vs N=100 | −0.0073 | [−0.0363, +0.0131] | 0.624 |

Same shape as §9.4 on a completely different corpus: **10 is too shallow, and 20
through 100 are indistinguishable.** Two conclusions replicating across corpora
is worth much more than either one alone — especially given that chunk size, the
last thing tested this way, did not replicate.

The N=30 vs N=50 row is a useful specimen: p = 0.035 clears the usual bar, and
the effect is **+0.0023** — two orders of magnitude below the rerank effect it
sits inside. It is statistically significant and practically meaningless. A
sufficiently tight interval will certify an effect far too small to care about.

### 10.4 Why it is bigger here — the obvious answer is measurably wrong

The natural explanation is that a single book is topically self-similar, so the
bi-encoder's candidates are bunched together and it cannot separate them. That is
testable, and it is **false**: mean score spread across each query's top-30 is
**0.2140 on SciFact and 0.2257 on DDIA** — essentially the same. (Within DDIA the
effect exists but is weak: queries where the bi-encoder is already confident gain
+0.1401 against +0.2189 for bunched ones, r = −0.195 over 91 queries.)

The real answer needs an **oracle rerank** — ordering the same top-30 pool
perfectly, which is the ceiling of any rerank stage, since it cannot retrieve what
retrieval missed:

| corpus | baseline | reranked | oracle | headroom | captured |
|---|---|---|---|---|---|
| SciFact | 0.6506 | 0.7143 | 0.8735 | 0.2228 | **28.6%** |
| DDIA | 0.5768 | 0.7755 | 0.9341 | 0.3573 | **55.6%** |

Two separate things are true. DDIA has **60% more headroom** — the bi-encoder
starts weaker on book prose than on abstracts — *and* the cross-encoder converts
**nearly twice as much** of what is available. The second is the interesting half,
and it is not explained by headroom.

A plausible reason, stated as a hypothesis and **not measured**: SciFact queries
are single-sentence scientific claims matched against abstracts, which is close to
what the bi-encoder was trained on, whereas DDIA queries are conversational
questions against continuous prose — close to the MS MARCO passage-ranking task
the *cross-encoder* was trained on.

### 10.5 Verdict

**§9.9 is confirmed and strengthened.** Reranking was already the only significant
quality lever measured in this document; on the corpus that actually resembles the
target workload it is three times larger, misfires on half as many queries, and
lifts top-1 accuracy from one query in three to three in five.

> **Lesson.** §8.3 concluded that a tuning result found on a benchmark did not
> survive contact with a real document. It would have been easy to generalize that
> into "benchmarks don't transfer." They do — *structural* findings replicated here
> across two corpora that disagree about everything else. What failed to transfer
> in §8.3 was a parameter fitted to one corpus's statistics. The distinction is
> worth keeping: benchmarks are weak evidence about tuning and strong evidence
> about architecture.

> **Corollary on eval power.** This experiment began by rebuilding the eval rather
> than running on the one that existed. With 22 queries the honest answer to "does
> reranking generalize?" would have been "cannot tell," and an eval that cannot
> reject anything will quietly launder null results into false reassurance.

---

## 11. Experiment G — The time complexity of indexing

Reranking (§9, §10) is a *search*-time cost and adds nothing to indexing. Indexing
has its own economics, and they turned out to contradict the intuition the rest of
this document was built on.

### 11.1 The forward pass is superlinear in length — and the textbook estimate was 4x wrong

Per transformer layer, attention costs about `L^2 * d` and the feed-forward network
about `L * d^2`. With `d = 384` and `L <= 256`, the ratio `L/d` never exceeds 0.67,
so the linear term should dominate and the quadratic term should be a rounding
error. **Predicted quadratic share at L=256: ~8%.**

Measured with uniform-length batches, so batch padding cannot contaminate it
(ONNX int8, 4 threads, batch 32):

| L | ms/chunk | µs/token | chunks/s |
|---|---|---|---|
| 16 | 0.439 | 27.46 | 2276 |
| 64 | 1.890 | 29.53 | 529 |
| 128 | 4.265 | 33.32 | 235 |
| 192 | 7.092 | 36.94 | 141 |
| 256 | 10.153 | 39.66 | 98.5 |

```
linear fit      T = 1.2910*L - 18.729                       R^2 = 0.99323
quadratic fit   T = 0.001580*L^2 + 0.8704*L - 0.855         R^2 = 0.99986
quadratic share of cost at L=256                            31.7%
```

**Not 8% — 31.7%.** A token costs **44% more** to embed inside a 256-token chunk
than inside a 16-token one. Counting FLOPs got the answer wrong by a factor of
four, because at int8 the cost is set by kernels, dequantization and memory
traffic rather than by arithmetic.

### 11.2 The cost model, validated against wall clock

That fit gives a predictive model. Total indexing time is

```
T  ~  corpus_tokens  x  A  x  cost_per_token(C)          A = C / (C - V)
```

for chunk size `C` and overlap `V`. Predicted against measured on DDIA (318,660
corpus tokens, batch 32):

| chunk/overlap | chunks | tokens fed | amplification | predicted | actual | error |
|---|---|---|---|---|---|---|
| 64 / 0 | 4,980 | 328,680 | 1.031 | 9.9 s | **10.4 s** | −5.1% |
| 128 / 0 | 2,490 | 323,681 | 1.016 | 10.8 s | 11.1 s | −2.6% |
| 128 / 30 | 3,252 | 422,740 | 1.327 | 14.2 s | 14.6 s | −2.7% |
| **200 / 30** *(shipped)* | 1,875 | 378,662 | 1.188 | 14.1 s | **14.6 s** | −3.1% |
| 256 / 0 | 1,245 | 321,141 | 1.008 | 12.8 s | 13.6 s | −5.6% |
| 256 / 50 | 1,547 | 399,035 | 1.252 | 16.1 s | 16.2 s | −0.7% |

Within 0.7–5.6%, consistently a little under (per-batch dispatch overhead the fit
does not include). **Indexing is linear in corpus size** — this is the property
that makes browser-side indexing viable at all; a 2x longer book takes 2x longer,
with no blow-up.

### 11.3 Chunk count is not the cost

`64/0` produces **four times as many chunks** as `256/0` and indexes **31% faster**
(10.4 s vs 13.6 s). Any reasoning of the form "fewer chunks, less work" is wrong;
what is conserved is roughly the token count, and per-token cost *falls* as chunks
get shorter.

This retires an assumption made early in this project — that chunk size drives
indexing cost through the number of embeddings produced. It drives it through
overlap.

### 11.4 Overlap is the expensive knob, and it costs exactly its amplification

Hold chunk size fixed and add overlap:

| | amplification | time | |
|---|---|---|---|
| 128 / 0 | 1.016 | 11.1 s | |
| 128 / 30 | 1.327 | 14.6 s | **+31.5% time for +30.6% amplification** |

The two match to within 1%. Overlap is not a subtle effect to be traded off — it
is a straight multiplier on indexing time, and `A = C/(C-V)` predicts it outright.

### 11.5 Padding waste is a property of the corpus, not of the configuration

A batch pads to its longest member. §6 measured 31.6% waste on SciFact and
concluded length-sorted batching would recover ~24%. That holds, but it is not a
general fact about indexing:

| | chunks | length sd | padding waste | wall-clock saved by sorting |
|---|---|---|---|---|
| SciFact 200/30 (5,183 abstracts) | 12,815 | 63 | **30.7%** | **28.8%** |
| SciFact 64/0 | 29,797 | 14 | 10.6% | — |
| DDIA 200/30 (one 613-page book) | 1,875 | **3** | **0.3%** | 0.6% *(noise)* |

**Sorting is worth 28.8% on a folder of short documents and nothing at all on a
single book**, because a book's chunks are all exactly full-length except the last
one. The app does it unconditionally, which is right — it is free — but its value
depends entirely on what the user drops in.

Note that sorting removes 23.2% of padded *tokens* and 28.8% of *time*. The gap is
§11.1 again: the padding it eliminates sits at the long end of the batch, where
per-token cost is highest.

### 11.6 Threads: past four, more is worse

The app requests `min(hardwareConcurrency, 8)` threads, which is the whole reason
the COOP/COEP work in §8.4 exists — without cross-origin isolation WASM silently
drops to one thread. Measured on a Mac (DDIA, ONNX int8):

| threads | seconds | chunks/s | speedup | efficiency |
|---|---|---|---|---|
| 1 | 28.1 | 66.6 | 1.00x | 100% |
| 2 | 19.3 | 97.3 | 1.46x | 73% |
| **4** | **15.3** | **122.3** | **1.84x** | 46% |
| 6 | 16.8 | 111.3 | 1.67x | 28% |
| 8 | 16.9 | 110.8 | 1.66x | 21% |
| 10 | 17.7 | 105.7 | 1.59x | 16% |

Threading is worth having — 1.84x is most of §8.4's justification — but it
**peaks at 4 and degrades after**, so the shipped `min(hardwareConcurrency, 8)`
would pick 8 on this machine and run ~10% slower than 4.

*Measured natively (ONNX Runtime on a Mac), not in a browser.* WASM threading uses
a different mechanism, so this is a flag for a browser measurement, not yet a
reason to change the constant.

### 11.7 The trade, stated plainly

For the shipped `200/30` against the fastest-indexing alternative:

| | 200/30 *(shipped)* | 64/0 |
|---|---|---|
| indexing time (DDIA) | 14.6 s | **10.4 s** (**1.40x faster**) |
| vector storage (DDIA) | **0.72 MB** | 1.91 MB (2.65x more) |
| retrieval quality | statistically indistinguishable (§8.3) | statistically indistinguishable (§8.3) |

So `200/30` buys **2.65x less storage for 40% more indexing time**. Indexing is
paid once; storage is paid permanently and must sit in memory for every search.
The existing choice survives — but on the cost axis it was previously justified by,
it is the *loser*, and §6's remark that "small chunks are faster" was right for a
reason it did not identify.

> **Lesson.** Three of this project's cost intuitions were derived from structure
> rather than measured: that transformers are quadratic in length here (31.7%,
> not the ~8% the FLOP ratio implies), that chunk count drives indexing cost (it
> does not — 4x the chunks ran 31% faster), and that padding waste is a property
> of the pipeline (it is a property of the corpus: 30.7% versus 0.3%). Complexity
> analysis tells you the *shape* of a curve. It does not tell you which term
> dominates at the sizes you actually run, and that is the only thing that decides
> anything.
