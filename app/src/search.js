/**
 * Vector storage and retrieval.
 *
 * There is no approximate-nearest-neighbour index here, deliberately. Scoring
 * every chunk is 2 × chunks × 384 multiply-adds: about 1 ms for a 600-page book
 * and 51 ms at 100,000 chunks. An ANN structure exists to avoid exactly that
 * scan, but the scan only becomes noticeable at a scale where browser memory
 * has already failed. Adding one would cost bytes and complexity to solve a
 * problem this application never reaches.
 */

export const DIMENSIONS = 384;

/**
 * Quantize L2-normalized float32 vectors to int8 with a single shared scale.
 *
 * Measured cost: Δ −0.0007 nDCG, p = 0.18 — indistinguishable from zero, for a
 * 4× reduction in memory and stored size.
 */
export function quantizeToInt8(values) {
  let peak = 0;
  for (let i = 0; i < values.length; i++) {
    const magnitude = Math.abs(values[i]);
    if (magnitude > peak) peak = magnitude;
  }
  const scale = (peak || 1) / 127;
  const data = new Int8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    data[i] = Math.round(values[i] / scale);
  }
  return { scale, data };
}

/**
 * All indexed vectors, held as one contiguous Int8Array.
 *
 * Rows are appended per file so that adding or removing a document does not
 * require re-encoding anything else.
 */
export class VectorIndex {
  constructor() {
    this.matrix = new Int8Array(0);
    this.scales = new Float32Array(0);
    this.rows = [];             // { fileId, ordinal } parallel to matrix rows
    this.count = 0;
  }

  get isEmpty() {
    return this.count === 0;
  }

  /**
   * Append one file's vectors.
   * @param {string} fileId
   * @param {number} scale
   * @param {Int8Array} data     chunkCount × DIMENSIONS int8 values
   * @param {boolean[]} [boiler] per-chunk "is book furniture" flags
   */
  add(fileId, scale, data, boiler = []) {
    const incoming = data.length / DIMENSIONS;
    if (!Number.isInteger(incoming)) {
      throw new Error(`Vector data for ${fileId} is not a multiple of ${DIMENSIONS}`);
    }

    const grown = new Int8Array(this.matrix.length + data.length);
    grown.set(this.matrix);
    grown.set(data, this.matrix.length);
    this.matrix = grown;

    const scales = new Float32Array(this.count + incoming);
    scales.set(this.scales);
    scales.fill(scale, this.count);
    this.scales = scales;

    for (let ordinal = 0; ordinal < incoming; ordinal++) {
      this.rows.push({ fileId, ordinal, boiler: Boolean(boiler[ordinal]) });
    }
    this.count += incoming;
  }

  /** Drop every row belonging to `fileId`, preserving the order of the rest. */
  remove(fileId) {
    const keep = [];
    for (let row = 0; row < this.rows.length; row++) {
      if (this.rows[row].fileId !== fileId) keep.push(row);
    }
    if (keep.length === this.rows.length) return;

    const matrix = new Int8Array(keep.length * DIMENSIONS);
    const scales = new Float32Array(keep.length);
    keep.forEach((row, target) => {
      matrix.set(this.matrix.subarray(row * DIMENSIONS, (row + 1) * DIMENSIONS),
                 target * DIMENSIONS);
      scales[target] = this.scales[row];
    });

    this.matrix = matrix;
    this.scales = scales;
    this.rows = keep.map((row) => this.rows[row]);
    this.count = keep.length;
  }

  /**
   * Score every chunk against `query` and return the best `k`.
   *
   * The query stays float32 and only stored vectors are quantized. The measured
   * quantization penalty above came from quantizing both sides, so this is
   * strictly less lossy than what was tested.
   *
   * @param {Float32Array} query  L2-normalized query embedding
   * @param {number} k
   * @returns {Array<{fileId: string, ordinal: number, score: number}>}
   */
  search(query, k = 10) {
    const { matrix, scales, count } = this;
    const best = [];   // kept sorted, descending; k is small so this is cheap
    let floor = -Infinity;

    for (let row = 0; row < count; row++) {
      const offset = row * DIMENSIONS;
      let dot = 0;
      for (let dim = 0; dim < DIMENSIONS; dim++) {
        dot += query[dim] * matrix[offset + dim];
      }
      const score = dot * scales[row];
      if (best.length === k && score <= floor) continue;

      const entry = { ...this.rows[row], score };
      const at = best.findIndex((candidate) => candidate.score < score);
      best.splice(at === -1 ? best.length : at, 0, entry);
      if (best.length > k) best.pop();
      floor = best[best.length - 1].score;
    }
    return best;
  }
}

/**
 * Thin the raw chunk hits into a readable result list.
 *
 * Two problems to solve. Neighbouring chunks overlap by design, so adjacent
 * ordinals return near-identical text and would show as duplicates. And one
 * large document can hold every good passage, so an unconstrained list would
 * let a single book crowd out a short document with the better answer.
 *
 * The per-file cap is a *preference*, not a hard rule. Applying it strictly
 * would mean a library of one document could never return more than
 * `maxPerFile` results however many the reader asked for. So it runs in two
 * passes: the first spreads results across documents, and the second fills any
 * remaining slots from wherever the best passages are.
 *
 * Passages flagged as book furniture — bibliographies, indexes, tables of
 * contents — are held back to a final tier. They embed convincingly (a citation
 * titled "Consistent Hashing and Random Trees" scores highly against a query
 * about consistent hashing) while containing a title rather than an
 * explanation. They stay searchable, but only surface once real content runs
 * out.
 *
 * The passes decide *which* passages appear; they do not decide the order they
 * are shown in. Each pass is internally sorted, but concatenating them is not:
 * a weak passage admitted by the first pass for diversity would otherwise sit
 * above a stronger one picked up by the second. So the selection is re-sorted
 * by score before returning, and the reader always sees descending relevance.
 *
 * @param {Array} hits          scored hits, best first
 * @param {number} maxPerFile   soft cap, relaxed if the library is too small
 * @param {number} limit        how many results the reader asked for
 */
export function diversify(hits, { maxPerFile = 3, limit = 8 } = {}) {
  const taken = new Map();   // fileId -> ordinals already chosen
  const chosen = [];

  const consider = (hit, cap) => {
    const ordinals = taken.get(hit.fileId) ?? [];
    if (ordinals.length >= cap) return;
    // Rejects both adjacent chunks (overlapping text) and exact repeats,
    // which is what makes the second pass safe to run over the same list.
    if (ordinals.some((ordinal) => Math.abs(ordinal - hit.ordinal) <= 1)) return;

    ordinals.push(hit.ordinal);
    taken.set(hit.fileId, ordinals);
    chosen.push(hit);
  };

  // Tier 1: real content, spread across documents.
  for (const hit of hits) {
    if (chosen.length >= limit) break;
    if (!hit.boiler) consider(hit, maxPerFile);
  }
  // Tier 2: real content, per-file cap relaxed to fill the request.
  if (chosen.length < limit) {
    for (const hit of hits) {
      if (chosen.length >= limit) break;
      if (!hit.boiler) consider(hit, Infinity);
    }
  }
  // Tier 3: book furniture, only because nothing better is left.
  if (chosen.length < limit) {
    for (const hit of hits) {
      if (chosen.length >= limit) break;
      consider(hit, Infinity);
    }
  }
  return chosen.sort((a, b) => b.score - a.score);
}
