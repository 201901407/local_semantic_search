/**
 * Telling prose apart from a book's furniture.
 *
 * A table of contents, a bibliography and an index are all *text*, and they
 * embed perfectly well — a citation titled "Consistent Hashing and Random
 * Trees" sits close to a query about consistent hashing and will outrank the
 * paragraph that actually explains it. The passage is a correct match and a
 * useless answer: it contains a title, not an explanation.
 *
 * Measured on a 613-page technical book, this score is strongly bimodal:
 *
 *     p50    2.3   ordinary prose
 *     p85    9.7
 *                  <- a clear gap
 *     p90   36.9   references, index, contents
 *     max  240.8   table-of-contents dot leaders
 *
 * Real prose that merely cites a source or mentions a year lands around 7-12,
 * so the cutoff sits in the gap at 25. A lower threshold is actively harmful:
 * at 12 it discarded a genuine passage about eventual consistency whose only
 * offence was carrying page cross-references.
 */

/** Mid-gap. Prose stays below ~12; real furniture starts around 37. */
export const BOILERPLATE_THRESHOLD = 25;

/**
 * Score how much a passage looks like reference material rather than prose.
 *
 * Every signal is normalised per 1000 characters so long and short passages are
 * judged alike. Digit and comma density carry an allowance, because ordinary
 * writing contains some of both — only the excess counts.
 *
 * @param {string} text
 * @returns {number} 0 for clean prose; tens to hundreds for furniture
 */
export function boilerplateScore(text) {
  const length = Math.max(text.length, 1);
  const per1k = 1000 / length;

  const citations = (text.match(/\[\s*\d+\s*\]/g) ?? []).length;
  const years = (text.match(/\b(?:19|20)\d{2}\b/g) ?? []).length;
  const etAl = (text.match(/\bet al\./g) ?? []).length;
  const links = (text.match(/https?:\/\/|www\.|\.com|\.org|doi:/gi) ?? []).length;
  const quoted = (text.match(/[“”]/g) ?? []).length;
  const dotLeaders = (text.match(/\. \. \./g) ?? []).length;

  let digits = 0;
  let commas = 0;
  for (const character of text) {
    if (character >= '0' && character <= '9') digits++;
    else if (character === ',') commas++;
  }

  return citations * 2.0 * per1k
    + years * 1.5 * per1k
    + etAl * 3.0 * per1k
    + links * 1.5 * per1k
    + quoted * 0.8 * per1k
    + dotLeaders * 4.0 * per1k
    + Math.max(0, digits / length - 0.04) * 60
    + Math.max(0, (commas / length) * 100 - 3.0) * 1.2;
}

/**
 * Is this passage book furniture rather than content?
 *
 * Flagged passages are still indexed and still searchable — they are simply
 * held back until the result list would otherwise be short. Deleting them would
 * make "which paper is cited for this?" unanswerable, and would discard text
 * the reader gave us.
 */
export const isBoilerplate = (text, threshold = BOILERPLATE_THRESHOLD) =>
  boilerplateScore(text) > threshold;
