/**
 * Choosing which part of a passage to show.
 *
 * A chunk is around 900 characters but only ~420 are displayed, so this choice
 * decides whether a reader sees the answer or the paragraph before it. Getting
 * it wrong looks exactly like a retrieval failure: a correct hit that appears
 * to be about something else.
 */

export const EXCERPT_CHARS = 420;

/** Characters of lead-in kept before the matched term. */
const LEAD_IN = 110;

/**
 * Words too common to locate a passage by.
 *
 * Without this, "how does consistent hashing work" anchors the excerpt on the
 * first "how" in the text, and the paragraph the reader wanted never appears.
 * Kept deliberately short: only words that are near-useless for locating a
 * passage. Nouns that could carry meaning in some query ("work", "state") stay
 * out of the list, because window scoring counts *distinct* terms and will
 * prefer a window covering two specific words over one repeating a vague one.
 */
export const STOPWORDS = new Set([
  'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'was', 'one',
  'how', 'does', 'did', 'what', 'why', 'when', 'where', 'who', 'which', 'with',
  'from', 'this', 'that', 'these', 'those', 'into', 'over', 'than', 'then',
  'they', 'them', 'has', 'have', 'had', 'its', 'about', 'would', 'could',
  'should', 'been', 'being', 'were', 'will', 'your', 'his', 'her', 'their',
]);

/**
 * Reduce a query to the words worth locating and highlighting.
 * @param {string} query
 * @returns {string[]} lowercased, de-duplicated, stopwords removed
 */
export function queryTerms(query) {
  return [...new Set(query.toLowerCase().split(/\W+/))]
    .filter((word) => word.length > 2 && !STOPWORDS.has(word));
}

/**
 * Pick the window of `text` that best covers `terms`.
 *
 * Windows are scored by how many *distinct* terms they contain rather than by
 * total matches, so a passage mentioning "hashing" once near the end beats one
 * repeating a vaguer word at the start. Total term length breaks ties, nudging
 * toward the more specific vocabulary.
 *
 * @param {string} text
 * @param {string[]} terms
 * @param {number} [budget]
 * @returns {string} the excerpt, with ellipses where it was cut
 */
export function selectExcerpt(text, terms, budget = EXCERPT_CHARS) {
  if (text.length <= budget) return text;

  const lower = text.toLowerCase();
  const marks = [];
  for (const term of terms) {
    for (let at = lower.indexOf(term); at !== -1; at = lower.indexOf(term, at + 1)) {
      marks.push({ at, term });
    }
  }
  if (marks.length === 0) return `${text.slice(0, budget).trim()}…`;

  const lastStart = Math.max(0, text.length - budget);
  let bestStart = 0;
  let bestScore = -1;

  for (const mark of marks) {
    const start = Math.max(0, Math.min(mark.at - LEAD_IN, lastStart));
    const covered = new Set();
    for (const other of marks) {
      if (other.at >= start && other.at + other.term.length <= start + budget) {
        covered.add(other.term);
      }
    }
    const score = covered.size * 1000
      + [...covered].reduce((total, term) => total + term.length, 0);
    if (score > bestScore) {
      bestScore = score;
      bestStart = start;
    }
  }

  // Snap forward to a word boundary so an excerpt never opens mid-word.
  let start = bestStart;
  if (start > 0) {
    const space = text.indexOf(' ', start);
    if (space !== -1 && space - start <= 40) start = space + 1;
  }
  const end = Math.min(text.length, start + budget);

  return `${start > 0 ? '…' : ''}${text.slice(start, end).trim()}${end < text.length ? '…' : ''}`;
}
