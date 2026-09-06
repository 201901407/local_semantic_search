/**
 * Bounding how long a unit of work may take.
 *
 * Every resource bomb this app defends against — a zip that expands to
 * gigabytes, a document larger than its declared size, a malformed PDF that
 * sends a parser in circles — is the same shape: unbounded work from bounded
 * input. Size limits stop the instances you thought of. A deadline stops the
 * category, including the ones you did not.
 *
 * Apply it per STEP, never per file. A whole-file budget would punish long
 * documents, which are legitimately slow: a 3,000-page book takes minutes and
 * is perfectly valid. Bounding each step lets total cost scale with document
 * size while guaranteeing that no single step can hang.
 */

/**
 * @param {Promise} promise  work to bound
 * @param {number} ms        deadline in milliseconds
 * @param {string} label     used in the error, so failures name themselves
 * @returns {Promise} resolves with `promise`, or rejects once `ms` elapses
 */
export function withTimeout(promise, ms, label = 'Operation') {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(
        () => reject(new Error(`${label} timed out after ${ms / 1000}s`)),
        ms,
      );
    }),
  ]).finally(() => clearTimeout(timer));
}
