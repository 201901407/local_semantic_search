/**
 * File admission checks.
 *
 * Threat model for a fully client-side app: we never execute an uploaded file,
 * so the danger is not "running" it. The two real risks are
 *
 *   1. Type confusion — something claiming to be .txt that is actually a binary,
 *      or a .pdf that is really a zip. A mismatched parser on hostile bytes is
 *      where parser exploits live, so we check the magic bytes rather than
 *      trusting the extension.
 *   2. Injection at render time — a document whose *text* contains markup that
 *      would execute if written into the DOM as HTML.
 *
 * Risk 2 is defended structurally, not here: everything user-supplied is
 * inserted with textContent (see ui.js), so markup in a document is inert text.
 * We still surface a warning, because a reader deserves to know. We deliberately
 * do NOT reject on it: this is a tool for searching technical documents, and
 * those legitimately contain code, script tags and shell snippets. Rejecting
 * every file containing `<script>` would break the primary use case while adding
 * no safety that textContent has not already provided.
 */

/** Extensions we can extract text from, with the magic bytes each must start with. */
const SIGNATURES = {
  pdf:  [[0x25, 0x50, 0x44, 0x46]],                 // "%PDF"
  docx: [[0x50, 0x4b, 0x03, 0x04],                  // "PK\x03\x04" — zip container
         [0x50, 0x4b, 0x05, 0x06]],                 // empty archive
  txt:  null,                                       // no signature; validated as text
  md:   null,
};

export const SUPPORTED_EXTENSIONS = Object.keys(SIGNATURES);

export const LIMITS = {
  maxFileBytes: 100 * 1024 * 1024,   // a 100 MB PDF is ~2,600 pages of text
  maxTextChars: 20 * 1024 * 1024,    // ~13,000 pages; enforced, not advisory
  maxChunksPerFile: 50000,           // bounds worker memory for one document
  maxDocxXmlBytes: 64 * 1024 * 1024, // uncompressed ceiling for word/document.xml
  sniffBytes: 4096,                  // classification only — never the sole check

  // Bounded work. Every bomb patched above was a case of unbounded work from
  // bounded input, so these close the category rather than one instance of it.
  // Note the units: a whole-file budget would punish long documents, which are
  // legitimately slow. Bounding each STEP lets cost scale with document size
  // while guaranteeing no single step can hang.
  extractTimeoutMs: 5 * 60 * 1000,   // parsing one document
  batchTimeoutMs: 60 * 1000,         // embedding one batch of 64 passages
  rerankTimeoutMs: 30 * 1000,        // rescoring one query's 30 candidates
  maxFilesPerDrop: 100,              // one drag-and-drop
  maxBytesPerDrop: 500 * 1024 * 1024,
};

/**
 * Bidirectional formatting characters.
 *
 * U+202E and friends reorder the glyphs that follow, so "report\u202Efdp.txt"
 * renders as "reporttxt.pdf". A reader trusting the displayed name cannot tell
 * what they actually added, so these are stripped before a name is shown.
 */
const BIDI_CONTROLS = /[\u202A-\u202E\u2066-\u2069\u200E\u200F\u061C]/g;

/** A filename safe to display: no bidi tricks, no control characters. */
export const sanitizeFilename = (name) =>
  name.replace(BIDI_CONTROLS, '')
      // eslint-disable-next-line no-control-regex
      .replace(/[\u0000-\u001F\u007F]/g, '')
      .trim() || 'unnamed';

/** Patterns worth telling the reader about. Informational only — never a rejection. */
const ACTIVE_CONTENT = [
  /<script[\s>]/i,
  /javascript:/i,
  /\son\w+\s*=\s*["']/i,             // inline event handlers: onerror=, onload=
  /<iframe[\s>]/i,
];

export function extensionOf(filename) {
  const dot = filename.lastIndexOf('.');
  return dot === -1 ? '' : filename.slice(dot + 1).toLowerCase();
}

const startsWith = (bytes, signature) =>
  signature.every((byte, i) => bytes[i] === byte);

/**
 * Does this look like text rather than a binary blob?
 *
 * NUL bytes are the strongest signal — they effectively never occur in real
 * UTF-8 prose but are everywhere in executables and media. We also reject a high
 * proportion of other control characters, which catches binaries that happen to
 * avoid NUL.
 */
function looksLikeText(bytes) {
  let control = 0;
  for (const byte of bytes) {
    if (byte === 0x00) return false;
    // Allow tab (0x09), newline (0x0a), form feed (0x0c), carriage return (0x0d).
    if (byte < 0x20 && byte !== 0x09 && byte !== 0x0a && byte !== 0x0c && byte !== 0x0d) control++;
  }
  return bytes.length === 0 || control / bytes.length < 0.05;
}

const reject = (reason) => ({ ok: false, reason, warnings: [] });

/**
 * Inspect a File before any parser touches it.
 *
 * @param {File} file
 * @returns {Promise<{ok: boolean, kind?: string, reason?: string, warnings: string[]}>}
 *          `kind` is the verified extension, safe to dispatch a parser on.
 */
export async function inspectFile(file) {
  const kind = extensionOf(file.name);

  if (!(kind in SIGNATURES)) {
    return reject(`Unsupported file type ".${kind || 'unknown'}"`);
  }
  if (file.size === 0) {
    return reject('File is empty');
  }
  if (file.size > LIMITS.maxFileBytes) {
    const mb = (LIMITS.maxFileBytes / 1024 / 1024) | 0;
    return reject(`File is larger than the ${mb} MB limit`);
  }

  const head = new Uint8Array(await file.slice(0, LIMITS.sniffBytes).arrayBuffer());
  const signatures = SIGNATURES[kind];

  // Binary formats: the bytes must match the extension's signature.
  if (signatures) {
    if (!signatures.some((signature) => startsWith(head, signature))) {
      return reject(`Contents do not match a .${kind} file — it may be mislabelled`);
    }
    return { ok: true, kind, warnings: [] };
  }

  // Text formats: must decode as UTF-8 and must not be a binary in disguise.
  if (!looksLikeText(head)) {
    return reject(`Contents look binary, not text — it may be mislabelled as .${kind}`);
  }
  try {
    new TextDecoder('utf-8', { fatal: true }).decode(head);
  } catch {
    return reject('File is not valid UTF-8 text');
  }

  return { ok: true, kind, warnings: [] };
}

/**
 * Scan extracted text for markup that would be dangerous if it were ever
 * rendered as HTML. Returns warnings to display; it never blocks indexing.
 */
export function scanExtractedText(text) {
  const warnings = [];
  if (ACTIVE_CONTENT.some((pattern) => pattern.test(text))) {
    warnings.push('Contains markup or script-like content. It is shown as plain text and cannot run.');
  }
  if (text.length > LIMITS.maxTextChars) {
    warnings.push('Unusually large amount of text; only the first portion was indexed.');
  }
  return warnings;
}

/**
 * Clamp extracted text to the ceiling, reporting whether it was cut.
 *
 * This is the enforcement point for `maxTextChars`. It was previously exported
 * and never called, which meant a 31 MB document sailed past a stated 20 MB
 * limit — a guard that exists only in documentation is not a guard.
 */
export function clampText(text) {
  if (text.length <= LIMITS.maxTextChars) return { text, clamped: false };
  return { text: text.slice(0, LIMITS.maxTextChars), clamped: true };
}

/**
 * Full-content check for text formats, run after decoding.
 *
 * `inspectFile` only sees the first few kilobytes, which a file can trivially
 * survive by placing prose at the front and binary after it. This inspects
 * everything that was actually decoded.
 */
export function verifyDecodedText(text) {
  if (text.includes('\u0000')) {
    return { ok: false, reason: 'Contains NUL bytes — this is not a text file' };
  }
  let control = 0;
  const sample = text.length > 1e6 ? text.slice(0, 1e6) : text;
  for (const character of sample) {
    const code = character.codePointAt(0);
    if (code < 0x20 && code !== 9 && code !== 10 && code !== 12 && code !== 13) control++;
  }
  if (sample.length && control / sample.length > 0.05) {
    return { ok: false, reason: 'Mostly control characters — this is not readable text' };
  }
  return { ok: true };
}
