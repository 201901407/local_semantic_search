/**
 * Text extraction, one parser per file type.
 *
 * Every parser is loaded with a dynamic import so it is fetched only when a file
 * of that type first appears. Someone searching Markdown never downloads pdf.js.
 * That lazy boundary is the single biggest lever on initial page weight.
 *
 * Extraction is cheap — 0.42% of indexing cost against 99.58% for the forward
 * pass — which is what makes it affordable to extract and count *before*
 * committing to embedding, and so quote the user a real estimate.
 */

import { LIMITS, verifyDecodedText } from './validate.js';

const PDFJS_VERSION = '4.7.76';
const FFLATE_VERSION = '0.8.2';

/** A 613-page technical book is ~1.4 MB of text; this is generous headroom. */
export const MAX_PDF_PAGES = 3000;

let pdfjsPromise = null;
let fflatePromise = null;

function loadPdfJs() {
  pdfjsPromise ??= (async () => {
    const base = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${PDFJS_VERSION}/build`;
    const pdfjs = await import(/* @vite-ignore */ `${base}/pdf.min.mjs`);
    pdfjs.GlobalWorkerOptions.workerSrc = `${base}/pdf.worker.min.mjs`;
    return pdfjs;
  })();
  return pdfjsPromise;
}

function loadFflate() {
  fflatePromise ??= import(
    /* @vite-ignore */ `https://cdn.jsdelivr.net/npm/fflate@${FFLATE_VERSION}/esm/browser.js`
  );
  return fflatePromise;
}

const decodeUtf8 = (bytes) => new TextDecoder('utf-8').decode(bytes);

async function extractPdf(buffer, onProgress) {
  const pdfjs = await loadPdfJs();
  const doc = await pdfjs.getDocument({
    data: buffer,
    // None of these matter for text extraction, and every one of them is a
    // way for a hostile document to reach more of the parser. A PDF carrying
    // an /OpenAction /JavaScript payload is inert with scripting off.
    disableFontFace: true,
    isEvalSupported: false,
    enableScripting: false,
    disableAutoFetch: true,
  }).promise;

  const pageCount = Math.min(doc.numPages, MAX_PDF_PAGES);
  const pages = [];
  for (let pageNumber = 1; pageNumber <= pageCount; pageNumber++) {
    const page = await doc.getPage(pageNumber);
    const content = await page.getTextContent();
    pages.push(content.items.map((item) => item.str).join(' '));
    page.cleanup();
    if (onProgress && pageNumber % 25 === 0) onProgress(pageNumber, pageCount);
  }
  await doc.destroy();

  return {
    text: pages.join('\n'),
    truncated: doc.numPages > MAX_PDF_PAGES,
    pages: pageCount,
  };
}

/**
 * A .docx is a zip whose word/document.xml holds the prose.
 *
 * We unzip with fflate (~8 KB) and pull the text runs out directly rather than
 * using a full docx library. Those libraries target high-fidelity HTML
 * conversion; we want plain text, so most of their weight would be unused.
 */
async function extractDocx(buffer) {
  const { unzipSync } = await loadFflate();

  // Decompress ONLY the one entry we read, and refuse it if the archive's own
  // header says it expands beyond the ceiling. Without this a 220 KB file
  // expands to 99 MB in half a second (a 456:1 ratio, measured), and at the
  // 100 MB file limit that is tens of gigabytes of memory.
  let declared = 0;
  const archive = unzipSync(new Uint8Array(buffer), {
    filter: (file) => {
      if (file.name !== 'word/document.xml') return false;
      declared = file.originalSize ?? 0;
      return declared <= LIMITS.maxDocxXmlBytes;
    },
  });

  const entry = archive['word/document.xml'];
  if (!entry) {
    throw new Error(declared > LIMITS.maxDocxXmlBytes
      ? 'Document body is implausibly large — refusing to decompress it'
      : 'No word/document.xml — not a Word document');
  }
  // The header can lie; the decompressed size cannot.
  if (entry.length > LIMITS.maxDocxXmlBytes) {
    throw new Error('Document body exceeded its declared size — refusing it');
  }

  const xml = decodeUtf8(entry);
  const paragraphs = xml.split(/<\/w:p>/).map((paragraph) => {
    const runs = [...paragraph.matchAll(/<w:t[^>]*>([\s\S]*?)<\/w:t>/g)];
    return runs.map((run) => run[1]).join('');
  });

  const text = paragraphs
    .join('\n')
    // The runs are XML-escaped; undo that so the stored excerpt reads correctly.
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&');

  return { text, truncated: false };
}

/**
 * Extract plain text from a supported file.
 *
 * @param {ArrayBuffer} buffer  raw file bytes
 * @param {string} kind         verified extension from validate.inspectFile
 * @param {Function} [onProgress] called as (done, total) for long documents
 * @returns {Promise<{text: string, truncated: boolean, pages?: number}>}
 */
export async function extractText(buffer, kind, onProgress) {
  switch (kind) {
    case 'txt':
    case 'md': {
      // inspectFile only sniffed the first few KB. Now that the whole file is
      // decoded, check all of it — prose at the front proves nothing.
      const text = decodeUtf8(new Uint8Array(buffer));
      const verdict = verifyDecodedText(text);
      if (!verdict.ok) throw new Error(verdict.reason);
      return { text, truncated: false };
    }
    case 'pdf':
      return extractPdf(buffer, onProgress);
    case 'docx':
      return extractDocx(buffer);
    default:
      throw new Error(`No extractor for .${kind}`);
  }
}

/**
 * A PDF with no text layer is a scan. It is not an error, and it is not an empty
 * document — it needs OCR, which v1 does not do. Saying so plainly is far better
 * than indexing nothing and appearing broken.
 */
export const isProbablyScanned = (kind, text) =>
  kind === 'pdf' && text.replace(/\s/g, '').length < 32;
