/**
 * Model layer: loads the ONNX sentiment classifier and aggregates its output.
 *
 * Model: Xenova/twitter-roberta-base-sentiment-latest
 *   = cardiffnlp/twitter-roberta-base-sentiment-latest exported to ONNX (q8)
 *   genuine three-class head: negative (0), neutral (1), positive (2)
 *   fine-tuned on TweetEval; trained on ~124M English tweets (2018-2021)
 */

import { env, pipeline } from '@huggingface/transformers';

export const MODEL = {
  id: 'Xenova/twitter-roberta-base-sentiment-latest',
  card: 'https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest',
  labels: ['negative', 'neutral', 'positive'],
  maxTokens: 512,
  approxBytes: 125905426,
};

// GitHub Pages is static hosting and cannot send COOP/COEP headers, so
// SharedArrayBuffer is unavailable. Force single-threaded WASM instead of
// letting onnxruntime try (and fail) to spawn worker threads.
env.backends.onnx.wasm.numThreads = 1;
// Nothing is bundled locally; always resolve weights from the Hugging Face CDN.
env.allowLocalModels = false;

let classifier = null;
let loading = null;
let progressHandler = null;
const downloads = new Map();
// Config and tokenizer files are a few hundred kB; including them makes the
// percentage jump to 100% and back. Only the weight file drives the progress bar.
const PROGRESS_MIN_BYTES = 10 * 1024 * 1024;

export function isReady() {
  return classifier !== null;
}

/** Register the download-progress callback (called with {percent, loadedBytes, totalBytes}). */
export function setProgressHandler(fn) {
  progressHandler = fn;
}

function trackProgress(event) {
  const { status, file, loaded, total } = event;
  if (!file || !progressHandler) return;

  if (status === 'initiate') {
    downloads.set(file, { loaded: 0, total: 0 });
  } else if (status === 'progress') {
    downloads.set(file, { loaded: loaded || 0, total: total || 0 });
  } else if (status === 'done' && downloads.has(file)) {
    const known = downloads.get(file);
    downloads.set(file, { loaded: known.total, total: known.total });
  } else {
    return;
  }

  let loadedBytes = 0;
  let totalBytes = 0;
  for (const entry of downloads.values()) {
    if (entry.total < PROGRESS_MIN_BYTES) continue;
    loadedBytes += entry.loaded;
    totalBytes += entry.total;
  }
  progressHandler({
    percent: totalBytes > 0 ? Math.min(100, (loadedBytes / totalBytes) * 100) : 0,
    loadedBytes,
    totalBytes,
  });
}

/** Load (once) and reuse the classifier. Rejects if loading fails; a retry may call it again. */
export function loadModel() {
  if (classifier) return Promise.resolve(classifier);
  if (loading) return loading;

  loading = pipeline('sentiment-analysis', MODEL.id, {
    dtype: 'q8',
    progress_callback: trackProgress,
  })
    .then((instance) => {
      classifier = instance;
      return instance;
    })
    .finally(() => {
      loading = null;
    });

  return loading;
}

/* --------------------------------------------------------------- tokenising */

function countTokens(tokenizer, text) {
  return tokenizer.encode(text, { add_special_tokens: false }).length;
}

function splitSentences(text) {
  const sentences = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const pieces = trimmed.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [trimmed];
    for (const piece of pieces) {
      const sentence = piece.trim();
      if (sentence) sentences.push(sentence);
    }
  }
  return sentences;
}

/** Hard-split a string that is too long to fit in a single window. */
function splitByLength(tokenizer, text, budget) {
  const pieces = [];
  let cursor = 0;
  while (cursor < text.length) {
    let size = Math.min(400, text.length - cursor);
    while (size > 1 && countTokens(tokenizer, text.slice(cursor, cursor + size)) > budget) {
      size = Math.floor(size / 2);
    }
    pieces.push(text.slice(cursor, cursor + size));
    cursor += size;
  }
  return pieces;
}

/**
 * Pack the text into windows that each fit the model's token limit, preferring
 * sentence boundaries so that each pass sees coherent language.
 */
function buildChunks(tokenizer, text, budget) {
  const chunks = [];
  let current = '';

  const flush = () => {
    if (current.trim()) chunks.push(current.trim());
    current = '';
  };

  for (const sentence of splitSentences(text)) {
    const candidate = current ? `${current} ${sentence}` : sentence;
    if (countTokens(tokenizer, candidate) <= budget) {
      current = candidate;
      continue;
    }

    flush();

    if (countTokens(tokenizer, sentence) <= budget) {
      current = sentence;
      continue;
    }

    // A single sentence longer than one window: pack words, then split by chars
    // as a last resort for pathological input without whitespace.
    let words = '';
    for (const word of sentence.split(/\s+/)) {
      const attempt = words ? `${words} ${word}` : word;
      if (countTokens(tokenizer, attempt) <= budget) {
        words = attempt;
        continue;
      }
      if (words) chunks.push(words);
      if (countTokens(tokenizer, word) <= budget) {
        words = word;
      } else {
        const pieces = splitByLength(tokenizer, word, budget);
        chunks.push(...pieces.slice(0, -1));
        words = pieces[pieces.length - 1];
      }
    }
    current = words;
  }

  flush();
  return chunks;
}

/* ---------------------------------------------------------------- inference */

/**
 * Classify a passage.
 * @param {string} text
 * @returns {Promise<{label: string, confidence: number, scores: Array<[string, number]>,
 *   margin: number, segments: number, tokens: number, ms: number, reused: boolean}>}
 */
export async function analyse(text) {
  const started = performance.now();
  const reused = isReady();
  const clf = await loadModel();
  const tokenizer = clf.tokenizer;

  const modelLimit = tokenizer.model_max_length && Number.isFinite(tokenizer.model_max_length)
    ? Math.min(tokenizer.model_max_length, MODEL.maxTokens)
    : MODEL.maxTokens;
  // Leave room for the two special tokens (<s> … </s>) added per pass.
  const budget = modelLimit - 16;

  const trimmed = text.trim();
  const tokens = countTokens(tokenizer, trimmed);
  const chunks = tokens <= budget ? [trimmed] : buildChunks(tokenizer, trimmed, budget);

  const passes = [];
  for (const chunk of chunks) {
    const raw = await clf(chunk, { top_k: null });
    const scores = new Map();
    for (const item of raw) scores.set(item.label, item.score);
    passes.push({ scores, weight: countTokens(tokenizer, chunk) });
    // Let the browser paint between passes on long input.
    if (chunks.length > 1) await new Promise((resolve) => setTimeout(resolve, 0));
  }

  // Token-weighted mean of per-segment class probabilities, then renormalise.
  const totals = new Map();
  let weightSum = 0;
  for (const pass of passes) {
    weightSum += pass.weight;
    for (const [label, score] of pass.scores) {
      totals.set(label, (totals.get(label) || 0) + score * pass.weight);
    }
  }

  const scores = [...totals.entries()]
    .map(([label, sum]) => [label, weightSum > 0 ? sum / weightSum : 0])
    .sort((a, b) => b[1] - a[1]);

  const normalized = scores.reduce((acc, [, score]) => acc + score, 0) || 1;
  const probabilities = scores.map(([label, score]) => [label, score / normalized]);

  return {
    label: probabilities[0][0],
    confidence: probabilities[0][1],
    scores: probabilities,
    margin: probabilities.length > 1 ? probabilities[0][1] - probabilities[1][1] : 0,
    segments: chunks.length,
    tokens,
    windowTokens: modelLimit,
    ms: Math.round(performance.now() - started),
    reused,
  };
}
