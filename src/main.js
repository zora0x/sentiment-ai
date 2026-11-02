import './style.css';

const MAX_CHARS = 6000;
const MAX_TOKENS = 512;

const EXAMPLES = {
  positive: 'The service was excellent and the staff were very helpful.',
  negative: 'The application keeps crashing and customer support was unhelpful.',
  neutral: 'The package arrived on Tuesday and contained two items.',
};

const ICONS = {
  positive: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6.5 9.5 17 4 11.5"/></svg>',
  negative: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/></svg>',
  neutral: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5.5 12h13"/></svg>',
};

const BASE_EXPLANATION = {
  positive:
    'The model reads this as a positive passage: the wording leans towards approval, satisfaction, or praise.',
  negative:
    'The model reads this as a negative passage: the wording leans towards complaint, disappointment, or criticism.',
  neutral:
    'The model reads this as neutral: the text appears descriptive or factual, without a clear positive or negative stance.',
};

const el = (id) => document.getElementById(id);
const ui = {
  input: el('input-text'),
  counter: el('char-count'),
  analyse: el('analyse-btn'),
  analyseLabel: document.querySelector('#analyse-btn .btn__label'),
  clear: el('clear-btn'),
  modelStatus: el('model-status'),
  modelStatusText: el('model-status-text'),
  modelStatusPct: el('model-status-pct'),
  emptyHint: el('empty-hint'),
  status: el('status'),
  statusText: el('status-text'),
  statusPct: el('status-pct'),
  statusBar: el('status-bar'),
  resultMeta: el('result-meta'),
  empty: el('result-empty'),
  loading: el('result-loading'),
  loadingTitle: el('loading-title'),
  loadingText: el('loading-text'),
  error: el('result-error'),
  errorText: el('error-text'),
  retry: el('retry-btn'),
  card: el('result-card'),
  icon: el('verdict-icon'),
  label: el('verdict-label'),
  scoreValue: el('score-value'),
  scoreTrack: el('score-track'),
  scoreFill: el('score-fill'),
  explain: el('verdict-explain'),
  scoresBody: el('scores-body'),
  metaList: el('meta-list'),
};

let busy = false;
let modelReady = false;
let modelCached = false;
let preloadStarted = false;
let worker = null;
let requestId = 0;
const pending = new Map();

/* ------------------------------------------------------------- model status */

const MODEL_CACHE = 'transformers-cache';
const MODEL_FILE = 'twitter-roberta-base-sentiment-latest';

/**
 * Transformers.js keeps downloaded files in the Cache API. Checking it costs
 * nothing and tells us whether a visit will pay the 130 MB download or not.
 */
async function modelIsCached() {
  if (!('caches' in window)) return false;
  try {
    const cache = await caches.open(MODEL_CACHE);
    const keys = await cache.keys();
    return keys.some((request) => request.url.includes(MODEL_FILE) && request.url.includes('model_quantized'));
  } catch {
    return false;
  }
}

function setModelState(state, text, pct = '') {
  ui.modelStatus.hidden = false;
  ui.modelStatus.dataset.state = state;
  ui.modelStatusText.textContent = text;
  ui.modelStatusPct.textContent = pct;
}

function setEmptyHint(cached) {
  ui.emptyHint.innerHTML = cached
    ? 'Add some text and select <strong>Analyse Sentiment</strong>. The model is already cached in this browser, so analysis runs offline and takes about a second.'
    : 'Add some text and select <strong>Analyse Sentiment</strong>. The model runs locally, so the first run downloads about 130&nbsp;MB of model and runtime files, then stays cached.';
}

/**
 * Warm the model as soon as the page is idle: fetch the weights on a cold visit
 * (so typing overlaps the download), or just build the session on a warm one
 * (so the first click is instant). Skipped on data-saver connections.
 */
function schedulePreload(cached) {
  if (navigator.connection && navigator.connection.saveData) {
    setModelState('idle', 'Model loads on first analysis · data saver is on');
    return;
  }

  if (navigator.deviceMemory && navigator.deviceMemory < 4) {
    setModelState(
      'idle',
      cached ? 'Model cached · loads on first analysis' : 'Model not loaded yet · one-time 130 MB download'
    );
    return;
  }

  const start = () => {
    if (preloadStarted || modelReady) return;
    preloadStarted = true;
    setModelState(
      'loading',
      cached ? 'Loading cached model into memory' : 'Preparing model · one-time download'
    );
    try {
      getWorker().postMessage({ type: 'load' });
    } catch {
      /* the first analysis will surface the error */
    }
  };

  const delay = cached ? 2000 : 1200;
  if ('requestIdleCallback' in window) requestIdleCallback(start, { timeout: delay + 1500 });
  else setTimeout(start, delay);
}

/* ------------------------------------------------------------------ worker */

function failAll(message) {
  for (const entry of pending.values()) entry.reject(new Error(message));
  pending.clear();
}

function getWorker() {
  if (worker) return worker;

  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

  worker.onmessage = (event) => {
    const message = event.data || {};

    if (message.type === 'ready') {
      modelReady = true;
      setModelState('ready', 'Model ready · runs locally in this tab');
      setEmptyHint(true);
      return;
    }

    if (message.type === 'load-error') {
      setModelState('error', 'Model unavailable · will retry when you analyse');
      return;
    }

    if (message.type === 'progress') {
      onDownloadProgress(message);
      return;
    }

    const entry = pending.get(message.id);
    if (!entry) return;
    pending.delete(message.id);

    if (message.type === 'result') {
      modelReady = true;
      entry.resolve(message.result);
    } else if (message.type === 'error') {
      entry.reject(new Error(message.message));
    }
  };

  worker.onerror = (event) => {
    const message = event.message || 'The inference worker failed to start.';
    worker = null;
    failAll(message);
  };

  return worker;
}

function analyseInWorker(text) {
  return new Promise((resolve, reject) => {
    const id = ++requestId;
    pending.set(id, { resolve, reject });
    try {
      getWorker().postMessage({ id, text });
    } catch (error) {
      pending.delete(id);
      reject(error);
    }
  });
}

function onDownloadProgress({ percent, loadedBytes, totalBytes }) {
  const known = totalBytes >= 1048576;
  const size = known
    ? loadedBytes >= 1048576
      ? `${mb(loadedBytes)} of ${mb(totalBytes)}`
      : mb(totalBytes)
    : '';
  setModelState(
    'loading',
    'Preparing model · one-time download',
    known ? `${Math.round(percent)}% · ${size}` : ''
  );

  if (busy) {
    if (known) setStatus(`Downloading model weights · ${size}`, percent);
    else setStatus('Downloading model weights', 0, true);
  }
}

/* ------------------------------------------------------------------ helpers */

const mb = (bytes) => `${Math.round(bytes / 1048576)} MB`;

function setBusy(state) {
  busy = state;
  ui.analyse.classList.toggle('is-busy', state);
  ui.analyse.disabled = state || !ui.input.value.trim();
  ui.clear.disabled = state;
  ui.analyseLabel.textContent = state ? 'Analysing…' : 'Analyse Sentiment';
  ui.input.setAttribute('aria-busy', String(state));
}

function updateCounter() {
  const length = ui.input.value.length;
  ui.counter.textContent = `${length} / ${MAX_CHARS}`;
  ui.counter.classList.toggle('counter--warn', length > MAX_CHARS - 500);
  if (!busy) ui.analyse.disabled = !ui.input.value.trim();
}

function setStatus(text, percent, indeterminate = false) {
  ui.status.hidden = false;
  ui.statusText.textContent = text;
  ui.statusBar.classList.toggle('status__bar--indeterminate', indeterminate);
  if (indeterminate) {
    ui.statusPct.textContent = '';
    ui.statusBar.style.width = '';
  } else {
    ui.statusBar.style.width = `${percent}%`;
    ui.statusPct.textContent = `${Math.round(percent)}%`;
  }
}

function showState(name) {
  ui.empty.hidden = name !== 'empty';
  ui.loading.hidden = name !== 'loading';
  ui.error.hidden = name !== 'error';
  ui.card.hidden = name !== 'card';
}

const pct = (score) => `${(score * 100).toFixed(1)}%`;

/* --------------------------------------------------------------- rendering */

function renderResult(result) {
  const { label, confidence, scores, margin, segments, tokens, ms } = result;
  const runnerUp = scores[1];

  showState('card');
  ui.status.hidden = true;
  ui.resultMeta.textContent = `${ms} ms · ${segments} pass${segments === 1 ? '' : 'es'}`;

  ui.card.className = `verdict verdict--${label}`;
  ui.icon.innerHTML = ICONS[label];
  ui.label.textContent = label;

  ui.scoreValue.textContent = `${(confidence * 100).toFixed(1)}%`;
  ui.scoreTrack.setAttribute('aria-valuenow', String(Math.round(confidence * 100)));
  ui.scoreFill.style.width = '0%';
  requestAnimationFrame(() => {
    ui.scoreFill.style.width = `${confidence * 100}%`;
  });

  const parts = [BASE_EXPLANATION[label]];
  if (margin < 0.15 && runnerUp) {
    parts.push(
      `It was a close call — ${runnerUp[0]} scored ${pct(runnerUp[1])} against ${pct(confidence)} for ${label}, so a small change in wording could flip the result.`
    );
  }
  if (segments > 1) {
    parts.push(
      `Your text was longer than the model's ${MAX_TOKENS}-token window, so it was analysed in ${segments} segments and the segment scores were combined in proportion to their length.`
    );
  }
  parts.push(
    'Scores are softmax outputs from a single model, not calibrated probabilities, and they can be wrong — especially on sarcasm or mixed opinions.'
  );
  ui.explain.textContent = parts.join(' ');

  ui.scoresBody.textContent = '';
  for (const [name, score] of scores) {
    const row = document.createElement('tr');
    if (name === label) row.dataset.active = 'true';
    const cell = document.createElement('td');
    cell.textContent = name;
    const value = document.createElement('td');
    value.textContent = pct(score);
    row.append(cell, value);
    ui.scoresBody.append(row);
  }

  const meta = [
    ['Model', 'Xenova/twitter-roberta-base-sentiment-latest'],
    ['Head', 'RobertaForSequenceClassification · 3 classes'],
    ['Labels', '0 negative · 1 neutral · 2 positive'],
    ['Weights', 'q8 quantised ONNX · 126 MB · cached after first load'],
    ['Runtime', 'ONNX Runtime Web (WebAssembly, single-threaded)'],
    ['Tokens', `${tokens} (window ${MAX_TOKENS})`],
    ['Inference', `${ms} ms across ${segments} pass${segments === 1 ? '' : 'es'}`],
  ];
  ui.metaList.textContent = '';
  for (const [key, value] of meta) {
    const dt = document.createElement('dt');
    dt.textContent = key;
    const dd = document.createElement('dd');
    dd.textContent = value;
    ui.metaList.append(dt, dd);
  }
}

function describeError(error) {
  const message = String((error && error.message) || error || '');
  if (
    !navigator.onLine ||
    /fetch|network|download|unauthorized|not found|404|502|503|504/i.test(message)
  ) {
    return [
      'The model files could not be downloaded from huggingface.co. The first analysis needs a connection and about 130 MB of files — check your network, then try again.',
      'Nothing you typed left this browser tab; there is no word-list fallback standing in for the model.',
    ].join(' ');
  }
  if (/memory|allocation|out of range|out of memory/i.test(message)) {
    return 'The browser could not allocate enough memory to run the model. Close other heavy tabs and try again.';
  }
  return 'The local model failed to run. Try again, and reload the page if the error keeps appearing.';
}

function renderError(error) {
  showState('error');
  ui.status.hidden = true;
  ui.resultMeta.textContent = '';
  ui.errorText.textContent = describeError(error);

  const detail = String((error && error.message) || error || '').trim();
  if (detail) {
    const code = document.createElement('code');
    code.className = 'err-detail';
    code.textContent = detail.slice(0, 220);
    ui.errorText.append(document.createElement('br'), code);
  }
}

/* ------------------------------------------------------------- interactions */

async function run() {
  const text = ui.input.value;
  if (!text.trim()) {
    ui.input.focus();
    return;
  }
  if (busy) return;

  setBusy(true);
  ui.resultMeta.textContent = '';
  showState('loading');

  const ready = modelReady;
  ui.loadingTitle.textContent = ready ? 'Analysing text' : 'Warming up the model';
  ui.loadingText.textContent = ready
    ? 'Running the transformer locally — a few seconds for longer passages.'
    : modelCached
      ? 'Loading the cached model into memory — usually a second or two.'
      : 'Downloading about 130 MB of model and runtime files. This happens once; afterwards your browser caches them.';

  if (!ready) setStatus('Downloading model weights', 0);
  else setStatus('Running inference', 0, true);

  try {
    const result = await analyseInWorker(text);
    modelReady = true;
    setModelState('ready', 'Model ready · runs locally in this tab');
    setEmptyHint(true);
    renderResult(result);
  } catch (error) {
    renderError(error);
  } finally {
    ui.status.hidden = true;
    setBusy(false);
  }
}

ui.input.addEventListener('input', updateCounter);

ui.input.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    event.preventDefault();
    run();
  }
});

ui.analyse.addEventListener('click', run);
ui.retry.addEventListener('click', run);

ui.clear.addEventListener('click', () => {
  ui.input.value = '';
  updateCounter();
  showState('empty');
  ui.resultMeta.textContent = '';
  ui.status.hidden = true;
  ui.input.focus();
});

for (const button of document.querySelectorAll('[data-example]')) {
  button.addEventListener('click', () => {
    ui.input.value = EXAMPLES[button.dataset.example];
    updateCounter();
    ui.input.focus();
  });
}

updateCounter();

/* ------------------------------------------------------------------- boot */

(async () => {
  const cached = await modelIsCached();
  modelCached = cached;
  setEmptyHint(cached);

  setModelState(
    cached ? 'ready' : 'idle',
    cached
      ? 'Model cached in this browser · ready to analyse'
      : 'Model not loaded yet · one-time 130 MB download'
  );

  schedulePreload(cached);
})();
