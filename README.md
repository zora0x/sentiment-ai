# SentimentAI

A browser-based sentiment analysis website that classifies English text as **positive**, **negative**, or **neutral** using a transformer model running entirely on the user's device.

Built as an **AI Applications Lab** project demonstrating NLP, text classification, and client-side AI deployment.

**Live site:** https://zora0x.github.io/sentiment-ai/

---

## Overview

The user pastes a sentence, review, or comment, presses **Analyse Sentiment**, and the page shows:

- the predicted sentiment, with a text label and a matching icon (never colour alone);
- the model's softmax score for the winning label;
- a short explanation of what that means;
- optional technical details — per-class scores, token count, number of inference passes, model id, runtime.

There is **no backend**. The model is downloaded from the Hugging Face CDN on first use and executed with WebAssembly inside a Web Worker, so the text never leaves the browser tab.

## Features

- Three-class sentiment classification from a model that genuinely outputs three classes.
- Client-side inference via Transformers.js (ONNX Runtime Web) — no server, no API key, no paid inference.
- Inference isolated in a Web Worker: the interface stays responsive while the model runs.
- Lazy loading: the 550 kB inference library and 126 MB of weights are fetched only when the user first analyses text.
- Download progress with a real percentage and megabyte counter, then a clear "analysing" state.
- Long text support: input is packed into sentence-aligned segments that fit the model's 512-token window and the per-segment scores are combined.
- Explicit error state with a retry button when model files cannot be downloaded — never a silent fallback.
- Responsive layout for mobile, tablet, and desktop; reduced-motion support; keyboard shortcut (<kbd>Ctrl</kbd>/<kbd>Cmd</kbd> + <kbd>Enter</kbd>).
- No storage of user text: no localStorage, cookies, analytics, database, or console logging of input.

## Technologies

| Layer | Choice |
| --- | --- |
| Markup / styling | HTML5, CSS3 (custom properties, grid, `color-mix`) |
| Application | Vanilla JavaScript (ES modules) |
| Build | Vite 8 |
| Inference | [Transformers.js](https://huggingface.co/docs/transformers.js) 4.x (ONNX Runtime Web, WASM) |
| Hosting | GitHub Pages (static) |

No framework is used; the whole application is four source files plus a stylesheet.

## Model

**Model:** `Xenova/twitter-roberta-base-sentiment-latest`

- Transformers.js build: https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest
- Original model card: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
- Base architecture: RoBERTa-base (12 layers, 125M parameters), 512-token limit
- Quantisation: `q8` (8-bit) ONNX, ~126 MB — chosen for browser download size
- Licence: **CC-BY-4.0** (attribution required); fine-tuned on the [TweetEval](https://github.com/cardiffnlp/tweeteval) benchmark by CardiffNLP, from a RoBERTa model trained on ~124M English tweets (2018–2021)
- Reference paper: [TimeLMs, arXiv:2202.03829](https://arxiv.org/abs/2202.03829)

### Labels

The model's `config.json` declares a three-way classification head — this is not a binary model renamed:

```
id2label: { "0": "negative", "1": "neutral", "2": "positive" }
label2id: { "negative": 0, "neutral": 1, "positive": 2 }
```

The interface displays these labels directly. No post-processing invents a neutral class or renames a binary prediction.

### How to read the score

The number shown as **Confidence** is the softmax output for the winning label across the model's three logits. It is a *relative model score*, not a calibrated probability that the prediction is correct. A 98% score does not mean the model is right 98% of the time; it means that class dominated the logits for that pass. The UI states this next to the score.

For multi-segment input, the per-segment softmax vectors are averaged, weighted by each segment's token count, and renormalised.

### Model selection note

`Xenova/distilbert-base-multilingual-cased-sentiments-student` (6 layers, faster) was benchmarked as an alternative and rejected: on the same test set it scored the negative example at 50.2% confidence and classified the neutral example ("The package arrived on Tuesday…") as positive, versus 93.6% and 90.1% for the RoBERTa model. Speed was not worth the accuracy loss.

## Project structure

```
sentiment-ai/
├── .github/
│   └── workflows/
│       └── deploy.yml        # build + deploy to GitHub Pages
├── public/
│   └── favicon.svg
├── src/
│   ├── main.js               # UI state machine, worker client, rendering
│   ├── sentiment.js          # model loading, tokenisation, chunking, aggregation
│   ├── worker.js             # inference worker entry point
│   └── style.css
├── index.html
├── package.json
├── vite.config.js            # base: './' so any Pages sub-path works
├── .gitignore
└── README.md
```

## Installation

Requires Node.js 20 or newer.

```bash
git clone https://github.com/zora0x/sentiment-ai.git
cd sentiment-ai
npm install
```

## Local development

```bash
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). The first analysis downloads the model; subsequent runs use the browser's HTTP cache.

## Production build

```bash
npm run build      # outputs dist/
npm run preview    # serve dist/ at http://localhost:4173
```

Build output:

| File | Size |
| --- | --- |
| `index.html` | 12 kB |
| `assets/index-*.js` (app) | 8 kB |
| `assets/index-*.css` | 15 kB |
| `assets/worker-*.js` (inference lib) | 550 kB (158 kB gzipped) |
| `assets/ort-wasm-simd-threaded.asyncify-*.wasm` | 27 MB (6.7 MB gzipped) |

`vite.config.js` sets `base: './'`, so all asset paths are relative and the build works from any GitHub Pages sub-path without hardcoding the repository name.

## GitHub Pages deployment

The workflow at `.github/workflows/deploy.yml` runs on every push to `main`:

1. checkout, 2. Node 22 with npm cache, 3. `npm ci`, 4. `npm run build`, 5. `actions/configure-pages`, 6. `actions/upload-pages-artifact` (`dist/`), 7. `actions/deploy-pages`.

One-time repository setup:

```bash
gh api -X POST repos/<user>/sentiment-ai/pages -f build_type=workflow
```

or in the browser: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Runtime behaviour and network requirements

- **The model is not bundled.** Nothing model-related ships with the build.
- **First analysis needs internet**, and downloads roughly 130 MB: 126 MB of `q8` ONNX weights plus the ~27 MB ONNX Runtime WASM binary (6.7 MB compressed). The UI shows progress; the download happens once.
- **Later analyses work offline** in practice: weights are cached by the browser (Cache API) and the runtime by the HTTP cache.
- All requests go to `cdn.huggingface.co` / `huggingface.co`. The application itself has no server.
- If the download fails, the UI shows an error with a **Try again** button. There is no keyword-based fallback pretending to be a model prediction.

## Testing

Executed against the production build (`npm run preview`) in Chrome, and the model outputs cross-checked with Transformers.js in Node.

| Case | Input | Result | Score |
| --- | --- | --- | --- |
| Positive | "The service was excellent and the staff were very helpful." | **positive** | 98.0% |
| Negative | "The application keeps crashing and customer support was unhelpful." | **negative** | 93.6% |
| Neutral | "The package arrived on Tuesday and contained two items." | **neutral** | 90.1% |
| Mixed | "The camera is excellent, but the battery life is disappointing." | **negative** | 78.1% |
| Sarcasm | "Oh fantastic, another update that breaks everything. Just what I needed." | **positive** (wrong) | 75.2% |
| Long | 4,016 characters / 832 tokens | **neutral**, 2 segments, close-call note | 51.3% |
| Empty / whitespace | `"   \n\t  "` | Analyse button disabled, no inference | — |
| Duplicate submit | Two rapid clicks | Second click ignored while busy | — |
| Model failure | Model id forced to a non-existent repo | Error card + working **Try again** | — |

Also verified: no console errors across a full session, no horizontal overflow at 390 px / 820 px / 1440 px, relative asset paths resolve under a sub-path, and the analysis button re-enables after each run.

The mixed example is classified negative and the sarcasm example positive — both are recorded as-is rather than tuned away, because they are genuine model behaviour and are exactly the failure modes described under Limitations.

## Known limitations

- **Trained on tweets.** Formal, technical, or domain-specific writing (legal, medical, financial) may be scored less reliably than conversational text.
- **English only.** The model card states it is suitable for English.
- **Sarcasm and irony** are regularly misread — the sarcasm test above is classified positive with 75% confidence.
- **One label for the whole passage.** The model does not do aspect-level analysis: in "the camera is excellent, but the battery is disappointing" it returns a single overall label and cannot say which clause drove it.
- **Mixed opinions** land wherever the token-weighted balance falls; a close-call note is shown when the top two scores are within 15 points.
- **Segmentation is a workaround.** Text beyond 512 tokens is split on sentence boundaries and the scores are averaged. This is a documented aggregation strategy, not the model's native behaviour, and it can hide a strong opinion confined to one segment.
- **Scores are uncalibrated** softmax outputs, not probabilities of correctness.
- **Single-threaded WASM.** GitHub Pages cannot send `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers, so `SharedArrayBuffer` is unavailable and ONNX Runtime runs with one thread. Measured on an Apple Silicon laptop: ~50 tokens ≈ 0.8 s, ~430 tokens ≈ 6.4 s, 832 tokens (2 passes) ≈ 10.8 s. Expect slower results on low-end hardware.
- **First load is heavy** (~130 MB). On a slow or metered connection this is a real cost; the progress bar exists so the wait is visible rather than silent.

## Responsible use

This is an educational demonstration. Predictions are automated estimates and **can be wrong**, especially on sarcasm, mixed opinions, and text unlike the training data.

- Do not use the output to make decisions about people — hiring, admissions, credit, moderation of individuals, or mental-health assessment.
- Do not treat a score as certainty or as a substitute for reading the text.
- Text you enter is processed in your browser tab, is never stored, and is never transmitted to the site's host. It is sent only to the model's CDN as part of downloading the model files — not as part of inference.
- The model is used under CC-BY-4.0 and was trained by CardiffNLP; it was not trained from scratch for this project.

## Licence and attribution

Application code: MIT.

Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`, CC-BY-4.0, CardiffNLP (Barbieri et al., *TimeLMs: Diachronic Language Models from Twitter*, 2022). ONNX conversion by Xenova.
