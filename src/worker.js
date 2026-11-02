/**
 * Inference worker. Transformers.js and ONNX Runtime live here so that
 * tokenising and the WASM forward pass never block the interface thread.
 */

import { analyse, setProgressHandler } from './sentiment.js';

setProgressHandler((progress) => {
  self.postMessage({ type: 'progress', ...progress });
});

self.onmessage = async (event) => {
  const { id, text } = event.data || {};
  try {
    const result = await analyse(text);
    self.postMessage({ type: 'result', id, result });
  } catch (error) {
    self.postMessage({
      type: 'error',
      id,
      message: String((error && error.message) || error || 'Unknown inference error'),
    });
  }
};
