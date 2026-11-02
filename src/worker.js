/**
 * Inference worker. Transformers.js and ONNX Runtime live here so that
 * tokenising and the WASM forward pass never block the interface thread.
 */

import { analyse, loadModel, setProgressHandler } from './sentiment.js';

setProgressHandler((progress) => {
  self.postMessage({ type: 'progress', ...progress });
});

self.onmessage = async (event) => {
  const { id, text, type } = event.data || {};

  // Preload: fetch the weights and build the session before the user asks.
  if (type === 'load') {
    try {
      await loadModel();
      self.postMessage({ type: 'ready' });
    } catch (error) {
      self.postMessage({
        type: 'load-error',
        message: String((error && error.message) || error || 'Unknown model error'),
      });
    }
    return;
  }

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
