import { defineConfig } from 'vite';

// `base: './'` keeps every asset path relative, so the same build works at
// https://<user>.github.io/<repo>/ without hardcoding the repository name.
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    assetsInlineLimit: 0,
  },
});
