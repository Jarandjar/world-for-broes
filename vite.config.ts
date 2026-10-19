import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: './index.html',
        lantern: './lantern-hollow.html',
        visual: './visual-shell.html',
      },
      output: {
        manualChunks: {
          three: ['three'],
          simplex: ['simplex-noise'],
        },
      },
    },
    chunkSizeWarningLimit: 800,
  },
  server: {
    port: 5175,
  },
});
