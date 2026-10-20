import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    target: 'es2022',
    sourcemap: true,
  },
  server: {
    watch: {
      ignored: [
        '**/tools/unreal/**',
        '**/tools/blender/**',
        '**/tools/vendor/**',
        '**/.git/**',
      ],
    },
  },
});
