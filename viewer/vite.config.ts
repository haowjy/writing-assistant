import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Built into the Python package, where `wa-bench view` serves it from (not committed).
  build: { outDir: '../src/writing_assistant/cli/view_dist', emptyOutDir: true, chunkSizeWarningLimit: 1000 },
  server: { proxy: { '/api': 'http://127.0.0.1:8767' } },
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
})
