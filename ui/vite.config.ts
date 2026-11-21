/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Local only: the dev server binds to 127.0.0.1 and proxies /api to the FastAPI server.
// The production build is served by `python -m options_copilot.cli ui` from ui/dist.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: false } },
  },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true },
  build: { outDir: "dist", assetsInlineLimit: 0, sourcemap: false },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
