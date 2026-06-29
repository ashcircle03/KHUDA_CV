import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_BASE = process.env.VITE_API_BASE_URL || "http://localhost:8000";

export default defineConfig({
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    warmup: {
      clientFiles: ["./src/main.jsx"],
    },
    proxy: {
      "/api": { target: API_BASE, changeOrigin: true },
      "/ws":  { target: API_BASE.replace("http", "ws"), ws: true, changeOrigin: true },
    },
  },
  plugins: [react()],
});
