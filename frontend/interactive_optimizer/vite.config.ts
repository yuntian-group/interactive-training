import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Proxying API requests from /api to the backend server
      "/api": {
        target: "http://127.0.0.1:7007",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://127.0.1:7007",
        changeOrigin: true,
      },
    },
  },
});
