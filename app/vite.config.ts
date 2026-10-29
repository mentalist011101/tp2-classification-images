import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Vite impose un export default pour son fichier de configuration.
export default defineConfig({
	plugins: [react(), tailwindcss()],
});
