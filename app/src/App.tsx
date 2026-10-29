import { Route, Routes } from "react-router";

import { Layout } from "./components/Layout.js";
import { AboutPage } from "./pages/AboutPage.js";
import { DiagnosticPage } from "./pages/DiagnosticPage.js";
import { ModelsPage } from "./pages/ModelsPage.js";
import { NotFoundPage } from "./pages/NotFoundPage.js";

export function App() {
	return (
		<Routes>
			<Route element={<Layout />}>
				<Route index element={<DiagnosticPage />} />
				<Route path="modeles" element={<ModelsPage />} />
				<Route path="a-propos" element={<AboutPage />} />
				<Route path="*" element={<NotFoundPage />} />
			</Route>
		</Routes>
	);
}
