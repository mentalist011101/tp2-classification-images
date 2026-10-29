import { Button } from "../components/paper.js";
import { PageHeader } from "../components/ui.js";

export function NotFoundPage() {
	return (
		<PageHeader eyebrow="Erreur 404" title="Cette page n'existe pas.">
			<p>Le lien est peut-être ancien. Le diagnostic, lui, est toujours sur la page d'accueil.</p>
			<div className="mt-8">
				<Button to="/">Revenir au diagnostic</Button>
			</div>
		</PageHeader>
	);
}
