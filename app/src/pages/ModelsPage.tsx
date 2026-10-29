import { Tape } from "../components/paper.js";
import { PageHeader } from "../components/ui.js";
import { DISEASES } from "../lib/diseases.js";
import { NOTEBOOK_URL } from "../lib/links.js";
import { NOTEBOOK_RESULTS, OUTSIDERS } from "../lib/results.js";

const MODEL_COLORS = ["var(--color-ink-soft)", "var(--color-kraft)", "var(--color-ember)"] as const;

const LESSONS = [
	{
		title: "Le transfert vaut surtout quand les données manquent",
		text: "Sur 4 200 images d'entraînement, EfficientNetB0 battait le CNN simple de 7 points de F1 macro (0,620 contre 0,547). Sur 8 400 images, l'écart tombe à 0,005 : les filtres appris sur ImageNet remplacent les données qu'on n'a pas.",
	},
	{
		title: "L'augmentation soigne le surapprentissage, pas le reste",
		text: "Le petit CNN sous-apprend déjà : ses courbes d'entraînement et de validation restent collées et basses. Lui montrer des images retournées et zoomées rend la tâche plus dure encore, et son F1 baisse.",
	},
	{
		title: "Les erreurs ont un sens visuel",
		text: "La confusion la plus fréquente, mosaïque prise pour marbrure verte (116 cas), oppose deux maladies aux motifs jaune-vert voisins. Plusieurs erreurs « sûres à 100 % » ressemblent à des erreurs d'étiquette du jeu de données.",
	},
] as const;

export function ModelsPage() {
	return (
		<>
			<PageHeader eyebrow="Trois approches, un même test" title="Quel modèle reconnaît le mieux les maladies ?">
				<p>
					Un CNN simple entraîné de zéro, le même avec augmentation de données, et EfficientNetB0 pré-entraîné sur
					ImageNet puis affiné : même découpage, même protocole d'entraînement, mêmes 1 800 photos de test. Le
					détail, cellule par cellule, est dans le{" "}
					<a href={NOTEBOOK_URL} target="_blank" rel="noreferrer noopener" className="text-rust underline underline-offset-4">
						notebook
					</a>
					.
				</p>
			</PageHeader>

			<section aria-labelledby="table-title" className="paper-card overflow-x-auto">
				<h2 id="table-title" className="sr-only">
					Tableau des résultats
				</h2>
				<table className="w-full min-w-[40rem] text-left text-sm">
					<thead>
						<tr className="border-b border-rule">
							{["Modèle", "Paramètres", "Entraînement", "Exactitude", "F1 macro", "Lecture des courbes"].map((heading) => (
								<th key={heading} scope="col" className="label-mono px-5 py-4 font-medium text-ink-faint">
									{heading}
								</th>
							))}
						</tr>
					</thead>
					<tbody>
						{NOTEBOOK_RESULTS.map((model, index) => (
							<tr key={model.name} className="border-b border-rule last:border-0">
								<th scope="row" className="px-5 py-4 font-display font-bold">
									<span className="flex items-center gap-2.5">
										<span className="size-2.5 shrink-0 rounded-full" style={{ backgroundColor: MODEL_COLORS[index] }} />
										{model.name}
									</span>
								</th>
								<td className="px-5 py-4 font-mono text-xs">{model.params}</td>
								<td className="px-5 py-4 font-mono text-xs">{model.minutes.toLocaleString("fr-FR")} min</td>
								<td className="px-5 py-4 font-mono text-xs">{model.accuracy.toLocaleString("fr-FR")}</td>
								<td className="px-5 py-4 font-mono text-xs font-semibold">{model.f1Macro.toLocaleString("fr-FR")}</td>
								<td className="max-w-xs px-5 py-4 text-ink-soft">{model.diagnosis}</td>
							</tr>
						))}
					</tbody>
				</table>
			</section>

			<section aria-labelledby="chart-title" className="relative mt-12 paper-card p-6 sm:p-8">
				<Tape className="-top-3 right-10" label="F1 par classe" />
				<h2 id="chart-title" className="text-2xl">
					F1 par maladie
				</h2>
				<p className="mt-2 max-w-2xl text-sm text-ink-soft">
					Le F1 combine précision et rappel. Le F1 macro, à droite, en fait la moyenne simple : chaque maladie compte
					autant, même la bactériose qui ne fait que 5 % des photos.
				</p>
				<F1Chart />
			</section>

			<section aria-labelledby="lessons-title" className="mt-16">
				<h2 id="lessons-title" className="text-[clamp(1.6rem,3.4vw,2.3rem)]">
					Ce que disent les résultats
				</h2>
				<div className="mt-8 grid gap-6 md:grid-cols-3">
					{LESSONS.map((lesson, index) => (
						<article key={lesson.title} className="paper-card p-6">
							<p className="label-mono text-ember">0{index + 1}</p>
							<h3 className="mt-3 text-lg leading-snug">{lesson.title}</h3>
							<p className="mt-3 text-sm leading-relaxed text-ink-soft">{lesson.text}</p>
						</article>
					))}
				</div>
			</section>

			<section aria-labelledby="ood-title" className="mt-16">
				<h2 id="ood-title" className="text-[clamp(1.6rem,3.4vw,2.3rem)]">
					Et si ce n'est pas du manioc ?
				</h2>
				<div className="mt-5 max-w-3xl space-y-4 text-[0.975rem] leading-relaxed text-ink-soft">
					<p>
						Un classifieur à cinq classes n'a pas de case « autre chose » : montré une capture d'écran, il répond quand
						même une maladie, parfois à 70 %. Sa confiance (softmax) ne dit pas « c'est du manioc », seulement « parmi
						les cinq, c'est celle-ci ».
					</p>
					<p>
						L'application mesure donc d'abord la <strong className="font-semibold text-ink">distance de Mahalanobis</strong>{" "}
						entre le résumé de l'image (1 280 nombres, juste avant la décision) et les photos de manioc d'entraînement.
						Le seuil laisse passer 95 % des vraies photos de manioc de validation ; au-delà, l'image est refusée. Testé
						sur 1 500 images qui ne sont pas du manioc :
					</p>
				</div>
				<div className="mt-8 paper-card overflow-x-auto">
					<table className="w-full min-w-[36rem] text-left text-sm">
						<thead>
							<tr className="border-b border-rule">
								{["Images", "Refusées", "AUROC distance", "AUROC confiance softmax"].map((heading) => (
									<th key={heading} scope="col" className="label-mono px-5 py-4 font-medium text-ink-faint">
										{heading}
									</th>
								))}
							</tr>
						</thead>
						<tbody>
							{OUTSIDERS.map((outsider) => (
								<tr key={outsider.name} className="border-b border-rule last:border-0">
									<th scope="row" className="px-5 py-4 font-display font-bold">
										{outsider.name}
									</th>
									<td className="px-5 py-4 font-mono text-xs font-semibold">{Math.round(outsider.rejected * 100)} %</td>
									<td className="px-5 py-4 font-mono text-xs">{outsider.aurocMahalanobis.toLocaleString("fr-FR")}</td>
									<td className="px-5 py-4 font-mono text-xs text-ink-faint">{outsider.aurocSoftmax.toLocaleString("fr-FR")}</td>
								</tr>
							))}
							<tr>
								<th scope="row" className="px-5 py-4 font-display font-bold">
									Manioc (test), à accepter
								</th>
								<td className="px-5 py-4 font-mono text-xs">acceptées à 95 %</td>
								<td className="px-5 py-4" />
								<td className="px-5 py-4" />
							</tr>
						</tbody>
					</table>
				</div>
				<p className="mt-4 max-w-3xl text-sm text-ink-faint">
					L'AUROC mesure la séparation entre manioc et intrus (1 : parfaite, 0,5 : hasard). La confiance softmax
					distingue mal le manioc des autres plantes (0,69) ; la distance les sépare presque parfaitement (0,985).
				</p>
				<div className="mt-8">
					<Figure
						src="/figures/ood_scores.png"
						alt="Histogrammes du score d'étrangeté pour le manioc et trois familles d'intrus, avec le seuil"
						caption="Score d'étrangeté (échelle logarithmique). Le manioc de test reste à gauche du seuil ; les autres plantes et les images courantes sont juste au-delà, les captures d'écran très loin."
					/>
				</div>
			</section>

			<section aria-labelledby="figures-title" className="mt-16 space-y-10">
				<h2 id="figures-title" className="text-[clamp(1.6rem,3.4vw,2.3rem)]">
					Les figures du notebook
				</h2>
				<Figure
					src="/figures/courbes_apprentissage.png"
					alt="Courbes de F1 macro et de perte, entraînement et validation, pour les trois modèles"
					caption="Courbes d'apprentissage. En haut le F1 macro, en bas la perte. Le transfert surapprend (écart croissant après le dégel, trait pointillé) ; le CNN augmenté sous-apprend (courbes basses et collées)."
				/>
				<Figure
					src="/figures/matrices_confusion.png"
					alt="Matrices de confusion normalisées des trois modèles"
					caption="Matrices de confusion normalisées par ligne : la diagonale donne le rappel de chaque classe. La colonne « Healthy » montre les feuilles malades prises pour saines."
				/>
			</section>
		</>
	);
}

function F1Chart() {
	const groups = [
		...DISEASES.map((disease, label) => ({ label: disease.code, values: NOTEBOOK_RESULTS.map((model) => model.f1PerClass[label]) })),
		{ label: "F1 macro", values: NOTEBOOK_RESULTS.map((model) => model.f1Macro) },
	];

	return (
		<figure className="mt-8">
			<div className="relative grid h-64 grid-cols-6 items-end gap-3 border-b border-rule-strong sm:gap-6">
				{[0.25, 0.5, 0.75].map((tick) => (
					<span key={tick} aria-hidden className="absolute inset-x-0 border-t border-dashed border-rule" style={{ bottom: `${tick * 100}%` }}>
						<span className="absolute -top-2.5 -left-1 -translate-x-full font-mono text-[0.625rem] text-ink-faint">
							{tick.toLocaleString("fr-FR")}
						</span>
					</span>
				))}
				{groups.map((group) => (
					<div key={group.label} className="relative flex h-full items-end justify-center gap-1">
						{group.values.map((value, index) => (
							<div
								key={NOTEBOOK_RESULTS[index].name}
								className="group relative w-full max-w-7 rounded-t-[2px]"
								style={{ height: `${value * 100}%`, backgroundColor: MODEL_COLORS[index] }}
								title={`${NOTEBOOK_RESULTS[index].name} : ${value.toLocaleString("fr-FR")}`}
							>
								<span className="absolute -top-5 left-1/2 hidden -translate-x-1/2 font-mono text-[0.625rem] text-ink sm:block">
									{value.toFixed(2).replace(".", ",")}
								</span>
							</div>
						))}
					</div>
				))}
			</div>
			<div className="mt-2 grid grid-cols-6 gap-3 text-center sm:gap-6">
				{groups.map((group) => (
					<span key={group.label} className="font-mono text-[0.6875rem] text-ink-soft">
						{group.label}
					</span>
				))}
			</div>
			<figcaption className="mt-6 flex flex-wrap gap-x-6 gap-y-2 text-sm text-ink-soft">
				{NOTEBOOK_RESULTS.map((model, index) => (
					<span key={model.name} className="flex items-center gap-2">
						<span className="size-2.5 rounded-full" style={{ backgroundColor: MODEL_COLORS[index] }} />
						{model.name}
					</span>
				))}
			</figcaption>
		</figure>
	);
}

function Figure({ src, alt, caption }: { readonly src: string; readonly alt: string; readonly caption: string }) {
	return (
		<figure className="paper-card overflow-hidden">
			<div className="overflow-x-auto bg-white p-3">
				<img src={src} alt={alt} loading="lazy" className="mx-auto block w-full min-w-[40rem]" />
			</div>
			<figcaption className="border-t border-rule px-5 py-4 text-sm leading-relaxed text-ink-soft">{caption}</figcaption>
		</figure>
	);
}
