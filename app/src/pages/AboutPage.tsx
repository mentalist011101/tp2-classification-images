import { Button, TornEdge } from "../components/paper.js";
import { PageHeader } from "../components/ui.js";
import { DATASET_URL, NOTEBOOK_URL, REPOSITORY_URL } from "../lib/links.js";
import { MODEL } from "../lib/results.js";

const STEPS = [
	{
		title: "La photo est réduite en 224 × 224",
		text: "Dans le navigateur, sans recadrage, comme les photos d'entraînement. Les pixels restent entre 0 et 255 : EfficientNet fait sa propre normalisation.",
	},
	{
		title: "Le réseau tourne en WebAssembly",
		text: `ONNX Runtime Web exécute le modèle exporté depuis Keras (${MODEL.sizeMb.toLocaleString("fr-FR")} Mo, téléchargés une fois puis gardés en cache). Aucun serveur : la photo reste sur l'appareil.`,
	},
	{
		title: "Une passe, deux réponses",
		text: "Le modèle renvoie les 5 probabilités et, pour chaque classe, une carte 7 × 7 de la zone qui l'évoque. Ces cartes (CAM) sont calculées dans le même graphe par une convolution 1 × 1 qui porte les poids de la couche de décision.",
	},
] as const;

export function AboutPage() {
	return (
		<>
			<PageHeader eyebrow="Des photos au diagnostic" title="Comment ça marche">
				<p>
					Foliole est le travail pratique image du cours INF5101 (Traitement multimédia des données, ENSPY) :
					construire une classification d'images par transfer learning, comparer CNN simple, modèle pré-entraîné et
					augmentation, analyser les erreurs, et tester une segmentation avec SAM.
				</p>
			</PageHeader>

			<figure className="paper-card overflow-hidden">
				<img
					src="/figures/architecture.png"
					alt="Architecture : données, trois modèles entraînés avec le même protocole, évaluation, analyse des erreurs et SAM"
					className="block w-full"
				/>
			</figure>

			<section aria-labelledby="app-title" className="mt-16">
				<h2 id="app-title" className="text-[clamp(1.6rem,3.4vw,2.3rem)]">
					Dans l'application
				</h2>
				<ol className="mt-8 grid gap-6 md:grid-cols-3">
					{STEPS.map((step, index) => (
						<li key={step.title} className="paper-card p-6">
							<p className="label-mono text-ember">Étape {index + 1}</p>
							<h3 className="mt-3 text-lg leading-snug">{step.title}</h3>
							<p className="mt-3 text-sm leading-relaxed text-ink-soft">{step.text}</p>
						</li>
					))}
				</ol>
			</section>

			<section aria-labelledby="limits-title" className="relative mt-24 mb-8 bg-night py-16 text-white night-grid">
				<TornEdge position="top" className="text-night" />
				<div className="mx-auto max-w-4xl px-6">
					<p className="eyebrow text-white/60">Ce qu'il faut savoir</p>
					<h2 id="limits-title" className="mt-2 text-[clamp(1.6rem,3.4vw,2.3rem)] text-white">
						Les limites du modèle
					</h2>
					<ul className="mt-8 space-y-5 text-[0.975rem] leading-relaxed text-white/75">
						<li>
							<strong className="text-white">Un score honnête mais modeste.</strong> F1 macro de{" "}
							{MODEL.f1Macro.toLocaleString("fr-FR")} sur 1 800 photos de test jamais vues. Les meilleures
							équipes de la compétition Kaggle dépassaient 90 % d'exactitude avec des modèles bien plus gros, toutes
							les images et une plus haute résolution.
						</li>
						<li>
							<strong className="text-white">Des étiquettes bruitées.</strong> Les organisateurs reconnaissent
							qu'une partie des photos porte une mauvaise étiquette : certaines erreurs « sûres d'elles » du modèle
							sont probablement justes.
						</li>
						<li>
							<strong className="text-white">Le contexte compte.</strong> Avec SAM, un clic au centre ne détoure
							souvent qu'une foliole ; sur cette seule partie, le F1 tombe de 0,70 à 0,19. Les symptômes se lisent
							sur toute la plante : photographiez le plant, pas un fragment de feuille.
						</li>
						<li>
							<strong className="text-white">Cinq classes seulement.</strong> Les images trop éloignées des photos de
							manioc sont refusées (94 % des autres plantes et 97 % des images courantes testées), mais une plante
							proche du manioc, une carence ou un insecte peuvent encore passer et être rangés de force dans une des
							cinq classes.
						</li>
					</ul>
				</div>
				<TornEdge position="bottom" className="text-night" />
			</section>

			<section aria-labelledby="data-title" className="mt-24 grid gap-10 md:grid-cols-[1.2fr_1fr]">
				<div>
					<h2 id="data-title" className="text-[clamp(1.6rem,3.4vw,2.3rem)]">
						Les données
					</h2>
					<p className="mt-5 text-[0.975rem] leading-relaxed text-ink-soft">
						21 397 photos de feuilles prises au champ en Ouganda avec des téléphones, annotées par des experts
						(Makerere AI Lab, compétition Kaggle <em>Cassava Leaf Disease Classification</em>). On en a tiré 12 000
						en respectant les proportions des classes : 8 400 pour l'entraînement, 1 800 pour la validation, 1 800
						pour le test. Les règles de la compétition interdisent de redistribuer les images : ce site n'en montre
						aucune, le notebook les relit sur Kaggle.
					</p>
				</div>
				<div className="flex flex-col items-start gap-4 md:pt-14">
					<Button href={NOTEBOOK_URL}>Lire le notebook</Button>
					<Button href={REPOSITORY_URL} variant="outline">
						Code source
					</Button>
					<Button href={DATASET_URL} variant="ghost">
						Jeu de données sur Kaggle
					</Button>
				</div>
			</section>
		</>
	);
}
