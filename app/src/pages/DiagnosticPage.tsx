import clsx from "clsx";
import { Camera, Eye, ImageUp, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Button, Tape } from "../components/paper.js";
import { ErrorNotice, PageHeader, Spinner } from "../components/ui.js";
import { DISEASES } from "../lib/diseases.js";
import { camImage, diagnose, INPUT_SIZE, loadModel } from "../lib/model.js";
import type { Diagnosis } from "../lib/model.js";
import { MODEL } from "../lib/results.js";

// En dessous, le modèle hésite trop pour qu'on affiche sa réponse comme un diagnostic.
const CONFIDENCE_THRESHOLD = 0.5;

type ModelState = { readonly status: "loading"; readonly progress: number } | { readonly status: "ready" } | { readonly status: "error" };

interface Photo {
	readonly url: string;
	readonly bitmap: ImageBitmap;
}

export function DiagnosticPage() {
	const [model, setModel] = useState<ModelState>({ status: "loading", progress: 0 });
	const [photo, setPhoto] = useState<Photo>();
	const [result, setResult] = useState<Diagnosis>();
	const [shownClass, setShownClass] = useState(0);
	const [showCam, setShowCam] = useState(true);
	const [error, setError] = useState<string>();
	const [busy, setBusy] = useState(false);

	useEffect(() => {
		loadModel((progress) => setModel({ status: "loading", progress })).then(
			() => setModel({ status: "ready" }),
			() => setModel({ status: "error" }),
		);
	}, []);

	useEffect(() => () => photo && URL.revokeObjectURL(photo.url), [photo]);

	const analyse = async (file: File) => {
		if (!file.type.startsWith("image/")) {
			setError("Ce fichier n'est pas une image.");
			return;
		}
		setError(undefined);
		setResult(undefined);
		setBusy(true);
		try {
			const bitmap = await createImageBitmap(file);
			setPhoto({ url: URL.createObjectURL(file), bitmap });
			const diagnosis = await diagnose(bitmap);
			setResult(diagnosis);
			setShownClass(argmax(diagnosis.probabilities));
			setShowCam(true);
		} catch {
			setError("Impossible d'analyser cette image. Essayez une autre photo (JPEG ou PNG).");
		} finally {
			setBusy(false);
		}
	};

	const reset = () => {
		setPhoto(undefined);
		setResult(undefined);
		setError(undefined);
	};

	return (
		<>
			<PageHeader
				eyebrow="Diagnostic au champ"
				title={
					<>
						Une photo de feuille,
						<br />
						cinq diagnostics possibles.
					</>
				}
			>
				<p>
					Photographiez une feuille de manioc : un réseau de neurones entraîné sur 8 400 photos prises au champ
					reconnaît quatre maladies ou une feuille saine, et montre la zone qui a guidé sa décision. Il tourne{" "}
					<strong className="font-semibold text-ink">dans votre navigateur</strong> : la photo ne quitte jamais
					votre appareil.
				</p>
			</PageHeader>

			<div className="grid items-start gap-8 lg:grid-cols-[1.15fr_0.85fr]">
				<section aria-label="Photo" className="relative paper-card p-3 sm:p-4">
					<Tape className="-top-3 left-8" label="photo" />
					{photo === undefined ? (
						<DropZone onFile={analyse} disabled={model.status === "error"} />
					) : (
						<div>
							<div className="relative overflow-hidden rounded-[2px] bg-paper-alt">
								<img
									src={photo.url}
									alt="Feuille à diagnostiquer"
									className={clsx(
										"block max-h-[70vh] w-full object-contain transition-[filter] duration-300",
										result !== undefined && showCam && "brightness-[0.6] saturate-50",
									)}
								/>
								{result !== undefined && showCam && (
									<CamOverlay cam={result.cams[shownClass]} />
								)}
								{busy && (
									<div className="absolute inset-0 grid place-items-center bg-paper/60 backdrop-blur-[2px]">
										<span className="flex items-center gap-2 text-sm text-ink">
											<Spinner /> Analyse en cours…
										</span>
									</div>
								)}
							</div>
							<div className="mt-3 flex flex-wrap items-center justify-between gap-3">
								{result !== undefined ? (
									<button
										type="button"
										onClick={() => setShowCam((value) => !value)}
										aria-pressed={showCam}
										className="inline-flex items-center gap-2 rounded-full border border-rule px-3.5 py-1.5 text-[0.8125rem] text-ink-soft transition-colors hover:bg-ink/5 hover:text-ink"
									>
										<Eye className="size-4" />
										{showCam ? `Zone regardée pour « ${DISEASES[shownClass].name} »` : "Afficher la zone regardée"}
									</button>
								) : (
									<span />
								)}
								<Button variant="outline" size="sm" onClick={reset}>
									<RotateCcw className="size-4" /> Nouvelle photo
								</Button>
							</div>
						</div>
					)}
				</section>

				<section aria-label="Résultat" className="flex flex-col gap-6">
					<ModelStatus state={model} />
					{error !== undefined && <ErrorNotice>{error}</ErrorNotice>}
					{result !== undefined ? (
						<Verdict result={result} shownClass={shownClass} onShowClass={setShownClass} />
					) : (
						<PhotoTips />
					)}
				</section>
			</div>

			<p className="mt-14 max-w-3xl border-t border-rule pt-6 text-sm leading-relaxed text-ink-faint">
				Outil pédagogique : sur 1 800 photos de test, le modèle obtient un F1 macro de{" "}
				{MODEL.f1Macro.toLocaleString("fr-FR")} et donne la bonne réponse {Math.round(MODEL.accuracy * 100)} fois sur
				100. Il se trompe surtout entre mosaïque et marbrure verte, et prend parfois une feuille malade pour une
				feuille saine. Il ne remplace pas l'avis d'un agronome.
			</p>
		</>
	);
}

function DropZone({ onFile, disabled }: { readonly onFile: (file: File) => void; readonly disabled: boolean }) {
	const cameraInput = useRef<HTMLInputElement>(null);
	const fileInput = useRef<HTMLInputElement>(null);
	const [dragging, setDragging] = useState(false);

	const pick = (files: FileList | null) => {
		const file = files?.[0];
		if (file !== undefined) onFile(file);
	};

	return (
		<div
			onDragOver={(event) => {
				event.preventDefault();
				setDragging(true);
			}}
			onDragLeave={() => setDragging(false)}
			onDrop={(event) => {
				event.preventDefault();
				setDragging(false);
				pick(event.dataTransfer.files);
			}}
			className={clsx(
				"grid min-h-[22rem] place-items-center rounded-[2px] border-2 border-dashed px-6 py-12 text-center transition-colors grid-paper",
				dragging ? "border-ember bg-ember/5" : "border-rule-strong",
			)}
		>
			<div className="flex max-w-sm flex-col items-center">
				<span className="grid h-14 w-14 place-items-center rounded-full bg-ink text-paper shadow-[var(--shadow-hard)]">
					<Camera className="size-6" />
				</span>
				<p className="mt-5 font-display text-xl font-extrabold tracking-tight">Une feuille, en gros plan</p>
				<p className="mt-2 text-sm text-ink-soft">Prenez une photo ou déposez une image ici.</p>
				<div className="mt-6 flex flex-wrap justify-center gap-3">
					<Button onClick={() => cameraInput.current?.click()} disabled={disabled}>
						<Camera className="size-4" /> Prendre une photo
					</Button>
					<Button variant="outline" onClick={() => fileInput.current?.click()} disabled={disabled}>
						<ImageUp className="size-4" /> Choisir une image
					</Button>
				</div>
				<input ref={cameraInput} type="file" accept="image/*" capture="environment" hidden onChange={(event) => pick(event.target.files)} />
				<input ref={fileInput} type="file" accept="image/*" hidden onChange={(event) => pick(event.target.files)} />
			</div>
		</div>
	);
}

function CamOverlay({ cam }: { readonly cam: Float32Array }) {
	const canvas = useRef<HTMLCanvasElement>(null);

	useEffect(() => {
		const small = new OffscreenCanvas(7, 7);
		small.getContext("2d")!.putImageData(camImage(cam), 0, 0);
		const context = canvas.current?.getContext("2d");
		if (!context) return;
		context.clearRect(0, 0, INPUT_SIZE, INPUT_SIZE);
		context.imageSmoothingQuality = "high";
		context.drawImage(small, 0, 0, INPUT_SIZE, INPUT_SIZE);
	}, [cam]);

	// La carte couvre la photo entière, étirée comme l'image l'a été pour le modèle.
	return (
		<canvas
			ref={canvas}
			width={INPUT_SIZE}
			height={INPUT_SIZE}
			aria-hidden
			className="pointer-events-none absolute inset-0 h-full w-full opacity-85"
		/>
	);
}

function ModelStatus({ state }: { readonly state: ModelState }) {
	if (state.status === "error") {
		return <ErrorNotice>Le modèle n'a pas pu être chargé. Vérifiez la connexion puis rechargez la page.</ErrorNotice>;
	}
	if (state.status === "loading") {
		return (
			<div className="paper-card p-4">
				<div className="flex items-center justify-between text-sm">
					<span className="flex items-center gap-2 text-ink-soft">
						<Spinner /> Chargement du modèle ({MODEL.sizeMb.toLocaleString("fr-FR")} Mo)
					</span>
					<span className="font-mono text-xs text-ink-faint">{Math.round(state.progress * 100)} %</span>
				</div>
				<div className="mt-3 h-1.5 overflow-hidden rounded-full bg-paper-alt">
					<div className="h-full bg-ember transition-[width] duration-200" style={{ width: `${state.progress * 100}%` }} />
				</div>
			</div>
		);
	}
	return (
		<p className="flex items-center gap-2 label-mono text-ink-faint">
			<span className="size-2 rounded-full bg-moss" /> Modèle prêt · calcul local
		</p>
	);
}

function Verdict({
	result,
	shownClass,
	onShowClass,
}: {
	readonly result: Diagnosis;
	readonly shownClass: number;
	readonly onShowClass: (label: number) => void;
}) {
	const best = argmax(result.probabilities);
	const confidence = result.probabilities[best];
	const disease = DISEASES[best];
	const uncertain = confidence < CONFIDENCE_THRESHOLD;

	return (
		<div className="flex animate-fade-in flex-col gap-6">
			<div className="relative paper-card p-6">
				<span aria-hidden className="absolute top-0 bottom-0 left-0 w-1.5" style={{ backgroundColor: disease.color }} />
				<p className="label-mono text-ink-faint">{uncertain ? "Diagnostic incertain" : "Diagnostic"}</p>
				<h2 className="mt-2 text-[2.2rem] leading-none">{disease.name}</h2>
				<p className="mt-2 font-editorial text-ink-soft italic">
					{disease.english} · {disease.code}
				</p>
				<p className="mt-5 flex items-baseline gap-2">
					<span className="font-display text-4xl font-extrabold tracking-tight">{Math.round(confidence * 100)} %</span>
					<span className="text-sm text-ink-soft">de confiance</span>
				</p>
				{uncertain && (
					<p className="mt-3 text-sm text-rust">
						Le modèle hésite entre plusieurs réponses : reprenez la photo plus près d'une feuille, en pleine lumière.
					</p>
				)}
				<p className="mt-5 text-sm leading-relaxed text-ink-soft">
					<span className="font-semibold text-ink">Symptômes : </span>
					{disease.symptoms}
				</p>
			</div>

			<div className="sticky-note rotate-[0.6deg] bg-kraft/60 p-5 text-[#16181d]">
				<p className="label-mono opacity-60">Que faire</p>
				<p className="mt-2 text-sm leading-relaxed">{disease.advice}</p>
			</div>

			<div className="paper-card p-5">
				<div className="flex items-baseline justify-between">
					<h3 className="label-mono text-ink-faint">Probabilités</h3>
					<span className="font-mono text-[0.6875rem] text-ink-faint">{Math.round(result.milliseconds)} ms</span>
				</div>
				<ul className="mt-4 space-y-2">
					{DISEASES.map((item, label) => (
						<li key={item.code}>
							<button
								type="button"
								onClick={() => onShowClass(label)}
								aria-pressed={shownClass === label}
								title={`Voir la zone qui évoque « ${item.name} »`}
								className={clsx(
									"w-full rounded-[4px] px-2 py-1.5 text-left transition-colors",
									shownClass === label ? "bg-ink/[0.06]" : "hover:bg-ink/[0.03]",
								)}
							>
								<span className="flex items-baseline justify-between text-sm">
									<span className={clsx(label === best ? "font-semibold text-ink" : "text-ink-soft")}>
										{item.name} <span className="font-mono text-[0.6875rem] text-ink-faint">{item.code}</span>
									</span>
									<span className="font-mono text-xs text-ink">{(result.probabilities[label] * 100).toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} %</span>
								</span>
								<span className="mt-1.5 block h-1.5 overflow-hidden rounded-full bg-paper-alt">
									<span
										className="block h-full rounded-full transition-[width] duration-500"
										style={{ width: `${result.probabilities[label] * 100}%`, backgroundColor: item.color }}
									/>
								</span>
							</button>
						</li>
					))}
				</ul>
				<p className="mt-4 text-xs leading-relaxed text-ink-faint">
					Cliquez sur une classe pour voir la zone de la photo qui l'évoque (carte d'activation, CAM).
				</p>
			</div>
		</div>
	);
}

function PhotoTips() {
	return (
		<div className="sticky-note -rotate-[0.8deg] bg-sun p-6 text-[#16181d]">
			<p className="label-mono opacity-60">Pour une bonne photo</p>
			<ul className="mt-3 space-y-2 text-sm leading-relaxed">
				<li>— une feuille ou un plant qui remplit le cadre ;</li>
				<li>— en plein jour, sans contre-jour ;</li>
				<li>— une image nette, les taches bien visibles ;</li>
				<li>— le modèle a appris sur des photos de champ : pas besoin de fond uni.</li>
			</ul>
		</div>
	);
}

function argmax(values: readonly number[]): number {
	return values.reduce((best, value, index) => (value > values[best] ? index : best), 0);
}
