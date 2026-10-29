export interface ModelResult {
	readonly name: string;
	readonly params: string;
	readonly minutes: number;
	readonly accuracy: number;
	readonly f1Macro: number;
	readonly f1PerClass: readonly number[];
	readonly diagnosis: string;
}

// Résultats du notebook (12 000 images, test sur 1 800), section 6.
export const NOTEBOOK_RESULTS: readonly ModelResult[] = [
	{
		name: "CNN simple",
		params: "0,39 M",
		minutes: 14.8,
		accuracy: 0.77,
		f1Macro: 0.635,
		f1PerClass: [0.474, 0.647, 0.569, 0.906, 0.581],
		diagnosis: "Ne surapprend presque pas ; atteint sa limite de 50 époques en progressant encore.",
	},
	{
		name: "CNN + augmentation",
		params: "0,39 M",
		minutes: 24.3,
		accuracy: 0.643,
		f1Macro: 0.54,
		f1PerClass: [0.433, 0.553, 0.449, 0.793, 0.471],
		diagnosis: "Sous-apprend : l'augmentation rend la tâche trop dure pour ce petit réseau.",
	},
	{
		name: "EfficientNetB0 (transfert)",
		params: "4,1 M",
		minutes: 32.5,
		accuracy: 0.745,
		f1Macro: 0.64,
		f1PerClass: [0.565, 0.605, 0.573, 0.874, 0.583],
		diagnosis: "Surapprend (0,85 en entraînement, 0,65 en validation) mais reste le meilleur sur les classes rares.",
	},
];

// Le modèle servi par l'application, réentraîné avec le même protocole puis exporté
// (kaggle/export/train_export.py) : ses scores sont mesurés sur le modèle ONNX lui-même. La
// version quantifiée en int8 (4,5 Mo) perdait près de 3 points de F1 macro : on garde float32.
export const MODEL = {
	file: "/models/manioc_effnetb0.onnx",
	precision: "float32",
	sizeMb: 16.2,
	accuracy: 0.739,
	f1Macro: 0.633,
} as const;
