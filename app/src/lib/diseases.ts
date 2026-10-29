export interface Disease {
	readonly code: string;
	readonly name: string;
	readonly english: string;
	readonly symptoms: string;
	readonly advice: string;
	readonly color: string;
}

// L'ordre suit les numéros de classe du jeu de données (0 à 4), donc celui des sorties du modèle.
export const DISEASES: readonly Disease[] = [
	{
		code: "CBB",
		name: "Bactériose",
		english: "Cassava Bacterial Blight",
		symptoms: "Taches angulaires brunes d'aspect huileux, souvent bordées de jaune ; flétrissement et chute des feuilles.",
		advice: "Bactérie transmise par les boutures et la pluie : écarter les plants atteints, planter des boutures saines.",
		color: "#8c6a3f",
	},
	{
		code: "CBSD",
		name: "Striure brune",
		english: "Cassava Brown Streak Disease",
		symptoms: "Taches jaunes le long des nervures secondaires ; plus tard, nécroses brunes dans les racines.",
		advice: "Virus porté par la mouche blanche : la feuille trompe, les racines peuvent être perdues. Arracher les plants atteints.",
		color: "#b4400f",
	},
	{
		code: "CGM",
		name: "Marbrure verte",
		english: "Cassava Green Mottle",
		symptoms: "Petites taches jaune-vert en marbrure, bords des feuilles déformés.",
		advice: "Virus transmis par de petits insectes : souvent moins grave, surveiller l'évolution du champ.",
		color: "#4a7a3a",
	},
	{
		code: "CMD",
		name: "Mosaïque",
		english: "Cassava Mosaic Disease",
		symptoms: "Mosaïque jaune et verte, feuilles tordues et rétrécies, plant rabougri.",
		advice: "La maladie la plus répandue en Afrique : utiliser des variétés tolérantes et des boutures saines.",
		color: "#e0661c",
	},
	{
		code: "Healthy",
		name: "Saine",
		english: "Healthy",
		symptoms: "Feuille d'un vert uniforme, sans tache ni déformation.",
		advice: "Aucun symptôme visible sur cette photo. Photographier d'autres feuilles du plant pour confirmer.",
		color: "#0f62ce",
	},
];
