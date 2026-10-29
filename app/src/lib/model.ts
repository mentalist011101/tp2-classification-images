import type * as Ort from "onnxruntime-web";

import { MODEL } from "./results.js";

export const INPUT_SIZE = 224;
const CAM_SIZE = 7;
const CLASS_COUNT = 5;

// ONNX Runtime est chargé depuis le CDN plutôt qu'empaqueté : le paquet npm ferait embarquer
// 29 Mo de WebAssembly au site. La version doit rester celle de package.json (types).
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

export interface Diagnosis {
	readonly probabilities: readonly number[];
	readonly cams: readonly Float32Array[];
	readonly milliseconds: number;
}

let runtime: Promise<typeof Ort> | undefined;
let session: Promise<Ort.InferenceSession> | undefined;

export function loadModel(onProgress: (fraction: number) => void): Promise<Ort.InferenceSession> {
	session ??= Promise.all([loadRuntime(), downloadModel(onProgress)]).then(([ort, bytes]) => ort.InferenceSession.create(bytes));
	return session;
}

export async function diagnose(bitmap: ImageBitmap): Promise<Diagnosis> {
	const [ort, model] = await Promise.all([loadRuntime(), loadModel(() => {})]);
	const input = new ort.Tensor("float32", toInput(bitmap), [1, INPUT_SIZE, INPUT_SIZE, 3]);
	const start = performance.now();
	const outputs = await model.run({ [model.inputNames[0]]: input });
	const milliseconds = performance.now() - start;
	const probabilities = Array.from(outputs[model.outputNames[0]].data as Float32Array);
	const cams = splitCams(outputs[model.outputNames[1]].data as Float32Array);
	return { probabilities, cams, milliseconds };
}

// Carte de chaleur d'une classe, en RGBA 7 × 7, à étirer sur la photo.
export function camImage(cam: Float32Array): ImageData {
	const max = Math.max(...cam, 1e-6);
	const image = new ImageData(CAM_SIZE, CAM_SIZE);
	cam.forEach((value, index) => {
		const intensity = Math.max(value, 0) / max;
		const [r, g, b] = inferno(intensity);
		// Les zones froides restent transparentes pour ne pas assombrir la photo.
		image.data.set([r, g, b, Math.round(230 * intensity)], index * 4);
	});
	return image;
}

function loadRuntime(): Promise<typeof Ort> {
	runtime ??= import(/* @vite-ignore */ `${ORT_URL}ort.wasm.min.mjs`).then((ort: typeof Ort) => {
		ort.env.wasm.wasmPaths = ORT_URL;
		return ort;
	});
	return runtime;
}

async function downloadModel(onProgress: (fraction: number) => void): Promise<Uint8Array> {
	const response = await fetch(MODEL.file);
	if (!response.ok || response.body === null) throw new Error(`Modèle introuvable (HTTP ${response.status})`);
	const total = Number(response.headers.get("Content-Length")) || MODEL.sizeMb * 1e6;
	const reader = response.body.getReader();
	const chunks: Uint8Array[] = [];
	let received = 0;
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		chunks.push(value);
		received += value.length;
		onProgress(Math.min(received / total, 1));
	}
	const bytes = new Uint8Array(received);
	let offset = 0;
	for (const chunk of chunks) {
		bytes.set(chunk, offset);
		offset += chunk.length;
	}
	return bytes;
}

// Même prétraitement qu'à l'entraînement : photo entière étirée en 224 × 224 (sans recadrage),
// pixels RVB bruts entre 0 et 255 ; EfficientNet fait sa propre normalisation.
function toInput(bitmap: ImageBitmap): Float32Array {
	const pixels = downscale(bitmap).getContext("2d")!.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
	const input = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
	for (let pixel = 0; pixel < INPUT_SIZE * INPUT_SIZE; pixel++) {
		input[pixel * 3] = pixels[pixel * 4];
		input[pixel * 3 + 1] = pixels[pixel * 4 + 1];
		input[pixel * 3 + 2] = pixels[pixel * 4 + 2];
	}
	return input;
}

// Réduire une photo de 4 000 pixels en 224 d'un seul coup crénèle l'image ; on divise par deux
// tant que c'est possible, ce qui approche le redimensionnement lissé (antialias) de TensorFlow.
function downscale(bitmap: ImageBitmap): OffscreenCanvas {
	let source: CanvasImageSource = bitmap;
	let width = bitmap.width;
	let height = bitmap.height;
	while (width > INPUT_SIZE * 2 && height > INPUT_SIZE * 2) {
		width = Math.round(width / 2);
		height = Math.round(height / 2);
		const step = new OffscreenCanvas(width, height);
		const context = step.getContext("2d")!;
		context.imageSmoothingQuality = "high";
		context.drawImage(source, 0, 0, width, height);
		source = step;
	}
	const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
	const context = canvas.getContext("2d")!;
	context.imageSmoothingQuality = "high";
	context.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE);
	return canvas;
}

// La sortie CAM est en (1, 7, 7, 5) : les 5 classes sont entrelacées pixel par pixel.
function splitCams(data: Float32Array): Float32Array[] {
	return Array.from({ length: CLASS_COUNT }, (_, label) =>
		Float32Array.from({ length: CAM_SIZE * CAM_SIZE }, (_, pixel) => data[pixel * CLASS_COUNT + label]),
	);
}

const INFERNO = [
	[0, 0, 4],
	[87, 16, 110],
	[188, 55, 84],
	[249, 142, 9],
	[252, 215, 60],
] as const;

function inferno(t: number): readonly [number, number, number] {
	const position = Math.min(Math.max(t, 0), 1) * (INFERNO.length - 1);
	const low = INFERNO[Math.floor(position)];
	const high = INFERNO[Math.min(Math.floor(position) + 1, INFERNO.length - 1)];
	const mix = position - Math.floor(position);
	const blend = (channel: 0 | 1 | 2) => Math.round(low[channel] * (1 - mix) + high[channel] * mix);
	return [blend(0), blend(1), blend(2)];
}
