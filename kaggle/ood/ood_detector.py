# Ajoute au modèle de l'application un détecteur d'images hors distribution : une photo qui
# n'est pas du manioc doit être refusée au lieu de recevoir l'une des 5 maladies.
#
# Méthode (Lee et al., 2018) : on mesure la distance de Mahalanobis entre le vecteur de
# caractéristiques de l'image (1 280 nombres, juste avant la couche de décision) et le centre
# de chaque classe, avec une covariance commune estimée sur l'entraînement. Le score est la
# plus petite de ces 5 distances ; au-delà d'un seuil réglé sur la validation, l'image est
# refusée. Le calcul est greffé dans le graphe ONNX : l'application reçoit le score en
# même temps que les probabilités.
import json
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import tensorflow as tf
from onnx import TensorProto, helper, numpy_helper
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
IMG_SIZE = 224
N_SAMPLE = 12_000
N_OOD = 500
# On accepte de refuser 5 % des vraies photos de manioc pour attraper le plus d'intrus possible.
ID_QUANTILE = 0.95
MODEL_URL = "https://foliole-manioc.vercel.app/models/manioc_effnetb0.onnx"
INPUT = Path("/kaggle/input")
DATA = INPUT / "competitions" / "cassava-leaf-disease-classification"
OUT = Path("/kaggle/working")
random.seed(SEED)
tf.config.set_visible_devices([], "GPU")

full = pd.read_csv(DATA / "train.csv")
sample, _ = train_test_split(full, train_size=N_SAMPLE, stratify=full["label"], random_state=SEED)
train_df, rest = train_test_split(sample, test_size=0.30, stratify=sample["label"], random_state=SEED)
val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)


def load_images(paths):
    def decode(path):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.cast(tf.round(tf.image.resize(image, (IMG_SIZE, IMG_SIZE), antialias=True)), tf.uint8)

    dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in paths]).map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return np.concatenate(list(dataset.batch(256).as_numpy_iterator()))


def images_under(folder_name):
    roots = [p for p in INPUT.rglob(folder_name) if p.is_dir()]
    files = sorted({f for root in roots for f in root.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}})
    print(f"{folder_name} : {len(files)} images trouvées")
    return random.sample(files, N_OOD)


def synthetic_screenshots(folder):
    # Captures d'écran plausibles : tableaux, textes, graphiques, thèmes clair et sombre.
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    paths = []
    for i in range(N_OOD):
        dark = rng.random() < 0.5
        fig = plt.figure(figsize=(8, 6), dpi=60, facecolor="#1e1f22" if dark else "white")
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax.set_facecolor("#2b2d31" if dark else "#f4f4f6")
        kind = i % 3
        ink = "white" if dark else "black"
        if kind == 0:
            rows = [[f"Ligne{r}{rng.integers(100)}", f"{rng.integers(1, 31)} oct. 2026", f"{rng.integers(1, 700)} jours",
                     f"{rng.random() * 9:.2f} MB"] for r in range(rng.integers(4, 9))]
            table = ax.table(cellText=rows, colLabels=["Site", "Dernier usage", "Inactivité", "Stockage"], loc="center")
            table.scale(1, 2)
            for cell in table.get_celld().values():
                cell.set_facecolor(ax.get_facecolor())
                cell.get_text().set_color(ink)
            ax.axis("off")
        elif kind == 1:
            for line in range(rng.integers(8, 16)):
                ax.text(0.02, 0.95 - line * 0.06, " ".join("lorem ipsum dolor sit amet".split()[: rng.integers(2, 6)] * 3),
                        color=ink, fontsize=11, transform=ax.transAxes)
            ax.axis("off")
        else:
            ax.bar(range(8), rng.random(8), color=rng.choice(["#4c78a8", "#f58518", "#54a24b", "#e45756"]))
            ax.tick_params(colors=ink)
        path = folder / f"capture_{i}.png"
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(path)
    return paths


# 1. Le modèle de l'application, avec une sortie de plus : le vecteur de caractéristiques.
urllib.request.urlretrieve(MODEL_URL, OUT / "base.onnx")
model = onnx.load(OUT / "base.onnx")
initializers = {init.name: init for init in model.graph.initializer}
decision = next(node for node in model.graph.node if node.op_type == "MatMul" and node.input[1] in initializers
                and list(initializers[node.input[1]].dims) == [1280, 5])
features_name = decision.input[0]
probe = onnx.ModelProto()
probe.CopyFrom(model)
probe.graph.output.append(helper.make_tensor_value_info(features_name, TensorProto.FLOAT, ["N", 1280]))
probe_session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
input_name = probe_session.get_inputs()[0].name


def features_and_probs(images):
    probs, features = [], []
    for i in range(0, len(images), 64):
        p, _, f = probe_session.run(None, {input_name: images[i:i + 64].astype(np.float32)})
        probs.append(p)
        features.append(f)
    return np.concatenate(features), np.concatenate(probs)


cassava = {name: load_images([DATA / "train_images" / i for i in df["image_id"]]) for name, df in
           (("train", train_df), ("val", val_df), ("test", test_df))}
outsiders = {
    "autres plantes (PlantVillage)": load_images(images_under("PlantVillage")),
    "images courantes (ImageNet)": load_images(images_under("imagenet-mini")),
    "captures d'écran": load_images(synthetic_screenshots(OUT / "captures")),
}
feats, probs = {}, {}
for name, images in {**cassava, **outsiders}.items():
    feats[name], probs[name] = features_and_probs(images)
    print(f"{name} : {len(images)} images")

# 2. Centres des classes et covariance commune (Ledoit-Wolf : 1 280 dimensions pour 8 400
# images, l'estimateur empirique serait mal conditionné).
y_train = train_df["label"].to_numpy()
centers = np.stack([feats["train"][y_train == c].mean(0) for c in range(5)])
covariance = LedoitWolf().fit(feats["train"] - centers[y_train]).covariance_
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
order = np.argsort(eigenvalues)[::-1]
eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]


def projection(k):
    return eigenvectors[:, :k] / np.sqrt(eigenvalues[:k])


def scores(f, k):
    w = projection(k)
    z, zc = f @ w, centers @ w
    return ((z[:, None, :] - zc[None]) ** 2).sum(-1).min(1)


def auroc(id_scores, ood_scores):
    return roc_auc_score(np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))], np.r_[id_scores, ood_scores])


all_ood = np.concatenate([feats[name] for name in outsiders])
# Restreindre la distance aux k directions principales réduit la taille de la matrice ajoutée
# au modèle ; on garde le plus petit k à moins de 0,005 d'AUROC du meilleur, choisi sur la
# validation pour ne pas régler sur le test.
by_k = {k: auroc(scores(feats["val"], k), scores(all_ood, k)) for k in (32, 64, 128, 256, 512, 1280)}
best = max(by_k.values())
K = min(k for k, value in by_k.items() if value >= best - 0.005)
threshold = float(np.quantile(scores(feats["val"], K), ID_QUANTILE))
print("AUROC (validation contre intrus) par k :", {k: round(v, 4) for k, v in by_k.items()}, "→ k =", K)

msp = {name: -probs[name].max(1) for name in probs}
report = {
    "methode": "distance de Mahalanobis minimale aux centres de classe (covariance commune Ledoit-Wolf)",
    "k": K,
    "seuil": threshold,
    "quantile_validation": ID_QUANTILE,
    "manioc_test_accepte": round(float((scores(feats["test"], K) <= threshold).mean()), 3),
    "intrus": {
        name: {
            "images": len(feats[name]),
            "refusees": round(float((scores(feats[name], K) > threshold).mean()), 3),
            "auroc_mahalanobis": round(auroc(scores(feats["test"], K), scores(feats[name], K)), 3),
            "auroc_confiance_softmax": round(auroc(msp["test"], msp[name]), 3),
            "confiance_softmax_moyenne": round(float(probs[name].max(1).mean()), 3),
        }
        for name in outsiders
    },
    "manioc_confiance_softmax_moyenne": round(float(probs["test"].max(1).mean()), 3),
}

# 3. Greffe du score dans le graphe : z = f·W, distances aux 5 centres projetés, minimum.
w = projection(K).astype(np.float32)
projected_centers = (centers @ projection(K)).astype(np.float32)[None]
model.graph.initializer.extend([
    numpy_helper.from_array(w, "ood_projection"),
    numpy_helper.from_array(projected_centers, "ood_centers"),
    numpy_helper.from_array(np.array([1], dtype=np.int64), "ood_axis_1"),
    numpy_helper.from_array(np.array([2], dtype=np.int64), "ood_axis_2"),
])
model.graph.node.extend([
    helper.make_node("MatMul", [features_name, "ood_projection"], ["ood_z"]),
    helper.make_node("Unsqueeze", ["ood_z", "ood_axis_1"], ["ood_z3"]),
    helper.make_node("Sub", ["ood_z3", "ood_centers"], ["ood_diff"]),
    helper.make_node("Mul", ["ood_diff", "ood_diff"], ["ood_sq"]),
    helper.make_node("ReduceSum", ["ood_sq", "ood_axis_2"], ["ood_dist"], keepdims=0),
    helper.make_node("ReduceMin", ["ood_dist"], ["ood_score"], axes=[1], keepdims=0),
])
model.graph.output.append(helper.make_tensor_value_info("ood_score", TensorProto.FLOAT, ["N"]))
onnx.checker.check_model(model)
final_path = OUT / "manioc_effnetb0_ood.onnx"
onnx.save(model, final_path)

session = ort.InferenceSession(str(final_path), providers=["CPUExecutionProvider"])
graph_scores = np.concatenate([session.run(["ood_score"], {input_name: cassava["test"][i:i + 64].astype(np.float32)})[0]
                               for i in range(0, 256, 64)])
report["ecart_graphe_numpy"] = float(np.abs(graph_scores - scores(feats["test"][:256], K)).max() / threshold)
report["taille_mo"] = round(final_path.stat().st_size / 1e6, 1)
(OUT / "ood.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))

fig, ax = plt.subplots(figsize=(10, 4.5))
bins = np.logspace(np.log10(np.percentile(scores(feats["train"], K), 1)), np.log10(scores(all_ood, K).max()), 60)
ax.hist(scores(feats["test"], K), bins=bins, alpha=0.7, label="manioc (test)", density=True)
for name in outsiders:
    ax.hist(scores(feats[name], K), bins=bins, alpha=0.5, label=name, density=True, histtype="stepfilled")
ax.axvline(threshold, color="black", linestyle="--", label=f"seuil ({ID_QUANTILE:.0%} du manioc de validation)")
ax.set_xscale("log")
ax.set_xlabel("score d'étrangeté (distance de Mahalanobis minimale)")
ax.set_yticks([])
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "ood_scores.png", dpi=120)
