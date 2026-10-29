# Aide-mémoire — TP 2, classification des maladies du manioc

Trois parties :

1. [Les notions](#1-les-notions) : ce qu'il faut savoir expliquer à l'oral.
2. [Les fonctions et classes utilisées](#2-les-fonctions-et-classes-utilisées) : un glossaire
   des bouts de code (`keras.Input`, `GlobalAveragePooling2D`, `plt.cm.inferno`…).
3. [Le notebook cellule par cellule](#3-le-notebook-cellule-par-cellule) : ce que fait chaque
   ligne.

---

## 1. Les notions

### Image, tenseur, lot

Une image couleur est un tableau de pixels de forme **(hauteur, largeur, 3)**. Les 3 canaux
sont rouge, vert et bleu, chacun entre 0 et 255 (un entier `uint8` sur un octet). Un réseau
traite les images par **lots** (*batches*) : un lot de 32 images de 224 × 224 est un tenseur
de forme **(32, 224, 224, 3)**.

### Redimensionnement (*resize*)

Les photos du jeu font 800 × 600. On les ramène en **224 × 224** pour deux raisons :

- EfficientNetB0 a été pré-entraîné à cette taille ;
- 12 fois moins de pixels, donc 12 fois moins de calcul et de mémoire.

On passe du format 4:3 au carré, donc les feuilles sont un peu écrasées. Ce n'est pas grave
puisque toutes les images subissent la même déformation, à l'entraînement comme au test.
`antialias=True` lisse l'image avant de la réduire, pour éviter le crénelage (des motifs
parasites qui apparaissent quand on supprime des pixels brutalement).

### Couche de convolution (Conv2D)

Un **filtre** est une petite grille de poids (3 × 3 × nombre de canaux) qu'on fait glisser
sur l'image. À chaque position, il calcule une somme pondérée des pixels voisins. Il répond
fort quand le motif local ressemble à ce qu'il a appris : un contour, une tache, une
nervure. Une couche de 32 filtres produit 32 **cartes de caractéristiques** (*feature
maps*), une par filtre.

- Les premières couches détectent des motifs simples (contours, couleurs).
- Les couches profondes combinent ces motifs en motifs complexes (tache brune à bord jaune,
  feuille tordue).

### Normalisation par lot (BatchNormalization)

Elle recentre et remet à l'échelle les sorties d'une couche (moyenne 0, écart-type 1), lot
par lot. L'entraînement devient plus rapide et plus stable.

À l'entraînement, elle utilise la moyenne du lot courant. En prédiction, elle utilise une
**moyenne glissante** accumulée pendant l'entraînement. Le paramètre `momentum` règle la
vitesse de cette moyenne glissante :

- 0,99 (valeur par défaut) : elle suit lentement ;
- 0,9 : elle suit plus vite.

Avec peu de pas par époque, une moyenne trop lente reste en retard sur les poids. La
validation, qui utilise cette moyenne, oscille alors. C'est ce qu'on a observé sur le
premier essai, d'où `momentum=0.9`.

### ReLU

La fonction d'activation `max(0, x)`. Elle garde les valeurs positives et met les négatives
à zéro. Sans activation non linéaire, empiler des couches reviendrait à une seule
transformation linéaire : le réseau ne pourrait rien apprendre de complexe.

### Max pooling

Il découpe chaque carte en carrés de 2 × 2 et garde le maximum de chaque carré. Résultat : la
résolution est divisée par 2 (224 → 112 → 56 → 28 → 14). Les couches suivantes « voient »
ainsi une zone plus large de l'image, et le calcul diminue.

### Global average pooling

Il remplace chaque carte de caractéristiques entière par **sa moyenne**. Un tenseur
(7, 7, 1280) devient un vecteur de 1 280 nombres.

L'alternative classique, `Flatten`, mettrait les 7 × 7 × 1 280 = 62 720 valeurs bout à bout.
La couche dense suivante aurait alors 62 720 × 5 poids au lieu de 1 280 × 5, ce qui favorise
fortement le surapprentissage. La moyenne rend aussi le réseau indifférent à **l'endroit**
où se trouve le symptôme dans l'image.

### Dropout

Pendant l'entraînement, il met à zéro au hasard une fraction des neurones à chaque pas : 30 %
avec `Dropout(0.3)`. Le réseau ne peut donc pas compter sur un neurone précis. Il doit
répartir l'information, ce qui réduit le **surapprentissage**. En prédiction, le dropout est
désactivé et tous les neurones servent.

### Couche dense et softmax

La couche **dense** (*fully connected*) relie chaque entrée à chaque sortie. Ici, elle relie
le vecteur de caractéristiques à 5 nombres, un score par classe. Le **softmax** transforme
ces scores en **probabilités** positives dont la somme vaut 1. La classe prédite est celle de
plus forte probabilité (`argmax`).

### Fonction de perte : entropie croisée

Elle mesure à quel point les probabilités prédites s'éloignent de la vraie classe :
`perte = −log(probabilité donnée à la bonne classe)`.

- 90 % sur la bonne classe donne une perte de 0,105.
- 10 % sur la bonne classe donne une perte de 2,3.

L'entraînement cherche à la faire baisser. Avec `categorical_crossentropy`, les étiquettes
sont en **one-hot** : la classe 3 devient `[0, 0, 0, 1, 0]`.

### Optimiseur Adam et taux d'apprentissage

À chaque pas, l'optimiseur modifie les poids dans la direction qui fait baisser la perte (le
**gradient**). Le **taux d'apprentissage** (*learning rate*) règle la taille du pas :

- trop grand, on oscille ;
- trop petit, on n'avance pas.

**Adam** adapte en plus la taille du pas pour chaque poids. `ReduceLROnPlateau` divise le
taux par 2 quand le score de validation stagne : de grands pas au début, de petits pas pour
affiner à la fin.

### Époque, arrêt anticipé

Une **époque** est un passage complet sur toutes les images d'entraînement. L'**arrêt
anticipé** (*early stopping*) surveille le F1 de validation. S'il ne progresse plus depuis
`patience` époques, on arrête et on revient aux poids de la meilleure époque
(`restore_best_weights=True`).

### Entraînement, validation, test

| Jeu | Rôle | Images |
|---|---|---|
| entraînement (70 %) | le modèle apprend dessus | 8 400 |
| validation (15 %) | choisir quand s'arrêter, régler le taux | 1 800 |
| test (15 %) | note finale, regardé **une seule fois** | 1 800 |

On ne règle rien sur le test, sinon la note finale serait optimiste. Le découpage est
**stratifié** : chaque jeu garde les proportions des 5 classes.

### Surapprentissage et sous-apprentissage

- **Surapprentissage** (*overfitting*) : le modèle apprend les images d'entraînement par cœur.
  Sur les courbes, le score d'entraînement monte et celui de validation stagne ou baisse :
  l'écart se creuse. Remèdes : augmentation, dropout, plus de données.
- **Sous-apprentissage** (*underfitting*) : le modèle n'arrive même pas à bien apprendre
  l'entraînement. Les deux courbes restent basses et proches. Remèdes : plus d'époques, un
  modèle plus puissant, le transfert.

### Déséquilibre des classes et pondération

La mosaïque (CMD) fait 61 % des images, la bactériose (CBB) 5 %. Sans précaution, le modèle
apprend surtout la mosaïque. `class_weight="balanced"` multiplie la perte de chaque image par
`n_images / (n_classes × n_images_de_sa_classe)` :

- CBB, rare, pèse environ 3,9 ;
- CMD, fréquente, pèse environ 0,33.

### Augmentation de données

À chaque époque, chaque image est transformée au hasard : retournée, tournée de ±15 % d'un
tour, zoomée de ±20 %, contraste et luminosité changés de ±20 %. Le réseau ne voit jamais deux
fois exactement la même image. C'est comme avoir beaucoup plus de données. On ne choisit que
des transformations **qui ne changent pas le diagnostic** : une feuille retournée reste
malade. Les couches d'augmentation ne s'activent qu'à l'entraînement.

### Transfer learning et fine-tuning

On réutilise un réseau déjà entraîné sur **ImageNet** (1,3 million d'images, 1 000 classes).
Ses filtres savent déjà voir contours, textures et formes. On retire sa dernière couche
(`include_top=False`) et on la remplace par la nôtre, à 5 classes.

1. **Tête seule** (`base.trainable = False`) : on gèle le réseau pré-entraîné et on n'entraîne
   que la nouvelle couche. Sinon, les gros gradients d'une tête initialisée au hasard
   détruiraient les filtres pré-entraînés.
2. **Fine-tuning** (`base.trainable = True`) : on dégèle tout et on continue avec un taux
   **10 fois plus petit** (1e-4), pour ajuster finement les filtres sans les détruire.

`training=False` à l'appel de la base garde ses BatchNorm en mode inférence. Leurs
statistiques ImageNet ne sont pas écrasées par celles de nos petits lots.

### EfficientNetB0

C'est une famille de réseaux (B0 à B7) dont la profondeur, la largeur et la résolution
d'entrée grandissent ensemble de façon équilibrée. B0 est le plus petit : 4 millions de
paramètres, 77 % de bonnes réponses sur ImageNet. Il fait sa propre normalisation des pixels
et prend donc directement des valeurs entre 0 et 255.

### Métriques

| Métrique | Définition | Lecture |
|---|---|---|
| exactitude (*accuracy*) | bonnes réponses / total | trompeuse si déséquilibre : « toujours CMD » donne 61 % |
| précision d'une classe | parmi les images prédites X, part vraiment X | « quand il dit CBB, a-t-il raison ? » |
| rappel d'une classe | parmi les vraies X, part trouvée | « trouve-t-il toutes les CBB ? » |
| F1 d'une classe | 2 × précision × rappel / (précision + rappel) | compromis des deux |
| **F1 macro** | moyenne simple des F1 des 5 classes | chaque maladie compte autant, notre métrique principale |
| moyenne pondérée | moyenne des F1 pondérée par l'effectif | dominée par CMD |

### Matrice de confusion

C'est un tableau de 5 × 5 : la ligne est la vraie classe, la colonne la classe prédite. Avec
`normalize="true"`, chaque ligne est divisée par son total :

- la **diagonale** donne le **rappel** de chaque classe ;
- une case hors diagonale donne la part d'une classe prise pour une autre.

### Grad-CAM / CAM

C'est une carte de chaleur qui montre **où le réseau a regardé** pour décider. Pour la classe
*c* :

    CAM_c(i, j) = Σ_k  w_{k,c} × A_k(i, j)

- `A_k` est la k-ième carte 7 × 7 de la dernière couche convolutive d'EfficientNet (il y en a
  1 280).
- `w_{k,c}` est le poids de la couche dense qui relie cette carte à la classe *c*.

Grad-CAM pondère en général les cartes par des **gradients**. Mais quand la dernière couche
convolutive est suivie d'une moyenne puis d'une couche dense, ce gradient vaut exactement
`w_{k,c}` à une constante près, et Grad-CAM est identique à la CAM. On garde ensuite les
valeurs positives (ReLU), on agrandit la carte 7 × 7 en 224 × 224 et on la superpose à la
photo.

### SAM (*Segment Anything*)

C'est un modèle de segmentation de Meta (2023), entraîné sur un milliard de masques. On lui
donne un **indice** (un point, une boîte) et il renvoie le masque de l'objet désigné, sans
entraînement spécifique. Il propose **3 masques** (partie, objet, ensemble) avec un score de
qualité estimé (`iou_scores`), et on garde le meilleur.

**U-Net**, l'autre option de l'énoncé, est un réseau de segmentation à entraîner soi-même. Il
faut alors des images annotées à la main avec leurs masques, et le jeu manioc n'en a pas.
D'où SAM.

---

## 2. Les fonctions et classes utilisées

### Python, NumPy, pandas

| Code | Ce que ça fait |
|---|---|
| `Path("/kaggle/input").exists()` | vrai si le dossier existe, donc si on tourne sur Kaggle |
| `Path(a) / "b"` | construit un chemin `a/b` |
| `np.flatnonzero(cond)` | indices où la condition est vraie : `np.flatnonzero(y == 2)` donne les positions des CGM |
| `np.argsort(-x)` | indices qui trient `x` par ordre **décroissant** (le signe moins inverse l'ordre) |
| `probs.argmax(1)` | pour chaque ligne, l'indice de la plus grande valeur, donc la classe prédite |
| `probs.max(1)` | pour chaque ligne, la plus grande valeur, donc la confiance |
| `np.where(masque, a, b)` | prend `a` là où le masque est vrai, `b` ailleurs : la feuille ou le gris |
| `np.concatenate(liste)` | colle des tableaux bout à bout |
| `image[None]` | ajoute une dimension de lot : (224, 224, 3) devient (1, 224, 224, 3) |
| `masks[..., None]` | ajoute une dimension à la fin : (n, 224, 224) devient (n, 224, 224, 1), pour multiplier par 3 canaux |
| `pd.read_csv` | lit un fichier CSV en tableau (DataFrame) |
| `value_counts().sort_index()` | compte les occurrences de chaque valeur, triées par valeur |
| `pd.crosstab(a, b, margins=True)` | tableau croisé des effectifs, avec les totaux |
| `df.assign(split="train")` | ajoute une colonne constante |
| `f"{x:.0%}"`, `f"{x:,}"`, `f"{x:.3f}"` | format : pourcentage sans décimale, milliers séparés, 3 décimales |
| `ids := expr` | l'« opérateur morse » : calcule `expr`, le range dans `ids` et le renvoie, dans la même expression |

### TensorFlow : lecture des images

| Code | Ce que ça fait |
|---|---|
| `tf.io.read_file(path)` | lit les octets bruts du fichier |
| `tf.io.decode_jpeg(..., channels=3)` | décode le JPEG en tableau (h, l, 3) |
| `tf.image.resize(img, (224, 224), antialias=True)` | redimensionne, avec lissage anti-crénelage |
| `tf.cast(tf.round(x), tf.uint8)` | arrondit puis convertit en entiers 0–255, 4 fois moins de mémoire que des flottants |
| `tf.data.Dataset.from_tensor_slices(x)` | crée un flux d'éléments à partir d'un tableau ou d'une liste |
| `.map(f, num_parallel_calls=AUTOTUNE)` | applique `f` à chaque élément, sur plusieurs cœurs en parallèle |
| `.shuffle(n, seed=…)` | mélange (tampon de `n` éléments, donc mélange complet si `n` = taille du jeu) |
| `.batch(32)` | regroupe par lots de 32 |
| `.prefetch(AUTOTUNE)` | prépare le lot suivant pendant que le GPU calcule le lot courant |
| `tf.config.set_visible_devices(gpus[:1], "GPU")` | TensorFlow n'utilise que le premier GPU |
| `set_memory_growth(gpu, True)` | réserve la mémoire du GPU au fur et à mesure, au lieu de tout prendre au démarrage |

### Keras : construire un modèle

| Code | Ce que ça fait |
|---|---|
| `keras.Input((224, 224, 3))` | déclare l'**entrée** du modèle et sa forme (sans la dimension du lot). C'est le point de départ de l'API fonctionnelle |
| `x = Couche(...)(x)` | API **fonctionnelle** : on crée une couche, puis on l'appelle sur le tenseur précédent. On chaîne ainsi les couches |
| `keras.Model(inputs, outputs)` | assemble le graphe entre l'entrée et la sortie en un modèle |
| `keras.Sequential([...])` | modèle (ou bloc) simple : les couches s'enchaînent dans l'ordre de la liste |
| `layers.Rescaling(1 / 255)` | multiplie les pixels par 1/255 pour passer de 0–255 à 0–1 |
| `layers.Conv2D(32, 3, padding="same", use_bias=False)` | 32 filtres 3 × 3. `same` complète les bords pour garder la taille. Pas de biais, car la BatchNorm qui suit en a un |
| `layers.BatchNormalization(momentum=0.9)` | normalisation par lot (voir partie 1) |
| `layers.ReLU()` | activation `max(0, x)` |
| `layers.MaxPooling2D()` | max sur des carrés de 2 × 2, résolution divisée par 2 |
| `layers.GlobalAveragePooling2D()` | moyenne de chaque carte : (h, l, c) devient (c,) |
| `layers.Dropout(0.3)` | éteint 30 % des neurones au hasard, à l'entraînement seulement |
| `layers.Dense(5, activation="softmax", name="probs")` | couche dense à 5 sorties, puis softmax. Le nom permet de la retrouver avec `get_layer("probs")` |
| `layers.RandomFlip("horizontal_and_vertical")` | retournement aléatoire gauche-droite et haut-bas |
| `layers.RandomRotation(0.15)` | rotation aléatoire de ±0,15 tour (±54°) |
| `layers.RandomZoom(0.2)` | zoom aléatoire de ±20 % |
| `layers.RandomContrast(0.2, value_range=(0, 255))` | contraste ±20 %. `value_range` indique que les pixels vont de 0 à 255 |
| `layers.RandomBrightness(0.2, value_range=(0, 255))` | luminosité ±20 % |
| `keras.applications.EfficientNetB0(include_top=False, weights="imagenet")` | télécharge EfficientNetB0 pré-entraîné, **sans** sa couche de classification ImageNet |
| `base.trainable = False / True` | gèle ou dégèle tous les poids de la base |
| `base(x, training=False)` | appelle la base en mode inférence (BatchNorm figées) |
| `model.count_params()` | nombre total de paramètres |
| `model.get_layer("probs").get_weights()` | renvoie `[noyau, biais]` de la couche. Le noyau fait (1280, 5) |
| `keras.utils.to_categorical(y, 5)` | étiquettes entières vers one-hot : 3 devient `[0, 0, 0, 1, 0]` |
| `keras.utils.set_random_seed(42)` | fixe le hasard (Python, NumPy, TensorFlow) pour des résultats reproductibles |

### Keras : entraîner et prédire

| Code | Ce que ça fait |
|---|---|
| `model.compile(optimizer, loss, metrics)` | choisit l'optimiseur, la perte et les métriques affichées. À refaire après avoir changé `trainable` |
| `keras.optimizers.Adam(1e-3)` | Adam avec un taux d'apprentissage de 0,001 |
| `keras.metrics.F1Score(average="macro", name="f1")` | F1 macro calculé à chaque époque. Apparaît sous le nom `f1` et `val_f1` |
| `model.fit(train, validation_data=val, epochs, class_weight, callbacks)` | lance l'entraînement. Renvoie un objet `History` |
| `history.history` | dictionnaire `{"loss": [...], "f1": [...], "val_loss": [...], "val_f1": [...], ...}`, une valeur par époque |
| `keras.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience, restore_best_weights=True)` | arrêt anticipé sur le F1 de validation (`max` : plus c'est haut, mieux c'est) |
| `keras.callbacks.ReduceLROnPlateau(monitor="val_f1", mode="max", factor=0.5, patience=3)` | divise le taux par 2 après 3 époques sans progrès |
| `model.predict(X, batch_size=64)` | renvoie les probabilités, forme (n_images, 5) |

### scikit-learn

| Code | Ce que ça fait |
|---|---|
| `train_test_split(df, train_size=…, stratify=df["label"], random_state=42)` | découpe au hasard en gardant les proportions des classes |
| `compute_class_weight("balanced", classes, y)` | poids inversement proportionnels à l'effectif de chaque classe |
| `accuracy_score(y, pred)` | exactitude |
| `f1_score(y, pred, average="macro")` | F1 macro. Avec `average=None`, un F1 par classe |
| `classification_report(y, pred, target_names, digits=3)` | tableau précision / rappel / F1 / effectif par classe |
| `ConfusionMatrixDisplay.from_predictions(y, pred, normalize="true", cmap="Oranges")` | calcule et dessine la matrice de confusion normalisée par ligne |

### Matplotlib et OpenCV

| Code | Ce que ça fait |
|---|---|
| `plt.subplots(2, 3, figsize=(16, 8))` | une figure de 2 × 3 graphiques. `figsize` en pouces |
| `sharey="row"` | les graphiques d'une même ligne partagent l'axe vertical |
| `squeeze=False` | `axes` reste un tableau 2D même avec une seule colonne |
| `layout="constrained"` | place automatiquement titres et images sans chevauchement |
| `ax.imshow(image)` | affiche une image |
| `ax.axis("off")` | masque les axes autour d'une image |
| `ax.plot(liste, label=…)` | trace une courbe |
| `ax.axvline(x, linestyle=":")` | trait vertical pointillé, ici le début du fine-tuning |
| `ax.bar(x, hauteurs, largeur)` / `ax.bar_label(barres)` | diagramme en barres, et valeurs écrites au-dessus |
| `plt.suptitle`, `plt.tight_layout`, `plt.savefig`, `plt.show` | titre général, ajustement des marges, enregistrement en PNG, affichage |
| `plt.cm.inferno(valeurs)` | **palette de couleurs** : transforme des valeurs entre 0 et 1 en couleurs (noir → violet → orange → jaune). Renvoie du RGBA entre 0 et 1, d'où `[..., :3]` (on retire la transparence) et `* 255` |
| `cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)` | agrandit la carte 7 × 7 en 224 × 224 avec une interpolation douce |
| `cv2.INTER_NEAREST` | interpolation « plus proche voisin », pour un masque qui doit rester en 0/1 |

### PyTorch et Hugging Face (SAM)

| Code | Ce que ça fait |
|---|---|
| `torch.cuda.device_count()` | nombre de GPU vus par PyTorch. `cuda:1` est le second |
| `SamModel.from_pretrained("facebook/sam-vit-base")` | télécharge le modèle SAM (version de base, 375 Mo) |
| `.to(device).eval()` | envoie le modèle sur le GPU, en mode inférence |
| `@torch.no_grad()` | ne calcule pas de gradients dans la fonction : plus rapide, moins de mémoire |
| `sam_processor(image, input_points=[[[x, y]]], return_tensors="pt")` | prépare l'image (redimensionnée en 1 024) et le point cliqué au format attendu par SAM |
| `post_process_masks(...)` | remet les masques prédits à la taille de l'image d'origine |
| `outputs.iou_scores` | qualité estimée de chacun des 3 masques proposés |

---

## 3. Le notebook cellule par cellule

### Cellule d'imports et de configuration

```python
import time                      # mesurer les durées d'entraînement
from pathlib import Path         # manipuler les chemins de fichiers
import cv2                       # OpenCV : redimensionner les cartes de chaleur et les masques
import keras                     # construire et entraîner les réseaux
import matplotlib.pyplot as plt  # graphiques
import numpy as np               # calcul sur tableaux
import pandas as pd              # tableaux de données (CSV, résultats)
import tensorflow as tf          # moteur de Keras, lecture des images
from keras import layers         # les couches (Conv2D, Dense…)
from sklearn.metrics import ...  # métriques et matrice de confusion
from sklearn.model_selection import train_test_split        # découpage stratifié
from sklearn.utils.class_weight import compute_class_weight # poids des classes
```

```python
FAST = False            # True = mini-exécution de test (1 200 images, 1 époque)
SEED = 42               # graine du hasard, pour pouvoir reproduire les résultats
IMG_SIZE = 224          # taille des images en entrée des réseaux
N_SAMPLE = 12_000       # nombre d'images tirées parmi les 21 397
EPOCHS = None           # None = utiliser le nombre d'époques prévu pour chaque modèle
```

- `IS_KAGGLE`, `DATA`, `OUT` : où lire les données et écrire les résultats, selon qu'on est
  sur Kaggle ou en local.
- `keras.utils.set_random_seed(SEED)` : même hasard à chaque exécution.
- Les 4 lignes GPU : TensorFlow ne prend que le premier T4 et réserve sa mémoire
  progressivement. Le second T4 reste libre pour SAM (PyTorch).

### 1.1 Les classes

```python
CLASSES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]   # l'ordre = le numéro de classe 0 à 4
LABELS = list(range(len(CLASSES)))                   # [0, 1, 2, 3, 4]
full = pd.read_csv(DATA / "train.csv")               # 21 397 lignes : image_id, label
counts = full["label"].value_counts().sort_index()   # nombre d'images par classe
pd.DataFrame({...})                                  # tableau affiché : classe, nom, effectif, part
```

### 1.2 Échantillon et découpage

```python
sample, _ = train_test_split(full, train_size=N_SAMPLE, stratify=full["label"], ...)
# garde 12 000 images en respectant les proportions ; « _ » jette le reste
train_df, rest = train_test_split(sample, test_size=0.30, stratify=..., ...)
# 70 % pour l'entraînement, 30 % mis de côté
val_df, test_df = train_test_split(rest, test_size=0.50, stratify=..., ...)
# le reste coupé en deux : 15 % validation, 15 % test
split.to_csv(OUT / "split.csv")      # garde la trace du découpage (identifiants seulement)
pd.crosstab(...)                     # vérifie les effectifs par classe et par jeu
```

### 1.3 Chargement des images

```python
def load_images(image_ids):
    paths = [str(DATA / "train_images" / image_id) for image_id in image_ids]  # chemins des JPEG

    def decode(path):                                       # appliquée à chaque fichier
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)   # octets → pixels
        return tf.cast(tf.round(tf.image.resize(image, (224, 224), antialias=True)), tf.uint8)

    dataset = tf.data.Dataset.from_tensor_slices(paths).map(decode, num_parallel_calls=AUTOTUNE)
    return np.concatenate(list(dataset.batch(256).as_numpy_iterator()))
    # décode par paquets de 256 en parallèle puis recolle tout en un seul tableau (n, 224, 224, 3)

X_train, X_val, X_test = (load_images(df["image_id"]) for df in (train_df, val_df, test_df))
y_train, y_val, y_test = (df["label"].to_numpy() for df in (...))   # les numéros de classe
```

On décode une seule fois et on garde tout en mémoire (≈ 1,8 Go) : relire et décoder les JPEG
à chaque époque ferait attendre le GPU.

### Galerie d'exemples

```python
fig, axes = plt.subplots(3, 5, ...)                          # 3 lignes × 5 classes
for label in LABELS:
    for row, index in enumerate(np.flatnonzero(y_train == label)[:3]):  # 3 premières images de la classe
        axes[row, label].imshow(X_train[index])
```

### 2. Protocole commun

```python
CLASS_WEIGHT = dict(enumerate(compute_class_weight("balanced", ...)))
# {0: 3.9, 1: 1.95, 2: 1.8, 3: 0.33, 4: 1.66} : la perte d'une CBB compte 12 fois plus qu'une CMD

def make_dataset(X, y, training):
    dataset = tf.data.Dataset.from_tensor_slices((X, keras.utils.to_categorical(y, 5)))  # (image, one-hot)
    if training:
        dataset = dataset.shuffle(len(X), seed=SEED)   # mélange à chaque époque (entraînement seulement)
    return dataset.batch(32).prefetch(AUTOTUNE)        # lots de 32, préparés à l'avance

val_ds = make_dataset(X_val, y_val, training=False)   # créé une fois, réutilisé par les 3 modèles

def fit(model, epochs, learning_rate, patience=5):
    model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy",
                  metrics=["accuracy", F1Score(average="macro", name="f1")])
    callbacks = [ReduceLROnPlateau(...), EarlyStopping(...)]
    history = model.fit(..., epochs=EPOCHS or epochs, class_weight=CLASS_WEIGHT, callbacks=callbacks, verbose=2)
    return history.history       # les courbes, époque par époque
```

`EPOCHS or epochs` vaut 1 en mode `FAST`, et sinon le nombre d'époques passé en paramètre.
`verbose=2` affiche une ligne par époque.

### 3. CNN simple

```python
def make_augmentation():                 # le bloc d'augmentation (voir partie 1)
    return keras.Sequential([RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness])

def simple_cnn(augment):
    inputs = keras.Input((224, 224, 3))                       # entrée : une image couleur
    x = make_augmentation()(inputs) if augment else inputs    # augmentation seulement si demandée
    x = layers.Rescaling(1 / 255)(x)                          # pixels 0–255 → 0–1
    for filters in (32, 64, 128, 256):                        # 4 blocs
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)                          # 224 → 112 → 56 → 28 → 14
    x = layers.GlobalAveragePooling2D()(x)                    # (14, 14, 256) → (256,)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(5, activation="softmax")(x)        # 5 probabilités
    return keras.Model(inputs, outputs)

cnn = simple_cnn(augment=False)
histories["CNN simple"] = fit(cnn, epochs=50, learning_rate=1e-3, patience=10)
```

Le réseau compte 391 141 paramètres, dont l'essentiel dans les filtres 3 × 3 × 128 × 256 du
dernier bloc.

### 4. CNN + augmentation

```python
cnn_aug = simple_cnn(augment=True)       # même architecture, avec le bloc d'augmentation
histories["CNN + augmentation"] = fit(cnn_aug, epochs=60, ...)   # plus d'époques : la tâche est plus dure
```

### 5. Transfer learning

```python
base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base.trainable = False                                   # temps 1 : base gelée
inputs = keras.Input((224, 224, 3))
features = base(make_augmentation()(inputs), training=False)   # cartes (7, 7, 1280)
x = layers.GlobalAveragePooling2D()(features)            # (1280,)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(5, activation="softmax", name="probs")(x)   # notre tête
effnet = keras.Model(inputs, outputs)

head = fit(effnet, epochs=5, learning_rate=1e-3)        # temps 1 : seule la tête apprend
head_epochs = len(head["f1"])                            # pour tracer le trait pointillé
base.trainable = True                                    # temps 2 : on dégèle
finetune = fit(effnet, epochs=20, learning_rate=1e-4, patience=6)   # fit() recompile, donc la modification est prise en compte
histories[...] = {key: head[key] + finetune[key] for key in head}   # recolle les deux courbes
```

### Courbes d'apprentissage

Il y a une figure de 2 × 3 graphiques : une colonne par modèle, le F1 macro en haut, la perte
en bas. Chaque graphique a deux courbes, entraînement et validation.

| Ce qu'on voit | Diagnostic |
|---|---|
| entraînement monte, validation stagne : l'écart se creuse | surapprentissage |
| deux courbes basses et proches | sous-apprentissage |
| validation en dents de scie | instabilité (taux trop grand, BatchNorm, classes très pondérées) |

### 6. Comparaison sur le test

```python
models = {"CNN simple": cnn, ...}
test_probs = {name: model.predict(X_test) for ...}       # probabilités (1 800, 5) par modèle
test_pred = {name: probs.argmax(1) for ...}              # classe prédite par modèle
pd.DataFrame({...}).T                                    # tableau : paramètres, durée, exactitude, F1
```

Le diagramme en barres compare, pour chaque classe et pour le F1 macro, les trois modèles
côte à côte. `width = 0.27` est la largeur d'une barre, et `(k - 1) * width` décale les barres
du modèle k à gauche, au centre ou à droite.

Les matrices de confusion sont dessinées en 3 côte à côte, normalisées par ligne. Le
`classification_report` donne précision, rappel et F1 par classe pour le meilleur modèle.

### 7.1 Confusions fréquentes

```python
best_pred = best_probs.argmax(1)        # prédictions du modèle transféré
confidence = best_probs.max(1)          # sa confiance pour chaque image
errors = pd.DataFrame({"vrai": ..., "prédit": ...})[best_pred != y_test]   # seulement les erreurs
errors.value_counts().head(8)           # les 8 couples (vrai, prédit) les plus fréquents
```

### 7.2 Grad-CAM

```python
feature_model = keras.Model(effnet.input, features)             # même réseau, mais sortie = cartes 7 × 7 × 1280
class_weights_dense = effnet.get_layer("probs").get_weights()[0]   # poids w, forme (1280, 5)

def heatmap(image, label):
    feature_map = feature_model.predict(image[None])[0]          # (7, 7, 1280)
    cam = np.maximum(feature_map @ class_weights_dense[:, label], 0)
    # @ = produit matriciel : somme pondérée des 1280 cartes → (7, 7), puis ReLU
    cam = cv2.resize(cam / (cam.max() + 1e-8), (224, 224), interpolation=cv2.INTER_CUBIC)
    # normalisée entre 0 et 1 (+1e-8 évite une division par zéro), puis agrandie
    colored = plt.cm.inferno(np.clip(cam, 0, 1))[..., :3] * 255  # valeurs → couleurs RVB 0–255
    return (0.55 * image + 0.45 * colored).astype(np.uint8)      # mélange 55 % photo + 45 % chaleur

def show_cases(indices, title):
    # 2 lignes : la photo (titre vrai / prédit / confiance) et la photo avec la carte de chaleur
```

`show_cases([...])` en fin de cellule affiche une bonne réponse par classe. `ids :=` garde les
indices des bonnes réponses de la classe, et on ne la montre que s'il y en a au moins une.

### 7.3 Erreurs les plus confiantes

```python
wrong = np.flatnonzero(best_pred != y_test)                      # indices des erreurs
most_confident_errors = wrong[np.argsort(-confidence[wrong])][:8]
# trie les erreurs par confiance décroissante et garde les 8 premières
show_cases(most_confident_errors[:4], ...)                       # affichées par 4
```

### 8. Bonus SAM

```python
TORCH_DEVICE = f"cuda:{torch.cuda.device_count() - 1}"   # le dernier GPU, donc le second T4
sam = SamModel.from_pretrained("facebook/sam-vit-base").to(TORCH_DEVICE).eval()

def leaf_mask(image_id):
    image = Image.open(...)                                       # la photo d'origine 800 × 600
    inputs = sam_processor(image, input_points=[[[l / 2, h / 2]]], ...)   # un clic au centre
    outputs = sam(**inputs)                                       # 3 masques candidats + scores
    masks = post_process_masks(...)[0][0]                         # masques à la taille d'origine
    best = masks[outputs.iou_scores[0, 0].argmax()]               # le meilleur des 3
    return cv2.resize(best, (224, 224), INTER_NEAREST).astype(bool)   # en 224 × 224, vrai/faux

masks = np.array([leaf_mask(i) for i in test_df["image_id"].iloc[:300]])
X_masked = np.where(masks[..., None], X_test[:300], 128)         # feuille gardée, fond gris (128)
masked_pred = effnet.predict(X_masked).argmax(1)                 # le modèle, sans réentraînement
pd.Series({...})   # surface de la feuille, F1 image entière vs feuille seule, part des prédictions changées
```

La dernière figure montre 6 images : en haut l'original, en bas la feuille détourée, avec la
prédiction dans les deux cas.
