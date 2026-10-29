# Réentraîne EfficientNetB0 avec le protocole et le découpage du notebook, puis l'exporte en
# ONNX pour l'application web : une entrée `image` (1, 224, 224, 3) en pixels 0-255, deux
# sorties, les probabilités (1, 5) et les cartes CAM (1, 7, 7, 5).
import json
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tf2onnx", "onnxruntime"], check=True)

import keras
import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorflow as tf
from keras import layers
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
IMG_SIZE = 224
N_SAMPLE = 12_000
CLASSES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
LABELS = list(range(len(CLASSES)))
DATA = Path("/kaggle/input/competitions/cassava-leaf-disease-classification")
OUT = Path("/kaggle/working")

keras.utils.set_random_seed(SEED)
gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(gpus[:1], "GPU")
for gpu in gpus[:1]:
    tf.config.experimental.set_memory_growth(gpu, True)

full = pd.read_csv(DATA / "train.csv")
sample, _ = train_test_split(full, train_size=N_SAMPLE, stratify=full["label"], random_state=SEED)
train_df, rest = train_test_split(sample, test_size=0.30, stratify=sample["label"], random_state=SEED)
val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)


def load_images(image_ids):
    paths = [str(DATA / "train_images" / image_id) for image_id in image_ids]

    def decode(path):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        return tf.cast(tf.round(tf.image.resize(image, (IMG_SIZE, IMG_SIZE), antialias=True)), tf.uint8)

    dataset = tf.data.Dataset.from_tensor_slices(paths).map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return np.concatenate(list(dataset.batch(256).as_numpy_iterator()))


X_train, X_val, X_test = (load_images(df["image_id"]) for df in (train_df, val_df, test_df))
y_train, y_val, y_test = (df["label"].to_numpy() for df in (train_df, val_df, test_df))
CLASS_WEIGHT = dict(enumerate(compute_class_weight("balanced", classes=np.array(LABELS), y=y_train)))


def make_dataset(X, y, training):
    dataset = tf.data.Dataset.from_tensor_slices((X, keras.utils.to_categorical(y, len(CLASSES))))
    if training:
        dataset = dataset.shuffle(len(X), seed=SEED)
    return dataset.batch(32).prefetch(tf.data.AUTOTUNE)


val_ds = make_dataset(X_val, y_val, training=False)


def fit(model, epochs, learning_rate, patience=5):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="macro", name="f1")],
    )
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_f1", mode="max", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=patience, restore_best_weights=True),
    ]
    model.fit(make_dataset(X_train, y_train, training=True), validation_data=val_ds, epochs=epochs,
              class_weight=CLASS_WEIGHT, callbacks=callbacks, verbose=2)


augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2, value_range=(0, 255)),
    layers.RandomBrightness(0.2, value_range=(0, 255)),
])
base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False
inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
x = layers.GlobalAveragePooling2D()(base(augmentation(inputs), training=False))
x = layers.Dropout(0.3)(x)
dense = layers.Dense(len(CLASSES), activation="softmax", name="probs")
effnet = keras.Model(inputs, dense(x))

fit(effnet, epochs=5, learning_rate=1e-3)
base.trainable = True
fit(effnet, epochs=20, learning_rate=1e-4, patience=6)

# Modèle d'export, sans augmentation : la CAM d'une classe est une convolution 1 × 1 qui porte
# les poids de la couche dense (même raisonnement que la section 7.2 du notebook).
image = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
features = base(image, training=False)
probs = dense(layers.GlobalAveragePooling2D()(features))
cam_layer = layers.Conv2D(len(CLASSES), 1, name="cam")
cam = cam_layer(features)
kernel, bias = dense.get_weights()
cam_layer.set_weights([kernel[None, None], bias])
export_model = keras.Model(image, [probs, cam])

# keras.Model.export(format="onnx") laisse les constantes de normalisation d'EfficientNet en
# entrées du graphe ; le passage par un SavedModel puis tf2onnx les fige correctement.
export_model.export(str(OUT / "saved"), format="tf_saved_model")
float_path = OUT / "manioc_effnetb0.onnx"
subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(OUT / "saved"),
                "--output", str(float_path), "--opset", "17"], check=True)
quant_path = OUT / "manioc_effnetb0_int8.onnx"
quantize_dynamic(str(float_path), str(quant_path), weight_type=QuantType.QUInt8)


def evaluate(path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    probs = np.concatenate([session.run(None, {name: X_test[i:i + 64].astype(np.float32)})[0]
                            for i in range(0, len(X_test), 64)])
    pred = probs.argmax(1)
    return {
        "taille_mo": round(path.stat().st_size / 1e6, 1),
        "exactitude": round(float(accuracy_score(y_test, pred)), 3),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 3),
        "f1_par_classe": dict(zip(CLASSES, f1_score(y_test, pred, labels=LABELS, average=None).round(3).tolist())),
        "matrice_confusion": confusion_matrix(y_test, pred, labels=LABELS).tolist(),
    }


keras_pred = effnet.predict(X_test, batch_size=64, verbose=0).argmax(1)
info = {
    "classes": CLASSES,
    "test_images": len(y_test),
    "keras_f1_macro": round(float(f1_score(y_test, keras_pred, average="macro")), 3),
    "onnx_float32": evaluate(float_path),
    "onnx_int8": evaluate(quant_path),
}
(OUT / "model_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(info, indent=2, ensure_ascii=False))
