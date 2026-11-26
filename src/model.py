"""Model creation, training, evaluation and retraining.

Architecture: MobileNetV2 (ImageNet weights) as a feature extractor plus a
small regularised classification head. Training is done in two phases:
  1. train the head with the backbone frozen
  2. fine-tune the top of the backbone with a lower learning rate
Optimisation techniques: data augmentation, dropout, L2 regularisation,
Adam optimiser, EarlyStopping and ReduceLROnPlateau.
"""
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone

import numpy as np

from src.preprocessing import (
    CLASS_NAMES, IMG_SIZE, ROOT, TEST_DIR, TRAIN_DIR, UPLOAD_DIR,
    list_images, make_dataset, train_val_split,
)

MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "bean_disease_model.h5")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
ARCHIVE_DIR = os.path.join(MODELS_DIR, "archive")
FINE_TUNE_AT = 100  # MobileNetV2 layers below this index stay frozen when fine-tuning
FINETUNE_LR = 3e-5
RETRAIN_LR = 1e-5


def build_model(num_classes=len(CLASS_NAMES), dropout=0.3, l2=1e-4, lr=1e-3, imagenet=True):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet" if imagenet else None)
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_to_minus1_1")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(l2), name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="bean_disease_classifier")
    compile_model(model, lr)
    return model


def compile_model(model, lr):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def get_backbone(model):
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2"):
            return layer
    raise ValueError("MobileNetV2 backbone not found in model")


def unfreeze_top(model, fine_tune_at=FINE_TUNE_AT):
    base = get_backbone(model)
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    # BatchNorm layers stay in inference mode for stable fine-tuning on small data
    for layer in base.layers:
        if layer.__class__.__name__ == "BatchNormalization":
            layer.trainable = False


def callbacks(patience=4):
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]


def train(epochs_head=10, epochs_finetune=20, batch_size=32, verbose=1):
    """Train from scratch (transfer learning) on data/train. Returns (model, history dict)."""
    paths, labels = list_images(TRAIN_DIR)
    tr_p, va_p, tr_y, va_y = train_val_split(paths, labels)
    train_ds = make_dataset(tr_p, tr_y, batch_size, shuffle=True)
    val_ds = make_dataset(va_p, va_y, batch_size)

    model = build_model()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_head,
                   callbacks=callbacks(), verbose=verbose)

    unfreeze_top(model)
    compile_model(model, lr=FINETUNE_LR)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetune,
                   callbacks=callbacks(), verbose=verbose)

    history = {k: h1.history[k] + h2.history.get(k, []) for k in h1.history if k != "learning_rate"}
    history["phase_boundary"] = len(h1.history["loss"])
    return model, history


def evaluate(model, data_dir=TEST_DIR, batch_size=32):
    """Compute accuracy, loss, precision, recall, F1 and the confusion matrix."""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, log_loss, precision_score, recall_score)

    paths, labels = list_images(data_dir)
    ds = make_dataset(paths, labels, batch_size)
    probs = model.predict(ds, verbose=0)
    y_true = np.array(labels)
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, probs, labels=list(range(len(CLASS_NAMES))))),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class": classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                           output_dict=True, zero_division=0),
        "n_test": int(len(y_true)),
    }


def load_model(path=MODEL_PATH):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def load_pretrained(path=MODEL_PATH):
    """Rebuild the architecture and load the saved model's weights into it.
    This is the starting point for retraining (the saved model acts as the
    pre-trained model) and keeps the result saveable as .h5 again."""
    model = build_model(imagenet=False)
    model.set_weights(load_model(path).get_weights())
    return model


def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


def save_model(model, metrics, history=None, trigger="initial", extra=None):
    """Archive the current model and write the new one + metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    old = load_metadata()
    version = int(old.get("version", 0)) + 1
    if os.path.exists(MODEL_PATH):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        shutil.copy(MODEL_PATH, os.path.join(ARCHIVE_DIR, f"bean_disease_model_v{old.get('version', 0)}.h5"))
    model.save(MODEL_PATH, include_optimizer=False)
    meta = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger,
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "metrics": metrics,
        "history": history or old.get("history"),
        **(extra or {}),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def retrain(new_paths, new_labels, epochs=4, replay_per_class=100, batch_size=32,
            min_accuracy_drop=0.03, trigger="manual", verbose=0):
    """Retrain the existing model (used as the pre-trained starting point) on
    newly uploaded data plus a replay sample of the original training data.

    The new model is evaluated on the held-out test set and only promoted if
    its accuracy does not drop by more than `min_accuracy_drop`.
    """
    start = time.time()
    model = load_pretrained()
    old_meta = load_metadata()
    old_acc = old_meta.get("metrics", {}).get("accuracy", 0.0)

    # replay sample from the original data to avoid catastrophic forgetting
    rng = random.Random(42)
    base_p, base_y = list_images(TRAIN_DIR)
    replay_p, replay_y = [], []
    for c in range(len(CLASS_NAMES)):
        idx = [i for i, y in enumerate(base_y) if y == c]
        for i in rng.sample(idx, min(replay_per_class, len(idx))):
            replay_p.append(base_p[i])
            replay_y.append(c)

    paths = list(new_paths) + replay_p
    labels = list(new_labels) + replay_y
    tr_p, va_p, tr_y, va_y = train_val_split(paths, labels, val_fraction=0.15)

    unfreeze_top(model)
    compile_model(model, lr=RETRAIN_LR)
    hist = model.fit(make_dataset(tr_p, tr_y, batch_size, shuffle=True),
                     validation_data=make_dataset(va_p, va_y, batch_size),
                     epochs=epochs, callbacks=callbacks(patience=2), verbose=verbose)

    metrics = evaluate(model)
    promoted = metrics["accuracy"] >= old_acc - min_accuracy_drop
    result = {
        "promoted": promoted,
        "previous_accuracy": old_acc,
        "new_metrics": metrics,
        "n_new_samples": len(new_paths),
        "n_replay_samples": len(replay_p),
        "epochs_run": len(hist.history["loss"]),
        "duration_sec": round(time.time() - start, 1),
    }
    if promoted:
        history = {k: [float(v) for v in vals] for k, vals in hist.history.items() if k != "learning_rate"}
        meta = save_model(model, metrics, trigger=trigger,
                          extra={"retrain_history": history, "n_new_samples": len(new_paths)})
        meta["history"] = old_meta.get("history")
        with open(METADATA_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        result["version"] = meta["version"]
    else:
        result["version"] = old_meta.get("version")
    return result


if __name__ == "__main__":
    m, h = train()
    met = evaluate(m)
    print(json.dumps({k: v for k, v in met.items() if k != "per_class"}, indent=2))
    save_model(m, met, h)
