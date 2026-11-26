"""Data processing utilities shared by training, retraining and prediction."""
import io
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(DATA_DIR, "uploads"))

IMG_SIZE = 224
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class InvalidImageError(ValueError):
    pass


def load_image(source):
    """Open an image from a path, raw bytes or file-like object as RGB."""
    try:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        img = Image.open(source)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Not a valid image: {exc}") from exc
    return img.convert("RGB")


def preprocess_image(source, size=IMG_SIZE):
    """Return a float32 array of shape (size, size, 3) in the [0, 255] range.

    Pixel scaling to [-1, 1] lives inside the model (Rescaling layer), so the
    same raw array works for training, evaluation and serving.
    """
    img = load_image(source).resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def preprocess_batch(sources, size=IMG_SIZE):
    return np.stack([preprocess_image(s, size) for s in sources])


def save_upload(content, filename, label, upload_dir=UPLOAD_DIR):
    """Validate an uploaded image, normalise it (RGB, IMG_SIZE) and store it
    under data/uploads/<label>/. Returns the saved path."""
    if label not in CLASS_NAMES:
        raise ValueError(f"Unknown label '{label}'. Expected one of {CLASS_NAMES}")
    if not filename.lower().endswith(ALLOWED_EXT):
        raise InvalidImageError(f"Unsupported file type: {filename}")
    img = load_image(content).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    out_dir = os.path.join(upload_dir, label)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(filename))[0][:60]
    path = os.path.join(out_dir, f"{stem}.jpg")
    i = 1
    while os.path.exists(path):
        path = os.path.join(out_dir, f"{stem}_{i}.jpg")
        i += 1
    img.save(path, quality=90)
    return path


def list_images(directory):
    """Return (paths, labels) for a directory laid out as <class>/<image>."""
    paths, labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(directory, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            if f.lower().endswith(ALLOWED_EXT):
                paths.append(os.path.join(cls_dir, f))
                labels.append(idx)
    return paths, labels


def make_dataset(paths, labels, batch_size=32, shuffle=False, seed=42):
    """Build a tf.data pipeline: read -> decode -> resize -> batch -> prefetch."""
    import tensorflow as tf

    def _load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_val_split(paths, labels, val_fraction=0.15, seed=42):
    from sklearn.model_selection import train_test_split

    return train_test_split(paths, labels, test_size=val_fraction, stratify=labels, random_state=seed)
