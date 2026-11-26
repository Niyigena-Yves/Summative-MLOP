"""Data acquisition for the Bean Leaf Disease classifier.

Downloads the iBeans dataset (Makerere AI Lab) and writes it into the
project layout:

    data/train/<class>/*.jpg   (train + validation splits merged)
    data/test/<class>/*.jpg

Images are resized to IMG_SIZE x IMG_SIZE once here so the repository
stays small and training is fast.

Usage:
    python -m src.data_acquisition
"""
import io
import os
import shutil
import urllib.request
import zipfile

from PIL import Image

BASE_URL = "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data"
SPLITS = {"train": "train", "validation": "train", "test": "test"}
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
STORE_SIZE = 224

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
DATA_DIR = os.path.join(ROOT, "data")


def download(split):
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{split}.zip")
    if not os.path.exists(path):
        url = f"{BASE_URL}/{split}.zip"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def extract(zip_path, target_split, prefix):
    counts = {c: 0 for c in CLASSES}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            label = os.path.basename(os.path.dirname(name))
            if label not in CLASSES:
                continue
            out_dir = os.path.join(DATA_DIR, target_split, label)
            os.makedirs(out_dir, exist_ok=True)
            img = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
            img = img.resize((STORE_SIZE, STORE_SIZE), Image.BILINEAR)
            fname = f"{prefix}_{os.path.splitext(os.path.basename(name))[0]}.jpg"
            img.save(os.path.join(out_dir, fname), quality=90)
            counts[label] += 1
    return counts


def main():
    for split in ("train", "test"):
        shutil.rmtree(os.path.join(DATA_DIR, split), ignore_errors=True)
    for split, target in SPLITS.items():
        counts = extract(download(split), target, split[:3])
        print(f"{split:>10} -> data/{target}: {counts}")


if __name__ == "__main__":
    main()
