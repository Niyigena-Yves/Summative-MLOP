"""Image feature extraction used for the dataset visualisations.

Features computed per image:
  * mean R, G, B intensity
  * brightness (mean of grayscale) and contrast (std of grayscale)
  * greenness index   = mean(G - (R + B) / 2)  -> amount of healthy green tissue
  * lesion ratio      = share of pixels that are brown/yellow (R >= G, not too dark)
  * texture (edge density) = mean absolute Laplacian-like gradient
"""
import json
import os

import numpy as np
from PIL import Image

from src.preprocessing import CLASS_NAMES, ROOT, TRAIN_DIR, list_images

INSIGHTS_PATH = os.path.join(ROOT, "models", "insights.json")


def image_features(path, size=128):
    a = np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    hsv = np.asarray(Image.open(path).convert("RGB").resize((size, size)).convert("HSV"), dtype=np.float32)
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    # brown/yellow lesion tissue: hue 15-60 deg with enough saturation and not dark
    lesion = (hue >= 15) & (hue <= 60) & (sat > 0.25) & (val > 0.25)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return {
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
        "brightness": float(gray.mean()), "contrast": float(gray.std()),
        "greenness": float((g - (r + b) / 2).mean()),
        "lesion_ratio": float(lesion.mean()),
        "edge_density": float((gx + gy) / 2),
    }


def compute_insights(data_dir=TRAIN_DIR, save=True):
    paths, labels = list_images(data_dir)
    rows = [dict(image_features(p), label=CLASS_NAMES[y]) for p, y in zip(paths, labels)]
    feats = [k for k in rows[0] if k != "label"]
    out = {"class_counts": {c: labels.count(i) for i, c in enumerate(CLASS_NAMES)},
           "features": {}, "histograms": {}}
    for f in feats:
        out["features"][f] = {c: round(float(np.mean([r[f] for r in rows if r["label"] == c])), 4)
                              for c in CLASS_NAMES}
        vals = np.array([r[f] for r in rows])
        bins = np.linspace(vals.min(), vals.max(), 21)
        out["histograms"][f] = {
            "bins": [round(float(x), 4) for x in bins],
            "counts": {c: np.histogram([r[f] for r in rows if r["label"] == c], bins=bins)[0].tolist()
                       for c in CLASS_NAMES},
        }
    if save:
        os.makedirs(os.path.dirname(INSIGHTS_PATH), exist_ok=True)
        with open(INSIGHTS_PATH, "w") as f:
            json.dump(out, f, indent=1)
    return out, rows


def load_insights():
    if not os.path.exists(INSIGHTS_PATH):
        return compute_insights()[0]
    with open(INSIGHTS_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    ins, _ = compute_insights()
    print(json.dumps(ins["features"], indent=2))
