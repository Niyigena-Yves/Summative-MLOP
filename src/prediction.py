"""Prediction: load the trained model and classify a single bean leaf image."""
import os
import threading
import time

import numpy as np

from src.model import MODEL_PATH, load_metadata, load_model
from src.preprocessing import CLASS_NAMES, preprocess_image

ADVICE = {
    "angular_leaf_spot": "Angular leaf spot (fungal). Remove infected leaves, rotate crops, "
                         "use certified seed and consider a copper-based fungicide.",
    "bean_rust": "Bean rust (fungal). Remove affected leaves, avoid overhead irrigation, "
                 "plant resistant varieties and apply a recommended fungicide early.",
    "healthy": "The leaf looks healthy. Keep monitoring the field regularly.",
}


class Predictor:
    """Thread-safe wrapper that reloads the model when a retrained version is saved."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._mtime = None
        self._lock = threading.Lock()
        self.version = None

    def _ensure_loaded(self):
        mtime = os.path.getmtime(self.model_path)
        if self._model is None or mtime != self._mtime:
            with self._lock:
                if self._model is None or mtime != self._mtime:
                    self._model = load_model(self.model_path)
                    self._mtime = mtime
                    self.version = load_metadata().get("version")
        return self._model

    @property
    def loaded(self):
        return self._model is not None

    def predict(self, source):
        model = self._ensure_loaded()
        start = time.perf_counter()
        x = preprocess_image(source)[np.newaxis, ...]
        probs = model(x, training=False).numpy()[0]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        return {
            "label": label,
            "confidence": float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            "advice": ADVICE[label],
            "model_version": self.version,
            "inference_ms": round((time.perf_counter() - start) * 1000, 2),
        }


_default = None


def get_predictor():
    global _default
    if _default is None:
        _default = Predictor()
    return _default


def predict_image(source):
    """Convenience function: predict one image (path, bytes or file object)."""
    return get_predictor().predict(source)


if __name__ == "__main__":
    import sys
    print(predict_image(sys.argv[1]))
