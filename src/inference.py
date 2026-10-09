"""Lightweight prediction code used by the web app (no training libraries needed)."""
import json

import joblib
import numpy as np

import config
from src.preprocess import clean_text


class KerasWrapper:
    """Gives the LSTM the same predict / predict_proba / classes_ interface as the sklearn models."""
    def __init__(self, model, classes):
        self.model, self.classes_ = model, np.array(classes)

    def predict_proba(self, texts):
        import tensorflow as tf
        return self.model(tf.constant(list(texts)), training=False).numpy()

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def load_best_model():
    if not config.BEST_MODEL_INFO.exists():
        raise FileNotFoundError("No trained model found (models/best_model.json is missing). "
                                "Train the models and run: python select_best.py")
    info = json.loads(config.BEST_MODEL_INFO.read_text())
    path = config.MODEL_DIR / info["model_file"]
    if info["type"] == "keras":
        import tensorflow as tf
        return KerasWrapper(tf.keras.models.load_model(path), info["labels"]), info
    return joblib.load(path), info


def predict_texts(model, texts):
    """Raw texts -> list of (label, confidence, {label: prob}). Cleaning is applied here, same as in training."""
    texts = list(texts)
    if not texts:
        return []
    proba = model.predict_proba([clean_text(t) for t in texts])
    classes = list(model.classes_)
    out = []
    for row in proba:
        k = int(np.argmax(row))
        out.append((classes[k], float(row[k]), {c: float(v) for c, v in zip(classes, row)}))
    return out
