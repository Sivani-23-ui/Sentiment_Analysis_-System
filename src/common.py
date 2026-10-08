"""Shared helpers: data loading, experiment runner (train/evaluate/save), model loading & prediction."""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

import config
from src.preprocess import clean_text

TEXT_COLS = ["text", "tweet", "review", "content", "sentence", "headline", "comment"]
LABEL_COLS = ["label", "sentiment", "airline_sentiment", "target", "polarity", "class"]
LABEL_MAP = {
    "positive": "positive", "pos": "positive", "negative": "negative", "neg": "negative",
    "neutral": "neutral", "neu": "neutral",
    "0": "negative", "1": "neutral", "2": "positive", "4": "positive",
}


# ----------------------------------------------------------------------------- data
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(config.DEFAULT_DATA), help="CSV with a text column and a sentiment column")
    p.add_argument("--text-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--max-rows", type=int, default=None, help="use a random subset (faster experiments)")
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def _find_col(df, preferred, candidates, what):
    if preferred:
        return preferred
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    raise SystemExit(f"Could not find the {what} column. Columns: {list(df.columns)}. Use --text-col / --label-col.")


def load_dataset(path, text_col=None, label_col=None, max_rows=None):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Dataset not found: {path}\nPut your CSV there, or run:  python make_sample_data.py")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    tcol = _find_col(df, text_col, TEXT_COLS, "text")
    lcol = _find_col(df, label_col, LABEL_COLS, "label")
    df = df[[tcol, lcol]].rename(columns={tcol: "text", lcol: "label"}).dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["label"]).drop_duplicates(subset="text")
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=config.RANDOM_STATE)
    return df.reset_index(drop=True)


def prepare_data(args):
    df = load_dataset(args.data, args.text_col, args.label_col, args.max_rows)
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.len() > 0]
    print(f"Dataset: {len(df)} rows | class counts: {df['label'].value_counts().to_dict()}")
    return train_test_split(df["clean"].values, df["label"].values, test_size=args.test_size,
                            stratify=df["label"], random_state=config.RANDOM_STATE)


# ----------------------------------------------------------------------------- evaluation
def measure_latency(predict_one, texts, n=200):
    texts = list(texts)[:n]
    start = time.perf_counter()
    for t in texts:
        predict_one(t)
    return (time.perf_counter() - start) / max(len(texts), 1) * 1000  # ms per prediction


def evaluate_and_save(name, title, y_true, y_pred, latency_ms, train_seconds, extra=None):
    labels = config.LABELS
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    macro_f1 = float(np.mean(f))
    result = {
        "model": name, "title": title,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1,
        "weighted_f1": float(np.average(f, weights=s)),
        "f1_negative": float(f[labels.index("negative")]),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "latency_ms": float(latency_ms), "train_seconds": float(train_seconds),
        "confusion_matrix": cm.tolist(), "labels": labels,
    }
    result.update(extra or {})
    (config.RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{title}\nConfusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"{name}_confusion.png", dpi=130)
    plt.close(fig)

    print(f"\n===== {title} =====")
    print(f"Accuracy {result['accuracy']:.4f} | Macro-F1 {macro_f1:.4f} | Negative-F1 {result['f1_negative']:.4f} "
          f"| Latency {latency_ms:.2f} ms | Train {train_seconds:.1f}s")
    for l in labels:
        c = result["per_class"][l]
        print(f"  {l:9s} P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f} (n={c['support']})")
    print(f"Saved -> results/{name}.json, results/{name}_confusion.png")
    return result


def run_experiment(name, title, clf, param_grid, args):
    """TF-IDF (unigrams+bigrams) + classifier, tuned with GridSearchCV on macro-F1. Fully independent per model."""
    X_tr, X_te, y_tr, y_te = prepare_data(args)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)),
        ("clf", clf),
    ])
    t0 = time.time()
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    train_seconds = time.time() - t0
    model = gs.best_estimator_
    y_pred = model.predict(X_te)
    latency = measure_latency(lambda t: model.predict([t]), X_te)
    joblib.dump(model, config.MODEL_DIR / f"{name}.joblib")
    return evaluate_and_save(name, title, y_te, y_pred, latency, train_seconds, extra={
        "type": "sklearn", "model_file": f"{name}.joblib", "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    })


# ----------------------------------------------------------------------------- loading / predicting
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
        raise FileNotFoundError("No best model yet. Train models and run:  python select_best.py")
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
