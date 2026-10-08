"""Algorithm 5 (deep learning): LSTM.  Needs:  pip install tensorflow   (Python 3.9-3.12)
Run: python train_lstm.py [--data ...] [--epochs 8]"""
import argparse
import time

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import config
from src.common import evaluate_and_save, measure_latency, prepare_data


def main():
    import tensorflow as tf
    from tensorflow.keras import layers

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DEFAULT_DATA))
    ap.add_argument("--text-col"); ap.add_argument("--label-col")
    ap.add_argument("--max-rows", type=int); ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(config.RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = prepare_data(args)
    classes = config.LABELS
    to_idx = {c: i for i, c in enumerate(classes)}
    y_tr_i = np.array([to_idx[y] for y in y_tr])

    vectorizer = layers.TextVectorization(max_tokens=20000, output_sequence_length=60)
    vectorizer.adapt(tf.constant(list(X_tr)))

    inp = tf.keras.Input(shape=(), dtype=tf.string)
    x = vectorizer(inp)
    x = layers.Embedding(20000, 64, mask_zero=True)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(len(classes), activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr_i)
    t0 = time.time()
    model.fit(tf.constant(list(X_tr)), y_tr_i, validation_split=0.1, epochs=args.epochs, batch_size=64,
              class_weight=dict(enumerate(weights)), verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    train_seconds = time.time() - t0

    proba = model.predict(tf.constant(list(X_te)), verbose=0)
    y_pred = np.array(classes)[proba.argmax(axis=1)]
    latency = measure_latency(lambda t: model(tf.constant([t]), training=False), X_te)
    model.save(config.MODEL_DIR / "lstm.keras")
    evaluate_and_save("lstm", "LSTM", y_te, y_pred, latency, train_seconds, extra={
        "type": "keras", "model_file": "lstm.keras", "best_params": {"epochs": str(args.epochs)},
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    })


if __name__ == "__main__":
    main()
