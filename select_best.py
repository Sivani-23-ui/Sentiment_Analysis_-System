"""Compares every model that has results/<model>.json, picks the best one and saves it as the project model.
Ranking: Macro-F1 (primary) -> Negative-class F1 -> lower latency.   Run: python select_best.py"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

rows = []
for f in sorted(config.RESULTS_DIR.glob("*.json")):
    r = json.loads(f.read_text())
    if "macro_f1" in r:
        rows.append(r)
if not rows:
    sys.exit("No results found. Run the train_*.py scripts first.")

df = pd.DataFrame([{
    "model": r["model"], "title": r["title"], "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
    "weighted_f1": r["weighted_f1"], "f1_negative": r["f1_negative"],
    "latency_ms": r["latency_ms"], "train_seconds": r["train_seconds"],
} for r in rows])
df["_f1r"], df["_negr"] = df["macro_f1"].round(4), df["f1_negative"].round(4)
df = df.sort_values(["_f1r", "_negr", "latency_ms"], ascending=[False, False, True]).drop(columns=["_f1r", "_negr"])
df = df.reset_index(drop=True)
df.to_csv(config.RESULTS_DIR / "comparison.csv", index=False)

best_name = df.loc[0, "model"]
best = next(r for r in rows if r["model"] == best_name)
config.BEST_MODEL_INFO.write_text(json.dumps({
    "model": best["model"], "title": best["title"], "type": best["type"],
    "model_file": best["model_file"], "labels": best["labels"],
    "macro_f1": best["macro_f1"], "accuracy": best["accuracy"],
}, indent=2))

fig, ax = plt.subplots(figsize=(8, 4.2))
x = range(len(df)); w = 0.27
ax.bar([i - w for i in x], df["accuracy"], w, label="Accuracy")
ax.bar(list(x), df["macro_f1"], w, label="Macro-F1")
ax.bar([i + w for i in x], df["f1_negative"], w, label="Negative-class F1")
ax.set_xticks(list(x), df["title"], rotation=15); ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
ax.set_title("Model comparison (higher is better)")
fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "comparison.png", dpi=130)

pd.set_option("display.width", 160)
print(df.round(4).to_string(index=False))
print(f"\nBEST MODEL: {best['title']}  (Macro-F1 {best['macro_f1']:.4f}) -> saved as the project model")
