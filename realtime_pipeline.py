"""Stages 1-4 as a background worker: collect -> preprocess+predict -> store, repeated on a schedule.
Run:  python realtime_pipeline.py --keyword "iPhone" --source rss --interval 60
      python realtime_pipeline.py --keyword "iPhone" --source simulated --once"""
import argparse
import time

import collector
import storage
from src.common import load_best_model, predict_texts

_cache = {}


def get_model():
    if "m" not in _cache:
        _cache["m"] = load_best_model()
    return _cache["m"]


def run_cycle(keyword, source="simulated", limit=20, burst=False):
    """One collection round. Returns the number of NEW posts analysed and stored."""
    storage.init_db()
    model, info = get_model()
    if source == "simulated":
        items = collector.collect_simulated(keyword, limit, negative_burst=burst)
    else:
        items = collector.collect(source, keyword, limit)
    if not items:
        return 0
    preds = predict_texts(model, [i["text"] for i in items])
    rows = [{**i, "keyword": keyword, "sentiment": lab, "confidence": conf, "model": info["model"]}
            for i, (lab, conf, _) in zip(items, preds)]
    return storage.insert_rows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--source", default="rss", choices=["rss", "reddit", "simulated", "all"])
    ap.add_argument("--interval", type=int, default=60, help="seconds between collection rounds")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    while True:
        n = run_cycle(a.keyword, a.source, a.limit)
        print(f"[{time.strftime('%H:%M:%S')}] stored {n} new posts for '{a.keyword}'")
        if a.once:
            break
        time.sleep(a.interval)
