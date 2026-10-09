"""Stages 1-4 as a worker: collect -> skip posts already stored -> preprocess + predict -> store.
Run:  python realtime_pipeline.py --keyword "iPhone" --source rss --interval 60
      python realtime_pipeline.py --keyword "iPhone" --source simulated --once"""
import argparse
import time

import collector
import storage
from src.inference import load_best_model, predict_texts

_cache = {}


def get_model():
    if "m" not in _cache:
        _cache["m"] = load_best_model()
    return _cache["m"]


def run_cycle(keyword, source="simulated", limit=20, burst=False):
    """One collection round. Returns {"fetched": n, "new": n, "skipped": n}.
    Posts already stored for this keyword are skipped BEFORE analysis, so nothing is analysed twice."""
    storage.init_db()
    model, info = get_model()
    if source == "simulated":
        items = collector.collect_simulated(keyword, limit, negative_burst=burst)
    else:
        items = collector.collect(source, keyword, limit)
    unique = list({i["uid"]: i for i in items}.values())          # drop repeats inside this batch
    known = storage.existing_uids(keyword, [i["uid"] for i in unique])
    fresh = [i for i in unique if i["uid"] not in known]
    result = {"fetched": len(items), "new": 0, "skipped": len(items) - len(fresh)}
    if fresh:
        preds = predict_texts(model, [i["text"] for i in fresh])
        rows = [{**i, "keyword": keyword, "sentiment": lab, "confidence": conf, "model": info["model"]}
                for i, (lab, conf, _) in zip(fresh, preds)]
        result["new"] = storage.insert_rows(rows)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--source", default="rss", choices=["rss", "reddit", "all", "simulated"])
    ap.add_argument("--interval", type=int, default=60, help="seconds between collection rounds")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    kw = a.keyword.strip().lstrip("#").lower()
    while True:
        r = run_cycle(kw, a.source, a.limit)
        print(f"[{time.strftime('%H:%M:%S')}] '{kw}': fetched {r['fetched']}, new {r['new']}, already stored {r['skipped']}")
        if a.once:
            break
        time.sleep(a.interval)
