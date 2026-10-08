"""Central settings for the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

DEFAULT_DATA = DATA_DIR / "sentiment.csv"      # your training dataset (text + label columns)
DB_PATH = DATA_DIR / "sentiment_results.db"    # SQLite file used as storage
BEST_MODEL_INFO = MODEL_DIR / "best_model.json"

LABELS = ["negative", "neutral", "positive"]   # fixed class order everywhere
RANDOM_STATE = 42
