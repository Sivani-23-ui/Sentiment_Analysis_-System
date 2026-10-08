# Real-Time Sentiment Analysis System (with Web Scraping)

Pipeline: **Collect → Preprocess → TF-IDF + ML model → Store → Streamlit dashboard**

## 1. Setup (VS Code terminal, Python 3.9–3.12)
```bash
python -m venv venv
venv\Scripts\activate          # Windows      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## 2. Get a training dataset
Download a real 3-class dataset and save it as `data/sentiment.csv` (or pass `--data path.csv` to every script).
Recommended: Kaggle **"Twitter US Airline Sentiment"** (`Tweets.csv`, columns `text` + `airline_sentiment`), ~14.6k tweets.
Column names are auto-detected (`text`/`tweet`/`review`..., `sentiment`/`label`/`airline_sentiment`...); otherwise use `--text-col` / `--label-col`.
Numeric labels are read as 0=negative, 1=neutral, 2=positive (4 = positive).

Just testing the code first? `python make_sample_data.py` makes a small synthetic dataset
(its scores are artificially high - never report them).

## 3. Train every algorithm independently
```bash
python train_naive_bayes.py
python train_logistic_regression.py
python train_svm.py
python train_random_forest.py
python train_lstm.py            # optional, needs: pip install tensorflow
```
Each run saves, in `models/` and `results/`: the model, `<name>.json` (accuracy, per-class precision/recall/F1,
macro-F1, latency, training time, best params), and `<name>_confusion.png`. Use `--data file.csv` or `--max-rows 5000`.

## 4. Pick the best model
```bash
python select_best.py
```
Ranks by macro-F1 → negative-class F1 → latency, writes `results/comparison.csv` + `comparison.png`
and saves the winner as the project model (`models/best_model.json`).

## 5. Run the interface
```bash
streamlit run app.py
```
Tabs: **Live Monitor** (distribution, trend, volume, keywords, recent predictions, negative-spike alert),
**Try a Text**, **Model Comparison**. Source `simulated` works offline; `rss` (Google News) and `reddit` need internet.

## 6. Optional: run collection as a separate background process (matches the 5-stage architecture)
```bash
python realtime_pipeline.py --keyword "iPhone" --source rss --interval 60
```
The dashboard reads from the same database, so it keeps working even if you close/reopen it.

## Files
| File | Role |
|---|---|
| `src/preprocess.py` | lowercase, URL/mention removal, tokenize, stopwords (negations kept), lemmatize |
| `src/common.py` | data loading, GridSearch training, metrics, model loading/prediction |
| `train_*.py` | one independent script per algorithm |
| `select_best.py` | comparison table/chart + best-model selection |
| `collector.py` | RSS, Reddit, BeautifulSoup scraper (`collect_web`), simulated source |
| `storage.py` | SQLite storage (swap for PostgreSQL/MongoDB here) |
| `realtime_pipeline.py` | collect → predict → store loop |
| `app.py` | Streamlit dashboard |

Responsible collection: only public data, obey robots.txt / terms of service / rate limits.
