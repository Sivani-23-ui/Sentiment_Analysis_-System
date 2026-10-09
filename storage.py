"""Stage 4 - Storage. SQLite (no install needed). To use PostgreSQL/MongoDB later, only this file has to change."""
import sqlite3

import pandas as pd

import config

SCHEMA = """CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL, keyword TEXT NOT NULL, source TEXT, text TEXT, url TEXT,
    sentiment TEXT, confidence REAL, model TEXT, ts TEXT NOT NULL,
    UNIQUE(uid, keyword))"""


def _conn():
    return sqlite3.connect(config.DB_PATH, timeout=30)


def init_db():
    with _conn() as c:
        c.execute(SCHEMA)
        c.execute("CREATE INDEX IF NOT EXISTS idx_kw_ts ON predictions(keyword, ts)")


def insert_rows(rows):
    """rows: list of dicts. Posts already stored (same uid + keyword) are skipped. Returns how many were really added."""
    if not rows:
        return 0
    added = 0
    with _conn() as c:
        for r in rows:
            cur = c.execute(
                "INSERT OR IGNORE INTO predictions (uid,keyword,source,text,url,sentiment,confidence,model,ts) "
                "VALUES (:uid,:keyword,:source,:text,:url,:sentiment,:confidence,:model,:ts)", r)
            added += max(cur.rowcount, 0)
    return added


def existing_uids(keyword, uids):
    """Which of these post ids are already stored for this keyword?"""
    uids = list(uids)
    found = set()
    with _conn() as c:
        for i in range(0, len(uids), 500):
            chunk = uids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            found.update(r[0] for r in c.execute(
                f"SELECT uid FROM predictions WHERE keyword = ? AND uid IN ({marks})", [keyword, *chunk]))
    return found


def load_df(keyword=None, since_minutes=None):
    q, params = "SELECT * FROM predictions WHERE 1=1", []
    if keyword:
        q += " AND keyword = ?"; params.append(keyword)
    if since_minutes:
        cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=since_minutes)).isoformat(timespec="seconds")
        q += " AND ts >= ?"; params.append(cutoff)
    with _conn() as c:
        df = pd.read_sql_query(q + " ORDER BY ts DESC", c, params=params)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def list_keywords():
    with _conn() as c:
        return [r[0] for r in c.execute("SELECT DISTINCT keyword FROM predictions ORDER BY keyword")]


def clear(keyword=None):
    with _conn() as c:
        if keyword:
            c.execute("DELETE FROM predictions WHERE keyword = ?", (keyword,))
        else:
            c.execute("DELETE FROM predictions")
