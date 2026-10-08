"""Stage 1 - Data collection. Every collector returns a list of dicts: {uid, source, text, url, ts}.
Use only public data and respect each site's robots.txt / terms of service / rate limits."""
import datetime as dt
import hashlib
import random
import uuid
from urllib.parse import quote_plus

HEADERS = {"User-Agent": "Mozilla/5.0 (academic sentiment-analysis project)"}


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _item(source, text, url="", uid=None):
    text = " ".join(str(text).split())
    uid = uid or hashlib.md5(f"{source}|{text.lower()}".encode()).hexdigest()
    return {"uid": uid, "source": source, "text": text, "url": url, "ts": _now()}


def collect_rss(keyword, limit=30):
    """Google News RSS search - an API-style source, no key needed. The headline is the analysed text."""
    import feedparser
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={quote_plus(keyword)}&hl=en-US&gl=US&ceid=US:en")
    return [_item("news_rss", e.get("title", "").rsplit(" - ", 1)[0], e.get("link", ""))
            for e in feed.entries[:limit] if e.get("title")]


def collect_reddit(keyword, limit=25):
    """Reddit public search JSON (rate-limited; fine for a small demo)."""
    import requests
    r = requests.get("https://www.reddit.com/search.json", headers=HEADERS, timeout=15,
                     params={"q": keyword, "sort": "new", "limit": limit})
    r.raise_for_status()
    items = []
    for c in r.json()["data"]["children"]:
        d = c["data"]
        text = f"{d.get('title', '')}. {d.get('selftext', '')[:300]}"
        items.append(_item("reddit", text, "https://reddit.com" + d.get("permalink", "")))
    return items


def collect_web(url, css_selector="h2", limit=30, source="web_scrape"):
    """Generic BeautifulSoup scraper: grabs the text of every element matching css_selector (e.g. 'h2', '.review-text').
    For JavaScript-rendered pages use Selenium to get the page HTML, then parse it the same way."""
    import requests
    from bs4 import BeautifulSoup
    html = requests.get(url, headers=HEADERS, timeout=15).text
    soup = BeautifulSoup(html, "html.parser")
    return [_item(source, el.get_text(" "), url) for el in soup.select(css_selector)[:limit]
            if len(el.get_text(strip=True)) > 15]


_SIM = {
    "positive": ["I love the new {k}, it works perfectly!", "{k} is amazing, best purchase this year",
                 "Really impressed with {k}. Fast and reliable", "Great experience with {k} today 😍"],
    "negative": ["{k} is terrible, completely disappointed", "Not happy with {k}. Keeps crashing and support is awful",
                 "Worst experience ever with {k} 😡", "{k} stopped working after the update, waste of money"],
    "neutral": ["{k} announced a new version today", "Reading the latest news about {k}",
                "{k} will be available in stores next week", "Anyone know when {k} gets the next update?"],
}


def collect_simulated(keyword, limit=10, negative_burst=False):
    """Offline demo source: random posts so the dashboard works without internet. Never use it for final results."""
    weights = [0.6, 0.2, 0.2] if negative_burst else [0.25, 0.45, 0.3]
    items = []
    for _ in range(limit):
        label = random.choices(["negative", "positive", "neutral"], weights)[0]
        items.append(_item("simulated", random.choice(_SIM[label]).format(k=keyword), uid=uuid.uuid4().hex))
    return items


def collect(source, keyword, limit=20):
    sources = {"rss": collect_rss, "reddit": collect_reddit, "simulated": collect_simulated}
    names = list(sources) if source == "all" else [source]
    out = []
    for n in names:
        try:
            out.extend(sources[n](keyword, limit))
        except Exception as exc:  # a failing source must never stop the pipeline
            print(f"[collector] {n} failed: {exc}")
    return out
