"""Real-Time Sentiment Analysis System - Streamlit app.   Run:  streamlit run app.py"""
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

import storage
from src.inference import load_best_model, predict_texts
from src.preprocess import STOPWORDS, clean_text

st.set_page_config(page_title="Sentiment Analysis System", page_icon="💬", layout="wide")
storage.init_db()

COLORS = {"positive": "#2e9e5b", "neutral": "#8a94a6", "negative": "#d64545"}
ORDER = ["positive", "neutral", "negative"]
SOURCES = {
    "rss": "News headlines (Google News RSS)",
    "reddit": "Reddit posts",
    "all": "News + Reddit",
    "simulated": "Demo data (simulated)",
}
WINDOWS = {"All time": None, "Last 15 minutes": 15, "Last 1 hour": 60, "Last 24 hours": 1440}

# ------------------------------------------------------------------ top bar
st.markdown(
    "<style>"
    "footer{visibility:hidden;}"
    ".topbar{background:linear-gradient(135deg,#0b3d91 0%,#0070C0 60%,#1aa3d9 100%);padding:1rem 1.6rem;"
    "border-radius:12px;margin-bottom:1rem;}"
    ".topbar h1{margin:0;padding:0;font-size:1.8rem;line-height:1.25;color:#ffffff;}"
    "</style>"
    "<div class='topbar'><h1>💬 Real-Time Sentiment Analysis System with Web Scraping</h1></div>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_best_model()


try:
    model, info = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

import realtime_pipeline  # noqa: E402  (after the model check so errors show nicely)

# ------------------------------------------------------------------ sidebar: no default values
st.sidebar.title("Get started")
st.sidebar.caption("Choose what to analyse, where to collect posts from, and how many to fetch.")
st.sidebar.markdown(f"Active model: **{info['title']}** (macro-F1 {info['macro_f1']:.3f})")

keyword = st.sidebar.text_input("Topic / keyword / hashtag", value="", placeholder="For example: electric cars")
keyword = keyword.strip().lstrip("#").lower()
source = st.sidebar.selectbox(
    "Data source", list(SOURCES), index=None, format_func=SOURCES.get, placeholder="Choose a source",
    help="News and Reddit collect real posts and need internet. Demo data is simulated and for demonstration only.")
limit = st.sidebar.number_input("Posts per collection", min_value=5, max_value=50, value=None, step=1,
                                placeholder="Choose 5 to 50 posts",
                                help="Each fetch asks the source for its latest posts. Posts already stored are skipped, "
                                     "so fetching 20 after 15 adds only the posts that are new.")
window_label = st.sidebar.selectbox("Time window", list(WINDOWS), index=None, placeholder="Choose a time window")
live = st.sidebar.toggle("Live mode (auto-collect + refresh)", value=False)
refresh = st.sidebar.slider("Refresh every (seconds)", 5, 120, 15)
threshold = st.sidebar.slider("Negative alert threshold (%)", 10, 90, 40)
burst = st.sidebar.checkbox("Simulate a negative burst", value=False) if source == "simulated" else False

missing = [name for name, val in [("topic", keyword), ("data source", source), ("number of posts", limit),
                                  ("time window", window_label)] if not val]
ready = not missing
minutes = WINDOWS.get(window_label)

if st.sidebar.button("Fetch now", use_container_width=True, type="primary"):
    if not ready:
        st.sidebar.warning("Please fill in: " + ", ".join(missing) + ".")
    else:
        with st.spinner("Collecting and analysing..."):
            res = realtime_pipeline.run_cycle(keyword, source, int(limit), burst)
        if res["fetched"] == 0:
            st.sidebar.warning("This source returned no posts. Check your internet connection, "
                               "or try another topic or source (Reddit sometimes blocks requests).")
        elif res["new"] == 0:
            st.sidebar.info(f"Fetched {res['fetched']} posts, but all were already stored. "
                            "No new posts on this topic yet.")
        else:
            st.sidebar.success(f"Fetched {res['fetched']} posts: {res['new']} new analysed, "
                               f"{res['skipped']} already stored.")

tab_live, tab_try = st.tabs(["📡 Live Monitor", "✍️ Try a Text"])

# ------------------------------------------------------------------ live monitor
with tab_live:
    @st.fragment(run_every=refresh if (live and ready) else None)
    def live_panel():
        if not ready:
            st.info("To get started, enter a topic, choose a data source, set the number of posts, and select a "
                    "time window. Then click **Fetch now** to collect and analyse posts.")
            return
        if live:
            realtime_pipeline.run_cycle(keyword, source, int(limit), burst)
        df = storage.load_df(keyword, minutes)
        if df.empty:
            st.info("No posts stored for this topic and time window yet. Click **Fetch now** "
                    "(or switch on Live mode) to collect some.")
            return
        total = len(df)
        pct = (df["sentiment"].value_counts(normalize=True) * 100).reindex(ORDER).fillna(0)

        recent = storage.load_df(keyword, 10)
        if len(recent) >= 10:
            neg_recent = (recent["sentiment"] == "negative").mean() * 100
            if neg_recent >= threshold:
                st.error(f"🚨 ALERT: {neg_recent:.0f}% of the last {len(recent)} posts (10 min) are NEGATIVE "
                         f"(threshold {threshold}%)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Posts analysed", total)
        c2.metric("Positive", f"{pct['positive']:.1f}%")
        c3.metric("Neutral", f"{pct['neutral']:.1f}%")
        c4.metric("Negative", f"{pct['negative']:.1f}%")

        left, right = st.columns([1, 2])
        pie = px.pie(df, names="sentiment", hole=0.45, color="sentiment", color_discrete_map=COLORS,
                     category_orders={"sentiment": ORDER})
        pie.update_layout(title="Sentiment distribution", margin=dict(t=50, b=10))
        left.plotly_chart(pie, use_container_width=True)

        span = df["ts"].max() - df["ts"].min()
        freq = "1min" if span < pd.Timedelta(hours=2) else "1h"
        trend = (df.set_index("ts").groupby([pd.Grouper(freq=freq), "sentiment"]).size()
                 .unstack(fill_value=0).reindex(columns=ORDER, fill_value=0))
        fig = px.line(trend, color_discrete_map=COLORS, markers=True)
        fig.update_layout(title=f"Sentiment trend (posts per {freq})", xaxis_title="", yaxis_title="posts",
                          legend_title="", margin=dict(t=50, b=10))
        right.plotly_chart(fig, use_container_width=True)

        k1, k2 = st.columns(2)
        kw_stop = set(clean_text(keyword).split())
        words = Counter(w for t in df["text"] for w in clean_text(t).split() if w not in STOPWORDS and w not in kw_stop)
        kdf = pd.DataFrame(words.most_common(12), columns=["keyword", "count"])
        k1.plotly_chart(px.bar(kdf, x="count", y="keyword", orientation="h", title="Top keywords")
                        .update_yaxes(autorange="reversed"), use_container_width=True)
        vol = df.set_index("ts").resample(freq).size().rename("posts").reset_index()
        k2.plotly_chart(px.bar(vol, x="ts", y="posts", title="Post volume"), use_container_width=True)

        st.subheader("Predictions")
        cols = ["ts", "source", "text", "sentiment", "confidence"]
        shown = min(total, 500)
        show = df.head(shown)[cols].copy()
        show["confidence"] = (show["confidence"] * 100).round(1).astype(str) + "%"
        st.caption(f"Showing the latest {shown} of {total} posts (newest first). Scroll inside the table to see more.")
        st.dataframe(show, use_container_width=True, hide_index=True, height=420)
        st.download_button("Download all posts (CSV)", df[cols].to_csv(index=False).encode("utf-8"),
                           file_name=f"{keyword.replace(' ', '_')}_sentiment.csv", mime="text/csv", key="download_posts")

    live_panel()

# ------------------------------------------------------------------ single text
with tab_try:
    txt = st.text_area("Type or paste a post, review or headline", value="",
                       placeholder="For example: The battery life is not good at all, very disappointed")
    if st.button("Analyse sentiment"):
        if not txt.strip():
            st.warning("Please type or paste some text first.")
        else:
            label, conf, probs = predict_texts(model, [txt])[0]
            st.markdown(f"### Prediction: **{label.upper()}**  ({conf * 100:.1f}% confidence)")
            pdf = pd.DataFrame({"sentiment": list(probs), "probability": list(probs.values())})
            st.plotly_chart(px.bar(pdf, x="sentiment", y="probability", color="sentiment", color_discrete_map=COLORS,
                                   range_y=[0, 1]), use_container_width=True)