"""Stage 5 - Streamlit dashboard.  Run:  streamlit run app.py"""
import json
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

import config
import storage
from src.common import load_best_model, predict_texts
from src.preprocess import STOPWORDS, clean_text

st.set_page_config(page_title="Real-Time Sentiment Monitor", page_icon="💬", layout="wide")
storage.init_db()
COLORS = {"positive": "#2e9e5b", "neutral": "#8a94a6", "negative": "#d64545"}
ORDER = ["positive", "neutral", "negative"]


@st.cache_resource
def get_model():
    return load_best_model()


try:
    model, info = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.code("python make_sample_data.py   # or put your CSV in data/sentiment.csv\n"
            "python train_naive_bayes.py\npython train_logistic_regression.py\npython train_svm.py\n"
            "python select_best.py")
    st.stop()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("💬 Sentiment Monitor")
st.sidebar.caption(f"Active model: **{info['title']}** (macro-F1 {info['macro_f1']:.3f})")
keyword = st.sidebar.text_input("Topic / keyword / hashtag", "iPhone")
source = st.sidebar.selectbox("Data source", ["simulated", "rss", "reddit", "all"],
                              help="'simulated' works offline for demos. rss = Google News headlines.")
limit = st.sidebar.slider("Posts per collection", 5, 50, 20)
window = st.sidebar.selectbox("Time window", ["All", "Last 15 min", "Last 1 hour", "Last 24 hours"])
live = st.sidebar.toggle("Live mode (auto-collect + refresh)", value=False)
refresh = st.sidebar.slider("Refresh every (seconds)", 5, 120, 15)
threshold = st.sidebar.slider("Negative alert threshold (%)", 10, 90, 40)
burst = st.sidebar.checkbox("Simulate a negative burst (demo)", value=False) if source == "simulated" else False

import realtime_pipeline  # noqa: E402  (after model check so errors show nicely)

if st.sidebar.button("Fetch now", use_container_width=True):
    with st.spinner("Collecting and analysing..."):
        n = realtime_pipeline.run_cycle(keyword, source, limit, burst)
    st.sidebar.success(f"{n} new posts analysed")
if st.sidebar.button("Clear stored data for this keyword", use_container_width=True):
    storage.clear(keyword)
    st.sidebar.info("Cleared.")

minutes = {"All": None, "Last 15 min": 15, "Last 1 hour": 60, "Last 24 hours": 1440}[window]

tab_live, tab_try, tab_models = st.tabs(["📡 Live Monitor", "✍️ Try a Text", "🏆 Model Comparison"])

# ------------------------------------------------------------------ live monitor
with tab_live:
    @st.fragment(run_every=refresh if live else None)
    def live_panel():
        if live:
            realtime_pipeline.run_cycle(keyword, source, limit, burst)
        df = storage.load_df(keyword, minutes)
        if df.empty:
            st.info("No data yet - press **Fetch now** in the sidebar (or turn on Live mode).")
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

        st.subheader("Recent predictions")
        show = df.head(25)[["ts", "source", "text", "sentiment", "confidence"]].copy()
        show["confidence"] = (show["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)

    live_panel()

# ------------------------------------------------------------------ single text
with tab_try:
    txt = st.text_area("Type or paste a post, review or headline", "The battery life is not good at all, very disappointed")
    if st.button("Analyse sentiment"):
        label, conf, probs = predict_texts(model, [txt])[0]
        st.markdown(f"### Prediction: **{label.upper()}**  ({conf * 100:.1f}% confidence)")
        pdf = pd.DataFrame({"sentiment": list(probs), "probability": list(probs.values())})
        st.plotly_chart(px.bar(pdf, x="sentiment", y="probability", color="sentiment", color_discrete_map=COLORS,
                               range_y=[0, 1]), use_container_width=True)
        st.caption(f"Cleaned text fed to the model: `{clean_text(txt)}`")

# ------------------------------------------------------------------ model comparison
with tab_models:
    comp = config.RESULTS_DIR / "comparison.csv"
    if not comp.exists():
        st.info("Run `python select_best.py` to generate the comparison.")
    else:
        cdf = pd.read_csv(comp)
        st.success(f"Best model: **{cdf.loc[0, 'title']}**  - selected on macro-F1, then negative-class F1, then latency")
        st.dataframe(cdf.drop(columns=["model"]).round(4), use_container_width=True, hide_index=True)
        png = config.RESULTS_DIR / "comparison.png"
        if png.exists():
            st.image(str(png))
        st.subheader("Per-model results")
        cols = st.columns(2)
        for i, row in cdf.iterrows():
            with cols[i % 2]:
                st.markdown(f"**{row['title']}**")
                cm_png = config.RESULTS_DIR / f"{row['model']}_confusion.png"
                if cm_png.exists():
                    st.image(str(cm_png))
                rj = config.RESULTS_DIR / f"{row['model']}.json"
                if rj.exists():
                    pc = json.loads(rj.read_text())["per_class"]
                    st.dataframe(pd.DataFrame(pc).T.round(3), use_container_width=True)
