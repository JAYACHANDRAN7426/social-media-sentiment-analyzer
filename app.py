"""
app.py
======
Social Media Sentiment Analyzer — Full Interactive Streamlit Dashboard
Integrates: Data Generator → HDFS → Spark/Pandas Processor → NLP → Charts

All data lives exclusively in Apache HDFS.
Start HDFS first:  start-dfs.cmd
Web UI:            http://localhost:9870/dfshealth.html#tab-overview

Run:
    streamlit run app.py
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import io, os, sys, time, warnings, subprocess
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Third-Party ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud

# ─── Local Modules ────────────────────────────────────────────────────────────
from sentiment_engine import analyze, clean_text, STOP_WORDS
from data_generator import fetch_all_sources, save_to_hdfs, PLATFORMS
from hdfs_manager import HDFSManager
try:
    import config as _cfg
except ImportError:
    _cfg = None

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# HDFS paths — all data lives exclusively in HDFS
HDFS_RAW       = "/bda/raw/raw_social_data.csv"
HDFS_PROCESSED = "/bda/processed/sentiment_output.csv"


@st.cache_resource(show_spinner=False)
def get_hdfs() -> HDFSManager | None:
    """Connect to HDFS once and cache the client for the session."""
    try:
        return HDFSManager()
    except (ConnectionError, Exception):
        return None

# ── Pre-load credentials from config.py into session state (first run only) ──
if "_config_loaded" not in st.session_state:
    st.session_state["_config_loaded"] = True
    if _cfg:
        if getattr(_cfg, "REDDIT_CLIENT_ID",     ""):
            st.session_state["reddit_id"]     = _cfg.REDDIT_CLIENT_ID
        if getattr(_cfg, "REDDIT_CLIENT_SECRET", ""):
            st.session_state["reddit_secret"] = _cfg.REDDIT_CLIENT_SECRET
        if getattr(_cfg, "YOUTUBE_API_KEY",      ""):
            st.session_state["youtube_key"]   = _cfg.YOUTUBE_API_KEY

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#94a3b8",
}
PLATFORM_COLORS = {
    "Twitter":   "#1d9bf0",
    "Instagram": "#e1306c",
    "Facebook":  "#1877f2",
}

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS (inline — no external .css file)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── metric cards ── */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg,#1e293b,#0f172a);
    border: 1px solid #334155; border-radius:14px;
    padding:18px 22px; box-shadow:0 4px 20px rgba(0,0,0,.35);
  }
  [data-testid="metric-container"] label        { color:#94a3b8 !important; font-size:.85rem !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"]
                                                { font-size:2rem !important; color:#f1f5f9 !important; font-weight:800 !important; }
  [data-testid="metric-container"] [data-testid="stMetricDelta"]
                                                { font-size:.8rem !important; }

  /* ── section headers ── */
  .sec-hdr {
    font-size:1.05rem; font-weight:700; color:#e2e8f0;
    border-left:4px solid #6366f1; padding-left:10px;
    margin:28px 0 12px;
  }

  /* ── pipeline step badges ── */
  .step-badge {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:.78rem; font-weight:700; margin:3px;
  }
  .step-done  { background:#166534; color:#bbf7d0; }
  .step-run   { background:#1e3a8a; color:#bfdbfe; }
  .step-idle  { background:#1e293b; color:#64748b; }

  /* ── sidebar ── */
  section[data-testid="stSidebar"] { background:#0f172a; border-right:1px solid #1e293b; }

  /* ── buttons ── */
  .stButton > button[kind="primary"] {
    background:linear-gradient(90deg,#6366f1,#8b5cf6);
    border:none; border-radius:10px; font-weight:700;
    transition: transform .15s, box-shadow .15s;
  }
  .stButton > button[kind="primary"]:hover {
    transform:translateY(-2px); box-shadow:0 6px 20px rgba(99,102,241,.5);
  }

  /* ── dataframe ── */
  [data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }

  /* ── expander ── */
  .streamlit-expanderHeader { font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def section(icon: str, title: str):
    st.markdown(f'<div class="sec-hdr">{icon} {title}</div>', unsafe_allow_html=True)


def badge(label: str, kind: str = "idle") -> str:
    return f'<span class="step-badge step-{kind}">{label}</span>'


def extract_words(series: pd.Series) -> list:
    words = []
    for text in series.dropna():
        for w in str(text).split():
            if w not in STOP_WORDS and len(w) > 2:
                words.append(w)
    return words


def style_sentiment(val: str) -> str:
    m = {"positive": "color:#22c55e;font-weight:700",
         "negative": "color:#ef4444;font-weight:700",
         "neutral":  "color:#94a3b8;font-weight:700"}
    return m.get(str(val).lower(), "")


@st.cache_data(show_spinner=False)
def cached_process(raw_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", on_bad_lines="skip")
    if "text" not in df.columns:
        return df
    df = df.copy()
    df["cleaned_text"] = df["text"].apply(clean_text)
    if "sentiment" not in df.columns or df["sentiment"].isna().all():
        results = df["text"].apply(lambda t: analyze(str(t) if pd.notna(t) else ""))
        rdf = pd.DataFrame(results.tolist())
        for c in ["sentiment", "vader_score", "textblob_polarity", "textblob_subjectivity"]:
            if c in rdf.columns:
                df[c] = rdf[c].values
        if "final_label" in rdf.columns:
            df["sentiment"] = rdf["final_label"].values
    # normalise labels
    df["sentiment"] = (df["sentiment"].astype(str).str.lower()
                       .replace({"pos": "positive", "neg": "negative", "neu": "neutral",
                                 "1": "positive", "-1": "negative", "0": "neutral"}))
    # timestamp
    for col in ["timestamp", "date", "created_at", "time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if col != "timestamp":
                df.rename(columns={col: "timestamp"}, inplace=True)
            break
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"), margin=dict(t=30, b=20, l=20, r=20),
)

def pie_chart(counts: pd.Series) -> go.Figure:
    labels = counts.index.tolist()
    colors = [SENTIMENT_COLORS.get(l, "#6366f1") for l in labels]
    fig = go.Figure(go.Pie(
        labels=[l.capitalize() for l in labels], values=counts.values,
        marker_colors=colors, hole=0.45,
        textinfo="label+percent", textfont_size=13,
    ))
    fig.update_layout(**_LAYOUT, legend=dict(font=dict(color="#e2e8f0")))
    return fig


def bar_chart(counts: pd.Series) -> go.Figure:
    labels = counts.index.tolist()
    colors = [SENTIMENT_COLORS.get(l, "#6366f1") for l in labels]
    fig = go.Figure(go.Bar(
        x=[l.capitalize() for l in labels], y=counts.values,
        marker_color=colors, text=counts.values, textposition="outside",
    ))
    fig.update_layout(**_LAYOUT,
        xaxis=dict(color="#94a3b8"),
        yaxis=dict(color="#94a3b8", gridcolor="#1e293b"),
    )
    return fig


def platform_bar(df: pd.DataFrame) -> go.Figure:
    if "platform" not in df.columns:
        return None
    grp = df.groupby(["platform", "sentiment"]).size().reset_index(name="count")
    fig = px.bar(grp, x="platform", y="count", color="sentiment",
                 color_discrete_map=SENTIMENT_COLORS, barmode="group",
                 labels={"platform": "Platform", "count": "Posts", "sentiment": "Sentiment"})
    fig.update_layout(**_LAYOUT, xaxis=dict(color="#94a3b8"),
                      yaxis=dict(color="#94a3b8", gridcolor="#1e293b"),
                      legend=dict(font=dict(color="#e2e8f0")))
    return fig


def wordcloud_fig(words: list, sentiment: str) -> plt.Figure | None:
    if not words:
        return None
    base = SENTIMENT_COLORS.get(sentiment, "#6366f1")
    rgb  = mcolors.hex2color(base)
    def cfunc(*a, **k):
        r = int(max(0, min(255, rgb[0]*255 + np.random.randint(-25, 25))))
        g = int(max(0, min(255, rgb[1]*255 + np.random.randint(-25, 25))))
        b = int(max(0, min(255, rgb[2]*255 + np.random.randint(-25, 25))))
        return f"rgb({r},{g},{b})"
    wc = WordCloud(width=760, height=300, background_color=None, mode="RGBA",
                   max_words=120, color_func=cfunc, collocations=False
                   ).generate(" ".join(words))
    fig, ax = plt.subplots(figsize=(7.6, 3)); fig.patch.set_alpha(0)
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def freq_bar(items: list, title: str, color: str) -> go.Figure | None:
    if not items:
        return None
    words, counts = zip(*items)
    fig = go.Figure(go.Bar(
        x=list(counts)[::-1], y=list(words)[::-1], orientation="h",
        marker_color=color, text=list(counts)[::-1], textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title=dict(text=title, font=dict(color="#e2e8f0", size=14)),
        xaxis=dict(color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(color="#94a3b8"),
        height=320,
        margin=dict(t=45, b=20, l=10, r=45),
    )
    return fig


def timeline_chart(df: pd.DataFrame) -> go.Figure | None:
    if "timestamp" not in df.columns:
        return None
    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
    tmp = tmp.dropna(subset=["timestamp"])
    if tmp.empty:
        return None
    tmp["date"] = tmp["timestamp"].dt.date
    trend = tmp.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig = px.line(trend, x="date", y="count", color="sentiment",
                  color_discrete_map=SENTIMENT_COLORS, markers=True,
                  labels={"date": "Date", "count": "Posts", "sentiment": "Sentiment"})
    fig.update_layout(**_LAYOUT,
        xaxis=dict(color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(color="#94a3b8", gridcolor="#1e293b"),
        legend=dict(font=dict(color="#e2e8f0")),
    )
    return fig


def scatter_polarity(df: pd.DataFrame) -> go.Figure | None:
    if "vader_score" not in df.columns or "textblob_polarity" not in df.columns:
        return None
    sample = df.dropna(subset=["vader_score", "textblob_polarity"]).sample(
        min(500, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="vader_score", y="textblob_polarity",
        color="sentiment", color_discrete_map=SENTIMENT_COLORS,
        opacity=0.65, size_max=6,
        labels={"vader_score": "VADER Score", "textblob_polarity": "TextBlob Polarity",
                "sentiment": "Sentiment"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#475569")
    fig.add_hline(y=0, line_dash="dash", line_color="#475569")
    fig.update_layout(**_LAYOUT,
        xaxis=dict(color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(color="#94a3b8", gridcolor="#1e293b"),
        legend=dict(font=dict(color="#e2e8f0")),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATION (HDFS-only — no CSV on disk)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(topic: str, n_records: int, hm: HDFSManager,
                 reddit_id: str, reddit_secret: str, youtube_key: str,
                 yt_videos: int = 20, yt_comments: int = 200) -> pd.DataFrame | None:
    """
    Full pipeline — fetch from Reddit + YouTube → HDFS → NLP → DataFrame
    """
    progress = st.progress(0, text="🔄 Initializing …")
    logs     = st.empty()

    def log(msg: str):
        logs.info(msg)

    # ── Step 1: Fetch from Reddit + YouTube ──────────────────────────────────
    progress.progress(10, text="📡 Fetching from Reddit + YouTube …")
    log(f"Fetching posts about '{topic}' from all connected sources …")
    try:
        records = fetch_all_sources(
            topic=topic,
            reddit_id=reddit_id,
            reddit_secret=reddit_secret,
            youtube_key=youtube_key,
            reddit_limit=n_records,
            youtube_max_videos=yt_videos,
            youtube_comments_per_video=yt_comments,
        )
    except Exception as e:
        st.error(f"❌ API error: {e}")
        progress.empty(); logs.empty()
        return None
    if not records:
        st.error("❌ No posts fetched. Check your keyword or API credentials.")
        progress.empty(); logs.empty()
        return None
    save_to_hdfs(records, HDFS_RAW, hm)
    log(f"✅ Fetched {len(records):,} posts → HDFS:{HDFS_RAW}")

    # ── Step 2: Process (spark_processor reads from / writes to HDFS) ─────────
    progress.progress(40, text="⚙️ Running NLP sentiment analysis …")
    log("Applying VADER + TextBlob ensemble …")

    cmd = [sys.executable, "spark_processor.py",
           "--input", HDFS_RAW, "--output", HDFS_PROCESSED]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=300,
        )
        if result.returncode != 0:
            st.error(f"Processor error:\n{result.stderr[:600]}")
            progress.empty(); logs.empty()
            return None
    except subprocess.TimeoutExpired:
        st.error("Processing timed out after 5 minutes.")
        progress.empty(); logs.empty()
        return None

    # ── Step 3: Read result back from HDFS ────────────────────────────────────
    progress.progress(85, text="📥 Reading results from HDFS …")
    df = hm.read_csv(HDFS_PROCESSED)
    if df is None or df.empty:
        st.error(f"Could not read processed data from HDFS: {HDFS_PROCESSED}")
        progress.empty(); logs.empty()
        return None

    progress.progress(100, text="✅ Pipeline complete!")
    time.sleep(0.4)
    progress.empty(); logs.empty()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:24px 0 4px">
      <span style="font-size:2.8rem;font-weight:800;
        background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        💬 Social Media Sentiment Analyzer
      </span>
      <p style="color:#475569;font-size:1rem;margin-top:6px;">
        Real-time sentiment intelligence across Twitter · Instagram · Facebook
        &nbsp;|&nbsp; Powered by VADER · TextBlob · Emoji NLP · Apache Spark · HDFS
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── HDFS connection (shared across entire function) ───────────────────────
    hm      = get_hdfs()
    hdfs_ok = hm is not None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Pipeline Controls")
        st.divider()

        st.markdown("### 🔍 Keyword & Data Volume")
        topic = st.text_input("Search keyword (e.g., iPhone)",
                              placeholder="iPhone, Tesla, ChatGPT …",
                              value=st.session_state.get("last_topic", "iPhone"),
                              key="topic_input")
        n_records     = st.slider("Reddit posts (max)",       50,  1000, 200,  50)
        yt_videos     = st.slider("YouTube videos to search",  1,    50,  20,   1)
        yt_comments   = st.slider("Comments per video",       50,   500, 200,  50)
        st.caption(f"⚡ Up to **{yt_videos * yt_comments:,}** YouTube comments total")

        st.divider()
        st.markdown("### 🔑 API Credentials")
        st.caption("🟢 Fill at least one source (Reddit or YouTube)")

        st.markdown("**🤖 Reddit** ([Create App](https://www.reddit.com/prefs/apps))")
        reddit_id = st.text_input(
            "Reddit Client ID",
            type="password",
            value=st.session_state.get("reddit_id", ""),
            placeholder="Reddit client_id …",
        )
        reddit_secret = st.text_input(
            "Reddit Client Secret",
            type="password",
            value=st.session_state.get("reddit_secret", ""),
            placeholder="Reddit client_secret …",
        )

        st.markdown("**🎥 YouTube** ([Get API Key](https://console.cloud.google.com))")
        youtube_key = st.text_input(
            "YouTube API Key",
            type="password",
            value=st.session_state.get("youtube_key", ""),
            placeholder="YouTube Data API v3 key …",
        )

        # Save to session state
        if reddit_id:    st.session_state["reddit_id"]     = reddit_id
        if reddit_secret: st.session_state["reddit_secret"] = reddit_secret
        if youtube_key:  st.session_state["youtube_key"]   = youtube_key

        creds_ok = bool(
            (reddit_id.strip() and reddit_secret.strip()) or youtube_key.strip()
        )
        if not creds_ok:
            st.warning("⚠️ Enter Reddit and/or YouTube credentials.")

        analyze_btn = st.button("🔍 Analyze Social Media", type="primary", width='stretch')

        st.divider()
        st.markdown("### 📂 Or Load Existing Data")
        uploaded   = st.file_uploader("Upload CSV", type=["csv"],
                                      help="Must have a 'text' column.")
        load_hdfs  = st.checkbox("Load from HDFS",
                                 value=False,
                                 help=f"Read {HDFS_PROCESSED} from HDFS storage")

        st.divider()
        st.markdown("### 🔎 Dashboard Filters")
        sentiment_filter = st.multiselect("Sentiment",
            ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"])
        platform_filter = st.multiselect("Platform",
            PLATFORMS, default=PLATFORMS)
        kw_refine = st.text_input("Refine keyword", placeholder="optional …")
        n_rows    = st.slider("Rows to display", 5, 500, 20, 5)

        st.divider()
        st.markdown("### ☁️ Word Cloud")
        wc_sent = st.selectbox("Sentiment for word cloud",
                               ["positive", "negative", "neutral", "all"])

        st.divider()
        # HDFS status
        st.markdown("### 🗄️ HDFS Storage Status")
        s  = hm.status() if hdfs_ok else {
            "mode": "⚠️ HDFS Offline",
            "endpoint": "localhost:9870",
            "web_ui": "http://localhost:9870/dfshealth.html#tab-overview",
            "connected": False,
        }
        if not hdfs_ok:
            st.error("❌ HDFS is offline!  Run `start-dfs.cmd` then refresh.")
            st.markdown(
                '<a href="http://localhost:9870/dfshealth.html#tab-overview" target="_blank"'
                ' style="color:#818cf8;font-size:.8rem;">🔗 Open HDFS Web UI</a>',
                unsafe_allow_html=True)
        else:
            mode_color  = "#22c55e"
            raw_exists  = hm.exists(HDFS_RAW)
            proc_exists = hm.exists(HDFS_PROCESSED)
            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:12px;">
              <b style="color:{mode_color};">● HDFS Connected</b><br>
              <a href="http://localhost:9870/dfshealth.html#tab-overview" target="_blank"
                 style="color:#64748b;font-size:.8rem;">🔗 {s['endpoint']}</a><br><br>
              <span style="color:{'#22c55e' if raw_exists else '#ef4444'};font-size:.8rem;">
                {'✅' if raw_exists else '❌'} Raw: {HDFS_RAW}
              </span><br>
              <span style="color:{'#22c55e' if proc_exists else '#ef4444'};font-size:.8rem;">
                {'✅' if proc_exists else '❌'} Processed: {HDFS_PROCESSED}
              </span>
            </div>
            """, unsafe_allow_html=True)

    # ── HDFS gate — block pipeline if HDFS is offline ────────────────────────
    if not hdfs_ok:
        st.error("⛔ HDFS is not running. Start it with `start-dfs.cmd` and refresh this page.")
        st.info("👉 Web UI: http://localhost:9870/dfshealth.html#tab-overview")
        return

    # ── Trigger pipeline ──────────────────────────────────────────────────────
    if analyze_btn and topic.strip() and creds_ok:
        st.session_state["pipeline_done"] = False
        st.session_state["last_topic"]    = topic.strip()
        with st.spinner("Fetching posts and running analysis …"):
            df_result = run_pipeline(
                topic.strip(), n_records, hm,
                reddit_id.strip(), reddit_secret.strip(),
                youtube_key.strip(),
                yt_videos, yt_comments,
            )
        if df_result is not None:
            st.session_state["pipeline_done"] = True
            st.session_state["pipeline_df"]   = df_result
    elif analyze_btn and not creds_ok:
        st.error("❌ Please enter Reddit and/or YouTube credentials in the sidebar.")
    elif analyze_btn:
        st.warning("Please enter a keyword before running the analysis.")

    # ── Architecture pipeline banner ──────────────────────────────────────────
    section("🏗️", "Architecture Pipeline")
    done = st.session_state.get("pipeline_done", False)
    st.markdown(
        badge("1 · Twitter/Instagram/Facebook", "done" if done else "idle") +
        " → " +
        badge("2 · Data Generator", "done" if done else "idle") +
        " → " +
        badge("3 · HDFS Storage", "done" if done else "idle") +
        " → " +
        badge("4 · Apache Spark / Pandas", "done" if done else "idle") +
        " → " +
        badge("5 · VADER + TextBlob NLP", "done" if done else "idle") +
        " → " +
        badge("6 · Dashboard", "done" if done else "run"),
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load data ─────────────────────────────────────────────────────────────
    df = None

    if uploaded:
        raw_bytes = uploaded.read()
        with st.spinner("Analyzing uploaded data …"):
            df = cached_process(raw_bytes)
        st.success(f"✅ Uploaded file processed — {len(df):,} records")

    elif load_hdfs and hm.exists(HDFS_PROCESSED):
        with st.spinner("Reading from HDFS …"):
            df = hm.read_csv(HDFS_PROCESSED)
        if df is not None and not df.empty:
            if "text" in df.columns and ("sentiment" not in df.columns or df["sentiment"].isna().all()):
                with st.spinner("Running NLP …"):
                    df = cached_process(df.to_csv(index=False).encode("utf-8"))
            st.success(f"✅ Loaded from HDFS:{HDFS_PROCESSED} — {len(df):,} records")
        else:
            st.warning(f"⚠️ Could not load from HDFS: {HDFS_PROCESSED}")
            return

    elif load_hdfs and not hm.exists(HDFS_PROCESSED):
        st.warning(f"⚠️ No data found at HDFS:{HDFS_PROCESSED}. Run the pipeline first.")
        return

    elif done and "pipeline_df" in st.session_state:
        df  = st.session_state["pipeline_df"]
        kw  = st.session_state.get("last_topic", "")
        st.success(f"✅ Pipeline complete — {len(df):,} records analyzed for **'{kw}'**")

    else:
        st.info("👈 Enter a keyword and click **🔍 Analyze Social Media** to begin,\n"
                "or upload a CSV / enable 'Load from HDFS' in the sidebar.")
        return

    if df is None or df.empty:
        st.warning("⚠️ Dataset is empty.")
        return
    if "text" not in df.columns:
        st.error("❌ Dataset must contain a **'text'** column.")
        return

    # normalise sentiment
    df["sentiment"] = (df["sentiment"].astype(str).str.lower()
        .replace({"pos": "positive", "neg": "negative", "neu": "neutral",
                  "1": "positive", "-1": "negative", "0": "neutral"}))

    # ── Apply dashboard filters ────────────────────────────────────────────────
    filtered = df[df["sentiment"].isin(sentiment_filter)].copy()

    if "platform" in filtered.columns and platform_filter:
        filtered = filtered[filtered["platform"].isin(platform_filter)]

    if kw_refine.strip():
        filtered = filtered[
            filtered["text"].str.lower().str.contains(
                kw_refine.strip().lower(), na=False, regex=False)
        ]

    if filtered.empty:
        st.warning("⚠️ No records match the current filters. Try broadening them.")
        return

    # ── Counts ────────────────────────────────────────────────────────────────
    total  = len(df)
    counts = df["sentiment"].value_counts()
    pct    = lambda s: round(counts.get(s, 0) / total * 100, 1)

    # ══════════════════════════════════════════════════════════════════════════
    # A. METRICS
    # ══════════════════════════════════════════════════════════════════════════
    section("📊", "Overview Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Total Records",  f"{total:,}")
    c2.metric("😊 Positive",       f"{pct('positive')}%",  f"{counts.get('positive', 0):,} posts")
    c3.metric("😠 Negative",       f"{pct('negative')}%",  f"{counts.get('negative', 0):,} posts")
    c4.metric("😐 Neutral",        f"{pct('neutral')}%",   f"{counts.get('neutral', 0):,} posts")
    c5.metric("🔍 After Filters",  f"{len(filtered):,}")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # B. DISTRIBUTION CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    section("📈", "Sentiment Distribution")
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander("🥧 Pie Chart", expanded=True):
            st.plotly_chart(pie_chart(counts), width='stretch')

    with col2:
        with st.expander("📊 Bar Chart", expanded=True):
            st.plotly_chart(bar_chart(counts), width='stretch')

    with col3:
        with st.expander("📱 By Platform", expanded=True):
            pfig = platform_bar(filtered)
            if pfig:
                st.plotly_chart(pfig, width='stretch')
            else:
                st.info("No platform column in data.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # C. NLP SCORES — VADER vs TextBlob scatter
    # ══════════════════════════════════════════════════════════════════════════
    sfig = scatter_polarity(filtered)
    if sfig:
        section("🔬", "VADER vs TextBlob Polarity Scatter")
        with st.expander("Score Correlation (sample ≤500)", expanded=True):
            st.plotly_chart(sfig, width='stretch')
        st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # D. WORD CLOUD
    # ══════════════════════════════════════════════════════════════════════════
    section("☁️", "Word Cloud")
    with st.expander("Word Cloud", expanded=True):
        wc_src  = filtered if wc_sent == "all" else filtered[filtered["sentiment"] == wc_sent]
        col_key = "cleaned_text" if "cleaned_text" in wc_src.columns else "text"
        words   = extract_words(wc_src[col_key])
        wfig    = wordcloud_fig(words, wc_sent if wc_sent != "all" else "positive")
        if wfig:
            st.pyplot(wfig, width='stretch')
            plt.close("all")
        else:
            st.info("Not enough text to build a word cloud.")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # E. TREND ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    tfig = timeline_chart(filtered)
    if tfig:
        section("📅", "Sentiment Trend Over Time")
        with st.expander("Trend Analysis", expanded=True):
            st.plotly_chart(tfig, width='stretch')
        st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # F. KEYWORD FREQUENCY
    # ══════════════════════════════════════════════════════════════════════════
    section("🔤", "Keyword Frequency Analysis")
    col_freq = "cleaned_text" if "cleaned_text" in filtered.columns else "text"

    with st.expander("Top 10 Overall Keywords", expanded=True):
        all_words = extract_words(filtered[col_freq])
        fig_all   = freq_bar(Counter(all_words).most_common(10), "Top 10 Keywords", "#818cf8")
        if fig_all:
            st.plotly_chart(fig_all, width='stretch')
        else:
            st.info("No words to display.")

    fp_col, fn_col = st.columns(2)
    with fp_col:
        with st.expander("✅ Top Positive Words", expanded=True):
            pw  = extract_words(filtered.loc[filtered["sentiment"] == "positive", col_freq])
            fpf = freq_bar(Counter(pw).most_common(10), "Positive Keywords", SENTIMENT_COLORS["positive"])
            st.plotly_chart(fpf, width='stretch') if fpf else st.info("No data.")

    with fn_col:
        with st.expander("❌ Top Negative Words", expanded=True):
            nw  = extract_words(filtered.loc[filtered["sentiment"] == "negative", col_freq])
            fnf = freq_bar(Counter(nw).most_common(10), "Negative Keywords", SENTIMENT_COLORS["negative"])
            st.plotly_chart(fnf, width='stretch') if fnf else st.info("No data.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # G. DATA TABLE
    # ══════════════════════════════════════════════════════════════════════════
    section("📋", "Filtered Dataset")
    show_cols = [c for c in ["id", "platform", "username", "text", "sentiment",
                              "vader_score", "textblob_polarity", "likes", "shares", "timestamp"]
                 if c in filtered.columns]
    disp_df = filtered[show_cols].head(n_rows)
    styled  = disp_df.style.applymap(
        style_sentiment,
        subset=["sentiment"] if "sentiment" in disp_df.columns else []
    )
    st.dataframe(styled, width='stretch', height=380)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # H. PIPELINE SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    section("🛠️", "Pipeline Summary")
    ps1, ps2, ps3, ps4 = st.columns(4)
    ps1.metric("Data Source",    "Simulated APIs")
    ps2.metric("Storage",        "Apache HDFS")
    ps3.metric("NLP Engine",     "VADER + TextBlob + Emoji")
    ps4.metric("Processing",     "PySpark / Pandas")
    st.divider()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:18px;color:#475569;font-size:.82rem;">
      Social Media Sentiment Analyzer &nbsp;·&nbsp;
      VADER · TextBlob · Emoji NLP · Apache Spark · HDFS · Streamlit &nbsp;
      <span style="color:#6366f1;">💜</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
