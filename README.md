# 💬 Social Media Sentiment Analyzer

A full-stack Big Data sentiment analysis system that collects social media data,
processes it with Apache Spark, applies VADER + TextBlob NLP, and presents
insights in an interactive Streamlit dashboard.

---

## 🏗️ Architecture

```
[Twitter / Instagram / Facebook APIs]
            ↓  (Simulated by data_generator.py)
    [raw_social_data.csv]
            ↓
    [HDFS Storage]  ← hdfs_manager.py (local fallback if HDFS offline)
            ↓
    [Apache Spark / Pandas Pipeline]  ← spark_processor.py
            ↓
    [VADER + TextBlob NLP Engine]  ← sentiment_engine.py
            ↓
    [sentiment_output.csv]
            ↓
    [Streamlit Dashboard]  ← app.py
```

---

## 📁 Project Structure

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit dashboard |
| `sentiment_engine.py` | VADER + TextBlob ensemble NLP engine |
| `data_generator.py` | Synthetic social media data generator |
| `spark_processor.py` | PySpark / pandas processing pipeline |
| `hdfs_manager.py` | HDFS operations with local fallback |
| `requirements.txt` | Python dependencies |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

### 2. Run the dashboard (auto-generates data inside the app)
```bash
streamlit run app.py
```

### OR — manual pipeline (optional)
```bash
# Step 1: Generate data
python data_generator.py --records 2000 --topic iPhone

# Step 2: Process with sentiment analysis
python spark_processor.py --input raw_social_data.csv --output sentiment_output.csv

# Step 3: Launch dashboard
streamlit run app.py
```

---

## 🎛️ Dashboard Features

| Section | Description |
|---|---|
| **Pipeline Controls** | Enter topic keyword → click Analyze → full pipeline runs |
| **Architecture Banner** | Live pipeline step tracker |
| **Metrics** | Total, % Positive / Negative / Neutral, filtered count |
| **Distribution Charts** | Pie, Bar, Platform breakdown |
| **NLP Scatter** | VADER compound vs TextBlob polarity per post |
| **Word Cloud** | Per-sentiment word cloud |
| **Trend Analysis** | Sentiment over time (if timestamp present) |
| **Keyword Frequency** | Top 10 overall + positive + negative words |
| **Data Table** | Color-coded, filterable, downloadable |
| **HDFS Status** | Live storage mode indicator |

---

## ⚙️ Configuration

| Setting | Default | Notes |
|---|---|---|
| HDFS URL | `http://localhost:9870` | Changed in `hdfs_manager.py` |
| HDFS user | `hadoop` | Changed in `hdfs_manager.py` |
| Raw CSV | `raw_social_data.csv` | Auto-generated |
| Output CSV | `sentiment_output.csv` | Used by dashboard |

> **Note:** HDFS is optional. The system automatically falls back to local
> filesystem storage when HDFS is not running.

---

## 🔗 Technologies Used

- **Python 3.10+**
- **Streamlit** — Interactive dashboard
- **VADER Sentiment** — Rule-based social media NLP
- **TextBlob** — Statistical NLP with subjectivity scoring
- **Apache Spark (PySpark)** — Large-scale distributed processing
- **HDFS** — Distributed file storage (local fallback included)
- **Plotly** — Interactive charts
- **WordCloud** — Word frequency visualization
- **Faker** — Realistic synthetic data generation
