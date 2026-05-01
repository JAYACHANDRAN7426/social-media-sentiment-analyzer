"""
spark_processor.py
==================
PySpark / pandas pipeline that:
  1. Reads raw social media data directly from HDFS (or hdfs_local/ fallback)
  2. Cleans text
  3. Applies VADER + TextBlob sentiment analysis
  4. Writes enriched results back to HDFS — no CSV files in the project root

Falls back to pandas when PySpark is unavailable.

Usage
-----
    python spark_processor.py
    python spark_processor.py --input /bda/raw/raw_social_data.csv \\
                              --output /bda/processed/sentiment_output.csv
    python spark_processor.py --input /bda/raw/raw_social_data.csv --spark
"""

import argparse
import io
import os
import sys
import time
import logging
import tempfile
import shutil

# ─── Windows UTF-8 console fix ───────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Local modules ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentiment_engine import analyze, clean_text
from hdfs_manager import HDFSManager

# ─── PySpark (optional) ───────────────────────────────────────────────────────
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import StringType, FloatType
    SPARK_OK = True
except ImportError:
    SPARK_OK = False
    logger.info("PySpark not available — falling back to pandas.")

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# SPARK PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_spark(hdfs_input: str, hdfs_output: str, hm: HDFSManager) -> None:
    """
    Read from *hdfs_input*, process with PySpark, write result to *hdfs_output*.
    Uses a temporary local file only for Spark I/O (Spark cannot read bytes);
    the temp file is deleted immediately after use.
    """
    print("🔥 Starting PySpark session …")
    spark = (
        SparkSession.builder
        .appName("SocialMediaSentimentAnalysis")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # ── Stage input to a temp file so Spark can read it ──────────────────────
    tmp_dir  = tempfile.mkdtemp(prefix="bda_spark_")
    tmp_in   = os.path.join(tmp_dir, "input.csv")
    tmp_out  = os.path.join(tmp_dir, "output.csv")

    try:
        raw = hm.read_bytes(hdfs_input)
        if raw is None:
            print(f"❌ Could not read from HDFS: {hdfs_input}")
            spark.stop()
            return
        with open(tmp_in, "wb") as f:
            f.write(raw)

        print(f"📂 Reading (via temp): {tmp_in}")
        df = spark.read.csv(tmp_in, header=True, inferSchema=True, multiLine=True, escape='"')

        if "text" not in df.columns:
            print("❌ No 'text' column found in input.")
            spark.stop()
            return

        # ── UDFs ──────────────────────────────────────────────────────────────
        @udf(StringType())
        def udf_clean(text):
            return clean_text(text or "")

        @udf(StringType())
        def udf_sentiment(text):
            return analyze(text or "")["final_label"]

        @udf(FloatType())
        def udf_vader_score(text):
            return float(analyze(text or "")["vader_score"])

        @udf(FloatType())
        def udf_tb_polarity(text):
            return float(analyze(text or "")["textblob_polarity"])

        @udf(FloatType())
        def udf_tb_subjectivity(text):
            return float(analyze(text or "")["textblob_subjectivity"])

        # ── Transform ─────────────────────────────────────────────────────────
        print("⚙️  Running sentiment analysis via Spark UDFs …")
        enriched = (
            df
            .withColumn("cleaned_text",          udf_clean(col("text")))
            .withColumn("sentiment",             udf_sentiment(col("cleaned_text")))
            .withColumn("vader_score",           udf_vader_score(col("cleaned_text")))
            .withColumn("textblob_polarity",     udf_tb_polarity(col("cleaned_text")))
            .withColumn("textblob_subjectivity", udf_tb_subjectivity(col("cleaned_text")))
        )

        # ── Collect to pandas and write to HDFS ───────────────────────────────
        result_df = enriched.toPandas()
        spark.stop()

        ok = hm.write_csv(result_df, hdfs_output)
        if ok:
            print(f"✅ Spark output saved → HDFS:{hdfs_output}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PANDAS PIPELINE (fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pandas(hdfs_input: str, hdfs_output: str, hm: HDFSManager) -> None:
    """Read from HDFS, process with pandas, write result back to HDFS."""
    print("🐼 Using pandas pipeline (PySpark unavailable).")
    print(f"📂 Reading from HDFS: {hdfs_input}")

    df = hm.read_csv(hdfs_input)
    if df is None or df.empty:
        print(f"⚠️ Could not read or empty: {hdfs_input}")
        return

    if "text" not in df.columns:
        print("❌ No 'text' column found.")
        return

    # ── Progress bar (tqdm optional) ──────────────────────────────────────────
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Analyzing sentiments")
        results = df["text"].progress_apply(lambda t: analyze(str(t) if pd.notna(t) else ""))
    except ImportError:
        print("⚙️  Analyzing sentiments …")
        results = df["text"].apply(lambda t: analyze(str(t) if pd.notna(t) else ""))

    result_df = pd.DataFrame(results.tolist())

    # Merge back
    for col_name in ["cleaned_text", "sentiment", "vader_score",
                     "textblob_polarity", "textblob_subjectivity", "final_label"]:
        if col_name in result_df.columns:
            df[col_name] = result_df[col_name].values

    if "final_label" in df.columns:
        df["sentiment"] = df["final_label"]
        df.drop(columns=["final_label"], inplace=True, errors="ignore")

    ok = hm.write_csv(df, hdfs_output)
    if ok:
        print(f"✅ Saved {len(df):,} records → HDFS:{hdfs_output}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Pipeline (HDFS-native)")
    parser.add_argument("--input",  default="/bda/raw/raw_social_data.csv",
                        help="HDFS input path  (default: /bda/raw/raw_social_data.csv)")
    parser.add_argument("--output", default="/bda/processed/sentiment_output.csv",
                        help="HDFS output path (default: /bda/processed/sentiment_output.csv)")
    parser.add_argument("--spark",  action="store_true",
                        help="Force PySpark even if pandas would suffice")
    args = parser.parse_args()

    hm = HDFSManager()

    if not hm.exists(args.input):
        print(f"❌ Input not found in HDFS: {args.input}")
        print("   Run data_generator.py first to create sample data.")
        sys.exit(1)

    t0 = time.time()

    use_spark = SPARK_OK and args.spark
    if use_spark:
        run_spark(args.input, args.output, hm)
    else:
        run_pandas(args.input, args.output, hm)

    elapsed = time.time() - t0
    print(f"⏱  Total processing time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
