"""
data_generator.py
=================
Fetches REAL posts from Reddit (PRAW) and/or YouTube (Data API v3).
Data is written directly to HDFS — no CSV files on disk.

Setup (both free):
  Reddit  → https://www.reddit.com/prefs/apps         (create "script" app)
  YouTube → https://console.cloud.google.com          (enable YouTube Data API v3)
"""

import io
import csv
import sys
from datetime import datetime, timezone

# ─── Windows UTF-8 console fix ───────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Optional: PRAW (Reddit) ─────────────────────────────────────────────────
try:
    import praw
    PRAW_OK = True
except ImportError:
    PRAW_OK = False

# ─── Optional: Google API (YouTube) ──────────────────────────────────────────
try:
    from googleapiclient.discovery import build
    YOUTUBE_OK = True
except ImportError:
    YOUTUBE_OK = False

PLATFORMS = ["Reddit", "YouTube"]


# ═══════════════════════════════════════════════════════════════════════════════
# REDDIT FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_reddit_posts(
    topic: str,
    client_id: str,
    client_secret: str,
    user_agent: str = "SentimentAnalyzer/1.0",
    limit: int = 200,
) -> list[dict]:
    """Fetch Reddit posts about *topic* using PRAW (free API)."""
    if not PRAW_OK:
        raise ImportError("Run: pip install praw")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )

    records = []
    try:
        for i, submission in enumerate(
            reddit.subreddit("all").search(topic, limit=limit, sort="new"), start=1
        ):
            text = submission.selftext.strip() or submission.title
            if text in ("[removed]", "[deleted]", ""):
                text = submission.title

            records.append({
                "id":           i,
                "platform":     "Reddit",
                "source":       f"r/{submission.subreddit}",
                "username":     str(submission.author) if submission.author else "deleted",
                "text":         text,
                "title":        submission.title,
                "score":        submission.score,
                "num_comments": submission.num_comments,
                "timestamp":    datetime.utcfromtimestamp(
                                    submission.created_utc
                                ).strftime("%Y-%m-%d %H:%M:%S"),
                "topic":        topic,
            })
    except Exception as e:
        print(f"⚠️  Reddit fetch error: {e}")

    print(f"✅ Reddit: fetched {len(records):,} posts about '{topic}'")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# YOUTUBE FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_youtube_comments(
    topic: str,
    api_key: str,
    max_videos: int = 10,
    comments_per_video: int = 50,
) -> list[dict]:
    """
    Search YouTube for *topic*, then fetch comments from top videos.
    max_videos * comments_per_video ≈ total records.
    Free quota: 10,000 units/day (each search = 100 units, comments = 1 unit).
    """
    if not YOUTUBE_OK:
        raise ImportError("Run: pip install google-api-python-client")

    youtube = build("youtube", "v3", developerKey=api_key)
    records = []
    record_id = 1

    # ── Search for videos ────────────────────────────────────────────────────
    try:
        search_resp = youtube.search().list(
            q=topic,
            part="id,snippet",
            type="video",
            maxResults=min(max_videos, 50),
            order="relevance",
        ).execute()
    except Exception as e:
        print(f"⚠️  YouTube search error: {e}")
        return records

    video_ids = [item["id"]["videoId"] for item in search_resp.get("items", [])]
    print(f"🎥 YouTube: found {len(video_ids)} videos for '{topic}'")

    # ── Fetch comments from each video (with pagination) ──────────────────────
    for vid_id in video_ids:
        fetched = 0
        next_page = None
        try:
            while fetched < comments_per_video:
                batch = min(100, comments_per_video - fetched)
                kwargs = dict(
                    part="snippet",
                    videoId=vid_id,
                    maxResults=batch,
                    textFormat="plainText",
                    order="relevance",
                )
                if next_page:
                    kwargs["pageToken"] = next_page

                comment_resp = youtube.commentThreads().list(**kwargs).execute()

                for item in comment_resp.get("items", []):
                    c = item["snippet"]["topLevelComment"]["snippet"]
                    published = c.get("publishedAt", "")
                    try:
                        ts = datetime.fromisoformat(
                            published.replace("Z", "+00:00")
                        ).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        ts = published

                    records.append({
                        "id":           record_id,
                        "platform":     "YouTube",
                        "source":       f"youtube.com/watch?v={vid_id}",
                        "username":     c.get("authorDisplayName", "unknown"),
                        "text":         c.get("textDisplay", ""),
                        "title":        "",
                        "score":        c.get("likeCount", 0),
                        "num_comments": 0,
                        "timestamp":    ts,
                        "topic":        topic,
                    })
                    record_id += 1
                    fetched += 1

                next_page = comment_resp.get("nextPageToken")
                if not next_page:
                    break   # no more pages
        except Exception as e:
            print(f"⚠️  YouTube comment fetch error (video {vid_id}): {e}")
            continue

    print(f"✅ YouTube: fetched {len(records):,} comments about '{topic}'")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_all_sources(
    topic: str,
    reddit_id: str = "",
    reddit_secret: str = "",
    youtube_key: str = "",
    reddit_limit: int = 200,
    youtube_max_videos: int = 10,
    youtube_comments_per_video: int = 50,
) -> list[dict]:
    """
    Fetch from Reddit and/or YouTube based on which credentials are provided.
    Returns combined list of records with unified schema.
    """
    all_records = []

    if reddit_id.strip() and reddit_secret.strip():
        try:
            reddit_records = fetch_reddit_posts(
                topic=topic,
                client_id=reddit_id,
                client_secret=reddit_secret,
                limit=reddit_limit,
            )
            all_records.extend(reddit_records)
        except Exception as e:
            print(f"⚠️  Reddit skipped: {e}")

    if youtube_key.strip():
        try:
            yt_records = fetch_youtube_comments(
                topic=topic,
                api_key=youtube_key,
                max_videos=youtube_max_videos,
                comments_per_video=youtube_comments_per_video,
            )
            # Re-number IDs to avoid collision
            offset = len(all_records)
            for r in yt_records:
                r["id"] = offset + r["id"]
            all_records.extend(yt_records)
        except Exception as e:
            print(f"⚠️  YouTube skipped: {e}")

    print(f"📦 Total: {len(all_records):,} records from all sources")
    return all_records


# ═══════════════════════════════════════════════════════════════════════════════
# HDFS I/O (no CSV on disk)
# ═══════════════════════════════════════════════════════════════════════════════

def records_to_bytes(records: list[dict]) -> bytes:
    """Serialize records to CSV bytes in memory (no file created)."""
    if not records:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue().encode("utf-8")


def save_to_hdfs(records: list[dict], hdfs_path: str, hm) -> bool:
    """Write records directly to HDFS — no CSV file on disk."""
    if not records:
        print("No records to save.")
        return False
    data = records_to_bytes(records)
    ok = hm.write_bytes(data, hdfs_path)
    if ok:
        print(f"✅ Saved {len(records):,} records → HDFS:{hdfs_path}")
    return ok
