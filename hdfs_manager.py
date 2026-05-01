"""
hdfs_manager.py
===============
HDFS client wrapper — connects exclusively to the real Apache Hadoop HDFS.
Start HDFS with:  start-dfs.cmd
Web UI:           http://localhost:9870/dfshealth.html#tab-overview

No local filesystem fallback. No CSV files on disk.
All data is read from / written to HDFS in memory.
"""

import io
import os
import sys
import logging

import pandas as pd

# ─── Windows UTF-8 console fix ───────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

# ─── HDFS client (required) ───────────────────────────────────────────────────
try:
    from hdfs import InsecureClient
except ImportError:
    raise ImportError(
        "The 'hdfs' package is not installed.\n"
        "Install it with:  pip install hdfs"
    )


class HDFSManager:
    """
    All reads and writes go directly to Apache HDFS.

    Start HDFS first:
        start-dfs.cmd
    Then verify it is running at:
        http://localhost:9870/dfshealth.html#tab-overview
    """

    HDFS_URL  = "http://localhost:9870"
    HDFS_USER = "User"

    def __init__(self):
        self._client = InsecureClient(self.HDFS_URL, user=self.HDFS_USER)
        # Verify connection — raises immediately if HDFS is not running
        try:
            self._client.status("/", strict=False)
            print(f"✅ Connected to HDFS at {self.HDFS_URL}")
        except Exception as e:
            raise ConnectionError(
                f"❌ Cannot reach HDFS at {self.HDFS_URL}\n"
                f"   Start HDFS with 'start-dfs.cmd' and retry.\n"
                f"   Details: {e}"
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_dir(self, hdfs_path: str) -> None:
        """Create parent directory in HDFS if it does not exist."""
        parent = "/".join(hdfs_path.rstrip("/").split("/")[:-1]) or "/"
        try:
            self._client.makedirs(parent)
        except Exception:
            pass   # already exists

    # ── Raw bytes I/O ─────────────────────────────────────────────────────────

    def write_bytes(self, data: bytes, hdfs_dest: str) -> bool:
        """Write raw bytes directly to *hdfs_dest* in HDFS."""
        try:
            self._ensure_dir(hdfs_dest)
            with self._client.write(hdfs_dest, overwrite=True) as writer:
                writer.write(data)
            print(f"📤 HDFS write: {len(data):,} bytes → {hdfs_dest}")
            return True
        except Exception as e:
            logger.error(f"write_bytes failed ({hdfs_dest}): {e}")
            return False

    def read_bytes(self, hdfs_src: str) -> bytes | None:
        """Read raw bytes from *hdfs_src* in HDFS. Returns None on failure."""
        try:
            with self._client.read(hdfs_src) as reader:
                data = reader.read()
            print(f"📥 HDFS read: {len(data):,} bytes ← {hdfs_src}")
            return data
        except Exception as e:
            logger.error(f"read_bytes failed ({hdfs_src}): {e}")
            return None

    # ── DataFrame I/O (no temp files) ────────────────────────────────────────

    def write_csv(self, df: pd.DataFrame, hdfs_dest: str) -> bool:
        """Serialize *df* to CSV bytes and store directly in HDFS."""
        csv_bytes = df.to_csv(index=False, encoding="utf-8").encode("utf-8")
        return self.write_bytes(csv_bytes, hdfs_dest)

    def read_csv(self, hdfs_src: str, **kwargs) -> pd.DataFrame | None:
        """
        Read a CSV stored in HDFS directly into a DataFrame (no temp file).
        Returns None if the file does not exist or cannot be read.
        """
        data = self.read_bytes(hdfs_src)
        if data is None:
            return None
        try:
            return pd.read_csv(io.BytesIO(data), encoding="utf-8",
                               on_bad_lines="skip", **kwargs)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1",
                               on_bad_lines="skip", **kwargs)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def exists(self, hdfs_path: str) -> bool:
        """Check if *hdfs_path* exists in HDFS."""
        try:
            return self._client.status(hdfs_path, strict=False) is not None
        except Exception:
            return False

    def list_files(self, hdfs_dir: str) -> list:
        """Return list of filenames in *hdfs_dir*."""
        try:
            return self._client.list(hdfs_dir)
        except Exception as e:
            logger.error(f"list failed ({hdfs_dir}): {e}")
            return []

    def delete(self, hdfs_path: str, recursive: bool = False) -> bool:
        """Delete *hdfs_path* from HDFS."""
        try:
            self._client.delete(hdfs_path, recursive=recursive)
            print(f"🗑️  HDFS deleted: {hdfs_path}")
            return True
        except Exception as e:
            logger.error(f"delete failed ({hdfs_path}): {e}")
            return False

    def status(self) -> dict:
        """Return connection status info."""
        return {
            "mode":      "HDFS",
            "endpoint":  self.HDFS_URL,
            "web_ui":    f"{self.HDFS_URL}/dfshealth.html#tab-overview",
            "connected": True,
        }


# ─── CLI Quick-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    hm = HDFSManager()
    print("Status:", hm.status())
    print("Root files:", hm.list_files("/"))
