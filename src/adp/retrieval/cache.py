from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS search_cache (
    query TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS page_cache (
    url TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class RetrievalCache:
    def __init__(self, db_path: str | Path, ttl_s: int) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = max(0, int(ttl_s))
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _fresh(self, created_at: float) -> bool:
        if self._ttl == 0:
            return True
        return (time.time() - created_at) <= self._ttl

    def get_search(self, query: str) -> list[dict[str, Any]] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, created_at FROM search_cache WHERE query = ?", (query,)
            ).fetchone()
        if row is None or not self._fresh(float(row["created_at"])):
            return None
        payload = json.loads(row["payload"])
        if not isinstance(payload, list):
            return None
        return [item for item in payload if isinstance(item, dict)]

    def put_search(self, query: str, results: list[dict[str, Any]]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO search_cache(query, payload, created_at) VALUES(?, ?, ?) "
                "ON CONFLICT(query) DO UPDATE SET payload=excluded.payload, "
                "created_at=excluded.created_at",
                (query, json.dumps(results, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def get_page(self, url: str) -> tuple[str, str] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT content, title, created_at FROM page_cache WHERE url = ?", (url,)
            ).fetchone()
        if row is None or not self._fresh(float(row["created_at"])):
            return None
        return str(row["content"]), str(row["title"])

    def put_page(self, url: str, content: str, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO page_cache(url, content, title, created_at) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(url) DO UPDATE SET content=excluded.content, title=excluded.title, "
                "created_at=excluded.created_at",
                (url, content, title, time.time()),
            )
            self._conn.commit()

    def purge_expired(self) -> int:
        if self._ttl == 0:
            return 0
        cutoff = time.time() - self._ttl
        with self._lock:
            cursor = self._conn.execute("DELETE FROM search_cache WHERE created_at < ?", (cutoff,))
            removed = cursor.rowcount or 0
            cursor = self._conn.execute("DELETE FROM page_cache WHERE created_at < ?", (cutoff,))
            removed += cursor.rowcount or 0
            self._conn.commit()
        return removed
