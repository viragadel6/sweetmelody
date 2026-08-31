from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from adp.config import AppConfig
from adp.memory.models import Discrepancy, FactRow, MemorySnapshot, SessionRow, TurnRow
from adp.providers.base import LLMProvider, ProviderError
from adp.schemas import ExtractedFactList, Message

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_id);
CREATE TABLE IF NOT EXISTS facts (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source_turn_id INTEGER NOT NULL,
    superseded_by INTEGER,
    created_at REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id, superseded_by);
CREATE INDEX IF NOT EXISTS idx_facts_sp ON facts(session_id, subject, predicate);
"""

FACT_SYSTEM = (
    "You extract atomic factual triples from a single user turn. Output exactly one JSON object: "
    '{"facts": [{"subject": str, "predicate": str, "object": str}]}\n'
    "Rules: extract only what the user asserts about themselves, their situation or their "
    "observations. Use short lowercase predicates such as 'lives in', 'works at', 'observed', "
    "'pays', 'received'. Do not invent facts. If the turn asserts nothing, return an empty list."
)


def _normalize_key(value: str) -> str:
    return " ".join(value.casefold().split())


class MemoryStore:
    def __init__(self, config: AppConfig, provider: LLMProvider | None = None) -> None:
        self._config = config
        self._provider = provider
        self._lock = threading.RLock()
        path = Path(config.paths.memory_db)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def ensure_session(self, session_id: str) -> SessionRow:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(session_id, created_at, updated_at, turn_count) "
                "VALUES(?, ?, ?, 0) ON CONFLICT(session_id) DO NOTHING",
                (session_id, now, now),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT session_id, created_at, updated_at, turn_count FROM sessions "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return SessionRow(
            session_id=str(row["session_id"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            turn_count=int(row["turn_count"]),
        )

    def append_turn(
        self, session_id: str, role: str, content: str, mode: str = "", category: str = ""
    ) -> int:
        self.ensure_session(session_id)
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO turns(session_id, created_at, role, content, mode, category) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (session_id, now, role, content, mode, category),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ?, turn_count = turn_count + 1 "
                "WHERE session_id = ?",
                (now, session_id),
            )
            self._conn.commit()
        return int(cursor.lastrowid or 0)

    def history(self, session_id: str, max_turns: int) -> list[Message]:
        limit = max(0, max_turns) * 2
        if limit == 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM turns WHERE session_id = ? "
                "ORDER BY turn_id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [Message(role=str(row["role"]), content=str(row["content"])) for row in reversed(rows)]

    def turns(self, session_id: str) -> list[TurnRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT turn_id, session_id, created_at, role, content, mode, category "
                "FROM turns WHERE session_id = ? ORDER BY turn_id ASC",
                (session_id,),
            ).fetchall()
        return [
            TurnRow(
                turn_id=int(row["turn_id"]),
                session_id=str(row["session_id"]),
                created_at=float(row["created_at"]),
                role=str(row["role"]),
                content=str(row["content"]),
                mode=str(row["mode"]),
                category=str(row["category"]),
            )
            for row in rows
        ]

    def active_facts(self, session_id: str, limit: int | None = None) -> list[FactRow]:
        query = (
            "SELECT fact_id, session_id, subject, predicate, object, source_turn_id, "
            "superseded_by, created_at FROM facts WHERE session_id = ? AND superseded_by IS NULL "
            "ORDER BY fact_id DESC"
        )
        params: tuple[object, ...] = (session_id,)
        if limit is not None:
            query += " LIMIT ?"
            params = (session_id, limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        facts = [
            FactRow(
                fact_id=int(row["fact_id"]),
                session_id=str(row["session_id"]),
                subject=str(row["subject"]),
                predicate=str(row["predicate"]),
                object=str(row["object"]),
                source_turn_id=int(row["source_turn_id"]),
                superseded_by=(
                    int(row["superseded_by"]) if row["superseded_by"] is not None else None
                ),
                created_at=float(row["created_at"]),
            )
            for row in rows
        ]
        facts.reverse()
        return facts

    def all_facts(self, session_id: str) -> list[FactRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT fact_id, session_id, subject, predicate, object, source_turn_id, "
                "superseded_by, created_at FROM facts WHERE session_id = ? ORDER BY fact_id ASC",
                (session_id,),
            ).fetchall()
        return [
            FactRow(
                fact_id=int(row["fact_id"]),
                session_id=str(row["session_id"]),
                subject=str(row["subject"]),
                predicate=str(row["predicate"]),
                object=str(row["object"]),
                source_turn_id=int(row["source_turn_id"]),
                superseded_by=(
                    int(row["superseded_by"]) if row["superseded_by"] is not None else None
                ),
                created_at=float(row["created_at"]),
            )
            for row in rows
        ]

    def supersede(self, old_fact_id: int, new_fact_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE facts SET superseded_by = ? WHERE fact_id = ?",
                (new_fact_id, old_fact_id),
            )
            self._conn.commit()

    def insert_fact(
        self, session_id: str, subject: str, predicate: str, obj: str, source_turn_id: int
    ) -> tuple[int, Discrepancy | None]:
        now = time.time()
        subject_key = _normalize_key(subject)
        predicate_key = _normalize_key(predicate)
        with self._lock:
            existing_rows = self._conn.execute(
                "SELECT fact_id, subject, predicate, object, source_turn_id FROM facts "
                "WHERE session_id = ? AND superseded_by IS NULL",
                (session_id,),
            ).fetchall()
            cursor = self._conn.execute(
                "INSERT INTO facts(session_id, subject, predicate, object, source_turn_id, "
                "superseded_by, created_at) VALUES(?, ?, ?, ?, ?, NULL, ?)",
                (session_id, subject.strip(), predicate.strip(), obj.strip(), source_turn_id, now),
            )
            new_id = int(cursor.lastrowid or 0)
            discrepancy: Discrepancy | None = None
            for row in existing_rows:
                if (
                    _normalize_key(str(row["subject"])) == subject_key
                    and _normalize_key(str(row["predicate"])) == predicate_key
                ):
                    if _normalize_key(str(row["object"])) == _normalize_key(obj):
                        self._conn.execute(
                            "UPDATE facts SET superseded_by = ? WHERE fact_id = ?",
                            (new_id, int(row["fact_id"])),
                        )
                        continue
                    self._conn.execute(
                        "UPDATE facts SET superseded_by = ? WHERE fact_id = ?",
                        (new_id, int(row["fact_id"])),
                    )
                    discrepancy = Discrepancy(
                        subject=subject.strip(),
                        predicate=predicate.strip(),
                        previous_object=str(row["object"]),
                        new_object=obj.strip(),
                        previous_turn_id=int(row["source_turn_id"]),
                        new_turn_id=source_turn_id,
                    )
            self._conn.commit()
        return new_id, discrepancy

    def extract_and_store_facts(
        self, session_id: str, user_text: str, source_turn_id: int
    ) -> tuple[list[FactRow], list[Discrepancy]]:
        if not self._config.memory.extract_facts or self._provider is None:
            return [], []
        messages = [
            Message(role="system", content=FACT_SYSTEM),
            Message(role="user", content=user_text.strip()),
        ]
        try:
            extracted = self._provider.generate_structured(messages, 0.0, ExtractedFactList)
        except ProviderError:
            return [], []
        stored: list[FactRow] = []
        discrepancies: list[Discrepancy] = []
        for fact in extracted.facts:
            if not fact.subject.strip() or not fact.predicate.strip() or not fact.object.strip():
                continue
            fact_id, discrepancy = self.insert_fact(
                session_id, fact.subject, fact.predicate, fact.object, source_turn_id
            )
            stored.append(
                FactRow(
                    fact_id=fact_id,
                    session_id=session_id,
                    subject=fact.subject.strip(),
                    predicate=fact.predicate.strip(),
                    object=fact.object.strip(),
                    source_turn_id=source_turn_id,
                    superseded_by=None,
                    created_at=time.time(),
                )
            )
            if discrepancy is not None:
                discrepancies.append(discrepancy)
        return stored, discrepancies

    def snapshot(self, session_id: str) -> MemorySnapshot:
        return MemorySnapshot(
            session_id=session_id,
            turns=self.turns(session_id),
            facts=self.all_facts(session_id),
            discrepancies=[],
        )
