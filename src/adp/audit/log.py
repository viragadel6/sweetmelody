from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

from adp.config import AppConfig
from adp.schemas import AuditRecord

TraceHook = Callable[[str, dict[str, object]], None]


class AuditLog:
    def __init__(self, config: AppConfig, trace_hook: TraceHook | None = None) -> None:
        self._config = config
        self._dir = Path(config.paths.audit_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._hook = trace_hook

    def set_trace_hook(self, hook: TraceHook | None) -> None:
        self._hook = hook

    def path_for(self, session_id: str) -> Path:
        safe = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_" for char in session_id
        )
        return self._dir / f"{safe or 'session'}.jsonl"

    def trace(self, event: str, payload: dict[str, object]) -> None:
        if not self._config.audit.tracing or self._hook is None:
            return
        self._hook(event, payload)

    def write(self, record: AuditRecord) -> None:
        if not self._config.audit.enabled:
            return
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        path = self.path_for(record.session_id)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        self.trace(
            "audit.write",
            {
                "session_id": record.session_id,
                "turn_id": record.turn_id,
                "mode": str(record.mode),
                "decision": record.final_decision,
            },
        )

    def read(self, session_id: str) -> list[AuditRecord]:
        path = self.path_for(session_id)
        if not path.exists():
            return []
        records: list[AuditRecord] = []
        with self._lock:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(AuditRecord.model_validate_json(stripped))
        return records

    def read_raw(self, session_id: str) -> list[dict[str, object]]:
        path = self.path_for(session_id)
        if not path.exists():
            return []
        payloads: list[dict[str, object]] = []
        with self._lock:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    payloads.append(parsed)
        return payloads
