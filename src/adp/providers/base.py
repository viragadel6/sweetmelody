from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from adp.schemas import Message

SchemaT = TypeVar("SchemaT", bound=BaseModel)

REPAIR_INSTRUCTION = (
    "Your previous output was not valid JSON matching the required schema. "
    "Return only a single JSON object, no prose, no markdown fences, matching exactly "
    "this JSON Schema:\n{schema}\nPrevious invalid output:\n{previous}\nError:\n{error}"
)


class ProviderError(RuntimeError):
    pass


class StructuredOutputError(ProviderError):
    pass


def messages_to_dicts(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


def extract_json_object(text: str) -> str:
    candidate = strip_code_fences(text)
    start = candidate.find("{")
    if start < 0:
        raise StructuredOutputError("no JSON object found in provider output")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : index + 1]
    raise StructuredOutputError("unbalanced JSON object in provider output")


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        raise AssertionError("LLMProvider.generate must be implemented by subclasses")

    @property
    def repair_retries(self) -> int:
        return 2

    def generate_structured(
        self,
        messages: list[Message],
        temperature: float,
        schema: type[SchemaT],
    ) -> SchemaT:
        attempt_messages = list(messages)
        last_error = ""
        last_output = ""
        for attempt in range(self.repair_retries + 1):
            raw = self.generate(attempt_messages, temperature, response_schema=schema)
            last_output = raw
            try:
                payload = json.loads(extract_json_object(raw))
            except (StructuredOutputError, json.JSONDecodeError) as exc:
                last_error = f"json parse error: {exc}"
            else:
                try:
                    return schema.model_validate(payload)
                except ValidationError as exc:
                    last_error = f"schema validation error: {exc}"
            if attempt == self.repair_retries:
                break
            attempt_messages = [
                *messages,
                Message(
                    role="user",
                    content=REPAIR_INSTRUCTION.format(
                        schema=json.dumps(schema.model_json_schema(), ensure_ascii=False),
                        previous=last_output[:4000],
                        error=last_error,
                    ),
                ),
            ]
        raise StructuredOutputError(
            f"structured output failed after {self.repair_retries + 1} attempts: {last_error}"
        )


class ReplayProvider(LLMProvider):
    def __init__(self, script: list[dict[str, Any]], strict: bool = True) -> None:
        self._script = script
        self._strict = strict
        self._cursor = 0
        self._used: list[dict[str, Any]] = []

    @classmethod
    def from_file(cls, path: str | Path, strict: bool = True) -> ReplayProvider:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = data["responses"] if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError("replay script must contain a list of response entries")
        return cls(entries, strict=strict)

    @property
    def remaining(self) -> int:
        return max(0, len(self._script) - self._cursor)

    @property
    def used(self) -> list[dict[str, Any]]:
        return list(self._used)

    def _match_tag(self, entry: dict[str, Any], schema: type[BaseModel] | None) -> bool:
        expected = entry.get("schema")
        if expected is None:
            return schema is None or not self._strict
        return schema is not None and schema.__name__ == expected

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        while self._cursor < len(self._script):
            entry = self._script[self._cursor]
            self._cursor += 1
            if not self._match_tag(entry, response_schema):
                continue
            self._used.append(
                {
                    "entry": entry,
                    "temperature": temperature,
                    "message_count": len(messages),
                }
            )
            content = entry.get("content", "")
            if isinstance(content, dict | list):
                return json.dumps(content, ensure_ascii=False)
            return str(content)
        raise ProviderError("replay script exhausted")
