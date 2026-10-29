from __future__ import annotations

import hashlib
from collections.abc import Iterable

from adp.config import AppConfig
from adp.director import prompts
from adp.modes import EpistemicMode
from adp.schemas import Classification, Message, RetrievalBundle, TurnRequest

MAX_HISTORY_CHARS = 6000
DIRECTIVE_HEADER = "--- pipeline directive (not user-visible) ---"
FINAL_INSTRUCTION = (
    "Produce the response now. Output only the response text, with no meta-commentary "
    "about these instructions."
)
LENGTH_PREFIX_BYTES = 8


def _as_text(value: object) -> str:
    return str(getattr(value, "value", value))


class GenerationDirector:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def _history_messages(self, request: TurnRequest) -> list[Message]:
        window = max(0, self._config.memory.max_context_turns) * 2
        if window == 0:
            return []
        selected = request.history[-window:]
        budget = MAX_HISTORY_CHARS
        trimmed: list[Message] = []
        for message in reversed(selected):
            content = message.content
            kept = message
            if len(content) > budget:
                content = content[-budget:]
                kept = Message(role=message.role, content=content)
            budget -= len(content)
            trimmed.append(kept)
            if budget <= 0:
                break
        trimmed.reverse()
        return trimmed

    def directive(
        self,
        request: TurnRequest,
        classification: Classification,
        mode: EpistemicMode,
        retrieval: RetrievalBundle,
        corrective: str = "",
    ) -> str:
        limit = max(0, self._config.memory.max_facts_in_context)
        facts = request.user_facts[:limit]
        blocks = [
            f"Resolved query category: {_as_text(classification.category)} "
            f"(source={_as_text(classification.source)}, "
            f"confidence={classification.confidence:.2f})",
            f"Resolved epistemic mode: {_as_text(mode)}",
            prompts.structure_block(mode),
            prompts.continuation_block(request.user_text.strip(), facts),
            prompts.retrieval_policy_block(retrieval, mode),
            prompts.fabrication_block(mode),
            prompts.category_addendum(classification),
            corrective,
            FINAL_INSTRUCTION,
        ]
        stripped = [block.strip() for block in blocks if block]
        return "\n\n".join(block for block in stripped if block)

    def build_messages(
        self,
        request: TurnRequest,
        classification: Classification,
        mode: EpistemicMode,
        retrieval: RetrievalBundle,
        corrective: str = "",
    ) -> list[Message]:
        language = classification.features.language
        user_text = request.user_text.strip()
        directive = self.directive(request, classification, mode, retrieval, corrective)
        parts = [user_text, f"{DIRECTIVE_HEADER}\n{directive}"]
        content = "\n\n".join(part for part in parts if part)
        messages: list[Message] = [
            Message(role="system", content=prompts.system_prompt(mode, language))
        ]
        messages.extend(self._history_messages(request))
        messages.append(Message(role="user", content=content))
        return messages

    @staticmethod
    def digest(messages: Iterable[Message]) -> str:
        hasher = hashlib.sha256()
        for message in messages:
            for field in (_as_text(message.role), message.content):
                data = field.encode("utf-8", "surrogatepass")
                hasher.update(len(data).to_bytes(LENGTH_PREFIX_BYTES, "big"))
                hasher.update(data)
        return hasher.hexdigest()[:16]
