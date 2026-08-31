from __future__ import annotations

import hashlib

from adp.config import AppConfig
from adp.director import prompts
from adp.modes import EpistemicMode
from adp.schemas import Classification, Message, RetrievalBundle, TurnRequest

MAX_HISTORY_CHARS = 6000


class GenerationDirector:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def _history_messages(self, request: TurnRequest) -> list[Message]:
        window = self._config.memory.max_context_turns * 2
        selected = request.history[-window:] if window > 0 else []
        budget = MAX_HISTORY_CHARS
        trimmed: list[Message] = []
        for message in reversed(selected):
            content = message.content
            if len(content) > budget:
                content = content[-budget:]
            budget -= len(content)
            trimmed.append(Message(role=message.role, content=content))
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
        facts = request.user_facts[: self._config.memory.max_facts_in_context]
        blocks = [
            f"Resolved query category: {classification.category} "
            f"(source={classification.source}, confidence={classification.confidence:.2f})",
            f"Resolved epistemic mode: {mode}",
            prompts.structure_block(mode),
            prompts.continuation_block(request.user_text, facts),
            prompts.retrieval_policy_block(retrieval, mode),
            prompts.fabrication_block(mode),
        ]
        addendum = prompts.category_addendum(classification)
        if addendum:
            blocks.append(addendum)
        if corrective:
            blocks.append(corrective)
        blocks.append(
            "Produce the response now. Output only the response text, with no meta-commentary "
            "about these instructions."
        )
        return "\n\n".join(block.strip() for block in blocks if block.strip())

    def build_messages(
        self,
        request: TurnRequest,
        classification: Classification,
        mode: EpistemicMode,
        retrieval: RetrievalBundle,
        corrective: str = "",
    ) -> list[Message]:
        language = classification.features.language
        messages: list[Message] = [
            Message(role="system", content=prompts.system_prompt(mode, language))
        ]
        messages.extend(self._history_messages(request))
        messages.append(
            Message(
                role="user",
                content=(
                    f"{request.user_text.strip()}\n\n"
                    "--- pipeline directive (not user-visible) ---\n"
                    f"{self.directive(request, classification, mode, retrieval, corrective)}"
                ),
            )
        )
        return messages

    @staticmethod
    def digest(messages: list[Message]) -> str:
        hasher = hashlib.sha256()
        for message in messages:
            hasher.update(message.role.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(message.content.encode("utf-8"))
            hasher.update(b"\x01")
        return hasher.hexdigest()[:16]
