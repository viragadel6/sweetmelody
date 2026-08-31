from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel

from adp.config import ProviderConfig
from adp.providers.base import LLMProvider, ProviderError
from adp.schemas import Message


class LocalServerProvider(LLMProvider):
    """Adapter for a local model server exposing a completion endpoint.

    Contract:
      POST {base_url}/completion
      body:    {"prompt": str, "temperature": float, "n_predict": int,
                optional "grammar_json_schema": {...}}
      200 body: {"content": str}  or  {"completion": str}  or  {"text": str}
    """

    ROLE_PREFIX = {
        "system": "### System",
        "user": "### User",
        "assistant": "### Assistant",
    }

    def __init__(self, config: ProviderConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=config.timeout_s)

    @property
    def repair_retries(self) -> int:
        return self._config.structured_repair_retries

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LocalServerProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _render_prompt(self, messages: list[Message]) -> str:
        blocks: list[str] = []
        for message in messages:
            prefix = self.ROLE_PREFIX.get(message.role, f"### {message.role.capitalize()}")
            blocks.append(f"{prefix}\n{message.content.strip()}")
        blocks.append("### Assistant\n")
        return "\n\n".join(blocks)

    def _payload(
        self,
        messages: list[Message],
        temperature: float,
        response_schema: type[BaseModel] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": self._render_prompt(messages),
            "temperature": temperature,
            "n_predict": self._config.max_output_tokens,
            "stream": False,
        }
        if response_schema is not None:
            payload["grammar_json_schema"] = response_schema.model_json_schema()
        return payload

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        url = self._config.base_url.rstrip("/") + "/completion"
        headers = {"content-type": "application/json", "accept": "application/json"}
        key = self._config.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            response = self._client.post(
                url,
                headers=headers,
                content=json.dumps(
                    self._payload(messages, temperature, response_schema), ensure_ascii=False
                ).encode("utf-8"),
                timeout=self._config.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"local_server transport failure: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"local_server returned {response.status_code}: {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(f"local_server returned non-JSON body: {exc}") from exc
        for key_name in ("content", "completion", "text"):
            value = body.get(key_name)
            if isinstance(value, str):
                return value
        raise ProviderError("local_server response lacked content/completion/text field")


def build_provider(config: ProviderConfig) -> LLMProvider:
    from adp.providers.http_chat import HttpChatProvider

    if config.kind == "http_chat":
        return HttpChatProvider(config)
    if config.kind == "local_server":
        return LocalServerProvider(config)
    raise ProviderError(f"provider kind '{config.kind}' cannot be constructed automatically")
