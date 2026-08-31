from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel

from adp.config import ProviderConfig
from adp.providers.base import LLMProvider, ProviderError, messages_to_dicts
from adp.schemas import Message


class HttpChatProvider(LLMProvider):
    """Adapter for OpenAI-compatible /chat/completions endpoints.

    Contract:
      POST {base_url}/chat/completions
      headers: Authorization: Bearer <key>, content-type: application/json
      body:    {"model": str, "messages": [{"role","content"}], "temperature": float,
                "max_tokens": int, optional "response_format": {"type": "json_object"}}
      200 body: {"choices": [{"message": {"content": str}}]}
    """

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

    def __enter__(self) -> HttpChatProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", "accept": "application/json"}
        key = self._config.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _payload(
        self,
        messages: list[Message],
        temperature: float,
        response_schema: type[BaseModel] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": self._config.max_output_tokens,
        }
        if response_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        try:
            response = self._client.post(
                url,
                headers=self._headers(),
                content=json.dumps(
                    self._payload(messages, temperature, response_schema), ensure_ascii=False
                ).encode("utf-8"),
                timeout=self._config.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"http_chat transport failure: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"http_chat provider returned {response.status_code}: {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(f"http_chat provider returned non-JSON body: {exc}") from exc
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("http_chat provider returned no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ProviderError("http_chat provider returned malformed choice")
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderError("http_chat provider returned non-string content")
        return content
