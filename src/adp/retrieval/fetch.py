from __future__ import annotations

import httpx

from adp.config import AppConfig
from adp.retrieval.cache import RetrievalCache
from adp.retrieval.extract import extract_main_content


class PageFetcher:
    def __init__(
        self,
        config: AppConfig,
        cache: RetrievalCache | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._cache = cache
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=config.retrieval.timeout_s,
            follow_redirects=True,
            headers={"user-agent": config.search.user_agent},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> PageFetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch(self, url: str) -> tuple[str, str, bool, str | None]:
        if self._cache is not None:
            cached = self._cache.get_page(url)
            if cached is not None:
                content, title = cached
                return content, title, True, None
        try:
            response = self._client.get(url, timeout=self._config.retrieval.timeout_s)
        except httpx.HTTPError as exc:
            return "", "", False, f"fetch transport error: {exc}"
        if response.status_code >= 400:
            return "", "", False, f"fetch status {response.status_code}"
        content_type = response.headers.get("content-type", "")
        if content_type and "html" not in content_type and "text" not in content_type:
            return "", "", False, f"unsupported content-type {content_type}"
        content, title = extract_main_content(
            response.text, self._config.retrieval.max_extract_chars
        )
        if len(content) < self._config.retrieval.min_extract_chars:
            return content, title, False, "extracted content below minimum length"
        if self._cache is not None:
            self._cache.put_page(url, content, title)
        return content, title, False, None
