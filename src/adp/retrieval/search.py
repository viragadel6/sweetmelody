from __future__ import annotations

import abc
import json
import math
import time
from typing import Any

import httpx
import regex as re

from adp.config import AppConfig
from adp.retrieval.cache import RetrievalCache
from adp.retrieval.fetch import PageFetcher
from adp.schemas import RetrievalBundle, RetrievalDocument

_TOKEN_RE = re.compile(r"\p{L}[\p{L}\p{M}\-']*|\d+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


class SearchProvider(abc.ABC):
    @abc.abstractmethod
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        raise AssertionError("SearchProvider.search must be implemented by subclasses")

    def close(self) -> None:
        return None


class HttpSearchProvider(SearchProvider):
    """Adapter for a JSON search API.

    Contract:
      GET {base_url}?q=<query>&count=<max_results>
      headers: Authorization: Bearer <key> when configured
      200 body: {"results": [{"url": str, "title": str, "snippet": str}]}
      Alternative accepted keys: "items"/"data" for the list, "link"/"href" for url,
      "description"/"content" for snippet.
    """

    def __init__(self, config: AppConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=config.retrieval.timeout_s,
            headers={"user-agent": config.search.user_agent},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @staticmethod
    def _normalize(raw: dict[str, Any]) -> dict[str, Any] | None:
        url = raw.get("url") or raw.get("link") or raw.get("href")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return None
        title = raw.get("title") or raw.get("name") or url
        snippet = raw.get("snippet") or raw.get("description") or raw.get("content") or ""
        return {"url": url, "title": str(title), "snippet": str(snippet)}

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        headers = {"accept": "application/json"}
        key = self._config.search.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        response = self._client.get(
            self._config.search.base_url,
            params={"q": query, "count": max_results},
            headers=headers,
            timeout=self._config.retrieval.timeout_s,
        )
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"search returned {response.status_code}",
                request=response.request,
                response=response,
            )
        body = response.json()
        if isinstance(body, list):
            entries = body
        elif isinstance(body, dict):
            candidate = body.get("results") or body.get("items") or body.get("data") or []
            entries = candidate if isinstance(candidate, list) else []
        else:
            entries = []
        normalized: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            item = self._normalize(entry)
            if item is not None:
                normalized.append(item)
            if len(normalized) >= max_results:
                break
        return normalized


class NullSearchProvider(SearchProvider):
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return []


class StaticSearchProvider(SearchProvider):
    def __init__(self, mapping: dict[str, list[dict[str, Any]]]) -> None:
        self._mapping = mapping

    @classmethod
    def from_payload(cls, payload: str) -> StaticSearchProvider:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("static search payload must be an object")
        mapping: dict[str, list[dict[str, Any]]] = {}
        for key, value in data.items():
            if isinstance(value, list):
                mapping[str(key)] = [item for item in value if isinstance(item, dict)]
        return cls(mapping)

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        exact = self._mapping.get(query)
        if exact is not None:
            return exact[:max_results]
        folded = query.casefold()
        for key, value in self._mapping.items():
            if key.casefold() in folded or folded in key.casefold():
                return value[:max_results]
        return self._mapping.get("*", [])[:max_results]


def rank_documents(query: str, documents: list[RetrievalDocument]) -> list[RetrievalDocument]:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        for index, document in enumerate(documents):
            document.rank = index + 1
        return documents
    scored: list[tuple[float, RetrievalDocument]] = []
    for index, document in enumerate(documents):
        haystack = f"{document.title} {document.snippet} {document.content[:2000]}"
        doc_tokens = _tokens(haystack)
        if not doc_tokens:
            score = 0.0
        else:
            token_set = set(doc_tokens)
            overlap = len(query_tokens & token_set)
            coverage = overlap / float(len(query_tokens))
            density = overlap / math.sqrt(float(len(doc_tokens)))
            score = (coverage * 0.7) + min(0.3, density * 0.3)
        position_penalty = index * 0.01
        content_bonus = 0.08 if document.content else 0.0
        scored.append((score + content_bonus - position_penalty, document))
    scored.sort(key=lambda item: item[0], reverse=True)
    ordered: list[RetrievalDocument] = []
    for position, (score, document) in enumerate(scored):
        document.rank = position + 1
        document.score = round(max(0.0, score), 6)
        ordered.append(document)
    return ordered


class RetrievalOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        provider: SearchProvider | None = None,
        fetcher: PageFetcher | None = None,
        cache: RetrievalCache | None = None,
    ) -> None:
        self._config = config
        self._cache = cache or RetrievalCache(config.paths.cache_db, config.retrieval.cache_ttl_s)
        if provider is not None:
            self._provider: SearchProvider = provider
        elif not config.search.enabled or config.search.kind == "null":
            self._provider = NullSearchProvider()
        else:
            self._provider = HttpSearchProvider(config)
        self._fetcher = fetcher or PageFetcher(config, cache=self._cache)

    def close(self) -> None:
        self._provider.close()
        self._fetcher.close()
        self._cache.close()

    def retrieve(self, query: str) -> RetrievalBundle:
        started = time.monotonic()
        if not self._config.search.enabled:
            return RetrievalBundle(
                query=query,
                executed=False,
                degraded=True,
                error="retrieval disabled by configuration",
                documents=[],
                elapsed_s=0.0,
            )
        cached = self._cache.get_search(query)
        raw_results: list[dict[str, Any]]
        error: str | None = None
        if cached is not None:
            raw_results = cached
        else:
            try:
                raw_results = self._provider.search(query, self._config.search.max_results)
            except Exception as exc:
                return RetrievalBundle(
                    query=query,
                    executed=True,
                    degraded=True,
                    error=f"search failure: {exc}",
                    documents=[],
                    elapsed_s=round(time.monotonic() - started, 6),
                )
            self._cache.put_search(query, raw_results)

        documents: list[RetrievalDocument] = []
        for entry in raw_results[: self._config.search.max_results]:
            url = str(entry.get("url", ""))
            if not url:
                continue
            documents.append(
                RetrievalDocument(
                    url=url,
                    title=str(entry.get("title", url)),
                    snippet=str(entry.get("snippet", "")),
                )
            )

        deadline = started + self._config.retrieval.timeout_s
        fetch_errors: list[str] = []
        for document in documents[: self._config.search.fetch_top_n]:
            if time.monotonic() >= deadline:
                fetch_errors.append("retrieval timeout reached before fetching all documents")
                break
            content, title, from_cache, fetch_error = self._fetcher.fetch(document.url)
            if content:
                document.content = content
                document.fetched = True
                document.from_cache = from_cache
                if title:
                    document.title = title
            if fetch_error:
                fetch_errors.append(f"{document.url}: {fetch_error}")

        ordered = rank_documents(query, documents)
        if not ordered:
            error = error or "search returned zero usable results"
        elif fetch_errors:
            error = "; ".join(fetch_errors)[:1000]

        return RetrievalBundle(
            query=query,
            executed=True,
            degraded=not ordered,
            error=error,
            documents=ordered,
            elapsed_s=round(time.monotonic() - started, 6),
        )
