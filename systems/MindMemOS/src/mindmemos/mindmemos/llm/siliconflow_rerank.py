"""SiliconFlow rerank provider - direct HTTP calls without litellm."""

from __future__ import annotations

from typing import Any

import httpx

from mindmemos.config import ModelEndpointConfig
from mindmemos.typing import RerankHit, RerankResponse, Usage


class SiliconFlowRerankProvider:
    """Direct SiliconFlow rerank provider."""

    NAME = "siliconflow"

    def __init__(self, endpoint: ModelEndpointConfig) -> None:
        self.endpoint = endpoint
        self.api_base = endpoint.api_base.rstrip("/")
        self.api_key = endpoint.api_key
        self.model = endpoint.extra_body.get("model") if endpoint.extra_body else endpoint.model
        self.timeout = endpoint.timeout or 60

    def _build_url(self) -> str:
        return f"{self.api_base}/rerank"

    def _build_payload(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def rerank(self, query: str, documents: list[str], top_n: int) -> RerankResponse:
        payload = self._build_payload(query, documents, top_n)
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self._build_url(), json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        hits = [
            RerankHit(index=r.get("index", 0), relevance_score=r.get("relevance_score", 0.0))
            for r in data.get("results", [])
        ]

        usage = data.get("usage", {})
        return RerankResponse(
            results=hits,
            model=data.get("model", self.model),
            usage=Usage(
                completion_tokens=usage.get("completion_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            raw_response=data,
        )


def create_siliconflow_rerank_provider(endpoint: ModelEndpointConfig) -> SiliconFlowRerankProvider:
    """Factory to create SiliconFlow rerank provider."""
    return SiliconFlowRerankProvider(endpoint)
