"""SiliconFlow embedding provider - direct HTTP calls without litellm."""

from __future__ import annotations

from typing import Any

import httpx

from mindmemos.config import ModelEndpointConfig
from mindmemos.typing import EmbeddingResponse, Usage


class SiliconFlowEmbeddingProvider:
    """Direct SiliconFlow embedding provider."""

    NAME = "siliconflow"

    def __init__(self, endpoint: EmbedEndpointConfig) -> None:
        self.endpoint = endpoint
        self.api_base = endpoint.api_base.rstrip("/")
        self.api_key = endpoint.api_key
        self.model = endpoint.extra_body.get("model") if endpoint.extra_body else endpoint.model
        self.dimensions = endpoint.dimensions
        self.timeout = endpoint.timeout or 60

    def _build_url(self) -> str:
        return f"{self.api_base}/embeddings"

    def _build_payload(self, texts: str | list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        return payload

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def embed(self, texts: str | list[str]) -> EmbeddingResponse:
        payload = self._build_payload(texts)
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self._build_url(), json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        embeddings: list[list[float]] = [
            item["embedding"] for item in data.get("data", [])
        ]
        usage = data.get("usage", {})
        return EmbeddingResponse(
            embeddings=embeddings,
            model=data.get("model", self.model),
            usage=Usage(
                completion_tokens=usage.get("completion_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            raw_response=data,
        )


def create_siliconflow_provider(endpoint: ModelEndpointConfig) -> SiliconFlowEmbeddingProvider:
    """Factory to create SiliconFlow provider."""
    return SiliconFlowEmbeddingProvider(endpoint)
