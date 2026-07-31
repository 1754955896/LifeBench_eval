"""
LLM Proxy utility for querying token statistics.
"""
from typing import Any, Dict, Optional


def query_llm_proxy(
    proxy_url: str = "http://localhost:18443",
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """
    Query the llm_proxy service for token statistics.

    Args:
        proxy_url: URL of the llm_proxy service.
        timeout: Request timeout in seconds.

    Returns:
        Token statistics dict with prompt_tokens, completion_tokens,
        total_tokens, request_count, or None on failure.
    """
    try:
        import httpx
        response = httpx.get(
            f"{proxy_url}/token-stats",
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None
