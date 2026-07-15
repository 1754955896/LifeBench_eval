"""
Context formatter for answer stage.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.search import SearchResult


def format_context(search_result: "SearchResult") -> str:
    """
    Format search result into context string.

    Args:
        search_result: Search result to format

    Returns:
        Formatted context string
    """
    formatted = search_result.retrieval_metadata.get("formatted_context", "")
    if formatted:
        return formatted

    parts = []
    top_k = search_result.retrieval_metadata.get("top_k", len(search_result.results))
    for idx, mem in enumerate(search_result.results[:top_k], 1):
        parts.append(f"{idx}. {mem.content}")

    context = "\n\n".join(parts)

    prefs = search_result.retrieval_metadata.get("preferences", {})
    if prefs.get("pref_string"):
        context += "\n\n" + prefs["pref_string"]

    return context
