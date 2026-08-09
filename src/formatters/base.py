"""
Context formatter for answer stage.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.search import SearchResult


def format_context(search_result: "SearchResult") -> str:
    """
    Format search result into context string.

    Each result is tagged with its source layer (entity / community / episode /
    edge) so both the LLM and human evaluators can distinguish summary-level
    knowledge from specific relationship facts.
    """
    formatted = search_result.retrieval_metadata.get("formatted_context", "")
    if formatted:
        return formatted

    _ICONS = {"entity": "[E]", "community": "[C]", "episode": "[P]", "edge": "[F]"}

    parts = []
    top_k = search_result.retrieval_metadata.get("top_k", len(search_result.results))
    for idx, mem in enumerate(search_result.results[:top_k], 1):
        layer = mem.metadata.get("layer", "") if mem.metadata else ""
        tag = _ICONS.get(layer, "")
        parts.append(f"{idx}. {tag} {mem.content}".strip())

    context = "\n\n".join(parts)

    prefs = search_result.retrieval_metadata.get("preferences", {})
    if prefs.get("pref_string"):
        context += "\n\n" + prefs["pref_string"]

    return context
