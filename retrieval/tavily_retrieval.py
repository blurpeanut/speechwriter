"""
Tavily live search retrieval — fourth retrieval stream.
Returns current statistics, recent policy announcements, and news.
Only permitted live-search provider: api.tavily.com.
"""
import logging
import os
import uuid

from tavily import TavilyClient

logger = logging.getLogger(__name__)

_DEFAULT_CONFIDENCE = 0.70   # used when Tavily does not return a score


def _client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY is not set. Add it to your .env file.")
    return TavilyClient(api_key=api_key)


def retrieve_live(query: str, top_k: int = 3) -> list[dict]:
    """
    Search Tavily for current content related to query.
    Returns up to top_k results as chunk-schema dicts with source_type='live'.
    Returns [] on API error (non-fatal — other retrieval streams still work).
    """
    try:
        client = _client()
        response = client.search(
            query=query,
            max_results=top_k,
            search_depth="basic",
            include_answer=False,
        )
    except EnvironmentError:
        raise
    except Exception:
        logger.exception("Tavily search failed for query '%s…'", query[:60])
        return []

    results: list[dict] = []
    for item in response.get("results", []):
        title   = item.get("title", "")
        url     = item.get("url", "")
        content = item.get("content", "").strip()
        score   = float(item.get("score", _DEFAULT_CONFIDENCE))

        if not content:
            continue

        source_name = f"Tavily — {title}" if title else f"Tavily — {url}"
        results.append({
            "chunk_id":         str(uuid.uuid4()),
            "source_name":      source_name,
            "source_type":      "live",
            "speaker":          "",
            "ministry":         "",
            "date":             "",
            "occasion":         url,
            "page_ref":         "",
            "style_tags":       [],
            "text":             content,
            "confidence_score": score,
        })

    logger.debug("Tavily retrieval for '%s…': %d results", query[:60], len(results))
    return results
