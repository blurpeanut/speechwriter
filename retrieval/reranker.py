"""
Reranker — merges results from all four retrieval streams, deduplicates,
and returns a single list sorted by confidence_score descending.

Deduplication rule: if two chunks have text similarity >= 0.95 (measured
by SequenceMatcher on the first 300 chars), keep the one with the higher
confidence_score and discard the other.
"""
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_DEDUP_THRESHOLD = 0.95
_DEDUP_WINDOW    = 300   # compare only first N chars for speed


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a[:_DEDUP_WINDOW], b[:_DEDUP_WINDOW]).ratio()


def _deduplicate(chunks: list[dict]) -> list[dict]:
    """
    Remove near-duplicate chunks (similarity >= threshold).
    O(n²) — acceptable for the small result sets (~15–20 chunks) we handle here.
    """
    kept: list[dict] = []
    for candidate in chunks:
        is_dup = False
        for existing in kept:
            if _similarity(candidate["text"], existing["text"]) >= _DEDUP_THRESHOLD:
                # Replace the existing entry if the candidate has a higher score.
                if candidate["confidence_score"] > existing["confidence_score"]:
                    kept[kept.index(existing)] = candidate
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)
    return kept


_CONTENT_THRESHOLD = 0.60   # public chunks below this score are style-only


def rerank(
    style_results: list[dict],
    content_results: list[dict],
    context_results: list[dict],
    live_results: list[dict],
) -> dict:
    """
    Merge all four retrieval streams, deduplicate, and split into four lists.

    Returns:
        {
          "content":    public chunks with confidence_score >= 0.60  (factual use),
          "style_only": public chunks with confidence_score <  0.60  (register reference only),
          "private":    private session chunks (always kept),
          "live":       Tavily live-search chunks (always kept),
        }
    """
    all_chunks = style_results + content_results + context_results + live_results

    before = len(all_chunks)
    merged = _deduplicate(all_chunks)
    after  = len(merged)

    if before != after:
        logger.debug("Deduplication: %d → %d chunks", before, after)

    merged.sort(key=lambda c: c.get("confidence_score", 0.0), reverse=True)

    content_chunks    = [c for c in merged
                         if c.get("source_type") == "public"
                         and c.get("confidence_score", 0.0) >= _CONTENT_THRESHOLD]
    style_only_chunks = [c for c in merged
                         if c.get("source_type") == "public"
                         and c.get("confidence_score", 0.0) < _CONTENT_THRESHOLD]
    private_chunks    = [c for c in merged if c.get("source_type") == "private"]
    live_chunks       = [c for c in merged if c.get("source_type") == "live"]

    logger.debug(
        "Reranker: %d content + %d style_only + %d private + %d live (from %d merged)",
        len(content_chunks), len(style_only_chunks),
        len(private_chunks), len(live_chunks), after,
    )
    return {
        "content":    content_chunks,
        "style_only": style_only_chunks,
        "private":    private_chunks,
        "live":       live_chunks,
    }
