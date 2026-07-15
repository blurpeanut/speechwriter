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


def rerank(
    style_results: list[dict],
    content_results: list[dict],
    context_results: list[dict],
    live_results: list[dict],
) -> list[dict]:
    """
    Merge all four retrieval streams, deduplicate, and sort by confidence_score.

    Every returned chunk carries: source_name, source_type, confidence_score,
    page_ref, text — the minimum required for the prompt builder and citation
    assembler in Phase 4.
    """
    all_chunks = style_results + content_results + context_results + live_results

    before = len(all_chunks)
    merged = _deduplicate(all_chunks)
    after  = len(merged)

    if before != after:
        logger.debug("Deduplication: %d → %d chunks", before, after)

    merged.sort(key=lambda c: c.get("confidence_score", 0.0), reverse=True)
    logger.debug(
        "Reranker: %d style + %d content + %d context + %d live → %d merged",
        len(style_results), len(content_results),
        len(context_results), len(live_results), after,
    )
    return merged
