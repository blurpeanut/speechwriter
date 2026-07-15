"""
Style retrieval — queries the FAISS style index for stylistically similar
speech excerpts. Used by the prompt builder to source style exemplars.
"""
import json
import logging
import os
from pathlib import Path

import faiss

from ingestion.embedder import embed_texts

logger = logging.getLogger(__name__)

_STYLE_INDEX_PATH = Path(os.getenv("FAISS_STYLE_INDEX_PATH", "storage/style_index"))

# Module-level cache — loaded once, reused across calls.
_index = None
_metadata: list[dict] = []


def _load() -> bool:
    """Lazy-load the FAISS style index and metadata. Returns False if not built yet."""
    global _index, _metadata
    if _index is not None:
        return True
    idx_file  = _STYLE_INDEX_PATH / "index.faiss"
    meta_file = _STYLE_INDEX_PATH / "style_metadata.json"
    if not idx_file.exists() or not meta_file.exists():
        logger.warning(
            "Style index not found at %s — run ingestion first.", _STYLE_INDEX_PATH
        )
        return False
    _index = faiss.read_index(str(idx_file))
    with open(meta_file, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
    logger.info("Style index loaded: %d vectors", _index.ntotal)
    return True


def retrieve_style(query: str, top_k: int = 3) -> list[dict]:
    """
    Return up to top_k chunks from the style index most similar to query.
    Each returned dict extends the chunk schema with confidence_score (float).
    Returns [] if the index has not been built.
    """
    if not _load():
        return []

    q_vec = embed_texts([query])   # shape (1, DIM), normalised
    scores, ids = _index.search(q_vec, top_k)

    results: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        chunk = dict(_metadata[idx])
        chunk["confidence_score"] = float(score)
        results.append(chunk)

    logger.debug("Style retrieval for '%s…': %d results", query[:60], len(results))
    return results
