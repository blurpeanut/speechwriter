"""
Content retrieval — queries the FAISS content index for topically relevant
speech precedents and factual material from the public corpus.
"""
import json
import logging
import os
from pathlib import Path

import faiss

from ingestion.embedder import embed_texts

logger = logging.getLogger(__name__)

_CONTENT_INDEX_PATH = Path(os.getenv("FAISS_CONTENT_INDEX_PATH", "storage/content_index"))

_index = None
_metadata: list[dict] = []


def _load() -> bool:
    global _index, _metadata
    if _index is not None:
        return True
    idx_file  = _CONTENT_INDEX_PATH / "index.faiss"
    meta_file = _CONTENT_INDEX_PATH / "content_metadata.json"
    if not idx_file.exists() or not meta_file.exists():
        logger.warning(
            "Content index not found at %s — run ingestion first.", _CONTENT_INDEX_PATH
        )
        return False
    _index = faiss.read_index(str(idx_file))
    with open(meta_file, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
    logger.info("Content index loaded: %d vectors", _index.ntotal)
    return True


def retrieve_content(query: str, top_k: int = 3) -> list[dict]:
    """
    Return up to top_k chunks from the content index most similar to query.
    Each returned dict extends the chunk schema with confidence_score (float).
    Returns [] if the index has not been built.
    """
    if not _load():
        return []

    q_vec = embed_texts([query])
    scores, ids = _index.search(q_vec, top_k)

    results: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        chunk = dict(_metadata[idx])
        chunk["confidence_score"] = float(score)
        results.append(chunk)

    logger.debug("Content retrieval for '%s…': %d results", query[:60], len(results))
    return results
