"""
Context retrieval — queries the Chroma in-memory session collection for
relevant facts from the user's private uploaded documents.

Session isolation is strictly enforced: only the collection belonging to
the given session_id is ever queried.
"""
import logging
import uuid

from ingestion.embedder import embed_texts
from session.manager import SessionManager

logger = logging.getLogger(__name__)


def retrieve_context(
    query: str,
    session_id: str,
    session_manager: SessionManager,
    top_k: int = 5,
) -> list[dict]:
    """
    Return up to top_k private chunks from this session's Chroma collection.

    Each returned dict follows the chunk schema with source_type='private'
    and confidence_score derived from the cosine distance returned by Chroma.
    Returns [] if the session has no documents or does not exist.
    """
    try:
        collection = session_manager.get_collection(session_id)
    except KeyError:
        logger.warning("No active session for id: %s", session_id)
        return []

    count = collection.count()
    if count == 0:
        logger.debug("Session %s collection is empty.", session_id)
        return []

    q_vec = embed_texts([query])   # shape (1, DIM), normalised
    actual_k = min(top_k, count)

    raw = collection.query(
        query_embeddings=q_vec.tolist(),
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict] = []
    docs      = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]   # cosine distance: 0 = identical

    for doc, meta, dist in zip(docs, metadatas, distances):
        # Floor at 0.70 — user explicitly uploaded these docs for this speech,
        # so low query-document cosine similarity ≠ irrelevant content.
        confidence = max(0.70, 1.0 - float(dist))
        chunk = {
            "chunk_id":        str(uuid.uuid4()),
            "source_name":     meta.get("source_name", ""),
            "source_type":     "private",
            "speaker":         meta.get("speaker", ""),
            "ministry":        meta.get("ministry", ""),
            "date":            meta.get("date", ""),
            "occasion":        meta.get("occasion", ""),
            "page_ref":        meta.get("page_ref", ""),
            "style_tags":      [],
            "text":            doc,
            "confidence_score": confidence,
        }
        results.append(chunk)

    logger.debug(
        "Context retrieval for '%s…' (session %s): %d results",
        query[:60], session_id, len(results),
    )
    return results
