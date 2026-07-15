"""
Citation assembler — maps generated speech paragraphs to source chunks
and produces the final citations list per the CLAUDE.md §6 schema.

Mapping strategy: for each chunk, find paragraphs in the speech that share
the most keyword overlap with the chunk text. This is a keyword-overlap
heuristic — sufficient for the prototype.
"""
import logging
import re

logger = logging.getLogger(__name__)

_WARNING_THRESHOLD  = 0.65
_EXCERPT_MAX_CHARS  = 120
_MIN_KEYWORD_LEN    = 4      # ignore stop-words shorter than this
_MIN_OVERLAP_SCORE  = 0.05   # minimum normalised overlap to consider a match


def _keywords(text: str) -> set[str]:
    """Extract lowercase meaningful words from text."""
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    return set(words)


def _overlap_score(chunk_text: str, speech_text: str) -> float:
    """
    Normalised keyword overlap: |intersection| / |chunk_keywords|.
    Returns 0.0 if chunk has no keywords.
    """
    ck = _keywords(chunk_text)
    sk = _keywords(speech_text)
    if not ck:
        return 0.0
    return len(ck & sk) / len(ck)


def assemble_citations(
    retrieved_chunks: list[dict],
    full_speech: str,
) -> list[dict]:
    """
    Build the citations list for the generated speech.

    For each retrieved chunk, compute how well it overlaps with the speech text.
    Only include chunks that have meaningful overlap (>= MIN_OVERLAP_SCORE).
    Assign sequential IDs and set warning flag for low-confidence citations.

    Args:
        retrieved_chunks: Merged list from reranker (each has confidence_score).
        full_speech:      The complete generated speech text.

    Returns:
        List of citation dicts matching CLAUDE.md §6 schema, sorted by id.
    """
    citations: list[dict] = []
    citation_id = 1

    for chunk in retrieved_chunks:
        overlap = _overlap_score(chunk.get("text", ""), full_speech)
        if overlap < _MIN_OVERLAP_SCORE:
            continue

        # Confidence: blend retrieval score with overlap signal.
        retrieval_score = chunk.get("confidence_score", 0.5)
        blended_score   = 0.7 * retrieval_score + 0.3 * overlap
        blended_score   = round(min(1.0, max(0.0, blended_score)), 4)

        source_name = chunk.get("source_name") or ""
        page_ref    = chunk.get("page_ref") or ""
        excerpt     = chunk.get("text", "")[:_EXCERPT_MAX_CHARS].strip()

        # These fields must never be empty (CLAUDE.md eval assertion)
        if not source_name or not excerpt:
            logger.warning(
                "Skipping citation with empty source_name or excerpt for chunk_id %s",
                chunk.get("chunk_id"),
            )
            continue

        citations.append({
            "id":               citation_id,
            "source_name":      source_name,
            "source_type":      chunk.get("source_type", "public"),
            "page_ref":         page_ref,
            "excerpt":          excerpt,
            "confidence_score": blended_score,
            "warning":          blended_score < _WARNING_THRESHOLD,
        })
        citation_id += 1

    logger.info(
        "Citations assembled: %d total, %d with warning flag",
        len(citations),
        sum(1 for c in citations if c["warning"]),
    )
    return citations


def compute_style_confidence(citations: list[dict]) -> float:
    """
    Average confidence score across public-source citations only.
    Returns 0.0 if there are no public citations.
    """
    public = [c for c in citations if c.get("source_type") == "public"]
    if not public:
        return 0.0
    return round(sum(c["confidence_score"] for c in public) / len(public), 4)
