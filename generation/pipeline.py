"""
End-to-end speech generation pipeline.
Wires retrieval (Phases 1–3) → prompt building → LLM generation → citations (Phase 4).
Called by the Streamlit UI (Phase 5) and the eval runner.

Returns the output dict defined in CLAUDE.md §7 Phase 4:
{
  "outline":                list[str],
  "full_speech":            str,
  "citations":              list[dict],
  "style_confidence_score": float,
  "word_count":             int,
}
"""
import logging

from retrieval.style_retrieval   import retrieve_style
from retrieval.content_retrieval import retrieve_content
from retrieval.context_retrieval import retrieve_context
from retrieval.tavily_retrieval  import retrieve_live
from retrieval.reranker          import rerank
from generation.prompt_builder   import build_prompt
from generation.llm              import generate_speech
from generation.citations        import assemble_citations, compute_style_confidence
from session.manager             import SessionManager

logger = logging.getLogger(__name__)


def run_pipeline(
    brief: dict,
    session_id: str,
    session_manager: SessionManager,
    style_top_k: int   = 3,
    content_top_k: int = 3,
    context_top_k: int = 5,
    live_top_k: int    = 3,
) -> dict:
    """
    Run the full RAG → generation pipeline for a speech brief.

    Args:
        brief:           Speech brief dict (speaker, event_name, date, audience,
                         length, tone, key_messages).
        session_id:      Active session UUID.
        session_manager: Shared SessionManager instance.

    Returns:
        Output dict with outline, full_speech, citations, style_confidence_score,
        word_count.
    """
    query = (
        f"{brief.get('event_name', '')} {brief.get('key_messages', '')} "
        f"{brief.get('tone', '')} Singapore public service"
    ).strip()

    logger.info("Pipeline: retrieving context for brief — '%s…'", query[:80])

    # ── Retrieval ──────────────────────────────────────────────────────────────
    style_results   = retrieve_style(query,   top_k=style_top_k)
    content_results = retrieve_content(query, top_k=content_top_k)
    context_results = retrieve_context(query, session_id, session_manager, top_k=context_top_k)
    live_results    = retrieve_live(query,    top_k=live_top_k)

    retrieved = rerank(style_results, content_results, context_results, live_results)
    logger.info("Pipeline: %d merged chunks after reranking", len(retrieved))

    # ── Prompt assembly ────────────────────────────────────────────────────────
    system_prompt, user_message = build_prompt(brief, retrieved)

    # ── Generation ────────────────────────────────────────────────────────────
    logger.info("Pipeline: calling GPT-4o …")
    llm_output = generate_speech(system_prompt, user_message)

    full_speech = llm_output.get("full_speech", "")
    outline     = llm_output.get("outline", [])
    style_score = float(llm_output.get("style_confidence_score", 0.0))

    # ── Citations ──────────────────────────────────────────────────────────────
    citations = assemble_citations(retrieved, full_speech)
    final_style_score = compute_style_confidence(citations)

    word_count = len(full_speech.split())
    logger.info(
        "Pipeline complete: %d words, %d citations, style_score=%.2f",
        word_count, len(citations), final_style_score,
    )

    return {
        "outline":                outline,
        "full_speech":            full_speech,
        "citations":              citations,
        "style_confidence_score": final_style_score or style_score,
        "word_count":             word_count,
    }
