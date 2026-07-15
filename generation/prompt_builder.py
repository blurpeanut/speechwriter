"""
Prompt builder — assembles the final augmented prompt sent to GPT-4o.

Priority order when trimming for the 8,000-token budget:
  1. Brief (always kept in full)
  2. Private context chunks (highest signal for this specific speech)
  3. Style exemplars (shape register and tone)
  4. Tavily live results (current facts)
  5. Content precedents (lowest priority — cut first)

Token counting uses a rough 4-chars-per-token heuristic to avoid a
tiktoken dependency; the 8,000-token limit has a 20% safety margin.
"""
import logging

logger = logging.getLogger(__name__)

_MAX_CHARS   = 8_000 * 4 * 0.80   # ~25,600 chars with 20% safety margin
_EXCERPT_LEN = 600                 # max chars per retrieved chunk in prompt


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert speech writer for the Singapore public service.
Your task is to draft ministerial speeches in the Singapore public service register.

REGISTER AND TONE RULES:
- Formal but not legalistic. Clear, direct sentences. No bureaucratic padding. Active voice preferred.
- Every factual claim must be traceable to the retrieved context provided. Do not invent statistics or policy details.
- If a claim cannot be supported by the context, do not include it.
- Avoid superlatives without evidence, unexplained acronyms, and overly emotive language.

STRUCTURE (follow this exactly):
1. Acknowledgement — greet the occasion and audience by name.
2. Statement of intent — "Let me share [N] things…" or equivalent signpost.
3. Substantive sections — one per key message, with clear section headings implied by the text.
4. Forward-looking close — return to the opening theme; end with a collective call to action.

RHETORICAL PATTERNS (use naturally, not mechanically):
- Rule of three for key points.
- Parallel construction for lists and comparisons.
- Deliberate repetition for emphasis on the most important idea.
- Collective call to action at the close ("build together", "get this right together").

TONE CALIBRATION:
- Budget speeches, Singapore Fintech Festival, major economic forums → highest register, most precise.
- Industry dinners, community events → slightly warmer, marginally less formal.

OUTPUT FORMAT:
Return a JSON object with exactly these keys:
{
  "outline": ["Section title 1", "Section title 2", ...],
  "full_speech": "The complete speech text...",
  "style_confidence_score": 0.0
}
The style_confidence_score should be your estimate (0.0–1.0) of how well the output matches
the Singapore public service register. Be conservative.
Do not include any text outside the JSON object."""


# ── Brief formatter ────────────────────────────────────────────────────────────

def _format_brief(brief: dict) -> str:
    length_map = {
        "5 min":  "~700 words",
        "10 min": "~1,400 words",
        "20 min": "~2,800 words",
    }
    length_hint = ""
    for key, words in length_map.items():
        if key in brief.get("length", ""):
            length_hint = f"{brief['length']} ({words})"
            break
    length_hint = length_hint or brief.get("length", "")

    return f"""SPEECH BRIEF
============
Speaker      : {brief.get('speaker', '')}
Event        : {brief.get('event_name', '')}
Date         : {brief.get('date', '')}
Audience     : {brief.get('audience', '')}
Length       : {length_hint}
Tone         : {brief.get('tone', 'Formal')}

Key messages (must all be addressed):
{brief.get('key_messages', '')}"""


# ── Chunk formatter ────────────────────────────────────────────────────────────

def _format_chunk(chunk: dict, label: str) -> str:
    source = chunk.get("source_name", "unknown")
    page   = chunk.get("page_ref", "")
    ref    = f"{source} — {page}" if page else source
    text   = chunk.get("text", "")[:_EXCERPT_LEN]
    return f"[{label}] {ref}\n{text}"


# ── Token budget trimmer ───────────────────────────────────────────────────────

def _trim_to_budget(sections: list[tuple[str, str]], budget: float) -> str:
    """
    Concatenate sections until the char budget is exhausted.
    Each section is (heading, body). The brief section is always included first
    and is not subject to trimming.
    """
    parts: list[str] = []
    used = 0.0
    for heading, body in sections:
        block = f"\n\n{'─' * 60}\n{heading}\n{'─' * 60}\n{body}"
        if used + len(block) > budget:
            remaining = int(budget - used)
            if remaining > 200:
                parts.append(block[:remaining] + "\n[truncated]")
            break
        parts.append(block)
        used += len(block)
    return "".join(parts)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_prompt(brief: dict, retrieved_chunks: list[dict]) -> tuple[str, str]:
    """
    Assemble the system prompt and user message for GPT-4o.

    Args:
        brief:            Speech brief dict (keys: speaker, event_name, date,
                          audience, length, tone, key_messages).
        retrieved_chunks: Merged, ranked list from the reranker.

    Returns:
        (system_prompt, user_message) — both strings ready to pass to the LLM.
    """
    # Separate chunks by source type
    private_chunks = [c for c in retrieved_chunks if c.get("source_type") == "private"]
    style_chunks   = [c for c in retrieved_chunks if c.get("source_type") == "public"]
    live_chunks    = [c for c in retrieved_chunks if c.get("source_type") == "live"]

    # Format each group
    def fmt_group(chunks, prefix):
        return "\n\n".join(
            _format_chunk(c, f"{prefix}{i+1}") for i, c in enumerate(chunks)
        )

    brief_text    = _format_brief(brief)
    private_text  = fmt_group(private_chunks, "PRIVATE")
    style_text    = fmt_group(style_chunks,   "STYLE")
    live_text     = fmt_group(live_chunks,    "LIVE")

    # Priority order: brief > private > style > live > (content already in style list)
    sections = [
        ("SPEECH BRIEF",             brief_text),
        ("PRIVATE CONTEXT (from uploaded documents — use these facts directly)",
                                     private_text),
        ("STYLE EXEMPLARS (match this register and tone)",
                                     style_text),
        ("LIVE SEARCH RESULTS (current statistics and announcements)",
                                     live_text),
    ]

    # Reserve the brief section fully; trim the rest to fit.
    brief_block    = f"\n\n{'─'*60}\n{sections[0][0]}\n{'─'*60}\n{sections[0][1]}"
    remaining_budget = _MAX_CHARS - len(brief_block)
    rest_block     = _trim_to_budget(sections[1:], remaining_budget)

    user_message = brief_block + rest_block + (
        "\n\n" + "═" * 60 +
        "\nUsing the brief and retrieved context above, draft the complete speech. "
        "Return ONLY valid JSON matching the output format in the system prompt."
    )

    total_chars = len(_SYSTEM_PROMPT) + len(user_message)
    logger.info(
        "Prompt built: ~%d tokens (%d chars). "
        "Private: %d chunks, Style: %d chunks, Live: %d chunks.",
        total_chars // 4, total_chars,
        len(private_chunks), len(style_chunks), len(live_chunks),
    )

    return _SYSTEM_PROMPT, user_message
