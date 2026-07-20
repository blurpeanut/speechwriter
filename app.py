"""
SpeechCraft · AI Speech Writer
Streamlit application entrypoint — Phase 5.
"""
import html
import json
import logging
import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config must be the first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="SpeechCraft",
    page_icon="✍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Citation badges — vivid enough to read on dark bg */
.c-pub  { background:#1A3D2B; color:#6EE7A0; padding:2px 9px; border-radius:20px;
           font-size:11px; font-weight:600; white-space:nowrap; border:1px solid #2D6A4F; }
.c-priv { background:#1E1535; color:#C4AFFE; padding:2px 9px; border-radius:20px;
           font-size:11px; font-weight:600; white-space:nowrap; border:1px solid #5B3FA6; }
.c-live { background:#2A1F00; color:#FBBF24; padding:2px 9px; border-radius:20px;
           font-size:11px; font-weight:600; white-space:nowrap; border:1px solid #92660A; }
/* Step rail */
.rail   { display:flex; gap:6px; align-items:center; margin-bottom:20px; flex-wrap:wrap; }
.s-done { background:#1A3D2B; color:#6EE7A0; padding:5px 14px; border-radius:20px;
           font-size:12.5px; font-weight:600; border:1px solid #2D6A4F; }
.s-now  { background:#4D7FE8; color:#fff; padding:5px 14px; border-radius:20px;
           font-size:12.5px; font-weight:600; }
.s-todo { background:#1C2B3A; color:#5A7A96; padding:5px 14px; border-radius:20px;
           font-size:12.5px; font-weight:500; border:1px solid #243547; }
.sep    { color:#2A3D52; font-size:14px; }
/* Speech body — light text on dark bg */
.speech { font-family:'Georgia',serif; font-size:17px; line-height:1.95; color:#D8E3ED; }
.speech p { margin-bottom:18px; position:relative; }
/* Inline citation superscripts */
.csup {
    display:inline-flex; align-items:center; justify-content:center;
    width:15px; height:15px; border-radius:50%;
    background:#1E3A5F; color:#7EB8F7;
    font-size:9px; font-weight:700; font-family:sans-serif;
    vertical-align:super; margin-left:2px; cursor:default;
    border:1px solid #2A5080;
}
/* Citation detail panel */
.cite-excerpt {
    background:#0B1520; border-left:3px solid #4D7FE8;
    padding:14px 16px; border-radius:0 8px 8px 0; margin:12px 0;
    font-family:'Georgia',serif; font-size:14px; line-height:1.75;
    color:#A8C0D4; font-style:italic;
}
.cite-warn {
    background:#2D0F0F; border:1px solid #7F1D1D; border-radius:6px;
    padding:9px 13px; color:#FCA5A5; font-size:12px; margin-top:10px;
}
</style>
""", unsafe_allow_html=True)


# ── Cached resources (shared across reruns, one per process) ──────────────────
@st.cache_resource
def _session_manager():
    from session.manager import SessionManager
    return SessionManager()


@st.cache_resource
def _ingestor():
    from session.ingestor import DocumentIngestor
    return DocumentIngestor(_session_manager())


# ── Session state bootstrap ───────────────────────────────────────────────────
def _init() -> None:
    if "ready" in st.session_state:
        return
    sid = str(uuid.uuid4())
    _session_manager().create_session(sid)
    st.session_state.update({
        "ready":           True,
        "step":            1,
        "session_id":      sid,
        "context":         "",         # free-text context from Step 1
        "ingested_files":  set(),
        "generation_done": False,
        "output":          None,
        "current_speech":  "",
        "versions":        [],         # list of {label, speech, citations}
        "cur_ver":         0,
        "chat_history":    [],
        "active_citation": None,
    })
    logger.info("Streamlit session initialised: %s", sid)


# ── Header ────────────────────────────────────────────────────────────────────
def _header() -> None:
    c1, c2 = st.columns([7, 1])
    with c1:
        st.markdown(
            "## ✍ &nbsp;SpeechCraft"
            "<span style='font-size:13px;color:#8EA3B6;font-weight:400;margin-left:10px;'>"
            "AI Speech Writer</span>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div style='text-align:right;padding-top:14px;'>"
            "<span style='background:#E8E0F8;color:#5B3FA6;padding:3px 11px;"
            "border-radius:20px;font-size:11px;font-weight:600;'>Beta</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()


# ── Step rail ─────────────────────────────────────────────────────────────────
def _step_rail() -> None:
    step   = st.session_state.step
    labels = ["1 · Upload & context", "2 · Generate & review"]
    parts  = []
    for i, label in enumerate(labels, 1):
        if i < step:
            parts.append(f'<span class="s-done">✓ {label}</span>')
        elif i == step:
            parts.append(f'<span class="s-now">{label}</span>')
        else:
            parts.append(f'<span class="s-todo">{label}</span>')
    st.markdown(
        '<div class="rail">' + '<span class="sep"> › </span>'.join(parts) + '</div>',
        unsafe_allow_html=True,
    )


# ── Citation helpers ──────────────────────────────────────────────────────────

def _citation_role(c: dict) -> str:
    """Derive citation_role from source_type and confidence_score."""
    if c["source_type"] == "private":
        return "private"
    if c["source_type"] == "live":
        return "live"
    return "content" if c["confidence_score"] >= 0.60 else "style_only"


_ROLE_STYLE: dict[str, tuple[str, str, str, str]] = {
    # role: (label, text_color, bg_color, border_color)
    "content":    ("Content",        "#6EE7A0", "#1A3D2B", "#2D6A4F"),
    "style_only": ("Style reference", "#A0AEC0", "#1C2B3A", "#2D3748"),
    "private":    ("Private doc",    "#C4AFFE", "#1E1535", "#5B3FA6"),
    "live":       ("Live search",    "#FBBF24", "#2A1F00", "#92660A"),
}


def _annotate_speech(text: str, citations: list[dict]) -> str:
    """
    Return speech HTML with citation superscripts added to each paragraph
    whose text has keyword overlap with a citation's excerpt.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    result: list[str] = []
    for para in paragraphs:
        para_words = set(re.findall(r'\b[a-z]{4,}\b', para.lower()))
        markers: list[int] = []
        for c in citations:
            exc_words = set(re.findall(r'\b[a-z]{4,}\b', c["excerpt"].lower()))
            if exc_words and para_words:
                overlap = len(para_words & exc_words) / len(exc_words)
                if overlap >= 0.12:
                    markers.append(c["id"])
        escaped = html.escape(para).replace("\n", "<br>")
        sups    = "".join(f'<sup class="csup">{m}</sup>' for m in markers)
        result.append(f"<p>{escaped}{sups}</p>")
    return "".join(result)


def _render_citation_detail(c: dict) -> None:
    """Render the full detail panel for one citation."""
    role  = _citation_role(c)
    label, tc, bg, border = _ROLE_STYLE.get(role, _ROLE_STYLE["content"])
    score = c["confidence_score"]
    bar_color = "#6EE7A0" if score >= 0.75 else "#FBBF24" if score >= 0.65 else "#F87171"

    st.markdown(f"**{c['source_name']}**")
    st.markdown(
        f'<span style="background:{bg};color:{tc};padding:2px 10px;border-radius:20px;'
        f'font-size:11px;font-weight:600;border:1px solid {border};">{label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="margin:10px 0 4px;">'
        f'<span style="color:#8EA3B6;font-size:11px;">Confidence</span>'
        f'<span style="color:{bar_color};font-weight:600;font-size:13px;'
        f'margin-left:8px;">{score:.2f}</span></div>'
        f'<div style="background:#1C2B3A;border-radius:4px;height:6px;overflow:hidden;">'
        f'<div style="width:{int(score*100)}%;height:6px;background:{bar_color};'
        f'border-radius:4px;"></div></div>',
        unsafe_allow_html=True,
    )
    if c.get("page_ref"):
        st.caption(f"📄 {c['page_ref']}")
    st.markdown(
        f'<div class="cite-excerpt">"{html.escape(c["excerpt"])}"</div>',
        unsafe_allow_html=True,
    )
    if c.get("warning") or score < 0.60:
        st.markdown(
            '<div class="cite-warn">⚠ Low confidence — verify this source before use.</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — UPLOAD & CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
def screen_upload() -> None:
    st.subheader("Upload & context")

    st.info(
        "Documents are embedded via OpenAI API (api.openai.com). "
        "Do not upload documents classified above Restricted.",
        icon="ℹ️",
    )

    ingestor = _ingestor()
    sid      = st.session_state.session_id
    ingested = st.session_state.ingested_files

    uploaded_files = st.file_uploader(
        "Upload your documents (brief, talking points, emails, stats)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="multi_upload",
    )

    for uploaded in (uploaded_files or []):
        if uploaded.name not in ingested:
            with st.spinner(f"Indexing {uploaded.name}…"):
                try:
                    n = ingestor.ingest(uploaded.getvalue(), uploaded.name, sid)
                    ingested.add(uploaded.name)
                    st.success(f"✓ {uploaded.name} — {n} chunks indexed")
                except Exception as exc:
                    st.error(f"Failed to process {uploaded.name}: {exc}")
                    logger.exception("Ingest error for %s", uploaded.name)
        else:
            st.success(f"✓ {uploaded.name} — already indexed")

    context = st.text_area(
        "Additional context for the AI",
        height=180,
        placeholder=(
            "e.g. Speaker: Minister Lawrence Wong. "
            "Occasion: Singapore Fintech Festival 2025. "
            "Audience: fintech founders and investors. "
            "Key messages: responsible AI adoption, Budget 2025 fintech support. "
            "Tone: formal."
        ),
        value=st.session_state.context,
    )

    st.divider()
    st.markdown("**Public knowledge base** (always active)")
    st.markdown(
        '<span class="c-pub">PMO speeches (2018–2025)</span> &nbsp;&nbsp;'
        '<span class="c-pub">Hansard debates</span> &nbsp;&nbsp;'
        '<span class="c-live">Tavily live search</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("✦ Generate speech", type="primary"):
        if not context.strip() and not ingested:
            st.error("Please upload at least one document or add context before generating.")
            return
        st.session_state.context         = context.strip()
        st.session_state.step            = 2
        st.session_state.generation_done = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — GENERATE & REVIEW
# ══════════════════════════════════════════════════════════════════════════════

def _run_generation() -> None:
    """Execute the pipeline with staged progress steps, then store results."""
    from retrieval.style_retrieval   import retrieve_style
    from retrieval.content_retrieval import retrieve_content
    from retrieval.context_retrieval import retrieve_context
    from retrieval.tavily_retrieval  import retrieve_live
    from retrieval.reranker          import rerank
    from generation.prompt_builder   import build_prompt
    from generation.llm              import generate_speech
    from generation.citations        import assemble_citations, compute_style_confidence

    context = st.session_state.context
    sid     = st.session_state.session_id
    smgr    = _session_manager()
    query   = f"{context} Singapore public service".strip()
    # Build a minimal brief dict so the prompt builder receives expected keys
    brief = {
        "speaker": "", "event_name": "", "date": "",
        "audience": "", "length": "", "tone": "",
        "key_messages": context,
    }

    with st.status("Generating your speech…", expanded=True) as status:

        status.write("🔍 Retrieving context from public corpus…")
        style_r   = retrieve_style(query,   top_k=3)
        content_r = retrieve_content(query, top_k=3)

        status.write("📄 Searching uploaded documents…")
        context_r = retrieve_context(query, sid, smgr, top_k=5)

        status.write("🌐 Running live web search…")
        live_r = retrieve_live(query, top_k=3)

        retrieved = rerank(style_r, content_r, context_r, live_r)
        total_chunks = sum(len(v) for v in retrieved.values())
        logger.info("Generation: %d chunks retrieved after reranking", total_chunks)

        status.write("📝 Building generation prompt…")
        system_prompt, user_message = build_prompt(brief, retrieved)

        status.write("✍ Drafting speech with GPT-4o…")
        try:
            llm_out = generate_speech(system_prompt, user_message)
        except Exception as exc:
            status.update(label="Generation failed", state="error")
            st.error(f"GPT-4o returned an error: {exc}")
            logger.exception("Generation error")
            return

        full_speech = llm_out.get("full_speech", "")
        outline     = llm_out.get("outline", [])
        style_score = float(llm_out.get("style_confidence_score", 0.0))

        # Flatten all chunks for citation assembly; style_only last (lowest priority)
        all_chunks = (retrieved["content"] + retrieved["private"]
                      + retrieved["live"] + retrieved["style_only"])
        citations        = assemble_citations(all_chunks, full_speech)
        final_style_score = compute_style_confidence(citations) or style_score

        status.update(label="Speech ready!", state="complete")

    st.session_state.output = {
        "outline":                outline,
        "full_speech":            full_speech,
        "citations":              citations,
        "style_confidence_score": final_style_score,
        "word_count":             len(full_speech.split()),
    }
    st.session_state.current_speech = full_speech
    st.session_state.versions       = [{"label": "v1 · Original",
                                         "speech": full_speech,
                                         "citations": citations}]
    st.session_state.cur_ver         = 0
    st.session_state.chat_history    = []
    st.session_state.active_citation = citations[0]["id"] if citations else None
    st.session_state.generation_done = True


def _refine(instruction: str) -> None:
    """Refine the current speech based on a chat instruction and add a new version."""
    from generation.llm import generate_speech

    current = st.session_state.current_speech
    system = (
        "You are editing a Singapore ministerial speech. "
        "Apply the user's instruction precisely, preserving the overall structure "
        "and Singapore public service register. "
        "Return JSON only: "
        '{"full_speech": "...", "change_summary": "one sentence describing the change"}'
    )
    user = (
        f"Current speech:\n\n{current}\n\n"
        f"Instruction: {instruction}"
    )

    with st.spinner("Refining speech…"):
        try:
            result = generate_speech(system, user)
        except Exception as exc:
            st.error(f"Refinement failed: {exc}")
            logger.exception("Refinement error")
            return

    new_speech = result.get("full_speech", current)
    summary    = result.get("change_summary", "Speech revised.")

    ver_num = len(st.session_state.versions) + 1
    st.session_state.versions.append({
        "label":     f"v{ver_num} · Chat edit",
        "speech":    new_speech,
        "citations": st.session_state.output["citations"],
    })
    st.session_state.cur_ver       = ver_num - 1
    st.session_state.current_speech = new_speech
    st.session_state.chat_history.append({"role": "assistant", "content": summary})


def screen_generate() -> None:
    # ── Run generation on first visit ─────────────────────────────────────────
    if not st.session_state.generation_done:
        _run_generation()
        if st.session_state.generation_done:
            st.rerun()
        return

    output    = st.session_state.output
    ver       = st.session_state.versions[st.session_state.cur_ver]
    citations = ver["citations"]

    # ── Meta bar ───────────────────────────────────────────────────────────────
    wc = len(st.session_state.current_speech.split())
    st.markdown(f"{wc:,} words")

    # ── Version selector ───────────────────────────────────────────────────────
    if len(st.session_state.versions) > 1:
        ver_labels = [v["label"] for v in st.session_state.versions]
        chosen     = st.selectbox(
            "Version", ver_labels, index=st.session_state.cur_ver
        )
        new_idx = ver_labels.index(chosen)
        if new_idx != st.session_state.cur_ver:
            st.session_state.cur_ver        = new_idx
            st.session_state.current_speech = st.session_state.versions[new_idx]["speech"]
            st.rerun()

    st.divider()

    # ── Outline / Full speech tabs ─────────────────────────────────────────────
    tab_outline, tab_speech = st.tabs(["📋 Outline", "📄 Full speech"])

    with tab_outline:
        outline = output.get("outline", [])
        if outline:
            for i, section in enumerate(outline, 1):
                with st.expander(f"{i}. {section}", expanded=(i == 1)):
                    st.caption("Expand to review this section.")
        else:
            st.info("No outline returned — the full speech is available in the next tab.")

        if st.button("View full speech →", type="primary"):
            # Can't programmatically switch Streamlit tabs; guide the user.
            st.toast("Click the 'Full speech' tab above.")

    with tab_speech:
        # ── Actions bar ────────────────────────────────────────────────────────
        dl_col, _, style_col = st.columns([2, 4, 3])
        with dl_col:
            st.download_button(
                "↓ Export .txt",
                data=st.session_state.current_speech,
                file_name="speech.txt",
                mime="text/plain",
            )
        with style_col:
            sc = output["style_confidence_score"]
            sc_color = "#6EE7A0" if sc >= 0.75 else "#FBBF24" if sc >= 0.65 else "#F87171"
            st.markdown(
                f"Style confidence: "
                f"<span style='color:{sc_color};font-weight:600;'>{sc:.0%}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Two-column: speech + source panel ──────────────────────────────────
        col_speech, col_panel = st.columns([3, 2], gap="large")

        with col_speech:
            annotated = _annotate_speech(st.session_state.current_speech, citations)
            st.markdown(f'<div class="speech">{annotated}</div>', unsafe_allow_html=True)

            # Citation pills — click to switch active citation
            if citations:
                st.markdown("")
                active_id = st.session_state.get("active_citation")
                pill_cols = st.columns(min(len(citations), 12))
                for col, c in zip(pill_cols, citations):
                    role = _citation_role(c)
                    tc   = _ROLE_STYLE.get(role, _ROLE_STYLE["content"])[1]
                    is_active = c["id"] == active_id
                    if col.button(
                        str(c["id"]),
                        key=f"pill_{c['id']}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state.active_citation = c["id"]
                        st.rerun()

        with col_panel:
            st.markdown(
                "<div style='position:sticky;top:80px;'>",
                unsafe_allow_html=True,
            )
            active_id = st.session_state.get("active_citation")
            active_c  = next((c for c in citations if c["id"] == active_id), None)

            if active_c:
                st.markdown(
                    f"<span style='font-size:10px;font-weight:600;color:#5A7A96;"
                    f"text-transform:uppercase;letter-spacing:0.1em;'>"
                    f"Source [{active_c['id']}]</span>",
                    unsafe_allow_html=True,
                )
                _render_citation_detail(active_c)

                # Collapsed list of all other citations
                if len(citations) > 1:
                    st.markdown("")
                    with st.expander("All sources", expanded=False):
                        for c in citations:
                            role  = _citation_role(c)
                            label = _ROLE_STYLE.get(role, _ROLE_STYLE["content"])[0]
                            if st.button(
                                f"[{c['id']}] {c['source_name'][:40]}",
                                key=f"src_list_{c['id']}",
                                use_container_width=True,
                            ):
                                st.session_state.active_citation = c["id"]
                                st.rerun()
            else:
                st.caption("Click a citation number to view its source.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # ── Chat refinement ────────────────────────────────────────────────────
        st.markdown("#### Refine with chat")
        st.caption("Each change is saved as a new version. Switch versions above.")

        quick_options = [
            "Make the opening more vivid and personal",
            "Shorten the speech by 20%",
            "Strengthen the closing paragraph",
            "Make the tone slightly less formal",
            "Add a concrete statistic to the key message section",
        ]
        quick_cols = st.columns(len(quick_options))
        quick_hit  = None
        for col, opt in zip(quick_cols, quick_options):
            if col.button(opt[:22] + "…" if len(opt) > 22 else opt,
                          use_container_width=True, key=f"qp_{opt[:10]}"):
                quick_hit = opt

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        prompt = st.chat_input("Instruct changes or ask questions…")
        if not prompt and quick_hit:
            prompt = quick_hit

        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            _refine(prompt)
            st.rerun()

    # ── Back button ────────────────────────────────────────────────────────────
    st.divider()
    if st.button("← Back"):
        st.session_state.step            = 1
        st.session_state.generation_done = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    _init()
    _header()
    _step_rail()

    step = st.session_state.step
    if step == 1:
        screen_upload()
    else:
        screen_generate()


if __name__ == "__main__":
    main()
