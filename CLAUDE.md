# CLAUDE.md — SpeechCraft AI Speech Writer

> Read this file fully before writing any code, creating any file, or making any architectural decision.
> When in doubt, re-read the relevant section rather than inferring.
> If anything is ambiguous, stop and ask — do not guess.

---

## 1. What this project is

SpeechCraft is a Streamlit-based AI speech drafting tool built for a hackathon prototype. It helps communications staff draft speeches for ministers and senior public servants.

Given a speech brief (speaker, occasion, audience, key messages, tone, length) and optional supporting documents uploaded for the session, the system:
1. Retrieves stylistically and topically relevant material from a public corpus of Singapore ministerial speeches (PMO + Hansard)
2. Retrieves relevant facts from any private documents uploaded for the session
3. Runs a live Tavily web search for current statistics and recent announcements
4. Uses GPT-4o to draft a full speech in Singapore public service register with every factual claim traced to a cited source
5. Lets the user refine the speech via a chat panel

**This is a prototype.** Users will not upload real sensitive documents during the hackathon. The full security hardening (strict local-only, no cloud calls) is noted for production but is not the focus now. Ship a working demo first.

---

## 2. Permitted external services

Three and only three:
- **`api.openai.com`** — embeddings (`text-embedding-3-small`) and generation (`gpt-4o`)
- **`api.tavily.com`** — live web search

Do not call any other external API.

---

## 3. Tech stack — do not deviate without explicit instruction

| Layer | Tool | Notes |
|---|---|---|
| Frontend | Streamlit | Single frontend — `app.py` is the entrypoint |
| LLM | OpenAI `gpt-4o` | Streaming via `openai` Python SDK |
| Embeddings | OpenAI `text-embedding-3-small` | Via `openai` Python SDK |
| Public style DB | FAISS | Persisted to `storage/style_index/` — load at startup |
| Public content DB | FAISS | Persisted to `storage/content_index/` — separate index |
| Private session DB | Chroma | **In-memory only** — `persist_directory` must NEVER be set |
| PDF parsing | `pdfplumber` | For both public corpus and private uploads — no LlamaParse |
| DOCX parsing | `python-docx` | For private DOCX uploads |
| HTML parsing | `BeautifulSoup` | For public corpus HTML files |
| Live search | Tavily | `tavily-python` SDK — fourth retrieval stream |

**LlamaParse is not used.** It calls an external API (`api.llama-cloud.com`) which is not on the permitted list. Use `pdfplumber` for all PDFs.

---

## 4. Project structure

```
speechcraft/
├── CLAUDE.md                    ← this file
├── .env                         ← API keys — gitignored
├── app.py                       ← Streamlit entrypoint
├── requirements.txt
├── specs/
│   ├── speechcraft_v3.html      ← UI layout reference (three-step flow)
│   └── architecture.html        ← system architecture reference
├── ingestion/
│   ├── run_ingestion.py         ← CLI: python ingestion/run_ingestion.py --source pmo|hansard
│   ├── chunker.py               ← paragraph-level chunking, metadata extraction
│   ├── style_extractor.py       ← tone label, sentence rhythm, rhetorical markers
│   └── embedder.py              ← OpenAI text-embedding-3-small wrapper
├── storage/
│   ├── style_index/             ← FAISS index + style_metadata.json (gitignored)
│   ├── content_index/           ← FAISS index + content_metadata.json (gitignored)
│   └── corpus/
│       ├── pmo/                 ← PMO speeches as PDF/HTML (manually downloaded)
│       └── hansard/             ← Hansard excerpts as PDF/HTML (manually downloaded)
├── retrieval/
│   ├── style_retrieval.py       ← queries FAISS style index
│   ├── content_retrieval.py     ← queries FAISS content index
│   ├── context_retrieval.py     ← queries Chroma session collection
│   ├── tavily_retrieval.py      ← Tavily live search
│   └── reranker.py              ← merges all four streams, deduplicates, sorts by score
├── generation/
│   ├── prompt_builder.py        ← assembles augmented prompt (<8,000 tokens)
│   ├── llm.py                   ← GPT-4o streaming call
│   └── citations.py             ← maps paragraphs to source chunks, flags low confidence
├── session/
│   ├── manager.py               ← Chroma in-memory collection per session UUID
│   └── ingestor.py              ← parse → chunk → embed → load into session collection
└── tests/
    └── evals/                   ← golden test cases (JSON)
```

---

## 5. Chunk metadata schema

Every chunk produced during ingestion or session upload must carry this metadata. Define it once in `ingestion/chunker.py` and reuse everywhere.

```python
{
  "chunk_id":    str,   # uuid
  "source_name": str,   # filename or URL e.g. "PM_Lawrence_Wong_NDP2024.pdf"
  "source_type": str,   # "public" | "private" | "live"
  "speaker":     str,   # e.g. "Lawrence Wong" — store for all public chunks
  "ministry":    str,   # e.g. "PMO" | "Hansard"
  "date":        str,   # ISO format e.g. "2024-08-18"
  "occasion":    str,   # e.g. "National Day Rally 2024"
  "page_ref":    str,   # e.g. "p.3" or "" if not available
  "style_tags":  list,  # e.g. ["formal", "rule-of-three", "budget-speech"]
  "text":        str    # the chunk text itself
}
```

**Speaker field:** Always store speaker as metadata on public corpus chunks. The reranker can use this to boost results from a specific speaker if requested — but the default retrieval does NOT filter by speaker. Store it, don't filter on it unless explicitly instructed.

---

## 6. Citation schema

Every retrieved chunk used in generation must produce a citation object. Use this schema consistently across the codebase:

```python
{
  "id":               int,   # sequential, starting from 1
  "source_name":      str,   # must match an actual file or known source — never hallucinated
  "source_type":      str,   # "public" | "private" | "live"
  "page_ref":         str,   # "p.2" or "" — never None
  "excerpt":          str,   # max 120 chars — never empty
  "confidence_score": float, # 0.0–1.0
  "warning":          bool   # True if confidence_score < 0.65
}
```

UI colour coding: public = sage green, private = violet, live = amber, warning = ⚠ flag.

---

## 7. Build sequence — follow this order strictly

Do not skip phases. Each phase must pass its verify step before the next begins.

### Phase 1 — Public corpus ingestion pipeline

**Prerequisite — corpus files must exist before the verify step:**
- `storage/corpus/pmo/` — ~40 PMO speeches (2018–2025) as PDF or HTML, manually downloaded from `pmo.gov.sg/newsroom`
- `storage/corpus/hansard/` — ~20 Hansard debate excerpts, filtered to budget/fiscal/digital economy topics, downloaded from `parliament.gov.sg/parliamentary-business/official-reports-(parl-debates)`
- The pipeline code can be built before the files exist, but the verify step needs real files

**Build:**
- `chunker.py` — reads PDF (pdfplumber) and HTML (BeautifulSoup) from a given folder, splits at paragraph boundaries, populates the chunk metadata schema above for every chunk
- `style_extractor.py` — takes chunk text, returns style_tags list: classify tone (formal/conversational/solemn), detect rule-of-three, detect parallel construction, detect direct address. Keep simple — keyword heuristics are fine.
- `embedder.py` — takes list of chunk texts, calls OpenAI `text-embedding-3-small` in batches of 100, returns list of embedding vectors
- `run_ingestion.py` — CLI that ties the above together: reads `--source pmo` or `--source hansard`, chunks all files, extracts style features, embeds, builds two FAISS indexes (`style_index` stores style-tagged chunks, `content_index` stores all chunks), saves indexes and a companion `metadata.json` to the respective storage paths

**Verify:** `python ingestion/run_ingestion.py --source pmo` completes without error. Run a test query and confirm top-5 results return with fully-populated metadata (speaker, date, occasion, style_tags).

---

### Phase 2 — Private document session pipeline

**Build:**
- `session/manager.py` — `SessionManager` class with methods: `create_session(session_id: str)`, `get_collection(session_id: str)`, `destroy_session(session_id: str)`. Each session gets its own Chroma in-memory collection. `persist_directory` is never set. Auto-destroy after `SESSION_TIMEOUT_MINUTES` (default 120) of inactivity. Log all create/destroy events.
- `session/ingestor.py` — `DocumentIngestor` class. Accepts file bytes + filename + session_id. Detects file type from extension: `.pdf` → pdfplumber, `.docx` → python-docx, `.txt` → plain read. Chunks the content, tags each chunk with `source_type: "private"`, `source_name: filename`, infers `doc_type` from filename keywords (brief/talking_points/email/stats). Embeds via OpenAI. Loads into the session Chroma collection.

**Verify:** Create a session, ingest a test PDF, query the collection with a sample phrase, print top-3 results with metadata. Confirm `storage/` contains no new files after the cycle. Destroy the session and confirm the collection is gone.

---

### Phase 3 — Retrieval layer

**Build four retrieval modules, all returning chunks in the citation schema format:**

- `retrieval/style_retrieval.py` — query FAISS style index, return top-k chunks, `source_type: "public"`
- `retrieval/content_retrieval.py` — query FAISS content index, return top-k chunks, `source_type: "public"`
- `retrieval/context_retrieval.py` — query Chroma session collection for given `session_id`, return top-k chunks, `source_type: "private"`. Must enforce session isolation — only query the collection belonging to the given session_id.
- `retrieval/tavily_retrieval.py` — call Tavily API with the speech brief as query, return top results as chunks with `source_type: "live"`, `confidence_score: 0.7` default (or use Tavily's returned score if available)
- `retrieval/reranker.py` — takes results from all four streams, deduplicates by text similarity (cosine > 0.95 = duplicate, keep higher-scored), sorts by `confidence_score` descending, returns a single merged list

**Verify:** Given a sample speech brief, all four streams return results. The merged list is correctly attributed (no chunk missing source_name or source_type). Deduplication removes obvious repeats.

---

### Phase 4 — Generation layer

**Build:**
- `generation/prompt_builder.py` — assembles the final prompt from: (a) system prompt encoding SG public service register rules (see Section 8 below), (b) top-3 style exemplars from style retrieval, (c) top-3 precedent structures from content retrieval, (d) all context retrieval chunks, (e) all Tavily results, (f) the speech brief. Total prompt must stay under 8,000 tokens — truncate lower-priority chunks if needed. Priority order: brief > private context > style exemplars > Tavily > precedents.
- `generation/llm.py` — calls `gpt-4o` with the assembled prompt, streams the response, returns the full speech text
- `generation/citations.py` — after generation, maps each paragraph of the speech to its most likely source chunk (keyword overlap), assembles the citations list, sets `warning: True` for `confidence_score < 0.65`

**Output structure** — the generation layer must return this exact dict:
```python
{
  "outline":               list,  # list of section title strings
  "full_speech":           str,   # the complete speech text
  "citations":             list,  # list of citation objects per schema above
  "style_confidence_score": float, # average confidence across public citations
  "word_count":            int
}
```

**Verify:** End-to-end test: speech brief → retrieval → prompt → generation → structured dict. Streaming works. All minimum eval assertions pass on 3 separate test briefs (see Section 9).

---

### Phase 5 — Streamlit UI

Wire Phases 1–4 into `app.py`. Match the three-step flow in `specs/speechcraft_v3.html`:

**Step 1 — Brief form:**
- Speaker dropdown (Minister Lawrence Wong, Minister of State Alvin Tan, Permanent Secretary)
- Event name text input
- Target audience text input
- Speech length select (5 min ~700 words / 10 min ~1,400 words / 20 min ~2,800 words)
- Date input
- Tone selector using `st.radio` (Formal / Conversational / Motivational / Celebratory / Solemn)
- Key messages `st.text_area`

**Step 2 — Document upload:**
- Four `st.file_uploader` widgets: background brief (required), agency talking points, email threads, statistics
- Show upload confirmation for each
- Display knowledge base pills as `st.info` badges: PMO corpus, Hansard, Tavily live search

**Step 3 — Generate and review:**
- Use `st.status()` to show generation progress (retrieval → prompt building → drafting)
- Outline tab: `st.expander` per section, with edit and regenerate buttons
- Full speech tab: `st.markdown` for speech body, `st.sidebar` for citations coloured by source type
- Chat refinement: `st.chat_input` + `st.chat_message`, full speech passed as context on every turn, AI response updates `st.session_state.current_speech`
- Version strip: store each version in `st.session_state.versions` list, `st.selectbox` to switch between versions
- Export: `st.download_button` to download speech as `.txt`

**Verify:** All five progress-bar steps complete end-to-end. Chat refinement updates the speech and adds a new version. Citations render with correct colours and ⚠ flags.

---

## 8. SG public service speech register — prompt builder rules

Embed these rules in the system prompt in `prompt_builder.py`:

- **Register:** Formal but not legalistic. Clear, direct sentences. No bureaucratic padding. Active voice preferred.
- **Structure:** (1) Acknowledge occasion and audience, (2) state intent ("Let me share three things..."), (3) substantive sections with clear signposting, (4) forward-looking close that returns to opening theme.
- **Rhetorical patterns:** Rule of three, parallel construction, deliberate repetition for emphasis, collective call to action at close ("build together", "get this right together").
- **Attribution:** Every factual claim must be traceable to a retrieved source. Do not invent statistics. If a claim cannot be cited, do not include it.
- **Tone calibration:** Budget speeches and major economic forums (SFF, SIFF) = highest register. Industry dinners and community events = slightly warmer, less formal.
- **Avoid:** Superlatives without evidence, overly emotive language, unexplained acronyms, and any claim not in the retrieved context.

*Note: `specs/domain_skill.md` does not exist yet. Derive all register knowledge from the rules above and the sample speech in `specs/speechcraft_v3.html`.*

---

## 9. Evals — run before marking Phase 4 complete

Golden test cases in `tests/evals/` as JSON files with `brief`, `documents`, `assertions` keys.

Minimum assertions for Phase 4 sign-off:
- Output speech contains at least one citation from each uploaded document
- No citation has `source_name` empty or `excerpt` empty
- All citations with `confidence_score < 0.65` have `warning: true`
- Word count within ±15% of requested length
- Output contains no placeholder text ("TBC", "[NAME]", "INSERT STATISTIC")
- No hallucinated source names — every `source_name` must match an actual file or known corpus document

Run with: `python tests/run_evals.py`

---

## 10. What "done" looks like

| Phase | Done when... |
|---|---|
| Phase 1 | `run_ingestion.py` completes, test query returns top-5 with full metadata |
| Phase 2 | Upload → query → destroy cycle works, nothing written to disk |
| Phase 3 | All four retrieval streams return attributed chunks, reranker merges correctly |
| Phase 4 | End-to-end produces valid output dict, streaming works, all evals pass |
| Phase 5 | Full UI flow works, chat refinement updates speech with version tracking |

---

## 11. Hard rules — never do these

- Set `persist_directory` on any Chroma collection
- Use LlamaParse — use pdfplumber instead
- Call any API other than `api.openai.com` and `api.tavily.com`
- Query one session's Chroma collection from another session
- Hardcode API keys — always load from `.env`
- Use `print()` for logging — use Python `logging` module
- Skip a phase verify step before moving to the next phase
- Commit anything under `storage/` or `.env` to git

---

## 12. Environment variables (.env)

```
OPENAI_API_KEY=           # gpt-4o + text-embedding-3-small
TAVILY_API_KEY=           # live web search
SESSION_TIMEOUT_MINUTES=120
FAISS_STYLE_INDEX_PATH=storage/style_index/
FAISS_CONTENT_INDEX_PATH=storage/content_index/
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o
LOG_LEVEL=INFO
```