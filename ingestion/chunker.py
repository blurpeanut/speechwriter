"""
Paragraph-level chunker for public corpus files (PDF and HTML).
Produces chunk dicts matching the schema in CLAUDE.md §5.
"""
import re
import uuid
import logging
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MIN_CHUNK_LEN = 60    # characters — skip boilerplate fragments
MAX_CHUNK_LEN = 800   # characters — split oversized blocks at sentence boundaries

# Titles stripped from the start of speaker names before storage.
_TITLES_RE = re.compile(
    r"^(Prime Minister|Deputy Prime Minister|DPM|Senior Minister|"
    r"Minister of State|Minister for|Minister of|Minister|"
    r"Permanent Secretary|MAS Managing Director|Managing Director|"
    r"Secretary-General)[,\s]+",
    re.IGNORECASE,
)


def normalise_speaker(name: str) -> str:
    """Strip common SG titles from the start of a speaker name.

    'Minister Lawrence Wong' → 'Lawrence Wong'
    Idempotent — safe to call on already-normalised names.
    """
    result = name.strip()
    while True:
        new = _TITLES_RE.sub("", result).strip()
        if new == result:
            break
        result = new
    return result


# ── Schema factory ────────────────────────────────────────────────────────────

def make_chunk(
    text: str,
    source_name: str,
    source_type: str = "public",
    speaker: str = "",
    ministry: str = "",
    date: str = "",
    occasion: str = "",
    page_ref: str = "",
    style_tags: list = None,
) -> dict:
    """Return a chunk dict matching CLAUDE.md §5 schema."""
    return {
        "chunk_id":    str(uuid.uuid4()),
        "source_name": source_name,
        "source_type": source_type,
        "speaker":     normalise_speaker(speaker),
        "ministry":    ministry,
        "date":        date,
        "occasion":    occasion,
        "page_ref":    page_ref,
        "style_tags":  style_tags or [],
        "text":        text.strip(),
    }


# ── Filename metadata extraction ───────────────────────────────────────────────

_PMO_SPEAKER_RE = re.compile(
    r"^(PM|DPM|SM|Senior Minister|Minister of State|Minister|Leader of the House)"
    r"[\s,]+([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})",
    re.IGNORECASE,
)
_PMO_OCCASION_RE = re.compile(r"\bat the (.+?)(?:\s*_|$)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _parse_pmo_filename(stem: str) -> dict:
    """
    Extract speaker / occasion / year from PMO-style filenames.
    E.g. 'PM Lawrence Wong at the 10th World Cities Summit _ Prime Minister's Office Singapore'
    """
    clean = stem.replace("_", " ").strip()
    meta = {"speaker": "", "ministry": "PMO", "date": "", "occasion": ""}

    m = _PMO_SPEAKER_RE.match(clean)
    if m:
        meta["speaker"] = m.group(2).strip()

    om = _PMO_OCCASION_RE.search(clean)
    if om:
        meta["occasion"] = om.group(1).strip()

    ym = _YEAR_RE.search(clean)
    if ym:
        meta["date"] = ym.group(1)      # year-only ISO prefix; full date not in filename

    return meta


def _parse_hansard_filename(stem: str) -> dict:
    """
    Hansard files are named by debate topic — speaker and date are not in the filename.
    """
    return {
        "speaker":  "",
        "ministry": "Hansard",
        "date":     "",
        "occasion": stem.replace("_", " ").strip(),
    }


# ── Text splitting ─────────────────────────────────────────────────────────────

def _split_text(text: str) -> list[str]:
    """
    Split raw text into paragraph-sized chunks.
    1. Split on double (or more) newlines.
    2. Collapse whitespace within each block.
    3. Further split blocks longer than MAX_CHUNK_LEN at sentence boundaries.
    """
    raw_blocks = re.split(r"\n{2,}", text)
    result: list[str] = []

    for block in raw_blocks:
        block = re.sub(r"[ \t]+", " ", block).strip()
        if not block:
            continue

        if len(block) <= MAX_CHUNK_LEN:
            result.append(block)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", block)
            current: list[str] = []
            acc = 0
            for sent in sentences:
                current.append(sent)
                acc += len(sent) + 1
                if acc >= MAX_CHUNK_LEN:
                    result.append(" ".join(current))
                    current, acc = [], 0
            if current:
                result.append(" ".join(current))

    return result


# ── Per-file parsers ───────────────────────────────────────────────────────────

def chunk_pdf(path: Path, ministry: str = "PMO") -> list[dict]:
    """Parse a PDF with pdfplumber and return paragraph-level chunks."""
    filename = path.name
    meta = (_parse_hansard_filename(path.stem) if ministry == "Hansard"
            else _parse_pmo_filename(path.stem))
    meta["ministry"] = ministry

    chunks: list[dict] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                for para in _split_text(raw):
                    if len(para) < MIN_CHUNK_LEN:
                        continue
                    chunks.append(make_chunk(
                        text=para,
                        source_name=filename,
                        speaker=meta["speaker"],
                        ministry=meta["ministry"],
                        date=meta["date"],
                        occasion=meta["occasion"],
                        page_ref=f"p.{page_num}",
                    ))
    except Exception:
        logger.exception("Failed to parse PDF: %s", path)

    logger.info("Chunked %s → %d chunks", filename, len(chunks))
    return chunks


def chunk_html(path: Path, ministry: str = "PMO") -> list[dict]:
    """Parse an HTML file with BeautifulSoup and return paragraph-level chunks."""
    filename = path.name
    meta = (_parse_hansard_filename(path.stem) if ministry == "Hansard"
            else _parse_pmo_filename(path.stem))
    meta["ministry"] = ministry

    chunks: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "lxml")
        for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()

        paras = soup.find_all("p")
        texts = (
            [p.get_text(separator=" ", strip=True) for p in paras]
            if paras
            else _split_text(soup.get_text(separator="\n"))
        )
        for para in texts:
            if len(para) < MIN_CHUNK_LEN:
                continue
            chunks.append(make_chunk(
                text=para,
                source_name=filename,
                speaker=meta["speaker"],
                ministry=meta["ministry"],
                date=meta["date"],
                occasion=meta["occasion"],
                page_ref="",
            ))
    except Exception:
        logger.exception("Failed to parse HTML: %s", path)

    logger.info("Chunked %s → %d chunks", filename, len(chunks))
    return chunks


# ── Folder entry point ─────────────────────────────────────────────────────────

def chunk_folder(folder: Path, ministry: str = "PMO") -> list[dict]:
    """Chunk all PDF and HTML files in a directory. Returns flat list of chunks."""
    all_chunks: list[dict] = []
    pdfs  = sorted(folder.glob("*.pdf"))
    htmls = sorted(folder.glob("*.html")) + sorted(folder.glob("*.htm"))

    logger.info("Folder %s: %d PDFs, %d HTML files", folder, len(pdfs), len(htmls))
    for p in pdfs:
        all_chunks.extend(chunk_pdf(p, ministry=ministry))
    for h in htmls:
        all_chunks.extend(chunk_html(h, ministry=ministry))

    logger.info("Total chunks from %s: %d", folder, len(all_chunks))
    return all_chunks
