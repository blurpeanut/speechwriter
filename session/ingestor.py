"""
Private document ingestor.

Accepts file bytes + filename + session_id.
Parses in memory (never written to disk), chunks, embeds via OpenAI,
and loads into the session's Chroma in-memory collection.

Supported formats: .pdf (pdfplumber), .docx (python-docx), .txt (plain read).
Document type is inferred from filename keywords.
"""
import io
import logging
import uuid
from pathlib import Path

import pdfplumber
import docx

from ingestion.chunker import make_chunk, _split_text, MIN_CHUNK_LEN
from ingestion.embedder import embed_texts
from session.manager import SessionManager

logger = logging.getLogger(__name__)

_DOC_TYPE_KEYWORDS: dict[str, list[str]] = {
    "brief":          ["brief", "background", "background_brief", "context"],
    "talking_points": ["talking", "tp_", "talking_points", "keypoints", "key_points"],
    "email":          ["email", "mail", "thread", "correspondence"],
    "stats":          ["stat", "data", "figure", "chart", "table", "numbers"],
}


def _infer_doc_type(filename: str) -> str:
    lower = filename.lower()
    for doc_type, keywords in _DOC_TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return doc_type
    return "brief"   # sensible default for unknown files


def _parse_pdf_bytes(data: bytes) -> list[str]:
    """Extract paragraph text from a PDF given as raw bytes."""
    paragraphs: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            paragraphs.extend(_split_text(raw))
    return paragraphs


def _parse_docx_bytes(data: bytes) -> list[str]:
    """Extract paragraph text from a DOCX given as raw bytes."""
    doc = docx.Document(io.BytesIO(data))
    paragraphs: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _parse_txt_bytes(data: bytes) -> list[str]:
    """Extract paragraph text from plain text bytes."""
    text = data.decode("utf-8", errors="ignore")
    return _split_text(text)


class DocumentIngestor:
    """
    Parses and embeds private documents into a session's Chroma collection.

    Documents are processed entirely in memory.
    Private embeddings are NEVER written to the FAISS disk indexes.
    """

    def __init__(self, session_manager: SessionManager) -> None:
        self._manager = session_manager

    def ingest(self, file_bytes: bytes, filename: str, session_id: str) -> int:
        """
        Parse, chunk, embed, and load a document into the session collection.
        Returns the number of chunks loaded.

        Raises:
            KeyError: if session_id has no active session.
            ValueError: if the file extension is unsupported.
        """
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            paragraphs = _parse_pdf_bytes(file_bytes)
        elif ext == ".docx":
            paragraphs = _parse_docx_bytes(file_bytes)
        elif ext == ".txt":
            paragraphs = _parse_txt_bytes(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Accepted: .pdf, .docx, .txt")

        doc_type = _infer_doc_type(filename)
        chunks = [
            make_chunk(
                text=para,
                source_name=filename,
                source_type="private",
                occasion=doc_type,   # re-use occasion field to carry doc_type
            )
            for para in paragraphs
            if len(para) >= MIN_CHUNK_LEN
        ]

        if not chunks:
            logger.warning("No usable chunks extracted from %s", filename)
            return 0

        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)

        collection = self._manager.get_collection(session_id)
        collection.add(
            ids        = [c["chunk_id"] for c in chunks],
            embeddings = vectors.tolist(),
            documents  = texts,
            metadatas  = [
                {k: v for k, v in c.items() if k not in ("text", "chunk_id", "style_tags")}
                for c in chunks
            ],
        )

        logger.info(
            "Ingested %s → %d chunks into session %s (doc_type: %s)",
            filename, len(chunks), session_id, doc_type,
        )
        return len(chunks)
