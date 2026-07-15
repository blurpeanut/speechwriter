"""
SpeechCraft corpus ingestion CLI.

Usage:
    python ingestion/run_ingestion.py --source pmo
    python ingestion/run_ingestion.py --source hansard
    python ingestion/run_ingestion.py --source all
    python ingestion/run_ingestion.py --query "fintech regulation"
    python ingestion/run_ingestion.py --query "budget fiscal policy" --top-k 3
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running from project root: python ingestion/run_ingestion.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from ingestion.chunker import chunk_folder
from ingestion.style_extractor import tag_chunks
from ingestion.embedder import embed_texts, EMBEDDING_DIM

STYLE_INDEX_PATH   = Path(os.getenv("FAISS_STYLE_INDEX_PATH",   "storage/style_index"))
CONTENT_INDEX_PATH = Path(os.getenv("FAISS_CONTENT_INDEX_PATH", "storage/content_index"))
CORPUS_PMO         = Path("storage/corpus/pmo")
CORPUS_HANSARD     = Path("storage/corpus/hansard")


# ── Index build/load helpers ───────────────────────────────────────────────────

def build_and_save_index(chunks: list[dict], index_dir: Path, meta_filename: str) -> None:
    """Embed chunks, build FAISS IndexFlatIP, and save index + metadata JSON."""
    index_dir.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks for %s …", len(texts), index_dir.name)

    vectors = embed_texts(texts)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    index_file = index_dir / "index.faiss"
    faiss.write_index(index, str(index_file))
    logger.info("Saved FAISS index: %s (%d vectors)", index_file, index.ntotal)

    meta_file = index_dir / meta_filename
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Saved metadata: %s", meta_file)


def load_index(index_dir: Path, meta_filename: str):
    """Load FAISS index and companion metadata. Returns (index, list[dict])."""
    index_file = index_dir / "index.faiss"
    meta_file  = index_dir / meta_filename
    if not index_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"Index not found at {index_dir}. Run ingestion first."
        )
    index = faiss.read_index(str(index_file))
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# ── Ingestion ──────────────────────────────────────────────────────────────────

def ingest_source(source: str) -> list[dict]:
    """Chunk, tag, and return all corpus chunks for the given source."""
    sources = ["pmo", "hansard"] if source == "all" else [source]
    all_chunks: list[dict] = []

    for src in sources:
        folder   = CORPUS_PMO     if src == "pmo" else CORPUS_HANSARD
        ministry = "PMO"          if src == "pmo" else "Hansard"

        if not folder.exists():
            logger.error(
                "Corpus folder not found: %s — populate it before running verify.", folder
            )
            continue

        logger.info("── Ingesting %s from %s", src.upper(), folder)
        chunks = chunk_folder(folder, ministry=ministry)
        tag_chunks(chunks)
        all_chunks.extend(chunks)
        logger.info("Source %s: %d chunks produced", src.upper(), len(chunks))

    return all_chunks


# ── Test query ─────────────────────────────────────────────────────────────────

def run_query(query: str, top_k: int = 5) -> None:
    """Query both FAISS indexes and print top-k results for manual verification."""
    q_vec = embed_texts([query])   # shape (1, DIM), already normalised

    for label, index_dir, meta_file in [
        ("STYLE",   STYLE_INDEX_PATH,   "style_metadata.json"),
        ("CONTENT", CONTENT_INDEX_PATH, "content_metadata.json"),
    ]:
        try:
            index, metadata = load_index(index_dir, meta_file)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            continue

        scores, ids = index.search(q_vec, top_k)
        print(f"\n{'═' * 64}")
        print(f"[{label} INDEX]  query: '{query}'")
        print("═" * 64)

        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            if idx < 0:
                continue
            c = metadata[idx]
            print(
                f"\n  #{rank}  score={score:.4f}\n"
                f"  source   : {c['source_name']}\n"
                f"  speaker  : {c['speaker'] or '—'}   ministry: {c['ministry']}"
                f"   date: {c['date'] or '—'}\n"
                f"  occasion : {c['occasion'] or '—'}\n"
                f"  page_ref : {c['page_ref'] or '—'}\n"
                f"  tags     : {c['style_tags']}\n"
                f"  excerpt  : {c['text'][:220]} …"
            )


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpeechCraft public corpus ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", choices=["pmo", "hansard", "all"],
        help="Corpus source to ingest and index",
    )
    parser.add_argument(
        "--query", type=str,
        help="Run a test query against the existing FAISS indexes",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return for --query (default: 5)",
    )
    args = parser.parse_args()

    if args.query:
        run_query(args.query, top_k=args.top_k)
        return

    if not args.source:
        parser.print_help()
        sys.exit(1)

    chunks = ingest_source(args.source)
    if not chunks:
        logger.error(
            "No chunks were produced. "
            "Ensure corpus files exist in storage/corpus/pmo/ and/or storage/corpus/hansard/ "
            "before running --source."
        )
        sys.exit(1)

    logger.info("Total chunks across all sources: %d", len(chunks))
    logger.info("Building FAISS indexes …")

    # Both indexes receive the same chunks; retrieval modules use them differently.
    build_and_save_index(chunks, STYLE_INDEX_PATH,   "style_metadata.json")
    build_and_save_index(chunks, CONTENT_INDEX_PATH, "content_metadata.json")

    logger.info("Ingestion complete. Run --query to verify.")


if __name__ == "__main__":
    main()
