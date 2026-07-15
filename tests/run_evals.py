"""
Eval runner — runs all JSON eval cases in tests/evals/ and checks minimum assertions.
Usage: python tests/run_evals.py

Assertions checked per CLAUDE.md §9:
  no_empty_source_name       — no citation has source_name == ""
  no_empty_excerpt           — no citation has excerpt == ""
  warning_flag_for_low_conf  — all citations with score < 0.65 have warning=True
  word_count_within_15pct    — speech word count within ±15% of requested length
  no_placeholder_text        — speech contains none of: TBC, [NAME], INSERT STATISTIC
  no_hallucinated_sources    — every citation source_name matches a known file or "Tavily"
"""
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from generation.pipeline import run_pipeline
from session.manager import SessionManager
from session.ingestor import DocumentIngestor

EVALS_DIR = Path(__file__).parent / "evals"

_PLACEHOLDER_PATTERNS = [
    r"\bTBC\b", r"\[NAME\]", r"INSERT STATISTIC", r"\bPLACEHOLDER\b",
]
_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE)

_LENGTH_MAP = {
    "700":   700,
    "1,400": 1400,
    "2,800": 2800,
}


def _target_word_count(length_str: str) -> int | None:
    for key, words in _LENGTH_MAP.items():
        if key in length_str:
            return words
    return None


def _check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"  ✓  {name}")
    else:
        print(f"  ✗  {name}" + (f" — {detail}" if detail else ""))
    return condition


def run_eval(eval_path: Path) -> bool:
    with open(eval_path, "r", encoding="utf-8") as f:
        case = json.load(f)

    brief      = case["brief"]
    doc_paths  = case.get("documents", [])
    assertions = set(case.get("assertions", []))

    print(f"\n{'═' * 60}")
    print(f"Eval: {eval_path.name}")
    print(f"Brief: {brief['speaker']} @ {brief['event_name']}")
    print("═" * 60)

    session_manager = SessionManager()
    session_id      = str(uuid.uuid4())
    session_manager.create_session(session_id)

    # Ingest any test documents
    if doc_paths:
        ingestor = DocumentIngestor(session_manager)
        for doc_path_str in doc_paths:
            doc_path = Path(doc_path_str)
            if not doc_path.exists():
                logger.warning("Test document not found: %s — skipping", doc_path)
                continue
            with open(doc_path, "rb") as f:
                ingestor.ingest(f.read(), doc_path.name, session_id)

    # Run pipeline
    try:
        output = run_pipeline(brief, session_id, session_manager)
    except Exception:
        logger.exception("Pipeline failed for %s", eval_path.name)
        session_manager.destroy_session(session_id)
        return False

    session_manager.destroy_session(session_id)

    speech     = output["full_speech"]
    citations  = output["citations"]
    word_count = output["word_count"]

    print(f"\nOutput: {word_count} words, {len(citations)} citations\n")

    results: list[bool] = []

    if "no_empty_source_name" in assertions:
        bad = [c for c in citations if not c.get("source_name")]
        results.append(_check(
            "no_empty_source_name",
            not bad,
            f"{len(bad)} citation(s) with empty source_name" if bad else "",
        ))

    if "no_empty_excerpt" in assertions:
        bad = [c for c in citations if not c.get("excerpt")]
        results.append(_check(
            "no_empty_excerpt",
            not bad,
            f"{len(bad)} citation(s) with empty excerpt" if bad else "",
        ))

    if "warning_flag_for_low_confidence" in assertions:
        bad = [c for c in citations
               if c.get("confidence_score", 1.0) < 0.65 and not c.get("warning")]
        results.append(_check(
            "warning_flag_for_low_confidence",
            not bad,
            f"{len(bad)} low-confidence citation(s) missing warning flag" if bad else "",
        ))

    if "word_count_within_15pct" in assertions:
        target = _target_word_count(brief.get("length", ""))
        if target:
            lo, hi = int(target * 0.85), int(target * 1.15)
            ok = lo <= word_count <= hi
            results.append(_check(
                "word_count_within_15pct",
                ok,
                f"{word_count} words (target {target}, allowed {lo}–{hi})" if not ok else "",
            ))

    if "no_placeholder_text" in assertions:
        match = _PLACEHOLDER_RE.search(speech)
        results.append(_check(
            "no_placeholder_text",
            not match,
            f"Found: '{match.group()}'" if match else "",
        ))

    if "no_hallucinated_sources" in assertions:
        known_files = {Path(p).name for p in doc_paths}
        bad = [
            c for c in citations
            if not c["source_name"].startswith("Tavily")
            and c["source_name"] not in known_files
            and not Path(f"storage/corpus/pmo/{c['source_name']}").exists()
            and not Path(f"storage/corpus/hansard/{c['source_name']}").exists()
        ]
        results.append(_check(
            "no_hallucinated_sources",
            not bad,
            f"Unrecognised: {[c['source_name'] for c in bad]}" if bad else "",
        ))

    passed = all(results)
    print(f"\n{'PASS' if passed else 'FAIL'} — {eval_path.name}")
    return passed


def main() -> None:
    eval_files = sorted(EVALS_DIR.glob("*.json"))
    if not eval_files:
        print("No eval files found in tests/evals/")
        sys.exit(1)

    results = [run_eval(f) for f in eval_files]
    total, passed = len(results), sum(results)
    print(f"\n{'═'*60}")
    print(f"Results: {passed}/{total} evals passed")
    print("═" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
