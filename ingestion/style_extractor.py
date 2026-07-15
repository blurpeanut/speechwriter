"""
Style feature extraction for public corpus chunks.
Uses keyword heuristics — no ML models required.
Tags: tone (formal / conversational / solemn), budget-speech,
      rule-of-three, parallel-construction, direct-address.
"""
import re

# ── Marker lists ───────────────────────────────────────────────────────────────

_FORMAL_MARKERS = [
    "ladies and gentlemen", "distinguished guests", "permit me to",
    "let me share", "let me turn to", "i am pleased to",
    "i wish to", "allow me to", "colleagues,", "mr speaker",
    "in conclusion,", "fundamentally,", "we must", "it is important",
    "i am confident", "i am glad", "honourable",
]

_CONVERSATIONAL_MARKERS = [
    "you know,", "i think,", "actually,", "so,", "well,", "okay,",
    "to be honest", "frankly,", "right?", "isn't it", "you see,",
    "kind of", "sort of",
]

_SOLEMN_MARKERS = [
    "we mourn", "in memory of", "sacrifice", "tragic", "condolences",
    "passed away", "loss of life", "solemn", "tribute to", "fallen",
    "grief", "remember those who", "honour their",
]

_BUDGET_MARKERS = [
    "budget", "fiscal", "revenue", "expenditure", "surplus", "deficit",
    "tax", "gst", "iras", "ministry of finance", "spending",
    "$", "billion", "million", "allocation", "disbursement",
]

# ── Compiled patterns ──────────────────────────────────────────────────────────

_RULE_OF_THREE_RE = re.compile(
    r"\b(first[,\.]?.{5,120}?second[,\.]?.{5,120}?third)\b",
    re.IGNORECASE | re.DOTALL,
)

_PARALLEL_PATTERNS = [
    re.compile(r"not only.{3,80}but also", re.IGNORECASE | re.DOTALL),
    re.compile(r"on the one hand.{3,120}on the other", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bwhether.{3,80}whether\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bif.{3,60}if.{3,60}if\b", re.IGNORECASE | re.DOTALL),
]

_DIRECT_ADDRESS_RE = re.compile(
    r"\b(ladies and gentlemen|dear friends|my fellow|colleagues,|"
    r"distinguished guests|honourable members|minister[s]?,)\b",
    re.IGNORECASE,
)


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_style_tags(text: str) -> list[str]:
    """Return a list of style tags for a single chunk of text."""
    tags: list[str] = []
    lower = text.lower()

    formal_hits  = sum(1 for m in _FORMAL_MARKERS        if m in lower)
    conv_hits    = sum(1 for m in _CONVERSATIONAL_MARKERS if m in lower)
    solemn_hits  = sum(1 for m in _SOLEMN_MARKERS         if m in lower)

    if solemn_hits >= 2:
        tags.append("solemn")
    elif conv_hits > formal_hits:
        tags.append("conversational")
    else:
        tags.append("formal")

    if sum(1 for m in _BUDGET_MARKERS if m in lower) >= 3:
        tags.append("budget-speech")

    if _RULE_OF_THREE_RE.search(text):
        tags.append("rule-of-three")

    if any(p.search(text) for p in _PARALLEL_PATTERNS):
        tags.append("parallel-construction")

    if _DIRECT_ADDRESS_RE.search(text):
        tags.append("direct-address")

    return tags


def tag_chunks(chunks: list[dict]) -> list[dict]:
    """Add style_tags in-place to every chunk that has no tags yet. Returns same list."""
    for chunk in chunks:
        if not chunk.get("style_tags"):
            chunk["style_tags"] = extract_style_tags(chunk["text"])
    return chunks
