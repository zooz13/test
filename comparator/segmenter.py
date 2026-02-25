import re
from typing import List

CONNECTORS = [
    "however",
    "but",
    "still",
    "therefore",
    "although",
    "while",
    "yet",
    "그러나",
    "하지만",
    "다만",
    "반면",
    "또한",
    "따라서",
    "그래서",
]


STRATEGY_STAGE_PATTERNS = [
    r"\b0\s*[-~]\s*5\s*명\b",
    r"\b5\s*[-~]\s*15\s*명\b",
    r"\b15\s*명\s*이상\b",
    r"\b0\s*to\s*5\b",
    r"\b5\s*to\s*15\b",
    r"\b15\+\b",
]


def _split_by_connectors(sentence: str) -> List[str]:
    chunks = [sentence]
    for c in CONNECTORS:
        next_chunks = []
        for chunk in chunks:
            parts = re.split(rf"\\b{re.escape(c)}\\b", chunk, flags=re.IGNORECASE)
            next_chunks.extend(parts)
        chunks = next_chunks
    return [c.strip(" ,;:-") for c in chunks if c.strip(" ,;:-")]


def _split_strategy_stages(sentence: str) -> List[str]:
    lowered = sentence.lower()
    if not any(re.search(p, lowered) for p in STRATEGY_STAGE_PATTERNS):
        return [sentence]
    # Stage-style lines often use ':' ';' '/' separators. Keep small clean chunks.
    parts = re.split(r"[;/]\s+|(?<=\))\s+(?=\d)|\s+(?=\d+\s*[-~]\s*\d+\s*명)|\s+(?=\d+\s*명\s*이상)", sentence)
    out = [p.strip(" ,;:-") for p in parts if len(p.strip(" ,;:-")) >= 8]
    return out if len(out) >= 2 else [sentence]


def segment_claims(text: str) -> List[str]:
    clean = re.sub(r"\s+", " ", text.strip())
    if not clean:
        return []

    # Support both English and Korean punctuation endings.
    sentences = re.split(r"(?<=[.!?。])\s+|\n+", clean)
    claims: List[str] = []

    for sent in sentences:
        sent = sent.strip(" \n\t")
        if not sent:
            continue
        stage_parts = _split_strategy_stages(sent)
        for part in stage_parts:
            parts = _split_by_connectors(part)
            claims.extend(parts if parts else [part])

    return [c for c in claims if len(c) >= 8]
