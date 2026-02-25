from typing import Dict, List

from .embedder import cosine, tokenize

NEG_CUES = {
    "not", "no", "never", "cannot", "can't", "isn't", "won't", "shouldn't",
    "don't", "doesn't", "without", "avoid", "against", "hardly",
    "아니", "않", "못", "없", "비추", "지양", "위험", "리스크", "어렵",
}

OPPOSITES = [
    ("increase", "decrease"),
    ("better", "worse"),
    ("safe", "risky"),
    ("recommended", "discouraged"),
    ("effective", "ineffective"),
    ("benefit", "harm"),
    ("원격", "대면"),
    ("원격", "온사이트"),
    ("완전", "하이브리드"),
    ("권장", "비권장"),
    ("추천", "비추천"),
]

RECO_POS = {
    "recommend", "recommended", "prefer", "best", "effective", "benefit",
    "권장", "추천", "유리", "장점", "효율", "확장", "절감", "좋",
}

RECO_NEG = {
    "discourage", "avoid", "risk", "risky", "harm", "ineffective",
    "비추천", "비권장", "위험", "리스크", "문제", "어려움", "저하", "느리", "불리",
}


def polarity_score(tokens: List[str]) -> float:
    neg_hits = sum(1 for t in tokens if t in NEG_CUES)
    return -1.0 if neg_hits > 0 else 1.0


def recommendation_polarity(tokens: List[str]) -> int:
    pos = sum(1 for t in tokens if t in RECO_POS)
    neg = sum(1 for t in tokens if t in RECO_NEG)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def contradiction_score(text_a: str, text_b: str, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    ta = tokenize(text_a)
    tb = tokenize(text_b)

    sim = cosine(vec_a, vec_b)
    if sim < 0.08:
        return 0.0

    pa = polarity_score(ta)
    pb = polarity_score(tb)
    ra = recommendation_polarity(ta)
    rb = recommendation_polarity(tb)
    score = 0.0

    if pa != pb:
        score += 0.45
    if ra != 0 and rb != 0 and ra != rb:
        score += 0.35

    sa = set(ta)
    sb = set(tb)
    for p, q in OPPOSITES:
        if (p in sa and q in sb) or (q in sa and p in sb):
            score += 0.2

    score += min(0.25, sim * 0.35)
    return min(1.0, score)
