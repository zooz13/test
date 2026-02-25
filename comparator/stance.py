from typing import List
import re

POSITIVE_KWS = {
    "장점", "유리", "효율", "절감", "확장", "권장", "추천", "도움", "강점",
    "benefit", "advantage", "improve", "effective", "recommended", "good",
}

NEGATIVE_KWS = {
    "단점", "위험", "리스크", "문제", "어려움", "어렵", "난이도", "저하", "불리", "부족", "느리", "단절", "지연",
    "harm", "risk", "problem", "issue", "difficult", "worse", "slow",
}

CONDITIONAL_KWS = {
    "조건", "경우", "단계", "상황", "depends", "if", "when", "only", "가급적", "현실적",
    "하이브리드", "부분", "전제",
}

META_KWS = {
    "결론", "요약", "정리", "체크리스트", "전략", "핵심",
    "in short", "summary", "conclusion",
}

POSITIVE_PATTERNS = [
    r"\bshould\b",
    r"\brecommend(ed)?\b",
    r"\badopt\b",
    r"\bworth\b",
    r"권장",
    r"추천",
    r"채택",
]

NEGATIVE_PATTERNS = [
    r"\bshould\s+not\b",
    r"\bnot\s+recommended\b",
    r"\bdo\s+not\b",
    r"\bdon't\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bavoid\b",
    r"\bdiscourage(d)?\b",
    r"비추천",
    r"비권장",
    r"권장하지",
    r"추천하지",
    r"하지\s*말",
]

CONDITIONAL_PATTERNS = [
    r"\bif\b",
    r"\bwhen\b",
    r"\bonly if\b",
    r"\bdepends?\b",
    r"\bunless\b",
    r"\bwithout\b",
    r"조건",
    r"경우",
    r"상황",
    r"단계",
    r"하이브리드",
    r"기본값",
    r"현실적인",
    r"전략",
]


def _count_hits(text: str, kws: List[str]) -> int:
    return sum(1 for k in kws if k in text)


def classify_stance(text: str) -> str:
    t = text.lower()

    neg_pat = sum(1 for p in NEGATIVE_PATTERNS if re.search(p, t))
    pos_pat = sum(1 for p in POSITIVE_PATTERNS if re.search(p, t))
    cond_pat = sum(1 for p in CONDITIONAL_PATTERNS if re.search(p, t))

    # Strategy/proposal language should not be treated as plain positive/negative.
    if re.search(r"하이브리드|기본값|단계|전략|현실적인", t):
        if neg_pat > 0:
            return "NEGATIVE"
        return "CONDITIONAL"

    # High-priority recommendation polarity
    if neg_pat > 0 and neg_pat >= pos_pat:
        return "NEGATIVE"
    if pos_pat > 0 and neg_pat == 0 and cond_pat == 0:
        return "POSITIVE"

    pos = _count_hits(t, list(POSITIVE_KWS))
    neg = _count_hits(t, list(NEGATIVE_KWS))
    cond = _count_hits(t, list(CONDITIONAL_KWS))
    meta = _count_hits(t, list(META_KWS))

    # Keep risks/problems as negative even when condition words co-appear.
    if neg >= 2 and neg >= pos:
        return "NEGATIVE"
    if neg > pos and neg > 0:
        return "NEGATIVE"
    if pos > neg and pos > 0 and cond_pat == 0:
        return "POSITIVE"
    if cond_pat > 0 and (pos > 0 or neg > 0 or pos_pat > 0 or neg_pat > 0):
        return "CONDITIONAL"
    if meta >= 2 and meta >= max(pos, neg, cond):
        return "META"
    if cond > 0 and (pos > 0 or neg > 0):
        return "CONDITIONAL"
    if cond >= 2 and cond >= max(pos, neg):
        return "CONDITIONAL"
    if pos > neg and pos > 0:
        return "POSITIVE"
    if cond > 0:
        return "CONDITIONAL"
    return "META"
