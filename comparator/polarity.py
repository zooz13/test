from typing import List

PRO_KEYWORDS = {
    "절감", "확장", "장점", "유리", "효율", "글로벌", "확보", "생산성", "유연", "자율",
    "benefit", "advantage", "efficient", "efficiency", "global", "save", "savings", "flexibility",
    "improve", "improved", "productivity",
}

CON_KEYWORDS = {
    "위험", "속도 저하", "문제", "어려움", "정렬", "리스크", "저하", "불리", "부족",
    "온보딩", "단절", "비효율", "느리", "갈등", "혼선",
    "risk", "risky", "problem", "issue", "slow", "slower", "misalignment", "friction",
    "difficult", "hard", "drop", "decline", "harm",
}


def _contains_any(text: str, keywords: List[str]) -> int:
    count = 0
    for kw in keywords:
        if kw in text:
            count += 1
    return count


def classify_polarity(text: str) -> str:
    t = text.lower()
    pro_score = _contains_any(t, list(PRO_KEYWORDS))
    con_score = _contains_any(t, list(CON_KEYWORDS))

    if pro_score > con_score and pro_score > 0:
        return "PRO"
    if con_score > pro_score and con_score > 0:
        return "CON"
    return "NEUTRAL"
