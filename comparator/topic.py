from typing import Dict, List
import re

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "비용": ["임대료", "관리비", "절감", "고정비", "비용", "burn", "runway", "cost"],
    "채용": ["인재", "채용", "글로벌", "지역 제약", "지역", "talent", "hire", "hiring"],
    "문화": [
        "문화", "신뢰", "결속", "전우애", "멘토링", "온보딩", "주니어", "피드백", "관계", "소통",
        "커뮤니케이션", "단절감", "협업", "브레인스토밍", "화이트보드", "우연한 대화",
        "onboarding", "mentoring", "trust", "cohesion", "communication",
    ],
    "속도": [
        "속도", "의사결정", "pmf", "피벗", "탐색", "빠른", "slow", "alignment", "실행", "동기화",
        "release", "cadence", "deployment", "daily", "ship",
    ],
    "생산성": ["생산성", "집중", "자율성", "자율", "효율", "성과", "productivity", "focus"],
    "전략": [
        "하이브리드", "단계", "조건부", "조건", "전략", "기본값", "기본", "초기", "권장", "추천",
        "policy", "hybrid", "운영", "default", "strategy",
    ],
}

META_KEYWORDS = [
    "정리", "결론", "핵심", "요약", "체크리스트", "표", "가이드", "말씀드리면", "드릴게요",
    "in short", "summary", "conclusion",
]


def _score_topic(text: str, keywords: List[str]) -> int:
    score = 0
    for kw in keywords:
        if kw.lower() in text:
            score += 1
    return score


def classify_topics(text: str, max_topics_per_claim: int = 2) -> List[str]:
    t = text.lower()
    # Stage strategy blocks should stay in strategy (optionally with one secondary topic).
    strategy_stage = bool(
        re.search(r"0\s*[-~]\s*5\s*명|5\s*[-~]\s*15\s*명|15\s*명\s*이상|0\s*to\s*5|5\s*to\s*15|15\+", t)
    )
    scored = []
    for topic, kws in TOPIC_KEYWORDS.items():
        s = _score_topic(t, kws)
        if s > 0:
            scored.append((topic, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    if strategy_stage:
        secondary = [topic for topic, _ in scored if topic != "전략"][:1]
        return ["전략"] + secondary
    selected = [topic for topic, _ in scored[:max_topics_per_claim]]
    if selected:
        return selected

    meta_score = _score_topic(t, META_KEYWORDS)
    if meta_score > 0:
        return ["META"]
    # Non-meta fallback: avoid showing classification-failure like "기타"
    return ["전략"]


def classify_topic(text: str) -> str:
    return classify_topics(text, max_topics_per_claim=1)[0]
