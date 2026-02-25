import math
import re
from typing import Dict, List

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "in", "on", "for", "with", "by", "as", "at",
    "that", "this", "it", "its", "from", "can", "may", "might", "should", "would",
    "will", "do", "does", "did", "than", "then", "so", "if", "into", "about",
    # Korean stopwords (minimal prototype set)
    "그리고", "또한", "하지만", "그러나", "때문", "정도", "경우", "대한", "에서", "으로",
    "이다", "있다", "없다", "한다", "한다면", "한다는", "수", "더", "좀",
}

NORMALIZE_MAP = {
    "startups": "startup",
    "startup's": "startup",
    "companies": "company",
    "recruit": "hire",
    "recruiting": "hire",
    "hiring": "hire",
    "talent": "hire",
    "remote-first": "remote",
    "remotely": "remote",
    "collaboration": "collaborate",
    "collaborative": "collaborate",
    "cohesion": "alignment",
    "onboarding": "alignment",
    "seed-stage": "early-stage",
    # Korean normalization (prototype)
    "원격근무": "원격",
    "재택근무": "원격",
    "하이브리드": "혼합",
    "온사이트": "오프라인",
    "오프사이트": "원격",
    "협업": "협력",
    "소통": "커뮤니케이션",
}


def _light_stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    # Lightweight Korean postposition stripping.
    for suf in ["에서", "으로", "에게", "들은", "들은", "들은", "들은", "는", "은", "이", "가", "을", "를", "도", "만"]:
        if len(token) > 2 and token.endswith(suf):
            return token[:-len(suf)]
    return token


def tokenize(text: str) -> List[str]:
    # English words + Hangul blocks.
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*|[가-힣]{2,}", text.lower())
    out = []
    for tok in tokens:
        tok = NORMALIZE_MAP.get(tok, tok)
        tok = _light_stem(tok)
        if tok not in STOPWORDS and len(tok) > 1:
            out.append(tok)
    return out


def build_tfidf_vectors(texts: List[str]) -> List[Dict[str, float]]:
    docs = [tokenize(t) for t in texts]
    n = len(docs)
    if n == 0:
        return []

    df: Dict[str, int] = {}
    for doc in docs:
        for tok in set(doc):
            df[tok] = df.get(tok, 0) + 1

    idf = {tok: math.log((1 + n) / (1 + freq)) + 1.0 for tok, freq in df.items()}

    vectors: List[Dict[str, float]] = []
    for doc in docs:
        tf: Dict[str, float] = {}
        for tok in doc:
            tf[tok] = tf.get(tok, 0.0) + 1.0
        if doc:
            length = float(len(doc))
            for tok in tf:
                tf[tok] = (tf[tok] / length) * idf.get(tok, 0.0)
        norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
        vectors.append({k: v / norm for k, v in tf.items()})

    return vectors


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(val * b.get(tok, 0.0) for tok, val in a.items())
