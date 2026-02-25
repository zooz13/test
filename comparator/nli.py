import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple


@lru_cache(maxsize=1)
def _load_classifier(backend: str) -> Tuple[Optional[Any], str]:
    if backend == "disabled_similarity_only":
        return None, "disabled_by_config"
    if backend == "api_nli":
        return None, "api_nli_not_configured"
    if backend != "local_transformers":
        return None, "unknown_backend"

    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        return None, "import_error"

    model_name = os.getenv("NLI_MODEL", "joeddav/xlm-roberta-large-xnli")
    try:
        return pipeline("text-classification", model=model_name), "ok"
    except Exception as e:
        msg = str(e).lower()
        if "out of memory" in msg or "oom" in msg:
            return None, "oom"
        if "not found" in msg or "404" in msg:
            return None, "model_not_found"
        if "token" in msg and "hf" in msg:
            return None, "auth_required"
        return None, "model_load_error"


def _normalize_label(label: str) -> str:
    x = (label or "").lower()
    if "contrad" in x:
        return "contradiction"
    if "entail" in x:
        return "entailment"
    if "neutral" in x:
        return "neutral"
    return x


def nli_status(requested: bool, backend: str = "local_transformers") -> Dict[str, Any]:
    if not requested:
        return {
            "requested": False,
            "effective": False,
            "backend": backend,
            "reason": "disabled_by_request",
        }
    clf, reason = _load_classifier(backend)
    return {
        "requested": True,
        "effective": clf is not None,
        "backend": backend,
        "reason": reason,
    }


def nli_scores(text_a: str, text_b: str, backend: str = "local_transformers") -> Optional[Dict[str, float]]:
    clf, _ = _load_classifier(backend)
    if clf is None:
        return None

    pairs = [{"text": text_a, "text_pair": text_b}]
    try:
        out = clf(pairs, truncation=True)
    except Exception:
        return None

    # pipeline may return list[dict] or list[list[dict]] depending on model config
    if not out:
        return None
    if isinstance(out[0], list):
        items = out[0]
    else:
        items = out

    scores = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}
    for item in items:
        label = _normalize_label(str(item.get("label", "")))
        score = float(item.get("score", 0.0))
        if label in scores:
            scores[label] = max(scores[label], score)

    return scores


def contradiction_with_optional_nli(
    heuristic_score: float,
    text_a: str,
    text_b: str,
    use_nli: bool = False,
    backend: str = "local_transformers",
) -> Dict[str, Any]:
    if not use_nli:
        return {"score": heuristic_score, "mode": "SIM", "nli_contradiction": None}

    scores = nli_scores(text_a, text_b, backend=backend)
    if not scores:
        return {"score": heuristic_score, "mode": "SIM", "nli_contradiction": None}

    # Blend heuristic and NLI contradiction score for stability.
    blended = min(1.0, (0.55 * heuristic_score) + (0.45 * scores["contradiction"]))
    return {"score": blended, "mode": "NLI", "nli_contradiction": scores["contradiction"]}
