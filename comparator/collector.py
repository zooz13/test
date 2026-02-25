import json
import os
import urllib.error
import urllib.request
from typing import List, Dict

DEFAULT_GPT_MODELS = ["gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
FALLBACK_MODEL = "gpt-4o-mini"


def _read_models() -> List[str]:
    raw = os.getenv("GPT_MODELS", "").strip()
    if not raw:
        return DEFAULT_GPT_MODELS
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models[:3] if models else DEFAULT_GPT_MODELS


def collect_gpt_responses(question: str, temperature: float = 0.3, max_tokens: int = 600) -> List[Dict[str, str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    responses: List[Dict[str, str]] = []
    for model in _read_models():
        tried = [model]
        used_model = model

        def _call(target_model: str) -> str:
            payload = {
                "model": target_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer concisely with 4-6 clear claims. If Korean question, answer in Korean.",
                    },
                    {"role": "user", "content": question},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        try:
            text = _call(model)
        except urllib.error.HTTPError:
            if model == FALLBACK_MODEL:
                raise
            tried.append(FALLBACK_MODEL)
            used_model = FALLBACK_MODEL
            try:
                text = _call(FALLBACK_MODEL)
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"Model calls failed: {', '.join(tried)} ({e.code})") from e

        responses.append({"model": used_model, "text": text})

    return responses
