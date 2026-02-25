import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

from comparator.collector import collect_gpt_responses
from comparator.pipeline import run_pipeline

ROOT = Path(__file__).parent
MANUAL_STORE_PATH = ROOT / "data" / "manual_batch_template_10.json"


def _validate_manual(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for i, item in enumerate(items[:3]):
        model = item.get("model", f"gpt-manual-{i+1}")
        text = item.get("text", "").strip()
        if text:
            out.append({"model": model, "text": text})
    return out


def _normalize_question(text: str) -> str:
    return " ".join(text.strip().split())


def _sanitize_json_like_text(raw: str) -> str:
    # Recover common invalid JSON pattern: raw newlines inside quoted strings.
    out = []
    in_string = False
    escaped = False
    for ch in raw:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch in ("\n", "\r"):
                out.append("\\n")
                continue
            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
            escaped = False

    return "".join(out)


def _load_store_json() -> Dict[str, Any]:
    raw = MANUAL_STORE_PATH.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = _sanitize_json_like_text(raw)
        return json.loads(fixed)


def _extract_from_json_like_store(question: str) -> List[Dict[str, str]]:
    if not MANUAL_STORE_PATH.exists():
        return []
    raw = MANUAL_STORE_PATH.read_text(encoding="utf-8")
    target = _normalize_question(question)

    q_pat = re.compile(r'"question"\s*:\s*"([^"]+)"')
    q_matches = list(q_pat.finditer(raw))
    for idx, qm in enumerate(q_matches):
        if _normalize_question(qm.group(1)) != target:
            continue

        start = qm.start()
        end = q_matches[idx + 1].start() if idx + 1 < len(q_matches) else len(raw)
        block = raw[start:end]

        models = []
        marker = '{"model":'
        pos = 0
        while True:
            m_start = block.find(marker, pos)
            if m_start == -1:
                break
            t_key = '"text": "'
            tk = block.find(t_key, m_start)
            if tk == -1:
                break
            model_m = re.search(r'"model"\s*:\s*"([^"]+)"', block[m_start:tk])
            if not model_m:
                pos = tk + len(t_key)
                continue
            model = model_m.group(1)

            text_start = tk + len(t_key)
            # End marker: quote + } before next model/item boundary.
            text_end = block.find('"}', text_start)
            while text_end != -1:
                tail = block[text_end:text_end + 80]
                if tail.startswith('"},') or tail.startswith('"}\n') or tail.startswith('"}\r'):
                    break
                text_end = block.find('"}', text_end + 2)
            if text_end == -1:
                break

            text = block[text_start:text_end]
            models.append({"model": model, "text": text})
            pos = text_end + 2

        return _validate_manual(models)

    return []


def _load_manual_responses_from_store(question: str) -> List[Dict[str, str]]:
    if not MANUAL_STORE_PATH.exists():
        return []

    try:
        data = _load_store_json()
    except Exception:
        return _extract_from_json_like_store(question)

    target = _normalize_question(question)
    for item in data.get("batch_items", []):
        q = _normalize_question(str(item.get("question", "")))
        if q != target:
            continue
        return _validate_manual(item.get("manual_responses", []))

    return []


def _analyze_single(question: str, mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if mode == "manual":
        manual = payload.get("manual_responses", [])
        responses = _validate_manual(manual)
        if len(responses) < 3:
            responses = _load_manual_responses_from_store(question)
        if len(responses) < 3:
            raise ValueError("manual mode requires 3 non-empty responses (or matching prefilled JSON data)")
    else:
        responses = collect_gpt_responses(question)
    use_nli = payload.get("use_nli")
    if use_nli is None:
        env = os.getenv("ENABLE_NLI", "1").strip().lower()
        use_nli = env in {"1", "true", "yes", "on"}
    else:
        use_nli = bool(use_nli)
    nli_backend = str(payload.get("nli_backend") or os.getenv("NLI_BACKEND", "local_transformers"))
    return run_pipeline(
        {"question": question, "responses": responses},
        use_nli=use_nli,
        nli_backend=nli_backend,
    )


def analyze_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(payload.get("mode", "gpt"))

    if mode == "manual-batch":
        items = payload.get("batch_items", [])
        if not isinstance(items, list) or not items:
            raise ValueError("manual-batch mode requires non-empty batch_items")

        results = []
        for idx, item in enumerate(items, start=1):
            question = str(item.get("question", "")).strip()
            if not question:
                raise ValueError(f"batch item #{idx} has empty question")
            single_payload = {"manual_responses": item.get("manual_responses", [])}
            result = _analyze_single(question, "manual", single_payload)
            results.append(result)

        return {
            "batch": True,
            "count": len(results),
            "results": results,
        }

    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")

    result = _analyze_single(question, mode, payload)
    return {
        "batch": False,
        "result": result,
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = (ROOT / "web" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/health":
            self._send_json(200, {"ok": True})
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self._send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        try:
            output = analyze_payload(payload)
            self._send_json(200, output)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            self._send_json(500, {"error": str(e)})


def run(host: str = "127.0.0.1", port: int = 8787) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
