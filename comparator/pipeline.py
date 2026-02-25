from dataclasses import dataclass
import re
from typing import Any, Dict, List, Set, Tuple

from .clusterer import cluster_by_similarity
from .conflict import contradiction_score
from .embedder import build_tfidf_vectors, cosine
from .nli import contradiction_with_optional_nli, nli_status
from .projection import normalize_points, vector_to_xy
from .segmenter import segment_claims
from .stance import classify_stance
from .topic import classify_topics


STANCE_CONFLICT_PAIRS = {
    ("POSITIVE", "NEGATIVE"),
    ("NEGATIVE", "POSITIVE"),
    ("POSITIVE", "CONDITIONAL"),
    ("CONDITIONAL", "POSITIVE"),
    ("NEGATIVE", "CONDITIONAL"),
    ("CONDITIONAL", "NEGATIVE"),
}

SIM_CONFLICT_PAIRS = {
    ("POSITIVE", "NEGATIVE"),
    ("NEGATIVE", "POSITIVE"),
}

UNIQUE_META_PATTERNS = [
    "정리해", "결론부터", "말씀드리면", "요약", "핵심은", "체크리스트", "가이드",
    "in short", "summary", "conclusion",
]

UNIQUE_ACTION_PATTERNS = [
    "권장", "추천", "하자", "해야", "채택", "도입", "기본값", "운영", "설계",
    "recommend", "should", "adopt", "set", "use",
]

UNIQUE_GENERIC_PATTERNS = [
    "상황에 따라", "케이스 바이 케이스", "정답은 없다", "균형이 중요", "중요하다",
    "depends", "it depends",
]


@dataclass
class Claim:
    id: int
    model: str
    text: str
    topic_labels: List[str]
    stance_label: str
    vector: Dict[str, float]
    response_idx: int
    sentence_index: int
    plot_cluster_id: int = -1


def _representative_text(members: List[Claim]) -> str:
    if not members:
        return ""
    return max(members, key=lambda m: sum(cosine(m.vector, x.vector) for x in members)).text


def _select_representative_claims(members: List[Claim], k: int = 3) -> List[Claim]:
    if not members:
        return []
    ranked = sorted(
        members,
        key=lambda m: sum(cosine(m.vector, x.vector) for x in members),
        reverse=True,
    )
    selected: List[Claim] = []
    used_models: Set[str] = set()
    for m in ranked:
        if m.model in used_models:
            continue
        selected.append(m)
        used_models.add(m.model)
        if len(selected) >= k:
            return selected
    for m in ranked:
        if m in selected:
            continue
        selected.append(m)
        if len(selected) >= k:
            return selected
    return selected


def _bucket_item(topic: str, stance: str, members: List[Claim], rep_k: int = 3) -> Dict[str, Any]:
    model_set = sorted(set(m.model for m in members))
    reps = _select_representative_claims(members, k=rep_k)
    return {
        "topic_label": topic,
        "stance_label": stance,
        "representative": _representative_text(members),
        "representative_claims": [{"id": m.id, "model": m.model, "text": m.text} for m in reps],
        "models": model_set,
        "model_count": len(model_set),
        "claims": [{"id": m.id, "model": m.model, "text": m.text} for m in members],
    }


def _cluster_topic_claims(topic_claims: List[Claim], threshold: float, merge_threshold: float = 0.82) -> Dict[int, List[Claim]]:
    if not topic_claims:
        return {}

    stance_groups: Dict[str, List[Claim]] = {}
    for c in topic_claims:
        stance_groups.setdefault(c.stance_label, []).append(c)

    out: Dict[int, List[Claim]] = {}
    next_id = 0
    for _, members in stance_groups.items():
        local_ids = cluster_by_similarity([c.vector for c in members], threshold=threshold)
        tmp: Dict[int, List[Claim]] = {}
        for c, lid in zip(members, local_ids):
            tmp.setdefault(lid, []).append(c)

        cluster_list = list(tmp.values())
        merged = [False] * len(cluster_list)
        for i in range(len(cluster_list)):
            if merged[i]:
                continue
            base = list(cluster_list[i])
            rep_i = _select_representative_claims(base, k=1)[0]
            for j in range(i + 1, len(cluster_list)):
                if merged[j]:
                    continue
                rep_j = _select_representative_claims(cluster_list[j], k=1)[0]
                if cosine(rep_i.vector, rep_j.vector) >= merge_threshold:
                    base.extend(cluster_list[j])
                    merged[j] = True
            out[next_id] = base
            next_id += 1
    return out


def _topic_overlap(a: Claim, b: Claim) -> List[str]:
    inter = sorted(set(a.topic_labels) & set(b.topic_labels))
    return inter


def _is_conflict_noise_text(text: str) -> bool:
    t = text.lower().strip()
    noise_tokens = ["1️⃣", "2️⃣", "3️⃣", "pros", "cons", "정리", "결론부터", "체크리스트", "핵심 요약"]
    return any(tok in t for tok in noise_tokens)


def _is_meta_like_text(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in UNIQUE_META_PATTERNS)


def _is_noisy_unique_text(text: str) -> bool:
    t = text.lower()
    if len(t) > 240:
        return True
    if "구분" in t and "유리한 경우" in t:
        return True
    if t.count(":") >= 3:
        return True
    return False


def _specificity_bonus(text: str) -> float:
    t = text.lower()
    score = 0.0
    if re.search(r"\d", t):
        score += 0.6
    if re.search(r"주\s*\d+\s*[~-]\s*\d+\s*회|0\s*[-~]\s*5\s*명|5\s*[-~]\s*15\s*명|15\s*명\s*이상|sprint", t):
        score += 0.8
    return score


def _action_bonus(text: str) -> float:
    t = text.lower()
    return 0.7 if any(p in t for p in UNIQUE_ACTION_PATTERNS) else 0.0


def _generic_penalty(text: str) -> float:
    t = text.lower()
    return 0.8 if any(p in t for p in UNIQUE_GENERIC_PATTERNS) else 0.0


def _pick_unique_highlights(
    global_unique: Dict[str, List[Dict[str, Any]]],
    claim_by_id: Dict[int, Claim],
    common_claims: List[Claim],
    top_k: int = 3,
    novelty_threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()

    for model, entries in global_unique.items():
        for entry in entries:
            topic = entry.get("topic_label", "META")
            if topic == "META":
                continue
            stance = entry.get("stance_label", "META")
            for c in entry.get("claims", []):
                cid = int(c.get("id", -1))
                claim = claim_by_id.get(cid)
                if claim is None:
                    continue
                text = claim.text.strip()
                if len(text) < 20:
                    continue
                if _is_meta_like_text(text):
                    continue
                if _is_noisy_unique_text(text):
                    continue
                if text in seen_texts:
                    continue

                max_common_sim = 0.0
                for cc in common_claims:
                    max_common_sim = max(max_common_sim, cosine(claim.vector, cc.vector))
                if max_common_sim >= novelty_threshold:
                    continue

                novelty_bonus = max(0.0, 1.0 - max_common_sim) * 1.2
                score = _specificity_bonus(text) + _action_bonus(text) + novelty_bonus - _generic_penalty(text)
                candidates.append(
                    {
                        "model": model,
                        "topic": topic,
                        "stance": stance,
                        "claim_id": f"c_{claim.id:03d}",
                        "text": text,
                        "score": round(score, 3),
                        "max_common_similarity": round(max_common_sim, 3),
                        "supporting_models": [model],
                        "supporting_count": 1,
                        "tag": "UNIQUE",
                    }
                )
                seen_texts.add(text)

    # model diversity first: max 1 per model
    best_per_model: Dict[str, Dict[str, Any]] = {}
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if c["model"] not in best_per_model:
            best_per_model[c["model"]] = c

    picked = sorted(best_per_model.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return picked


def _select_topic_diverse(entries: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    if not entries:
        return []
    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        by_topic.setdefault(e.get("topic_label", "기타"), []).append(e)

    selected: List[Dict[str, Any]] = []
    used_text: Set[str] = set()
    # First pass: 1 per topic.
    for topic in sorted(by_topic.keys()):
        for e in by_topic[topic]:
            txt = e.get("representative", "")
            if txt in used_text:
                continue
            selected.append(e)
            used_text.add(txt)
            break
        if len(selected) >= k:
            return selected[:k]
    # Second pass: fill the rest by original order.
    for e in entries:
        txt = e.get("representative", "")
        if txt in used_text:
            continue
        selected.append(e)
        used_text.add(txt)
        if len(selected) >= k:
            return selected[:k]
    return selected[:k]


def run_pipeline(
    payload: Dict[str, Any],
    cluster_threshold: float = 0.1,
    conflict_threshold: float = 0.55,
    use_nli: bool = True,
    nli_backend: str = "local_transformers",
    representative_k: int = 3,
    unique_highlight_k: int = 3,
    unique_novelty_threshold: float = 0.8,
    summary_top_k: int = 3,
) -> Dict[str, Any]:
    question = payload.get("question", "")
    responses = payload.get("responses", [])
    model_order = []
    for r in responses:
        m = str(r.get("model", "unknown"))
        if m not in model_order:
            model_order.append(m)

    raw_claims: List[Dict[str, Any]] = []
    for ridx, resp in enumerate(responses):
        model = resp.get("model", "unknown")
        text = resp.get("text", "")
        for sidx, c in enumerate(segment_claims(text)):
            raw_claims.append({"model": model, "text": c, "response_idx": ridx, "sentence_index": sidx})

    texts = [x["text"] for x in raw_claims]
    vectors = build_tfidf_vectors(texts)

    claims: List[Claim] = []
    for i, item in enumerate(raw_claims):
        txt = item["text"]
        topics = classify_topics(txt, max_topics_per_claim=2)
        claims.append(
            Claim(
                id=i,
                model=item["model"],
                text=txt,
                topic_labels=topics,
                stance_label=classify_stance(txt),
                vector=vectors[i] if i < len(vectors) else {},
                response_idx=int(item.get("response_idx", -1)),
                sentence_index=int(item.get("sentence_index", -1)),
            )
        )

    # Use primary topic for aggregation to avoid duplicate output across topics.
    topic_claims_map: Dict[str, List[Claim]] = {}
    for c in claims:
        primary = c.topic_labels[0] if c.topic_labels else "META"
        topic_claims_map.setdefault(primary, []).append(c)

    nli_state = nli_status(requested=bool(use_nli), backend=nli_backend)
    nli_requested = bool(nli_state["requested"])
    nli_effective = bool(nli_state["effective"])
    nli_reason = str(nli_state["reason"])
    effective_conf_threshold = conflict_threshold if nli_effective else 0.55

    print(
        f"[pipeline] NLI: {'ON' if nli_requested else 'OFF'} "
        f"(effective={'ON' if nli_effective else 'OFF'}, backend={nli_backend}, reason={nli_reason})"
    )

    # Topic-local clustering for plotting/inspector.
    plot_cluster_counter = 0
    for topic, topic_claims in sorted(topic_claims_map.items(), key=lambda kv: len(kv[1]), reverse=True):
        topic_threshold = cluster_threshold * (0.9 if topic in {"문화", "속도", "비용", "채용"} else 1.0)
        clusters = _cluster_topic_claims(topic_claims, threshold=topic_threshold)
        for local_id, members in clusters.items():
            gid = plot_cluster_counter + local_id
            for m in members:
                if m.plot_cluster_id == -1:
                    m.plot_cluster_id = gid
        plot_cluster_counter += max(len(clusters), 1)

    # Global conflict generation with topic overlap.
    candidate_pairs_count = 0
    global_conflicts: List[Dict[str, Any]] = []
    weak_conflicts: List[Dict[str, Any]] = []
    pair_seen: Set[Tuple[int, int]] = set()

    for i, a in enumerate(claims):
        for j in range(i + 1, len(claims)):
            b = claims[j]
            if a.model == b.model:
                continue
            if a.stance_label == "META" or b.stance_label == "META":
                continue
            pair_set = STANCE_CONFLICT_PAIRS if nli_effective else SIM_CONFLICT_PAIRS
            if (a.stance_label, b.stance_label) not in pair_set:
                continue
            overlap = _topic_overlap(a, b)
            if not overlap:
                continue
            if not nli_effective:
                a_primary = a.topic_labels[0] if a.topic_labels else "META"
                b_primary = b.topic_labels[0] if b.topic_labels else "META"
                if a_primary != b_primary:
                    continue
                if _is_conflict_noise_text(a.text) or _is_conflict_noise_text(b.text):
                    continue

            key = (min(a.id, b.id), max(a.id, b.id))
            if key in pair_seen:
                continue
            pair_seen.add(key)
            candidate_pairs_count += 1

            heuristic = contradiction_score(a.text, b.text, a.vector, b.vector)
            score_payload = contradiction_with_optional_nli(
                heuristic,
                a.text,
                b.text,
                use_nli=nli_effective,
                backend=nli_backend,
            )
            score = float(score_payload["score"])

            if {a.stance_label, b.stance_label} == {"POSITIVE", "NEGATIVE"}:
                score = min(1.0, score + 0.12)
            elif "CONDITIONAL" in {a.stance_label, b.stance_label}:
                score = min(1.0, score + 0.1)

            topic_label = overlap[0]
            rec = {
                "topic_label": topic_label,
                "topics": overlap,
                "claim_id_a": a.id,
                "claim_id_b": b.id,
                "model_a": a.model,
                "model_b": b.model,
                "stance_a": a.stance_label,
                "stance_b": b.stance_label,
                "claim_a": a.text,
                "claim_b": b.text,
                "score": round(score, 3),
                "score_mode": score_payload["mode"],
                "mode": score_payload["mode"],
                "nli_contradiction": (
                    round(float(score_payload["nli_contradiction"]), 3)
                    if score_payload.get("nli_contradiction") is not None
                    else None
                ),
            }

            if score >= effective_conf_threshold:
                global_conflicts.append(rec)
            elif score >= (0.4 if nli_effective else 0.5):
                weak_conflicts.append(rec)

    global_conflicts.sort(key=lambda x: x["score"], reverse=True)
    weak_conflicts.sort(key=lambda x: x["score"], reverse=True)
    print(
        f"[pipeline] candidate_pairs={candidate_pairs_count} "
        f"conflicts={len(global_conflicts)} weak_conflicts={len(weak_conflicts)}"
    )

    topic_sections: List[Dict[str, Any]] = []
    common_pros_compat: List[Dict[str, Any]] = []
    common_cons_compat: List[Dict[str, Any]] = []
    global_unique: Dict[str, List[Dict[str, Any]]] = {}
    global_unique_seen: Dict[str, Set[Tuple[str, str, int]]] = {}
    common_claim_ids: Set[int] = set()

    for topic, topic_claims in sorted(topic_claims_map.items(), key=lambda kv: len(kv[1]), reverse=True):
        buckets: Dict[Tuple[str, str], List[Claim]] = {}
        for c in topic_claims:
            buckets.setdefault((topic, c.stance_label), []).append(c)

        common_positive: List[Dict[str, Any]] = []
        common_negative: List[Dict[str, Any]] = []
        common_conditional: List[Dict[str, Any]] = []
        meta_statements: List[Dict[str, Any]] = []
        unique_by_model: Dict[str, List[Dict[str, Any]]] = {}

        for (_, stance), members in buckets.items():
            item = _bucket_item(topic, stance, members, rep_k=representative_k)
            distinct_models = set(m.model for m in members)

            if stance == "META":
                meta_statements.append(item)
                continue

            if len(distinct_models) >= 2:
                for m in members:
                    common_claim_ids.add(m.id)
                if stance == "POSITIVE":
                    common_positive.append(item)
                    for rep in item["representative_claims"]:
                        common_pros_compat.append(
                            {
                                "topic_label": topic,
                                "representative": rep["text"],
                                "model": rep["model"],
                                "models": item["models"],
                                "supporting_models": item["models"],
                                "supporting_count": len(item["models"]),
                                "tag": "COMMON",
                            }
                        )
                elif stance == "NEGATIVE":
                    common_negative.append(item)
                    for rep in item["representative_claims"]:
                        common_cons_compat.append(
                            {
                                "topic_label": topic,
                                "representative": rep["text"],
                                "model": rep["model"],
                                "models": item["models"],
                                "supporting_models": item["models"],
                                "supporting_count": len(item["models"]),
                                "tag": "COMMON",
                            }
                        )
                elif stance == "CONDITIONAL":
                    common_conditional.append(item)
            else:
                only_model = next(iter(distinct_models))
                unique_by_model.setdefault(only_model, []).append(item)
                key = (topic, stance, members[0].id)
                seen = global_unique_seen.setdefault(only_model, set())
                if key not in seen:
                    seen.add(key)
                    global_unique.setdefault(only_model, []).append(item)

        topic_conflicts = [c for c in global_conflicts if topic in c.get("topics", [])]
        topic_weak_conflicts = [c for c in weak_conflicts if topic in c.get("topics", [])]

        topic_sections.append(
            {
                "topic_label": topic,
                "stats": {
                    "claim_count": len(topic_claims),
                    "cluster_count": len(set(m.plot_cluster_id for m in topic_claims)),
                    "conflict_count": len(topic_conflicts),
                },
                "common_positive": common_positive,
                "common_negative": common_negative,
                "common_conditional": common_conditional,
                "conflicts": topic_conflicts,
                "weak_conflicts": topic_weak_conflicts,
                "unique": unique_by_model,
                "meta_statements": meta_statements,
            }
        )

    coords = normalize_points([vector_to_xy(c.vector) for c in claims])
    claim_points = []
    for claim, (x, y) in zip(claims, coords):
        top_terms = sorted(claim.vector.items(), key=lambda kv: kv[1], reverse=True)[:6]
        claim_points.append(
            {
                "id": claim.id,
                "model": claim.model,
                "text": claim.text,
                "topic_labels": claim.topic_labels,
                "stance_label": claim.stance_label,
                "cluster_id": claim.plot_cluster_id,
                "x": round(x, 4),
                "y": round(y, 4),
                "top_terms": [{"term": t, "weight": round(w, 4)} for t, w in top_terms],
            }
        )

    cluster_members: Dict[int, List[Claim]] = {}
    for c in claims:
        cid = c.plot_cluster_id if c.plot_cluster_id >= 0 else c.id
        cluster_members.setdefault(cid, []).append(c)
    clusters = []
    for cid, members in sorted(cluster_members.items(), key=lambda kv: kv[0]):
        rep = max(members, key=lambda m: sum(cosine(m.vector, x.vector) for x in members))
        primary_topics = {}
        stances = {}
        for m in members:
            t = m.topic_labels[0] if m.topic_labels else "META"
            primary_topics[t] = primary_topics.get(t, 0) + 1
            s = m.stance_label
            stances[s] = stances.get(s, 0) + 1
        topic = max(primary_topics.items(), key=lambda kv: kv[1])[0]
        stance = max(stances.items(), key=lambda kv: kv[1])[0]
        clusters.append(
            {
                "cluster_id": f"k_{cid}",
                "topic": topic,
                "stance": stance,
                "member_claim_ids": [f"c_{m.id:03d}" for m in members],
                "representative_claim_id": f"c_{rep.id:03d}",
            }
        )

    stance_counts: Dict[str, int] = {"POSITIVE": 0, "NEGATIVE": 0, "CONDITIONAL": 0, "META": 0}
    for c in claims:
        stance_counts[c.stance_label] = stance_counts.get(c.stance_label, 0) + 1

    stance_axis = ["POSITIVE", "NEGATIVE", "CONDITIONAL"]
    topics_axis = sorted(topic_claims_map.keys(), key=lambda t: (t == "META", t))
    bucket_map: Dict[Tuple[str, str], List[Claim]] = {}
    for c in claims:
        topic = c.topic_labels[0] if c.topic_labels else "META"
        if c.stance_label in stance_axis:
            bucket_map.setdefault((topic, c.stance_label), []).append(c)

    buckets = []
    for topic in topics_axis:
        for stance in stance_axis:
            members = bucket_map.get((topic, stance), [])
            if not members:
                continue
            models = sorted(set(m.model for m in members))
            reps = _select_representative_claims(members, k=representative_k)
            rep_claims = [{"text": c.text, "model": c.model, "id": f"c_{c.id:03d}"} for c in reps]

            ccount = 0
            for cf in global_conflicts:
                if topic not in cf.get("topics", []):
                    continue
                if cf["stance_a"] == stance or cf["stance_b"] == stance:
                    ccount += 1

            buckets.append(
                {
                    "topic": topic,
                    "stance": stance,
                    "claim_ids": [f"c_{c.id:03d}" for c in members],
                    "distinct_models": models,
                    "representative_claims": rep_claims,
                    "is_common": len(models) >= 2,
                    "conflict_count": ccount,
                }
            )

    consensus = common_pros_compat + common_cons_compat
    claim_by_id = {c.id: c for c in claims}
    common_claims = [claim_by_id[cid] for cid in sorted(common_claim_ids) if cid in claim_by_id]
    unique_highlights = _pick_unique_highlights(
        global_unique=global_unique,
        claim_by_id=claim_by_id,
        common_claims=common_claims,
        top_k=unique_highlight_k,
        novelty_threshold=unique_novelty_threshold,
    )
    summary_common_pros = _select_topic_diverse(common_pros_compat, k=summary_top_k)
    summary_common_cons = _select_topic_diverse(common_cons_compat, k=summary_top_k)

    # Claim-level provenance/support tags based on primary (topic, stance) bucket.
    primary_bucket_models: Dict[Tuple[str, str], List[str]] = {}
    for (topic, stance), members in bucket_map.items():
        distinct = sorted(set(m.model for m in members), key=lambda x: model_order.index(x) if x in model_order else 10_000)
        primary_bucket_models[(topic, stance)] = distinct

    claim_conflict_meta: Dict[int, Dict[str, Any]] = {}
    for cf in global_conflicts:
        for cid in (cf["claim_id_a"], cf["claim_id_b"]):
            prev = claim_conflict_meta.get(cid)
            if prev is None or float(cf["score"]) > float(prev["score"]):
                claim_conflict_meta[cid] = {
                    "score": float(cf["score"]),
                    "mode": cf.get("score_mode", "SIM"),
                }

    return {
        "question": question,
        "responses": responses,
        "topics_axis": topics_axis,
        "stances": stance_axis,
        "buckets": buckets,
        "claims": [
            {
                "id": f"c_{c.id:03d}",
                "display_id": c.id + 1,
                "text": c.text,
                "model": c.model,
                "primary_topic": c.topic_labels[0] if c.topic_labels else "META",
                "topic_labels": c.topic_labels,
                "stance": c.stance_label,
                "cluster_id": f"k_{(c.plot_cluster_id if c.plot_cluster_id >= 0 else c.id)}",
                "emb2d": [p["x"], p["y"]],
                "supporting_models": primary_bucket_models.get(
                    ((c.topic_labels[0] if c.topic_labels else "META"), c.stance_label),
                    [c.model],
                ),
                "supporting_count": len(
                    primary_bucket_models.get(
                        ((c.topic_labels[0] if c.topic_labels else "META"), c.stance_label),
                        [c.model],
                    )
                ),
                "tag": (
                    "COMMON"
                    if len(
                        primary_bucket_models.get(
                            ((c.topic_labels[0] if c.topic_labels else "META"), c.stance_label),
                            [c.model],
                        )
                    ) >= 2
                    else "UNIQUE"
                ),
                "conflict_badge": (
                    {
                        "is_conflict": True,
                        "score": round(claim_conflict_meta[c.id]["score"], 3),
                        "mode": claim_conflict_meta[c.id]["mode"],
                    }
                    if c.id in claim_conflict_meta
                    else {"is_conflict": False}
                ),
                "source": {"run_id": f"r_{c.response_idx+1:02d}", "sentence_index": c.sentence_index},
            }
            for c, p in zip(claims, claim_points)
        ],
        "clusters": clusters,
        "visual_conflicts": [
            {
                "claim_id_a": f"c_{c.get('claim_id_a'):03d}",
                "claim_id_b": f"c_{c.get('claim_id_b'):03d}",
                "topics": c.get("topics", [c.get("topic_label")]),
                "score": c.get("score"),
                "stance_pair": [c.get("stance_a"), c.get("stance_b")],
                "score_mode": c.get("score_mode", "SIM"),
            }
            for c in global_conflicts
        ],
        "runs": [
            {"run_id": f"r_{idx+1:02d}", "model": (r.get("model", "unknown")), "raw_response": r.get("text", "")}
            for idx, r in enumerate(responses)
        ],
        "topics": topic_sections,
        "stats": {
            "response_count": len(responses),
            "claim_count": len(claims),
            "topic_count": len(topic_claims_map),
            "cluster_count": sum(t["stats"]["cluster_count"] for t in topic_sections),
            "positive_claim_count": stance_counts.get("POSITIVE", 0),
            "negative_claim_count": stance_counts.get("NEGATIVE", 0),
            "conditional_claim_count": stance_counts.get("CONDITIONAL", 0),
            "meta_claim_count": stance_counts.get("META", 0),
            "common_pro_count": len(common_pros_compat),
            "common_con_count": len(common_cons_compat),
            "summary_common_pro_count": len(summary_common_pros),
            "summary_common_con_count": len(summary_common_cons),
            "unique_highlight_count": len(unique_highlights),
            "candidate_pairs_count": candidate_pairs_count,
            "conflict_count": len(global_conflicts),
            "weak_conflict_count": len(weak_conflicts),
            "nli_requested": nli_requested,
            "nli_effective": nli_effective,
            "nli_reason": nli_reason,
            "nli_backend": nli_backend,
            "conflict_threshold_effective": effective_conf_threshold,
        },
        "common_pros": common_pros_compat,
        "common_cons": common_cons_compat,
        "summary_common_pros": summary_common_pros,
        "summary_common_cons": summary_common_cons,
        "conflicts": global_conflicts,
        "weak_conflicts": weak_conflicts,
        "unique": global_unique,
        "unique_highlights": unique_highlights,
        "consensus": consensus,
        "claim_points": claim_points,
    }
