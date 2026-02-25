import argparse
import json
from pathlib import Path

from comparator.pipeline import run_pipeline


def print_result(result: dict, top_k: int) -> None:
    print(f"Question: {result['question']}")
    print("=" * 80)
    stats = result["stats"]
    print(
        "Stats: "
        f"responses={stats['response_count']}, "
        f"claims={stats['claim_count']}, "
        f"topics={stats.get('topic_count', 0)}, "
        f"positive={stats.get('positive_claim_count', 0)}, "
        f"negative={stats.get('negative_claim_count', 0)}, "
        f"conditional={stats.get('conditional_claim_count', 0)}, "
        f"clusters={stats['cluster_count']}, "
        f"common_pros={stats.get('common_pro_count', 0)}, "
        f"common_cons={stats.get('common_con_count', 0)}, "
        f"candidate_pairs={stats.get('candidate_pairs_count', 0)}, "
        f"conflicts={stats['conflict_count']}, "
        f"weak={stats.get('weak_conflict_count', 0)}, "
        f"nli_requested={'on' if stats.get('nli_requested', False) else 'off'}, "
        f"nli_effective={'on' if stats.get('nli_effective', False) else 'off'}, "
        f"reason={stats.get('nli_reason', '-')}"
    )
    print(
        "Common lengths: "
        f"pros={len(result.get('common_pros', []))}, "
        f"cons={len(result.get('common_cons', []))}"
    )

    print("\n[Topics]")
    topics = result.get("topics", [])[:top_k]
    if not topics:
        print("(none)")
    for t in topics:
        print(f"- TOPIC: {t['topic_label']} (claims={t['stats']['claim_count']}, conflicts={t['stats']['conflict_count']})")
        if t["topic_label"] == "META":
            meta_items = t.get("meta_statements", [])[:3]
            if not meta_items:
                print("  * Statements: (none)")
            for m in meta_items:
                print(f"  * Statements: {m['representative']}")
            continue

        pos = t.get("common_positive", [])[:2]
        neg = t.get("common_negative", [])[:2]
        cond = t.get("common_conditional", [])[:2]
        if pos:
            for p in pos:
                print(f"  + Common Positive: {p['representative']}")
        if neg:
            for n in neg:
                print(f"  - Common Negative: {n['representative']}")
        if cond:
            for c in cond:
                print(f"  ~ Common Conditional: {c['representative']}")

    print("\n[Conflicts]")
    conflicts = result["conflicts"][:top_k]
    if not conflicts:
        print("(none)")
    for idx, c in enumerate(conflicts, start=1):
        mode = c.get("score_mode", "SIM")
        print(
            f"{idx}. [topic={c['topic_label']} score={c['score']} mode={mode}] "
            f"({c['model_a']}) {c['claim_a'][:120]}  vs  ({c['model_b']}) {c['claim_b'][:120]}"
        )

    print("\n[Weak Conflicts]")
    weak_conflicts = result.get("weak_conflicts", [])[:top_k]
    if not weak_conflicts:
        print("(none)")
    for idx, c in enumerate(weak_conflicts, start=1):
        print(f"{idx}. score={c['score']} | {c['model_a']} vs {c['model_b']}")
        print(f"   A: {c['claim_a']}")
        print(f"   B: {c['claim_b']}")

    print("\n[Unique]")
    if not result["unique"]:
        print("(none)")
    for model, entries in result["unique"].items():
        print(f"- {model}")
        for e in entries[:top_k]:
            print(f"  * {e['representative']}")

    print("\n[Unique Highlights]")
    highlights = result.get("unique_highlights", [])[:top_k]
    if not highlights:
        print("(none)")
    for idx, h in enumerate(highlights, start=1):
        print(f"{idx}. ({h['model']}/{h['topic']}) score={h['score']}")
        print(f"   {h['text']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-LLM perspective comparator prototype")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K items per section")
    parser.add_argument("--cluster-threshold", type=float, default=0.1, help="Similarity threshold for clustering")
    parser.add_argument("--conflict-threshold", type=float, default=0.55, help="Minimum contradiction score")
    parser.add_argument("--use-nli", action="store_true", default=True, help="Enable NLI-based contradiction scoring")
    parser.add_argument("--no-nli", action="store_true", help="Disable NLI and use heuristic-only conflict scoring")
    parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = run_pipeline(
        payload,
        cluster_threshold=args.cluster_threshold,
        conflict_threshold=args.conflict_threshold,
        use_nli=(False if args.no_nli else args.use_nli),
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result, top_k=args.top_k)


if __name__ == "__main__":
    main()
