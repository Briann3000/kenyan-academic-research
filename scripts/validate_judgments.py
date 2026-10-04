"""Validation script for human relevance judgments.

Audits data/evaluation/judgments/ against data/evaluation/pool.json to verify:
- Exact 496 query-candidate coverage
- Absence of unjudged/missing candidates
- Validity of 0/1/2 labels
- Absence of duplicate or ambiguous judgments
- Completed vs incomplete queries breakdown
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.config import PROJECT_ROOT

POOL_FILE = PROJECT_ROOT / "data" / "evaluation" / "pool.json"
JUDGMENTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "judgments"

BATCH_FILES = [
    "batch1_health.json",
    "batch2_agriculture.json",
    "batch3_economics.json",
    "batch4_climate.json",
]


def validate_all_judgments(verbose: bool = True) -> Dict[str, Any]:
    """Performs full audit of judgment batch files against pool.json."""
    if not POOL_FILE.exists():
        raise FileNotFoundError(f"Pool file not found at {POOL_FILE}")

    with open(POOL_FILE, "r", encoding="utf-8") as f:
        pool_data = json.load(f)

    # Collect expected (query_id, paper_id) pairs
    expected_pairs: Set[Tuple[str, str]] = set()
    query_candidate_map: Dict[str, Set[str]] = {}
    for q_id, q_pool in pool_data.items():
        query_candidate_map[q_id] = set()
        for cand in q_pool["candidates"]:
            p_id = cand["paper_id"]
            expected_pairs.add((q_id, p_id))
            query_candidate_map[q_id].add(p_id)

    total_expected = len(expected_pairs)

    # Inspect batch files
    judged_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    duplicates: List[Tuple[str, str, str]] = []
    invalid_labels: List[Tuple[str, str, Any, str]] = []
    batch_stats: Dict[str, Dict[str, int]] = {}

    for batch_file_name in BATCH_FILES:
        b_path = JUDGMENTS_DIR / batch_file_name
        if not b_path.exists():
            batch_stats[batch_file_name] = {"total": 0, "completed": 0, "missing": 0}
            continue

        with open(b_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)

        items = b_data.get("items", [])
        b_completed = 0
        b_missing = 0

        for item in items:
            q_id = item["query_id"]
            p_id = item["paper_id"]
            rel = item.get("relevance")
            pair = (q_id, p_id)

            if pair in judged_records:
                duplicates.append((q_id, p_id, batch_file_name))

            if rel is None:
                b_missing += 1
            else:
                if rel not in (0, 1, 2) or isinstance(rel, bool) or not isinstance(rel, int):
                    invalid_labels.append((q_id, p_id, rel, batch_file_name))
                else:
                    b_completed += 1
                judged_records[pair] = {
                    "relevance": rel,
                    "notes": item.get("notes", ""),
                    "batch": batch_file_name,
                }

        batch_stats[batch_file_name] = {
            "total": len(items),
            "completed": b_completed,
            "missing": b_missing,
        }

    completed_pairs = set(judged_records.keys())
    missing_pairs = expected_pairs - completed_pairs
    unexpected_pairs = completed_pairs - expected_pairs

    # Per-query completion
    completed_queries = []
    incomplete_queries = []

    for q_id, q_cand_set in sorted(query_candidate_map.items()):
        judged_in_q = sum(1 for p_id in q_cand_set if (q_id, p_id) in judged_records)
        if judged_in_q == len(q_cand_set):
            completed_queries.append(q_id)
        else:
            incomplete_queries.append((q_id, judged_in_q, len(q_cand_set)))

    audit_result = {
        "expected_pairs": total_expected,
        "total_judgments_recorded": len(completed_pairs),
        "missing_judgments": len(missing_pairs),
        "unexpected_judgments": len(unexpected_pairs),
        "duplicate_judgments": len(duplicates),
        "invalid_labels": len(invalid_labels),
        "completed_queries_count": len(completed_queries),
        "incomplete_queries_count": len(incomplete_queries),
        "is_ready_for_qrels": (
            len(completed_pairs) == total_expected
            and len(missing_pairs) == 0
            and len(invalid_labels) == 0
            and len(duplicates) == 0
        ),
        "batch_stats": batch_stats,
        "completed_queries": completed_queries,
        "incomplete_queries": incomplete_queries,
    }

    if verbose:
        print("=" * 70)
        print("Human Relevance Judgment Audit Report")
        print("=" * 70)
        print(f"Total Expected Pool Pairs : {total_expected}")
        print(f"Total Completed Judgments : {len(completed_pairs)} / {total_expected} ({len(completed_pairs)/total_expected*100:.1f}%)")
        print(f"Missing Judgments         : {len(missing_pairs)}")
        print(f"Invalid Labels            : {len(invalid_labels)}")
        print(f"Duplicate Judgments       : {len(duplicates)}")
        print(f"Completed Queries         : {len(completed_queries)} / {len(query_candidate_map)}")
        print(f"Incomplete Queries        : {len(incomplete_queries)} / {len(query_candidate_map)}")

        print("\nBatch Status Breakdown:")
        for b_name, s in batch_stats.items():
            pct = (s["completed"] / s["total"] * 100) if s["total"] > 0 else 0.0
            print(f"  - {b_name:25s}: {s['completed']:3d} / {s['total']:3d} completed ({pct:5.1f}%), {s['missing']:3d} remaining")

        if incomplete_queries and len(completed_pairs) > 0:
            print("\nIncomplete Queries Detail:")
            for q_id, done_c, total_c in incomplete_queries[:10]:
                print(f"  - {q_id}: {done_c}/{total_c} judged ({total_c - done_c} remaining)")

        if audit_result["is_ready_for_qrels"]:
            print("\n[SUCCESS] All 496/496 judgments are complete, valid, and ready for qrels.json generation.")
        else:
            print(f"\n[PENDING] {len(missing_pairs)} judgments remaining before qrels.json can be generated.")

    return audit_result


if __name__ == "__main__":
    res = validate_all_judgments(verbose=True)
    if not res["is_ready_for_qrels"]:
        sys.exit(1)
    sys.exit(0)
