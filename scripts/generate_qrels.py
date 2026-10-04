"""Compiles data/evaluation/qrels.json from human judgment batch files.

Executes validation check first; strictly rejects generation if any judgment is missing or invalid.
"""

import sys
import json
from pathlib import Path
from collections import Counter

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.config import PROJECT_ROOT
from scripts.validate_judgments import validate_all_judgments, BATCH_FILES, JUDGMENTS_DIR, POOL_FILE

QRELS_FILE = PROJECT_ROOT / "data" / "evaluation" / "qrels.json"


def generate_qrels() -> bool:
    print("=" * 70)
    print("Generating Final Human Ground Truth Relevance Judgments (qrels.json)")
    print("=" * 70)

    # 1. Run strict validation
    audit = validate_all_judgments(verbose=True)
    if not audit["is_ready_for_qrels"]:
        print(f"\n[ERROR] Cannot generate qrels.json: {audit['missing_judgments']} items remaining unjudged.")
        print("Please complete all human judgments in data/evaluation/judgments/ first.")
        return False

    # 2. Compile qrels mapping
    qrels_dict = {}
    label_counts = Counter()

    for batch_file_name in BATCH_FILES:
        b_path = JUDGMENTS_DIR / batch_file_name
        with open(b_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)

        for item in b_data.get("items", []):
            q_id = item["query_id"]
            p_id = item["paper_id"]
            rel = item["relevance"]

            if q_id not in qrels_dict:
                qrels_dict[q_id] = {}

            qrels_dict[q_id][p_id] = rel
            label_counts[rel] += 1

    # 3. Verify exact candidate pool match
    with open(POOL_FILE, "r", encoding="utf-8") as f:
        pool_data = json.load(f)

    for q_id, q_pool in pool_data.items():
        if q_id not in qrels_dict:
            print(f"[ERROR] Query {q_id} missing from compiled qrels.")
            return False
        for cand in q_pool["candidates"]:
            p_id = cand["paper_id"]
            if p_id not in qrels_dict[q_id]:
                print(f"[ERROR] Candidate {p_id} in {q_id} missing from compiled qrels.")
                return False

    # 4. Save qrels.json
    with open(QRELS_FILE, "w", encoding="utf-8") as f:
        json.dump(qrels_dict, f, indent=2)

    total_judged = sum(len(docs) for docs in qrels_dict.values())
    print(f"\n[SUCCESS] Successfully compiled {total_judged} human judgments into: {QRELS_FILE}")
    print("\nRelevance Label Distribution across all 496 judgments:")
    print(f"  - 0 (Irrelevant)         : {label_counts[0]:3d} ({label_counts[0]/total_judged*100:.1f}%)")
    print(f"  - 1 (Partially Relevant) : {label_counts[1]:3d} ({label_counts[1]/total_judged*100:.1f}%)")
    print(f"  - 2 (Highly Relevant)    : {label_counts[2]:3d} ({label_counts[2]/total_judged*100:.1f}%)")

    return True


if __name__ == "__main__":
    success = generate_qrels()
    if not success:
        sys.exit(1)
    sys.exit(0)
