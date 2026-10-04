"""Initializes the 4 masked human annotation batch files in data/evaluation/judgments/.

Extracts only Query ID, Query Text, Paper ID, Title, and Abstract from pool.json,
strictly omitting all retrieval system identities, ranks, and scores.
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.config import PROJECT_ROOT

POOL_FILE = PROJECT_ROOT / "data" / "evaluation" / "pool.json"
JUDGMENTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "judgments"

BATCHES = [
    {
        "batch_id": "batch1_health",
        "domain": "Health & Epidemiology",
        "query_ids": ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"],
        "filename": "batch1_health.json",
    },
    {
        "batch_id": "batch2_agriculture",
        "domain": "Agriculture & Food Security",
        "query_ids": ["Q07", "Q08", "Q09", "Q10", "Q11", "Q12", "Q13"],
        "filename": "batch2_agriculture.json",
    },
    {
        "batch_id": "batch3_economics",
        "domain": "Economics, FinTech & Business",
        "query_ids": ["Q14", "Q15", "Q16", "Q17", "Q18", "Q19"],
        "filename": "batch3_economics.json",
    },
    {
        "batch_id": "batch4_climate",
        "domain": "Climate, Ecology & Water",
        "query_ids": ["Q20", "Q21", "Q22", "Q23", "Q24", "Q25"],
        "filename": "batch4_climate.json",
    },
]


def main():
    print("=" * 70)
    print("Initializing Masked Human Relevance Judgment Batches")
    print("=" * 70)

    if not POOL_FILE.exists():
        raise FileNotFoundError(f"Pool file not found at {POOL_FILE}. Run scripts/build_evaluation_pool.py first.")

    with open(POOL_FILE, "r", encoding="utf-8") as f:
        pool_data = json.load(f)

    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    total_items = 0
    for batch_info in BATCHES:
        batch_filename = JUDGMENTS_DIR / batch_info["filename"]
        
        # If batch file already exists, preserve existing human judgments
        existing_judgments = {}
        if batch_filename.exists():
            with open(batch_filename, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    for item in existing_data.get("items", []):
                        key = (item["query_id"], item["paper_id"])
                        if item.get("relevance") is not None:
                            existing_judgments[key] = {
                                "relevance": item["relevance"],
                                "notes": item.get("notes", "")
                            }
                except Exception as exc:
                    print(f"Warning: Could not read existing {batch_filename}: {exc}")

        batch_items = []
        for q_id in batch_info["query_ids"]:
            if q_id not in pool_data:
                continue
            q_pool = pool_data[q_id]
            q_text = q_pool["query_text"]

            for cand in q_pool["candidates"]:
                p_id = cand["paper_id"]
                title = cand["title"]
                abstract = cand.get("abstract")

                prev_rel = existing_judgments.get((q_id, p_id), {}).get("relevance", None)
                prev_note = existing_judgments.get((q_id, p_id), {}).get("notes", "")

                # Strictly masked item: ONLY query_id, query_text, paper_id, title, abstract
                batch_items.append(
                    {
                        "query_id": q_id,
                        "query_text": q_text,
                        "paper_id": p_id,
                        "title": title,
                        "abstract": abstract,
                        "relevance": prev_rel,
                        "notes": prev_note,
                    }
                )

        batch_record = {
            "batch_id": batch_info["batch_id"],
            "domain": batch_info["domain"],
            "query_ids": batch_info["query_ids"],
            "total_items": len(batch_items),
            "completed_items": sum(1 for item in batch_items if item["relevance"] is not None),
            "items": batch_items,
        }

        with open(batch_filename, "w", encoding="utf-8") as f:
            json.dump(batch_record, f, indent=2)

        print(f"  [OK] {batch_info['filename']}: {len(batch_items)} masked items initialized ({batch_record['completed_items']} completed).")
        total_items += len(batch_items)

    print(f"\nTotal masked query-candidate items initialized across 4 batches: {total_items}")


if __name__ == "__main__":
    main()
