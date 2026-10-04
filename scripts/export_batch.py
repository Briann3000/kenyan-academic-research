"""Exports Batch 1 (Health & Epidemiology) to a clean, readable Markdown judging document."""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.config import PROJECT_ROOT

BATCH1_FILE = PROJECT_ROOT / "data" / "evaluation" / "judgments" / "batch1_health.json"
EXPORT_MD = PROJECT_ROOT / "data" / "evaluation" / "batch1_health_export.md"


def main():
    if not BATCH1_FILE.exists():
        raise FileNotFoundError(f"{BATCH1_FILE} does not exist.")

    with open(BATCH1_FILE, "r", encoding="utf-8") as f:
        batch_data = json.load(f)

    items = batch_data.get("items", [])
    print(f"Loaded {len(items)} items from {BATCH1_FILE.name}")

    # Group by Query ID
    grouped = {}
    for it in items:
        q_id = it["query_id"]
        grouped.setdefault(q_id, []).append(it)

    lines = []
    lines.append("# Batch 1: Health & Epidemiology (106 Masked Candidates)")
    lines.append("")
    lines.append("> **Relevance Scale:**")
    lines.append("> - `0` = **Irrelevant / Off-topic** (does not address information need)")
    lines.append("> - `1` = **Partially Relevant** (related domain/topic, but does not directly address specific need)")
    lines.append("> - `2` = **Highly Relevant** (directly and substantively addresses the query in Kenya)")
    lines.append("")
    lines.append("---")
    lines.append("")

    total_idx = 0
    for q_id, q_items in grouped.items():
        q_text = q_items[0]["query_text"]
        lines.append(f"## Query {q_id}: *\"{q_text}\"*")
        lines.append(f"**Total Pooled Candidates**: {len(q_items)}\n")

        for idx, item in enumerate(q_items, start=1):
            total_idx += 1
            p_id = item["paper_id"]
            title = item["title"]
            abstract = item["abstract"] or "(No abstract available in OpenAlex corpus)"
            rel_status = f"`{item['relevance']}`" if item["relevance"] is not None else "[Unjudged]"

            lines.append(f"### Item {total_idx} ({q_id} — Candidate {idx} of {len(q_items)})")
            lines.append(f"- **Query ID**: `{q_id}`")
            lines.append(f"- **Query Text**: {q_text}")
            lines.append(f"- **Paper ID**: `{p_id}`")
            lines.append(f"- **Title**: **{title}**")
            lines.append(f"- **Abstract**:\n  > {abstract}\n")
            lines.append(f"- **Current Relevance**: {rel_status}")
            lines.append("- **Relevance Score**: `[  ]` *(Assign 0, 1, or 2)*")
            lines.append("- **Assessor Notes**: ` `")
            lines.append("\n---\n")

    with open(EXPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Successfully exported 106 candidates to: {EXPORT_MD}")


if __name__ == "__main__":
    main()
