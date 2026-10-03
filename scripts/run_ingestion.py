"""CLI script to run full ingestion pipeline: Ingest -> Cache -> Normalize -> Profile -> Report."""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.config import (
    DEFAULT_MAX_WORKS,
    DEFAULT_PAGE_SIZE,
    RAW_WORKS_FILE,
    INGESTION_METADATA_FILE,
    NORMALIZED_WORKS_FILE,
    PROFILING_METRICS_FILE,
    DATA_QUALITY_REPORT_FILE,
    INSTITUTIONS_FILE
)
from src.ingestion.fetch_works import fetch_kenyan_works
from src.ingestion.cache import RawDataCache
from src.ingestion.normalizer import normalize_work
from src.ingestion.profiling import profile_dataset
from src.ingestion.institution_registry import InstitutionRegistry


def generate_markdown_report(metrics: dict, metadata: dict, registry_count: int) -> str:
    """Formats profiling results into an empirical markdown report."""
    completeness = metrics.get("completeness", [])
    dist = metrics.get("distributions", {})
    aff = metrics.get("affiliations", {})
    edges = metrics.get("edge_cases", {})
    total = metrics.get("total_records", 0)

    lines = []
    lines.append("# Data Quality & Metadata Profiling Report (Phase 1)")
    lines.append("")
    lines.append("> **Project:** Kenyan Academic Research Knowledge Graph  ")
    lines.append(f"> **Report Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    lines.append(f"> **Status:** Completed (Seed Corpus Ingestion & Audit)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Dataset Overview & Ingestion Parameters")
    lines.append("")
    lines.append(f"- **Data Source:** OpenAlex API (`https://api.openalex.org`)")
    lines.append(f"- **Filter Query Used:** `{metadata.get('filter_query', 'institutions.country_code:KE')}`")
    lines.append(f"- **Total Records Ingested:** **{total}**")
    lines.append(f"- **Ingestion Duration:** {metadata.get('duration_seconds', 0)} seconds")
    lines.append(f"- **Registered Kenyan Institutions:** {registry_count} institutions in registry")
    lines.append(f"- **Polite Rate Limiting:** Enforced with email identification")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Metadata Field Completeness Matrix")
    lines.append("")
    lines.append("| Field Name | Present Count | Missing Count | Completeness % | Missing % |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for f in completeness:
        lines.append(f"| **{f['field']}** | {f['present']} | {f['missing']} | {f['present_pct']}% | {f['missing_pct']}% |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Key Distribution Metrics")
    lines.append("")
    lines.append("### A. Entity Cardinalities per Paper")
    lines.append("")
    lines.append("| Dimension | Min | Median | Mean | Max |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for key, label in [
        ("authors_per_work", "Authors per Paper"),
        ("institutions_per_work", "Institutions per Paper"),
        ("topics_per_work", "Topics per Paper"),
        ("concepts_per_work", "Concepts per Paper"),
        ("citations_per_work", "Citations per Paper"),
        ("abstract_word_length", "Abstract Word Length (when present)")
    ]:
        stats = dist.get(key, {})
        lines.append(f"| **{label}** | {stats.get('min', 0)} | {stats.get('median', 0)} | {stats.get('mean', 0)} | {stats.get('max', 0)} |")

    lines.append("")
    lines.append("### B. Work Type Distribution")
    lines.append("")
    lines.append("| Work Type | Count | Percentage |")
    lines.append("| :--- | :---: | :---: |")
    for wtype, count in dist.get("work_types", {}).items():
        pct = (count / total * 100) if total else 0
        lines.append(f"| `{wtype}` | {count} | {pct:.1f}% |")

    lines.append("")
    lines.append("### C. Open Access (OA) Status Breakdown")
    lines.append("")
    lines.append("| OA Status | Count | Percentage |")
    lines.append("| :--- | :---: | :---: |")
    for oa, count in dist.get("oa_statuses", {}).items():
        pct = (count / total * 100) if total else 0
        lines.append(f"| `{oa}` | {count} | {pct:.1f}% |")

    lines.append("")
    lines.append("### D. Top Primary Research Domains & Fields")
    lines.append("")
    lines.append("| Domain / Field | Work Count |")
    lines.append("| :--- | :---: |")
    for field, count in dist.get("topic_domains", {}).items():
        lines.append(f"| **Domain:** {field} | {count} |")
    for field, count in list(dist.get("topic_fields", {}).items())[:8]:
        lines.append(f"| Field: {field} | {count} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Affiliation & Institutional Collaboration Analysis")
    lines.append("")
    lines.append(f"- **International Collaboration Rate:** {aff.get('international_collaboration_pct', 0)}% of papers ({aff.get('international_collaboration_count', 0)} papers) involve co-authors from institutions outside Kenya.")
    lines.append("")
    lines.append("### Top Domestic (Kenyan) Institutions:")
    lines.append("")
    lines.append("| Institution Name | Indexed Works Count |")
    lines.append("| :--- | :---: |")
    for inst, cnt in list(aff.get("top_domestic_institutions", {}).items())[:12]:
        lines.append(f"| {inst} | {cnt} |")

    lines.append("")
    lines.append("### Top Collaborating Partner Countries:")
    lines.append("")
    lines.append("| Country Code | Co-authored Papers |")
    lines.append("| :--- | :---: |")
    for ccode, cnt in list(aff.get("top_collaborating_countries", {}).items())[:10]:
        lines.append(f"| `{ccode}` | {cnt} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Identified Edge Cases & Data Anomalies")
    lines.append("")
    lines.append(f"- **Duplicate DOIs:** {edges.get('duplicate_dois_count', 0)}")
    lines.append(f"- **Duplicate Titles:** {edges.get('duplicate_titles_count', 0)}")
    lines.append(f"- **Duplicate OpenAlex IDs:** {edges.get('duplicate_ids_count', 0)}")
    lines.append(f"- **Records with Country=KE but Unresolved Domestic Institution:** {edges.get('records_without_identified_ke_institution', 0)}")
    lines.append(f"- **Anomalous Publication Years (<1960 or >2027):** {edges.get('anomalous_years_count', 0)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Engineering Decisions & Recommendations for Phase 2")
    lines.append("")
    lines.append("1. **Abstract Incompleteness & Semantic Fallback:**")
    lines.append(f"   - Observation: Abstracts are present in only **{next((f['present_pct'] for f in completeness if f['field'] == 'Abstract (Reconstructed)'), 'N/A')}%** of works.")
    lines.append("   - Engineering Decision: Semantic embeddings cannot rely strictly on abstracts. The system MUST construct a composite search text: `Title + Topic labels + Abstract (when present)` to prevent vector index gaps.")
    lines.append("")
    lines.append("2. **High International Collaboration Density:**")
    lines.append(f"   - Observation: Over **{aff.get('international_collaboration_pct', 0)}%** of works contain co-authors from international institutions (US, GB, ZA, UG, etc.).")
    lines.append("   - Engineering Decision: In Phase 2, graph modeling should distinguish domestic parent affiliations (`AFFILIATED_WITH`) from cross-border institutional collaborations (`COLLABORATES_WITH`).")
    lines.append("")
    lines.append("3. **Topic Modeling Consistency:**")
    lines.append(f"   - Observation: Primary topics and OpenAlex topics are present in **{next((f['present_pct'] for f in completeness if f['field'] == 'Topics List'), 'N/A')}%** of records.")
    lines.append("   - Engineering Decision: `(:Paper)-[:ABOUT]->(:Topic)` will serve as a highly dense and reliable spine for graph traversals and hybrid retrieval.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run OpenAlex ingestion and profiling pipeline.")
    parser.add_argument("--max-works", type=int, default=DEFAULT_MAX_WORKS, help="Maximum number of works to ingest")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Page size for OpenAlex pagination")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching and profile existing cache")
    args = parser.parse_args()

    # Step 1: Ensure Institution Registry exists
    registry = InstitutionRegistry(INSTITUTIONS_FILE)
    if not registry.institutions:
        print("Discovering Kenyan institutions...")
        registry.discover_from_openalex()
        registry.save()

    cache = RawDataCache(RAW_WORKS_FILE, INGESTION_METADATA_FILE)

    # Step 2: Fetch works if not skipping
    if not args.skip_fetch:
        print(f"Fetching up to {args.max_works} Kenyan works from OpenAlex...")
        fetch_kenyan_works(max_works=args.max_works, per_page=args.page_size, cache=cache)
    else:
        print(f"Skipping fetch. Using existing raw cache ({cache.count_records()} records)...")

    # Step 3: Normalize works
    print("Normalizing records...")
    normalized_records = []
    with open(NORMALIZED_WORKS_FILE, "w", encoding="utf-8") as f_norm:
        for raw_work in cache.stream_records():
            norm = normalize_work(raw_work)
            normalized_records.append(norm)
            f_norm.write(json.dumps(norm, ensure_ascii=False) + "\n")

    print(f"Normalized {len(normalized_records)} records to {NORMALIZED_WORKS_FILE}")

    # Step 4: Profile dataset
    print("Profiling metadata quality and completeness...")
    metrics = profile_dataset(normalized_records)

    with open(PROFILING_METRICS_FILE, "w", encoding="utf-8") as f_met:
        json.dump(metrics, f_met, indent=2, ensure_ascii=False)
    print(f"Saved profiling metrics to {PROFILING_METRICS_FILE}")

    # Step 5: Generate DATA_QUALITY_REPORT.md
    metadata = cache.load_metadata() or {}
    report_md = generate_markdown_report(metrics, metadata, len(registry.institutions))

    with open(DATA_QUALITY_REPORT_FILE, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_md)
    print(f"Generated {DATA_QUALITY_REPORT_FILE}")
    print("=" * 60)
    print(" PIPELINE EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
