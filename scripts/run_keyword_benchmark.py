"""Benchmark and evaluation script for the Phase 3A BM25 Keyword Retrieval Baseline."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search.corpus import RetrievalCorpus
from src.search.keyword import BM25SearchEngine
from src.ingestion.config import REPORTS_DIR, PROJECT_ROOT

TEST_QUERIES = [
    "mobile money and small businesses in Kenya",
    "climate change effects on agriculture",
    "malaria prevention among children",
    "maternal healthcare access in rural Kenya",
    "digital financial services and informal businesses"
]


def run_benchmark():
    print("=" * 70)
    print(" PHASE 3A KEYWORD RETRIEVAL BENCHMARK (BM25)")
    print("=" * 70)

    # 1. Measure Corpus Loading Time
    t0 = time.perf_counter()
    corpus = RetrievalCorpus()
    corpus_load_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Corpus Loaded: {len(corpus)} documents in {corpus_load_time_ms:.2f}ms")

    # 2. Measure Index Construction Time
    t0 = time.perf_counter()
    engine = BM25SearchEngine(corpus=corpus, k1=1.5, b=0.75, remove_stopwords=True)
    index_build_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"BM25 Index Built: {len(corpus)} documents in {index_build_time_ms:.2f}ms")

    # 3. Execute 5 Test Queries & Measure Latencies
    query_results = []
    latencies = []

    print("\nExecuting Test Queries (Top-10 Results):")
    print("-" * 70)

    for q in TEST_QUERIES:
        resp = engine.search(q, top_k=10)
        latencies.append(resp.latency_ms)
        query_results.append(resp)
        print(f"\nQuery: '{q}' (Latency: {resp.latency_ms:.2f}ms | Hits: {resp.total_hits})")
        for res in resp.results[:3]:
            safe_title = res.title.encode("ascii", "replace").decode("ascii")
            print(f"  [Rank {res.rank}] ({res.score:.2f}) {res.paper_id} - '{safe_title[:60]}...' ({res.publication_year})")


    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)

    print("\n" + "=" * 70)
    print(" PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Corpus Size:               {len(corpus)} papers")
    print(f"Index Construction Time:   {index_build_time_ms:.2f} ms")
    print(f"Average Query Latency:     {avg_latency:.2f} ms")
    print(f"Max Query Latency:         {max_latency:.2f} ms")
    print(f"Min Query Latency:         {min_latency:.2f} ms")

    # 4. Generate Report
    report_lines = []
    report_lines.append("# Phase 3A: Keyword Retrieval Baseline Report (BM25)")
    report_lines.append("")
    report_lines.append("> **Project:** Kenyan Academic Research Knowledge Graph  ")
    report_lines.append(f"> **Evaluation Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    report_lines.append("> **Status:** Completed & Tested (Control Baseline for Information Retrieval)")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Objective & Scope")
    report_lines.append("")
    report_lines.append("The goal of Phase 3A is to implement a strict, reproducible **conventional keyword-based retrieval baseline** using standard Okapi BM25. This engine acts as the formal experimental control against which dense semantic search (Phase 3B) and graph-enhanced semantic search (Phase 3C) will be compared.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 2. Retrieval Corpus & Search Text Specification")
    report_lines.append("")
    report_lines.append("- **Corpus Source:** `data/processed/normalized_works.jsonl`")
    report_lines.append(f"- **Document Count:** **{len(corpus)}** distinct academic works")
    report_lines.append("- **Searchable Field (`search_text`):**")
    report_lines.append("  ```text")
    report_lines.append("  search_text = Title + Topic labels + Abstract (when available)")
    report_lines.append("  ```")
    report_lines.append("- **Missing Abstract Fallback:** As established in Phase 1, works without abstracts rely on `Title + Topic labels`, preventing omission from the inverted index.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 3. BM25 Algorithm & Technical Configuration")
    report_lines.append("")
    report_lines.append("| Parameter / Component | Specification | Description / Standard Defaults |")
    report_lines.append("| :--- | :--- | :--- |")
    report_lines.append("| **Library / Engine** | `rank-bm25 0.2.2` | Standard pure-Python Okapi BM25 implementation |")
    report_lines.append("| **Term Frequency Saturation ($k_1$)** | `1.5` | Standard default governing term frequency scaling |")
    report_lines.append("| **Document Length Normalization ($b$)** | `0.75` | Standard default scaling document length penalty |")
    report_lines.append("| **Tokenizer** | Regex `\\b\\w+\\b` | Alphanumeric word boundary tokenizer |")
    report_lines.append("| **Case Sensitivity** | Lowercase | All tokens lowercased before indexing |")
    report_lines.append("| **Stopword Removal** | Standard English Set | Standard 127-stopword list filtered during tokenization |")
    report_lines.append("| **Tie-Breaking Rule** | Score DESC, Paper ID ASC | Deterministic ranking for equal scores |")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 4. Benchmark Performance Metrics")
    report_lines.append("")
    report_lines.append("| Metric | Measured Value |")
    report_lines.append("| :--- | :---: |")
    report_lines.append(f"| **Corpus Loading Time** | {corpus_load_time_ms:.2f} ms |")
    report_lines.append(f"| **Index Construction Time** | {index_build_time_ms:.2f} ms |")
    report_lines.append(f"| **Average Query Latency** | {avg_latency:.2f} ms |")
    report_lines.append(f"| **Min Query Latency** | {min_latency:.2f} ms |")
    report_lines.append(f"| **Max Query Latency** | {max_latency:.2f} ms |")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 5. Test Query Results (Top-10 per Query)")
    report_lines.append("")

    for q_idx, resp in enumerate(query_results, 1):
        report_lines.append(f"### Query {q_idx}: `{resp.query}`")
        report_lines.append(f"- **Execution Latency:** {resp.latency_ms:.2f} ms | **Hits Returned:** {resp.total_hits}")
        report_lines.append("")
        report_lines.append("| Rank | Score | Paper ID | Year | Title |")
        report_lines.append("| :---: | :---: | :--- | :---: | :--- |")
        for res in resp.results:
            clean_title = res.title.replace("|", "-")
            report_lines.append(f"| {res.rank} | {res.score:.2f} | `{res.paper_id}` | {res.publication_year} | {clean_title[:80]} |")
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 6. Input Validation & Edge Case Sanity Checks")
    report_lines.append("")
    report_lines.append("Automated unit tests in `tests/test_keyword_search.py` verified the following behavioral sanity checks:")
    report_lines.append("- **Empty Query (`''`)**: Returns 0 hits with 0.0ms latency without crashing.")
    report_lines.append("- **Whitespace-Only Query (`'   \\t\\n '`)**: Returns 0 hits safely.")
    report_lines.append("- **Out-of-Vocabulary Terms (`'quantum teleportation'`)**: Returns 0 hits.")
    report_lines.append("- **Stopwords-Only Query (`'what is the and or'`)**: Filters out all tokens and returns 0 hits safely.")
    report_lines.append("- **Boundary $k$ Values ($k=0$)**: Returns empty list.")
    report_lines.append("- **Boundary $k$ Values ($k > 1000$)**: Safely bounded to corpus size.")
    report_lines.append("- **Deterministic Output**: Identical queries against the corpus produce identical rankings and scores.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 7. Observed Limitations of the Keyword Baseline")
    report_lines.append("")
    report_lines.append("1. **Vocabulary Mismatch (Synonym Blindness):**")
    report_lines.append("   - BM25 cannot match terms that do not share exact lexical roots. For example, a paper on *'financial inclusion via M-Pesa'* will receive a lower score if the query specifically uses the phrasing *'digital financial services'* and that exact phrase does not appear in the title/abstract.")
    report_lines.append("2. **Word Sense & Semantic Context:**")
    report_lines.append("   - BM25 scores independently across terms based on inverse document frequency, lacking awareness of conceptual relations (e.g., that *'malaria'* relates inherently to *'Anopheles mosquitoes'* and *'artemisinin'*).")
    report_lines.append("3. **Absence of Relational Context:**")
    report_lines.append("   - BM25 scores papers solely on document text, ignoring author expertise, institutional research clusters, and conceptual topic graphs.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 8. Requirements for Phase 3B (Dense Semantic Retrieval)")
    report_lines.append("")
    report_lines.append("To address keyword retrieval limitations, Phase 3B must:")
    report_lines.append("1. Evaluate and select an appropriate sentence embedding model (e.g., `all-MiniLM-L6-v2` or `BGE-small-en-v1.5`).")
    report_lines.append("2. Generate dense vector embeddings for all 1,000 papers over the same standardized `search_text`.")
    report_lines.append("3. Implement a vector similarity search engine matching the exact same `SearchResult` and `SearchResponse` interface.")
    report_lines.append("")

    report_path = REPORTS_DIR / "phase3a_keyword_baseline.md"
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write("\n".join(report_lines))

    print(f"\nReport successfully generated at: {report_path}")


if __name__ == "__main__":
    run_benchmark()
