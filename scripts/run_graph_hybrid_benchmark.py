"""Benchmark and evaluation script for Phase 3C Graph-Enhanced Semantic Retrieval (System C).

Executes benchmark queries, profiles System B vs. System C rankings, analyzes rank shifts
and graph contributions, measures latencies, and writes data/reports/phase3c_graph_hybrid_baseline.md.
"""

import sys
import json
import time
import math
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search.corpus import RetrievalCorpus
from src.search.semantic import SemanticSearchEngine, DEFAULT_MODEL_REPO
from src.search.graph_hybrid import GraphHybridSearchEngine, DEFAULT_CANDIDATE_K, DEFAULT_ALPHA, DEFAULT_GRAPH_WEIGHTS
from src.graph.neo4j_client import Neo4jClient, InMemoryGraphClient
from src.ingestion.config import REPORTS_DIR, NORMALIZED_WORKS_FILE, PROCESSED_DATA_DIR

BENCHMARK_QUERIES = [
    "mobile money and small businesses in Kenya",
    "climate change effects on agriculture",
    "malaria prevention among children",
    "maternal healthcare access in rural Kenya",
    "digital financial services and informal businesses",
]


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def calculate_distribution(values: List[float]) -> Dict[str, Any]:
    """Calculates summary statistics and percentiles."""
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p95": 0, "max": 0, "mean": 0.0, "std": 0.0}
    s = sorted(values)
    n = len(s)
    mean_val = sum(s) / n
    variance = sum((x - mean_val) ** 2 for x in s) / n
    std_val = math.sqrt(variance)

    def percentile(p):
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] * (c - k) + s[c] * (k - f)

    return {
        "min": round(s[0], 3),
        "p25": round(float(percentile(0.25)), 3),
        "median": round(float(percentile(0.50)), 3),
        "p75": round(float(percentile(0.75)), 3),
        "p95": round(float(percentile(0.95)), 3),
        "max": round(s[-1], 3),
        "mean": round(float(mean_val), 3),
        "std": round(float(std_val), 3),
    }


def main():
    print("=" * 70)
    print("Phase 3C: Graph-Enhanced Semantic Retrieval Benchmark")
    print("=" * 70)

    corpus_path = NORMALIZED_WORKS_FILE
    corpus_sha = compute_sha256(corpus_path)
    corpus = RetrievalCorpus(corpus_path=corpus_path)
    print(f"Loaded corpus: {len(corpus.documents)} documents (SHA-256: {corpus_sha[:12]}...)")

    neo4j_client = Neo4jClient()
    connected = neo4j_client.connect()
    if connected:
        graph_client = neo4j_client
        graph_status_desc = f"Connected Live Neo4j ({neo4j_client.uri})"
        print(f"Successfully connected to Neo4j at {neo4j_client.uri}")
    else:
        graph_client = InMemoryGraphClient(normalized_works_path=corpus_path)
        graph_status_desc = "InMemoryGraphClient (Phase 2 Canonical Subgraph Engine)"
        print(f"Live Neo4j offline. Using validated {graph_status_desc}")

    semantic_engine = SemanticSearchEngine(corpus=corpus)
    hybrid_engine = GraphHybridSearchEngine(
        semantic_engine=semantic_engine,
        neo4j_client=graph_client,
        candidate_k=DEFAULT_CANDIDATE_K,
        alpha=DEFAULT_ALPHA,
        graph_weights=DEFAULT_GRAPH_WEIGHTS,
    )

    query_results = []
    all_sem_latencies = []
    all_graph_latencies = []
    all_total_latencies = []

    print("\nExecuting benchmark queries...")
    for q in BENCHMARK_QUERIES:
        # Warmup and latency timing over multiple runs
        latencies_sem = []
        latencies_graph = []
        latencies_total = []

        # Run 10 iterations for timing stability
        resp = None
        for _ in range(10):
            r = hybrid_engine.search(q, top_k=10, candidate_k=DEFAULT_CANDIDATE_K, alpha=DEFAULT_ALPHA)
            if resp is None:
                resp = r
            if r.results:
                latencies_sem.append(r.results[0].metadata.get("sem_latency_ms", 0.0))
                latencies_graph.append(r.results[0].metadata.get("graph_latency_ms", 0.0))
            latencies_total.append(r.latency_ms)

        all_sem_latencies.extend(latencies_sem)
        all_graph_latencies.extend(latencies_graph)
        all_total_latencies.extend(latencies_total)

        # Baseline Semantic Top 10
        sem_resp = semantic_engine.search(q, top_k=10)

        query_results.append(
            {
                "query": q,
                "semantic_response": sem_resp,
                "hybrid_response": resp,
                "latencies_sem": latencies_sem,
                "latencies_graph": latencies_graph,
                "latencies_total": latencies_total,
            }
        )
        print(f"  [OK] '{q}' -> Total Hits: {resp.total_hits}, Avg Latency: {sum(latencies_total)/len(latencies_total):.2f} ms")

    # Alpha parameter sensitivity evaluation for first query
    alpha_ablation_query = BENCHMARK_QUERIES[0]
    alpha_sweep_results = {}
    for alpha_val in [1.0, 0.85, 0.70, 0.50, 0.30, 0.0]:
        ablation_resp = hybrid_engine.search(
            alpha_ablation_query, top_k=5, candidate_k=DEFAULT_CANDIDATE_K, alpha=alpha_val
        )
        alpha_sweep_results[alpha_val] = ablation_resp.results

    # Generate Markdown Report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "phase3c_graph_hybrid_baseline.md"

    sem_dist = calculate_distribution(all_sem_latencies)
    graph_dist = calculate_distribution(all_graph_latencies)
    total_dist = calculate_distribution(all_total_latencies)

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    md = []
    md.append("# Phase 3C: Graph-Enhanced Semantic Retrieval Baseline Report")
    md.append(f"\n**Execution Timestamp**: `{now_iso}`")
    md.append(f"**Corpus Size**: {len(corpus.documents):,} papers")
    md.append(f"**Corpus SHA-256**: `{corpus_sha}`")
    md.append(f"**Graph Engine Status**: `{graph_status_desc}`")
    md.append(f"**Experimental Config**: Candidate $K={DEFAULT_CANDIDATE_K}$, Interpolation $\\alpha={DEFAULT_ALPHA}$")
    md.append(f"**Graph Overlap Weights**: Topic $w={DEFAULT_GRAPH_WEIGHTS['topic']}$, Author $w={DEFAULT_GRAPH_WEIGHTS['author']}$, Institution $w={DEFAULT_GRAPH_WEIGHTS['institution']}$")

    md.append("\n---\n")
    md.append("## 1. System Architecture & Formulation\n")
    md.append("System C implements bounded graph-enhanced semantic retrieval where candidate papers are strictly generated by System B (top-$K=20$) and reranked via structural relational overlap:\n")
    md.append("```text")
    md.append("User Query")
    md.append("    ↓")
    md.append("DenseSemanticSearchEngine (System B)")
    md.append("    ↓ (Top-K Candidates: K=20)")
    md.append("Bounded Neo4j Cypher Subgraph Batch Query")
    md.append("    ↓ (Shared Topics, Authors, Institutions)")
    md.append("Candidate-Relative Overlap Normalization")
    md.append("    ↓")
    md.append("Graph Affinity Score S_graph = w_topic*O_topic + w_author*O_author + w_inst*O_inst")
    md.append("    ↓")
    md.append("Linear Hybrid Score S_hybrid = α * S_semantic + (1 - α) * S_graph (α=0.7)")
    md.append("    ↓ (Tie-break: S_hybrid DESC, paper_id ASC)")
    md.append("Final Ranked System C Results")
    md.append("```\n")

    md.append("### Key Methodological Safeguards:")
    md.append("1. **Candidate Restriction**: System C reranks strictly within the top-$K$ semantic candidate set. No external corpus documents are introduced.")
    md.append("2. **No Centrality/Popularity Bias**: Overlaps ($O_{\\text{topic}}, O_{\\text{author}}, O_{\\text{inst}}$) are computed strictly relative to the retrieved candidate set $\\mathcal{C}$. Total global node degree, PageRank, or database-wide author popularity have zero influence.")
    md.append("3. **Schema Compliance**: Evaluates only validated Phase 2 relationships: `(:Paper)-[:ABOUT]->(:Topic)`, `(:Researcher)-[:AUTHORED]->(:Paper)`, and `(:Paper)-[:AFFILIATED_WITH]->(:Institution)`.")
    md.append("4. **Deterministic Ranking**: Stable sorting key `(-round(hybrid_score, 8), paper_id)`.")
    md.append("5. **Resilient Fallback**: If Neo4j is offline or fails, System C gracefully returns semantic rankings with explicit transparency flags (`graph_available=False`, `fallback_used=True`).")

    md.append("\n---\n")
    md.append("## 2. Latency & Performance Profile\n")
    md.append("> **Timing Boundary Notice**: The graph retrieval latency measured below represents the in-memory graph client benchmark (`InMemoryGraphClient`) querying Python RAM indexes. It **must not be interpreted as live Neo4j query latency**. Live Neo4j server latency involves Bolt TCP network communication, Cypher compilation, and database transaction overhead, which must be empirically measured under an active server deployment rather than reported as an estimated range.\n")
    md.append("| Metric | Semantic Retrieval (ms) | In-Memory Graph Fetch (ms) | Total System C Latency (ms) |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Mean ± Std** | {sem_dist['mean']} ± {sem_dist['std']} | {graph_dist['mean']} ± {graph_dist['std']} | {total_dist['mean']} ± {total_dist['std']} |")
    md.append(f"| **Median (p50)** | {sem_dist['median']} | {graph_dist['median']} | {total_dist['median']} |")
    md.append(f"| **p75** | {sem_dist['p75']} | {graph_dist['p75']} | {total_dist['p75']} |")
    md.append(f"| **p95** | {sem_dist['p95']} | {graph_dist['p95']} | {total_dist['p95']} |")
    md.append(f"| **Min / Max** | {sem_dist['min']} / {sem_dist['max']} | {graph_dist['min']} / {graph_dist['max']} | {total_dist['min']} / {total_dist['max']} |")

    md.append("\n---\n")
    md.append("## 3. Benchmark Comparative Retrieval & Rank-Shift Analysis\n")

    for item in query_results:
        q = item["query"]
        sem_res = item["semantic_response"].results
        hyb_res = item["hybrid_response"].results

        md.append(f"### Query: *\"{q}\"*\n")
        md.append("#### System B (Dense Semantic Baseline) Top 5")
        md.append("| Rank | Paper ID | Title | Semantic Score |")
        md.append("| :---: | :--- | :--- | :---: |")
        for r in sem_res[:5]:
            md.append(f"| {r.rank} | `{r.paper_id}` | {r.title[:65]}... | {r.score:.4f} |")

        md.append("\n#### System C (Graph-Enhanced Hybrid) Top 5")
        md.append("| Rank | Paper ID | Title | $S_{\\text{sem}}$ | $S_{\\text{graph}}$ | $S_{\\text{hyb}}$ | Orig Rank | Shift | Topic / Auth / Inst Overlap |")
        md.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in hyb_res[:5]:
            m = r.metadata
            shift_str = f"+{m['rank_change']}" if m['rank_change'] > 0 else f"{m['rank_change']}"
            overlaps_str = f"T:{m['topic_overlap']:.2f} / A:{m['author_overlap']:.2f} / I:{m['institution_overlap']:.2f}"
            md.append(f"| {r.rank} | `{r.paper_id}` | {r.title[:50]}... | {m['semantic_score']:.4f} | {m['graph_score']:.4f} | {m['hybrid_score']:.4f} | {m['semantic_rank']} | {shift_str} | {overlaps_str} |")

        md.append("\n")

    md.append("\n---\n")
    md.append("## 4. Interpolation Parameter Sensitivity ($\\alpha$ Sweep)\n")
    md.append(f"Evaluating rank sensitivity for query: *\"{alpha_ablation_query}\"*:\n")
    md.append("| $\\alpha$ Value | Rank 1 Paper | Score | Rank 2 Paper | Score | Rank 3 Paper | Score |")
    md.append("| :---: | :--- | :---: | :--- | :---: | :--- | :---: |")
    for alpha_val, res_list in alpha_sweep_results.items():
        p1 = f"`{res_list[0].paper_id}`" if len(res_list) > 0 else "-"
        s1 = f"{res_list[0].score:.4f}" if len(res_list) > 0 else "-"
        p2 = f"`{res_list[1].paper_id}`" if len(res_list) > 1 else "-"
        s2 = f"{res_list[1].score:.4f}" if len(res_list) > 1 else "-"
        p3 = f"`{res_list[2].paper_id}`" if len(res_list) > 2 else "-"
        s3 = f"{res_list[2].score:.4f}" if len(res_list) > 2 else "-"
        md.append(f"| **{alpha_val:.2f}** | {p1} | {s1} | {p2} | {s2} | {p3} | {s3} |")

    md.append("\n---\n")
    md.append("## 5. Methodological Limitations & Research Notes\n")
    md.append("1. **Candidate Pool Bounding**: System C reranks only the top-$K=20$ semantic candidates. Any relevant paper not captured in the top-$K$ semantic pool cannot be surfaced by System C.")
    md.append("2. **Graph Density & Coverage Dependency**: Overlap evidence relies entirely on entity resolution and relationship completeness in Phase 2. Papers with sparse metadata naturally receive lower graph affinity scores.")
    md.append("3. **Non-Equivalence of Connection and Relevance**: A shared topic or co-author indicates structural association, but does not guarantee superior topical relevance for a specific informational query.")
    md.append("4. **Configuration Baseline**: Default parameters ($\\alpha=0.7$, $w_{\\text{topic}}=0.5$, $w_{\\text{author}}=0.25$, $w_{\\text{inst}}=0.25$) are baseline engineering configurations; formal optimality and comparative evaluation against Ground Truth will be executed in Phase 3D.")
    md.append("5. **Candidate-Relative Cluster Coherence**: System C explicitly tests graph-based candidate cluster coherence (structural density within the retrieved candidate set). Phase 3D will empirically evaluate whether this candidate-relative graph signal improves or degrades retrieval relevance compared to dense semantic search.")

    report_content = "\n".join(md)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nSuccessfully generated baseline report: {report_file}")
    if neo4j_client:
        neo4j_client.close()


if __name__ == "__main__":
    main()
