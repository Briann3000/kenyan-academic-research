"""Candidate pool generation script for Phase 3D.

Executes Systems A, B, and C on the 25 benchmark queries at depth 10,
unions and de-duplicates the retrieved documents, and saves data/evaluation/pool.json.
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.config import NORMALIZED_WORKS_FILE, PROJECT_ROOT
from src.search.corpus import RetrievalCorpus
from src.search.keyword import BM25SearchEngine
from src.search.semantic import SemanticSearchEngine
from src.search.graph_hybrid import GraphHybridSearchEngine
from src.graph.neo4j_client import Neo4jClient, InMemoryGraphClient
from src.evaluation.models import BenchmarkQuery
from src.evaluation.runner import build_candidate_pool

QUERIES_FILE = PROJECT_ROOT / "data" / "evaluation" / "benchmark_queries.json"
POOL_FILE = PROJECT_ROOT / "data" / "evaluation" / "pool.json"


def main():
    print("=" * 70)
    print("Phase 3D: Generating Relevance Assessment Candidate Pool")
    print("=" * 70)

    # Load queries
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)
    queries = [BenchmarkQuery(**q) for q in raw_queries]
    print(f"Loaded {len(queries)} benchmark queries across 4 domains.")

    # Load corpus
    corpus = RetrievalCorpus(corpus_path=NORMALIZED_WORKS_FILE)
    print(f"Loaded corpus: {len(corpus.documents)} documents.")

    # Initialize retrieval systems
    sys_a = BM25SearchEngine(corpus=corpus)
    sys_b = SemanticSearchEngine(corpus=corpus)

    neo4j_client = Neo4jClient()
    connected = neo4j_client.connect()
    graph_client = neo4j_client if connected else InMemoryGraphClient(normalized_works_path=NORMALIZED_WORKS_FILE)

    sys_c = GraphHybridSearchEngine(
        semantic_engine=sys_b,
        neo4j_client=graph_client,
        candidate_k=20,
        alpha=0.70,
        graph_weights={"topic": 0.50, "author": 0.25, "institution": 0.25},
    )

    print("\nRunning depth-10 pooling across Systems A, B, and C...")
    pools = build_candidate_pool(queries, corpus, sys_a, sys_b, sys_c, top_k=10)

    total_pool_documents = sum(p.total_candidates for p in pools.values())
    all_unique_papers = set()
    for p in pools.values():
        for c in p.candidates:
            all_unique_papers.add(c.paper_id)

    print(f"Total query-candidate pairs pooled across 25 queries: {total_pool_documents}")
    print(f"Total distinct research papers across all query pools: {len(all_unique_papers)}")

    # Save pool.json
    POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    serialized_pools = {q_id: pool.model_dump() for q_id, pool in pools.items()}
    with open(POOL_FILE, "w", encoding="utf-8") as f:
        json.dump(serialized_pools, f, indent=2)

    print(f"\nSaved pooled candidate records to: {POOL_FILE}")

    # Breakdown per domain
    domain_counts = {}
    for p in pools.values():
        domain_counts.setdefault(p.domain, []).append(p.total_candidates)

    print("\nCandidate Pool Size Summary by Domain:")
    for dom, counts in domain_counts.items():
        avg_c = sum(counts) / len(counts)
        print(f"  - {dom} ({len(counts)} queries): Total = {sum(counts)} candidates (Mean = {avg_c:.1f} / query, Min = {min(counts)}, Max = {max(counts)})")


if __name__ == "__main__":
    main()
