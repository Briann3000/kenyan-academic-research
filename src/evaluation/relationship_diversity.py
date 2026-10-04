"""Relationship Discovery & Entity Diversity Evaluation (Phase 4).

Quantifies how effectively each search paradigm uncovers connected research entities
(researchers, institutions, topics) across top-K retrieved papers, measuring:
1. Entity Coverage / Discovery Count: Total unique entities discovered in top-K.
2. Entity Diversity (Gini-Simpson & Shannon Entropy): Dispersion of retrieved entities across academic domains/institutions.
3. Cross-Institutional Collaboration Reach: Unique domestic and international institutions connected via the top-K papers.
"""

import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import Counter

from src.evaluation.models import BenchmarkQuery
from src.search.corpus import RetrievalCorpus
from src.search.keyword import BM25SearchEngine
from src.search.semantic import SemanticSearchEngine
from src.search.graph_hybrid import GraphHybridSearchEngine
from src.graph.neo4j_client import InMemoryGraphClient
from src.ingestion.config import DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

BENCHMARK_QUERIES_FILE = DATA_DIR / "evaluation" / "benchmark_queries.json"

logger = logging.getLogger(__name__)


def compute_shannon_entropy(items: List[str]) -> float:
    """Computes Shannon entropy (base 2) for a list of categorical items."""
    if not items:
        return 0.0
    counts = Counter(items)
    n = len(items)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def compute_simpson_diversity(items: List[str]) -> float:
    """Computes Gini-Simpson diversity index: 1 - sum(p_i^2).
    
    Ranges from 0.0 (all items belong to one entity) to 1.0 (infinitely diverse).
    """
    if not items or len(items) <= 1:
        return 0.0
    counts = Counter(items)
    n = len(items)
    sum_sq = sum((c / n) ** 2 for c in counts.values())
    return round(1.0 - sum_sq, 4)


def evaluate_relationship_diversity(
    queries: List[BenchmarkQuery],
    system_a: BM25SearchEngine,
    system_b: SemanticSearchEngine,
    system_c: GraphHybridSearchEngine,
    graph_client: InMemoryGraphClient,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Evaluates entity discovery and diversity metrics across Systems A, B, and C."""
    systems = {
        "system_a": system_a,
        "system_b": system_b,
        "system_c": system_c,
    }

    per_query_results = []
    summary: Dict[str, Dict[str, List[float]]] = {
        sys_name: {
            "unique_authors": [],
            "unique_institutions": [],
            "unique_topics": [],
            "author_entropy": [],
            "institution_entropy": [],
            "topic_entropy": [],
            "institution_diversity": [],
            "topic_diversity": [],
        }
        for sys_name in systems
    }

    for q in queries:
        q_entry: Dict[str, Any] = {
            "query_id": q.query_id,
            "domain": q.domain,
            "query_text": q.query_text,
            "systems": {},
        }

        for sys_name, engine in systems.items():
            resp = engine.search(q.query_text, top_k=top_k)
            retrieved_ids = [r.paper_id for r in resp.results]

            # Fetch connected entities from graph index
            all_authors: List[str] = []
            all_institutions: List[str] = []
            all_topics: List[str] = []

            for p_id in retrieved_ids:
                authors = list(graph_client.paper_authors.get(p_id, set()))
                insts = list(graph_client.paper_institutions.get(p_id, set()))
                topics = list(graph_client.paper_topics.get(p_id, set()))

                all_authors.extend(authors)
                all_institutions.extend(insts)
                all_topics.extend(topics)

            n_unique_authors = len(set(all_authors))
            n_unique_insts = len(set(all_institutions))
            n_unique_topics = len(set(all_topics))

            author_ent = compute_shannon_entropy(all_authors)
            inst_ent = compute_shannon_entropy(all_institutions)
            topic_ent = compute_shannon_entropy(all_topics)

            inst_div = compute_simpson_diversity(all_institutions)
            topic_div = compute_simpson_diversity(all_topics)

            # Record in summary
            summary[sys_name]["unique_authors"].append(n_unique_authors)
            summary[sys_name]["unique_institutions"].append(n_unique_insts)
            summary[sys_name]["unique_topics"].append(n_unique_topics)
            summary[sys_name]["author_entropy"].append(author_ent)
            summary[sys_name]["institution_entropy"].append(inst_ent)
            summary[sys_name]["topic_entropy"].append(topic_ent)
            summary[sys_name]["institution_diversity"].append(inst_div)
            summary[sys_name]["topic_diversity"].append(topic_div)

            q_entry["systems"][sys_name] = {
                "top_k_papers": len(retrieved_ids),
                "unique_authors": n_unique_authors,
                "unique_institutions": n_unique_insts,
                "unique_topics": n_unique_topics,
                "author_entropy": author_ent,
                "institution_entropy": inst_ent,
                "topic_entropy": topic_ent,
                "institution_diversity": inst_div,
                "topic_diversity": topic_div,
            }

        per_query_results.append(q_entry)

    # Compute macro averages
    macro_summary: Dict[str, Dict[str, float]] = {}
    num_q = len(queries)

    for sys_name, metrics in summary.items():
        macro_summary[sys_name] = {
            "mean_unique_authors_at_10": round(sum(metrics["unique_authors"]) / num_q, 2),
            "mean_unique_institutions_at_10": round(sum(metrics["unique_institutions"]) / num_q, 2),
            "mean_unique_topics_at_10": round(sum(metrics["unique_topics"]) / num_q, 2),
            "mean_author_entropy": round(sum(metrics["author_entropy"]) / num_q, 4),
            "mean_institution_entropy": round(sum(metrics["institution_entropy"]) / num_q, 4),
            "mean_topic_entropy": round(sum(metrics["topic_entropy"]) / num_q, 4),
            "mean_institution_diversity": round(sum(metrics["institution_diversity"]) / num_q, 4),
            "mean_topic_diversity": round(sum(metrics["topic_diversity"]) / num_q, 4),
        }

    return {
        "num_queries": num_q,
        "top_k": top_k,
        "macro_summary": macro_summary,
        "per_query_results": per_query_results,
    }


def run_diversity_evaluation():
    """Main execution function to run and save relationship discovery evaluation."""
    logger.info("Initializing search engines and graph client for diversity evaluation...")
    corpus = RetrievalCorpus()
    graph_client = InMemoryGraphClient()

    system_a = BM25SearchEngine(corpus=corpus)
    system_b = SemanticSearchEngine(corpus=corpus)
    system_c = GraphHybridSearchEngine(
        semantic_engine=system_b,
        corpus=corpus,
        neo4j_client=graph_client,
        candidate_k=20,
        alpha=0.70,
    )

    with open(BENCHMARK_QUERIES_FILE, "r", encoding="utf-8") as f:
        queries_data = json.load(f)
    queries = [BenchmarkQuery(**q) for q in queries_data]

    results = evaluate_relationship_diversity(
        queries=queries,
        system_a=system_a,
        system_b=system_b,
        system_c=system_c,
        graph_client=graph_client,
        top_k=10,
    )

    output_path = PROCESSED_DATA_DIR.parent / "reports" / "relationship_discovery_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(" RELATIONSHIP DISCOVERY & ENTITY DIVERSITY EVALUATION (TOP-10)")
    print("=" * 70)
    print(f"{'Metric':<35} | {'System A (BM25)':<12} | {'System B (BGE)':<12} | {'System C (Hybrid)':<12}")
    print("-" * 79)
    
    ms = results["macro_summary"]
    metrics_to_show = [
        ("mean_unique_authors_at_10", "Mean Unique Authors"),
        ("mean_unique_institutions_at_10", "Mean Unique Institutions"),
        ("mean_unique_topics_at_10", "Mean Unique Topics"),
        ("mean_author_entropy", "Author Shannon Entropy"),
        ("mean_institution_entropy", "Institution Shannon Entropy"),
        ("mean_topic_entropy", "Topic Shannon Entropy"),
        ("mean_institution_diversity", "Institution Gini-Simpson"),
        ("mean_topic_diversity", "Topic Gini-Simpson"),
    ]

    for key, label in metrics_to_show:
        val_a = ms["system_a"][key]
        val_b = ms["system_b"][key]
        val_c = ms["system_c"][key]
        print(f"{label:<35} | {val_a:<12} | {val_b:<12} | {val_c:<12}")

    print("=" * 70)
    print(f"Results successfully saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_diversity_evaluation()
