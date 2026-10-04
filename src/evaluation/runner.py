"""Evaluation pipeline runner: pools candidate papers across Systems A, B, and C, evaluates metrics, and runs statistical comparisons."""

import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.evaluation.models import (
    BenchmarkQuery,
    PooledCandidate,
    QueryPool,
    QueryEvaluationMetrics,
    DomainSummaryMetrics,
    SystemSummaryMetrics,
)
from src.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_ndcg_at_k,
)
from src.evaluation.significance import paired_t_test, wilcoxon_signed_rank_test
from src.search.corpus import RetrievalCorpus
from src.search.keyword import BM25SearchEngine
from src.search.semantic import SemanticSearchEngine
from src.search.graph_hybrid import GraphHybridSearchEngine

logger = logging.getLogger(__name__)


def build_candidate_pool(
    queries: List[BenchmarkQuery],
    corpus: RetrievalCorpus,
    system_a: BM25SearchEngine,
    system_b: SemanticSearchEngine,
    system_c: GraphHybridSearchEngine,
    top_k: int = 10,
) -> Dict[str, QueryPool]:
    """Generates a de-duplicated depth-K candidate pool across Systems A, B, and C.

    Args:
        queries: List of 25 benchmark queries.
        corpus: Document retrieval corpus.
        system_a: System A (BM25) search engine.
        system_b: System B (Dense Semantic) search engine.
        system_c: System C (Graph-Hybrid) search engine.
        top_k: Pooling depth cutoff per system (default: 10).

    Returns:
        Dictionary mapping query_id -> QueryPool containing unique pooled candidates.
    """
    pools: Dict[str, QueryPool] = {}

    for q in queries:
        q_id = q.query_id
        q_text = q.query_text

        # 1. Retrieve top-K from each system
        res_a = system_a.search(q_text, top_k=top_k)
        res_b = system_b.search(q_text, top_k=top_k)
        res_c = system_c.search(q_text, top_k=top_k)

        candidate_map: Dict[str, PooledCandidate] = {}

        def record_results(results, sys_name):
            for rank_idx, r in enumerate(results, start=1):
                p_id = r.paper_id
                if p_id not in candidate_map:
                    doc = corpus.doc_map.get(p_id)
                    title = doc.title if doc else r.title
                    abstract = doc.abstract if doc else None
                    candidate_map[p_id] = PooledCandidate(
                        paper_id=p_id,
                        title=title,
                        abstract=abstract,
                        retrieved_by=[sys_name],
                        source_ranks={sys_name: rank_idx},
                    )
                else:
                    cand = candidate_map[p_id]
                    if sys_name not in cand.retrieved_by:
                        cand.retrieved_by.append(sys_name)
                    cand.source_ranks[sys_name] = rank_idx

        record_results(res_a.results, "system_a")
        record_results(res_b.results, "system_b")
        record_results(res_c.results, "system_c")

        # Stable sort by paper_id for deterministic un-biased presentation
        sorted_candidates = sorted(candidate_map.values(), key=lambda c: c.paper_id)

        pools[q_id] = QueryPool(
            query_id=q_id,
            domain=q.domain,
            query_text=q_text,
            total_candidates=len(sorted_candidates),
            candidates=sorted_candidates,
        )

    return pools


def evaluate_system_on_queries(
    system_engine: Any,
    system_name: str,
    queries: List[BenchmarkQuery],
    qrels: Dict[str, Dict[str, int]],
    top_k: int = 10,
    search_kwargs: Optional[Dict[str, Any]] = None,
) -> List[QueryEvaluationMetrics]:
    """Evaluates a retrieval system on all benchmark queries against qrels."""
    search_kwargs = search_kwargs or {}
    results = []

    for q in queries:
        q_id = q.query_id
        q_text = q.query_text
        query_qrels = qrels.get(q_id, {})

        t0 = time.perf_counter()
        resp = system_engine.search(q_text, top_k=top_k, **search_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_ids = [r.paper_id for r in resp.results]
        judged_count = sum(1 for p_id in retrieved_ids[:top_k] if p_id in query_qrels)

        p5 = compute_precision_at_k(retrieved_ids, query_qrels, k=5, relevance_threshold=1)
        p10 = compute_precision_at_k(retrieved_ids, query_qrels, k=10, relevance_threshold=1)
        r10 = compute_recall_at_k(retrieved_ids, query_qrels, k=10, relevance_threshold=1)
        mrr = compute_reciprocal_rank(retrieved_ids, query_qrels, relevance_threshold=1)
        ndcg5 = compute_ndcg_at_k(retrieved_ids, query_qrels, k=5)
        ndcg10 = compute_ndcg_at_k(retrieved_ids, query_qrels, k=10)

        results.append(
            QueryEvaluationMetrics(
                query_id=q_id,
                system_name=system_name,
                p_at_5=round(p5, 4),
                p_at_10=round(p10, 4),
                recall_at_10=round(r10, 4),
                mrr=round(mrr, 4),
                ndcg_at_5=round(ndcg5, 4),
                ndcg_at_10=round(ndcg10, 4),
                latency_ms=round(latency_ms, 2),
                judged_hits_at_10=judged_count,
            )
        )

    return results


def summarize_system_metrics(
    eval_metrics: List[QueryEvaluationMetrics],
    queries_map: Dict[str, BenchmarkQuery],
) -> SystemSummaryMetrics:
    """Computes overall mean metrics and per-domain breakdown."""
    if not eval_metrics:
        raise ValueError("Cannot summarize empty metric list.")

    sys_name = eval_metrics[0].system_name
    n = len(eval_metrics)

    def mean_of(attr):
        return round(sum(getattr(m, attr) for m in eval_metrics) / n, 4)

    # Domain groupings
    domain_groups: Dict[str, List[QueryEvaluationMetrics]] = {}
    for m in eval_metrics:
        dom = queries_map[m.query_id].domain
        domain_groups.setdefault(dom, []).append(m)

    domain_summaries = {}
    for dom, d_metrics in domain_groups.items():
        dn = len(d_metrics)
        domain_summaries[dom] = DomainSummaryMetrics(
            domain=dom,
            num_queries=dn,
            mean_p_at_5=round(sum(m.p_at_5 for m in d_metrics) / dn, 4),
            mean_p_at_10=round(sum(m.p_at_10 for m in d_metrics) / dn, 4),
            mean_recall_at_10=round(sum(m.recall_at_10 for m in d_metrics) / dn, 4),
            mean_mrr=round(sum(m.mrr for m in d_metrics) / dn, 4),
            mean_ndcg_at_5=round(sum(m.ndcg_at_5 for m in d_metrics) / dn, 4),
            mean_ndcg_at_10=round(sum(m.ndcg_at_10 for m in d_metrics) / dn, 4),
            mean_latency_ms=round(sum(m.latency_ms for m in d_metrics) / dn, 2),
        )

    return SystemSummaryMetrics(
        system_name=sys_name,
        num_queries=n,
        mean_p_at_5=mean_of("p_at_5"),
        mean_p_at_10=mean_of("p_at_10"),
        mean_recall_at_10=mean_of("recall_at_10"),
        mean_mrr=mean_of("mrr"),
        mean_ndcg_at_5=mean_of("ndcg_at_5"),
        mean_ndcg_at_10=mean_of("ndcg_at_10"),
        mean_latency_ms=round(sum(m.latency_ms for m in eval_metrics) / n, 2),
        domain_breakdowns=domain_summaries,
    )
