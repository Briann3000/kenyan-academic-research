"""Information Retrieval (IR) metrics implementation.

Implements Precision@K, Recall@K (against judged pool), MRR, and graded nDCG@K.
"""

import math
from typing import List, Dict, Optional, Set
from src.search.models import SearchResult


def compute_precision_at_k(
    retrieved_doc_ids: List[str],
    qrels: Dict[str, int],
    k: int = 10,
    relevance_threshold: int = 1
) -> float:
    """Computes Precision@K.

    Args:
        retrieved_doc_ids: Ranked list of retrieved document/paper IDs.
        qrels: Mapping of paper_id -> graded relevance (0, 1, or 2).
        k: Cutoff rank.
        relevance_threshold: Minimum relevance score to be considered relevant (default: 1).

    Returns:
        Precision@K in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_doc_ids:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    relevant_count = 0
    for doc_id in top_k:
        rel = qrels.get(doc_id, 0)
        if rel >= relevance_threshold:
            relevant_count += 1

    return relevant_count / float(k)


def compute_recall_at_k(
    retrieved_doc_ids: List[str],
    qrels: Dict[str, int],
    k: int = 10,
    relevance_threshold: int = 1
) -> float:
    """Computes Recall@K against the set of judged relevant documents in the pooled qrels.

    Args:
        retrieved_doc_ids: Ranked list of retrieved document/paper IDs.
        qrels: Mapping of paper_id -> graded relevance (0, 1, or 2).
        k: Cutoff rank.
        relevance_threshold: Minimum relevance score to be considered relevant (default: 1).

    Returns:
        Recall@K in [0.0, 1.0]. Returns 0.0 if there are zero relevant documents in qrels.
    """
    if k <= 0 or not retrieved_doc_ids or not qrels:
        return 0.0

    total_relevant = sum(1 for rel in qrels.values() if rel >= relevance_threshold)
    if total_relevant == 0:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    retrieved_relevant = 0
    for doc_id in top_k:
        rel = qrels.get(doc_id, 0)
        if rel >= relevance_threshold:
            retrieved_relevant += 1

    return retrieved_relevant / float(total_relevant)


def compute_reciprocal_rank(
    retrieved_doc_ids: List[str],
    qrels: Dict[str, int],
    relevance_threshold: int = 1
) -> float:
    """Computes Reciprocal Rank (RR) for a single query.

    Args:
        retrieved_doc_ids: Ranked list of retrieved document/paper IDs.
        qrels: Mapping of paper_id -> graded relevance (0, 1, or 2).
        relevance_threshold: Minimum relevance score to count as the first relevant hit (default: 1).

    Returns:
        1.0 / rank of first relevant doc, or 0.0 if no relevant doc found.
    """
    if not retrieved_doc_ids or not qrels:
        return 0.0

    for rank_idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        rel = qrels.get(doc_id, 0)
        if rel >= relevance_threshold:
            return 1.0 / float(rank_idx)

    return 0.0


def compute_dcg_at_k(
    retrieved_doc_ids: List[str],
    qrels: Dict[str, int],
    k: int = 10
) -> float:
    """Computes Discounted Cumulative Gain (DCG@K) using standard graded relevance gains (2^rel - 1).

    Formula: sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)

    Args:
        retrieved_doc_ids: Ranked list of retrieved document/paper IDs.
        qrels: Mapping of paper_id -> graded relevance (0, 1, 2).
        k: Cutoff rank.

    Returns:
        DCG@K value.
    """
    if k <= 0 or not retrieved_doc_ids or not qrels:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    dcg = 0.0
    for rank_idx, doc_id in enumerate(top_k, start=1):
        rel = qrels.get(doc_id, 0)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(rank_idx + 1.0)
        dcg += gain / discount

    return dcg


def compute_idcg_at_k(
    qrels: Dict[str, int],
    k: int = 10
) -> float:
    """Computes Ideal Discounted Cumulative Gain (IDCG@K) from all judged documents for the query.

    Args:
        qrels: Mapping of paper_id -> graded relevance (0, 1, 2).
        k: Cutoff rank.

    Returns:
        IDCG@K value.
    """
    if k <= 0 or not qrels:
        return 0.0

    # Sort all available relevance judgments in descending order
    sorted_relevances = sorted(qrels.values(), reverse=True)
    top_k_ideal = sorted_relevances[:k]

    idcg = 0.0
    for rank_idx, rel in enumerate(top_k_ideal, start=1):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(rank_idx + 1.0)
        idcg += gain / discount

    return idcg


def compute_ndcg_at_k(
    retrieved_doc_ids: List[str],
    qrels: Dict[str, int],
    k: int = 10
) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@K) using graded 0/1/2 relevance.

    Formula: DCG@K / IDCG@K

    Args:
        retrieved_doc_ids: Ranked list of retrieved document/paper IDs.
        qrels: Mapping of paper_id -> graded relevance (0, 1, 2).
        k: Cutoff rank.

    Returns:
        nDCG@K in [0.0, 1.0]. Returns 0.0 if IDCG is 0.0.
    """
    idcg = compute_idcg_at_k(qrels, k=k)
    if idcg <= 0.0:
        return 0.0

    dcg = compute_dcg_at_k(retrieved_doc_ids, qrels, k=k)
    return min(1.0, dcg / idcg)
