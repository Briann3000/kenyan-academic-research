"""Unit tests for Information Retrieval metrics and paired significance tests."""

import math
import pytest

from src.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_dcg_at_k,
    compute_idcg_at_k,
    compute_ndcg_at_k,
)
from src.evaluation.significance import (
    paired_t_test,
    wilcoxon_signed_rank_test,
)


def test_perfect_ranking():
    """Verifies that an ideal top-K ranking gives 1.0 across all metrics."""
    retrieved = ["D1", "D2", "D3", "D4", "D5"]
    qrels = {"D1": 2, "D2": 2, "D3": 2, "D4": 2, "D5": 2}

    assert compute_precision_at_k(retrieved, qrels, k=5) == 1.0
    assert compute_recall_at_k(retrieved, qrels, k=5) == 1.0
    assert compute_reciprocal_rank(retrieved, qrels) == 1.0
    assert compute_ndcg_at_k(retrieved, qrels, k=5) == 1.0


def test_completely_irrelevant_ranking():
    """Verifies that ranking only irrelevant documents gives 0.0."""
    retrieved = ["D1", "D2", "D3", "D4", "D5"]
    qrels = {"D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0, "D_REL": 2}

    assert compute_precision_at_k(retrieved, qrels, k=5) == 0.0
    assert compute_recall_at_k(retrieved, qrels, k=5) == 0.0
    assert compute_reciprocal_rank(retrieved, qrels) == 0.0
    assert compute_ndcg_at_k(retrieved, qrels, k=5) == 0.0


def test_zero_relevant_judgments_in_qrels():
    """Verifies safe handling when qrels has no relevant documents."""
    retrieved = ["D1", "D2"]
    qrels = {"D1": 0, "D2": 0}

    assert compute_precision_at_k(retrieved, qrels, k=5) == 0.0
    assert compute_recall_at_k(retrieved, qrels, k=5) == 0.0
    assert compute_reciprocal_rank(retrieved, qrels) == 0.0
    assert compute_ndcg_at_k(retrieved, qrels, k=5) == 0.0


def test_fewer_than_k_results():
    """Verifies metric calculations when fewer than K results are retrieved."""
    retrieved = ["D1", "D2"]
    qrels = {"D1": 2, "D2": 1, "D3": 2, "D4": 2}

    # 2 relevant in retrieved out of k=5 -> precision = 2 / 5 = 0.4
    assert compute_precision_at_k(retrieved, qrels, k=5) == 0.4
    # 2 relevant retrieved out of 4 total relevant in pool -> recall = 2 / 4 = 0.50
    assert compute_recall_at_k(retrieved, qrels, k=5) == 0.50


def test_reciprocal_rank_threshold():
    """Verifies MRR calculation with first relevant at rank 3."""
    retrieved = ["D1", "D2", "D3", "D4"]
    qrels = {"D1": 0, "D2": 0, "D3": 1, "D4": 2}

    assert compute_reciprocal_rank(retrieved, qrels, relevance_threshold=1) == pytest.approx(1.0 / 3.0, abs=1e-5)
    # If threshold is 2, first highly relevant is at rank 4 -> 1/4
    assert compute_reciprocal_rank(retrieved, qrels, relevance_threshold=2) == pytest.approx(1.0 / 4.0, abs=1e-5)


def test_hand_calculated_ndcg_graded():
    """Verifies graded nDCG calculation against manual mathematical calculation.

    Retrieved list: [D1, D2, D3]
    Relevance: D1=2, D2=0, D3=1
    Pool contains: D1=2, D2=0, D3=1, D4=2

    DCG@3:
    rank 1: (2^2 - 1) / log2(2) = 3 / 1.0 = 3.0
    rank 2: (2^0 - 1) / log2(3) = 0 / 1.58496 = 0.0
    rank 3: (2^1 - 1) / log2(4) = 1 / 2.0 = 0.5
    Total DCG@3 = 3.5

    Ideal ranking for top 3 from pool: [2, 2, 1]
    rank 1: (2^2 - 1) / log2(2) = 3.0
    rank 2: (2^2 - 1) / log2(3) = 3 / 1.5849625 = 1.892789
    rank 3: (2^1 - 1) / log2(4) = 1 / 2.0 = 0.5
    Total IDCG@3 = 3.0 + 1.892789 + 0.5 = 5.392789

    nDCG@3 = 3.5 / 5.392789 = 0.64898
    """
    retrieved = ["D1", "D2", "D3"]
    qrels = {"D1": 2, "D2": 0, "D3": 1, "D4": 2}

    dcg = compute_dcg_at_k(retrieved, qrels, k=3)
    idcg = compute_idcg_at_k(qrels, k=3)
    ndcg = compute_ndcg_at_k(retrieved, qrels, k=3)

    assert dcg == pytest.approx(3.5, abs=1e-4)
    expected_idcg = 3.0 + (3.0 / math.log2(3)) + 0.5
    assert idcg == pytest.approx(expected_idcg, abs=1e-4)
    assert ndcg == pytest.approx(3.5 / expected_idcg, abs=1e-4)


def test_paired_t_test_identical_and_differing():
    """Verifies paired t-test for identical distributions and known difference."""
    # Identical
    res_ident = paired_t_test([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert res_ident["mean_diff"] == 0.0
    assert res_ident["p_value"] == 1.0

    # System B uniformly higher than System A by +0.10
    a = [0.5, 0.6, 0.7, 0.4, 0.8]
    b = [0.6, 0.7, 0.8, 0.5, 0.9]
    res = paired_t_test(a, b)
    assert res["mean_diff"] == pytest.approx(0.10, abs=1e-5)
    assert res["std_diff"] == pytest.approx(0.0, abs=1e-5)


def test_wilcoxon_signed_rank_test_basic():
    """Verifies Wilcoxon signed-rank test calculation."""
    # 5 pairs where B is consistently better
    a = [0.1, 0.2, 0.3, 0.4, 0.5]
    b = [0.3, 0.4, 0.5, 0.6, 0.7]
    res = wilcoxon_signed_rank_test(a, b)

    assert res["n_nonzero"] == 5
    assert res["w_plus"] == 15.0  # Ranks 1+2+3+4+5 = 15
    assert res["w_minus"] == 0.0
    assert res["w_stat"] == 0.0
    assert res["rank_biserial_r"] == 1.0
