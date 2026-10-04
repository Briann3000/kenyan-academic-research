"""Statistical significance testing for paired IR evaluation metrics.

Implements paired Student's t-test and Wilcoxon signed-rank test without external dependencies.
"""

import math
from typing import List, Dict, Any, Tuple


def _normal_cdf(x: float) -> float:
    """Standard Normal Cumulative Distribution Function (Phi(x)) using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_distribution_two_tailed_p(t_stat: float, df: int) -> float:
    """Calculates two-tailed p-value for Student's t-distribution with df degrees of freedom.

    Uses the regularized incomplete beta function expansion for exact/high-precision p-values.
    """
    if df <= 0:
        return 1.0

    t2 = t_stat * t_stat
    x = df / (df + t2)

    # Incomplete beta function approximation via continued fraction / series
    # For Student t: p = I_x(df/2, 1/2)
    a = df / 2.0
    b = 0.5

    # Continued fraction for regularized incomplete beta I_x(a, b)
    def betacf(a_val: float, b_val: float, x_val: float) -> float:
        max_iter = 200
        eps = 3.0e-7
        qab = a_val + b_val
        qap = a_val + 1.0
        qam = a_val - 1.0
        c = 1.0
        d = 1.0 - qab * x_val / qap
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        d = 1.0 / d
        h = d

        for m in range(1, max_iter + 1):
            m2 = 2 * m
            # Even step
            aa = m * (b_val - m) * x_val / ((qam + m2) * (a_val + m2))
            d = 1.0 + aa * d
            if abs(d) < 1.0e-30:
                d = 1.0e-30
            c = 1.0 + aa / c
            if abs(c) < 1.0e-30:
                c = 1.0e-30
            d = 1.0 / d
            h *= d * c

            # Odd step
            aa = -(a_val + m) * (qab + m) * x_val / ((a_val + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1.0e-30:
                d = 1.0e-30
            c = 1.0 + aa / c
            if abs(c) < 1.0e-30:
                c = 1.0e-30
            d = 1.0 / d
            del_val = d * c
            h *= del_val
            if abs(del_val - 1.0) < eps:
                break
        return h

    # ln(Beta(a, b)) = lgamma(a) + lgamma(b) - lgamma(a+b)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    bt = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) if 0.0 < x < 1.0 else 0.0

    if x < (a + 1.0) / (a + b + 2.0):
        ibeta = bt * betacf(a, b, x) / a
    else:
        ibeta = 1.0 - (bt * betacf(b, a, 1.0 - x) / b)

    return max(0.0, min(1.0, float(ibeta)))


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float]
) -> Dict[str, Any]:
    """Computes paired Student's t-test comparing two retrieval systems across matched queries.

    Args:
        scores_a: List of per-query metric values for System A.
        scores_b: List of per-query metric values for System B.

    Returns:
        Dictionary containing:
          - n_queries: number of matched query pairs
          - mean_diff: mean of paired differences (scores_b - scores_a)
          - std_diff: sample standard deviation of differences
          - t_stat: Student's t-statistic
          - p_value: two-tailed p-value
          - cohens_d: paired effect size (Cohen's d_z = mean_diff / std_diff)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Score lists must have identical lengths, got {len(scores_a)} vs {len(scores_b)}")

    n = len(scores_a)
    if n < 2:
        return {
            "n_queries": n,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
        }

    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = math.sqrt(var_d)

    if std_d < 1e-12:
        # All differences are identical
        return {
            "n_queries": n,
            "mean_diff": round(mean_d, 6),
            "std_diff": 0.0,
            "t_stat": 0.0 if abs(mean_d) < 1e-12 else (float("inf") if mean_d > 0 else float("-inf")),
            "p_value": 1.0 if abs(mean_d) < 1e-12 else 0.0,
            "cohens_d": 0.0,
        }

    se_d = std_d / math.sqrt(n)
    t_stat = mean_d / se_d
    df = n - 1
    p_val = _t_distribution_two_tailed_p(t_stat, df)
    cohens_d = mean_d / std_d

    return {
        "n_queries": n,
        "mean_diff": round(mean_d, 6),
        "std_diff": round(std_d, 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 6),
        "cohens_d": round(cohens_d, 4),
    }


def wilcoxon_signed_rank_test(
    scores_a: List[float],
    scores_b: List[float]
) -> Dict[str, Any]:
    """Computes Wilcoxon signed-rank test for paired metric differences with Pratt/zero-handling.

    Args:
        scores_a: List of per-query metric values for System A.
        scores_b: List of per-query metric values for System B.

    Returns:
        Dictionary containing:
          - n_total: total query pairs
          - n_nonzero: non-zero difference query pairs
          - w_stat: Wilcoxon test statistic (min(W+, W-))
          - z_stat: asymptotic standard normal z-statistic
          - p_value: two-tailed p-value
          - rank_biserial_r: matched-pairs rank biserial correlation effect size
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Score lists must have identical lengths, got {len(scores_a)} vs {len(scores_b)}")

    n_total = len(scores_a)
    diffs = [round(b - a, 8) for a, b in zip(scores_a, scores_b)]

    # Filter out exact zero differences
    non_zero_items = [(abs(d), 1 if d > 0 else -1) for d in diffs if abs(d) > 1e-9]
    n_nonzero = len(non_zero_items)

    if n_nonzero == 0:
        return {
            "n_total": n_total,
            "n_nonzero": 0,
            "w_stat": 0.0,
            "z_stat": 0.0,
            "p_value": 1.0,
            "rank_biserial_r": 0.0,
        }

    # Sort by absolute difference
    non_zero_items.sort(key=lambda x: x[0])

    # Assign ranks with fractional tie-breaking
    ranks = [0.0] * n_nonzero
    i = 0
    tie_groups = []
    while i < n_nonzero:
        j = i
        while j < n_nonzero and abs(non_zero_items[j][0] - non_zero_items[i][0]) < 1e-9:
            j += 1
        group_size = j - i
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        if group_size > 1:
            tie_groups.append(group_size)
        i = j

    w_plus = sum(ranks[k] for k in range(n_nonzero) if non_zero_items[k][1] > 0)
    w_minus = sum(ranks[k] for k in range(n_nonzero) if non_zero_items[k][1] < 0)
    w_stat = min(w_plus, w_minus)

    # Effect size: Rank Biserial Correlation r = (W+ - W-) / (W+ + W-)
    total_w = w_plus + w_minus
    rank_biserial_r = (w_plus - w_minus) / total_w if total_w > 0 else 0.0

    # Normal approximation with continuity correction and tie adjustment
    e_w = n_nonzero * (n_nonzero + 1) / 4.0
    var_w = (n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1)) / 24.0

    # Tie correction for variance
    for t_k in tie_groups:
        var_w -= (t_k ** 3 - t_k) / 48.0

    if var_w <= 0:
        z_stat = 0.0
        p_val = 1.0
    else:
        # Continuity correction
        std_w = math.sqrt(var_w)
        if w_stat < e_w:
            z_stat = (w_stat - e_w + 0.5) / std_w
        elif w_stat > e_w:
            z_stat = (w_stat - e_w - 0.5) / std_w
        else:
            z_stat = 0.0

        p_val = 2.0 * (1.0 - _normal_cdf(abs(z_stat)))

    return {
        "n_total": n_total,
        "n_nonzero": n_nonzero,
        "w_stat": round(w_stat, 2),
        "w_plus": round(w_plus, 2),
        "w_minus": round(w_minus, 2),
        "z_stat": round(z_stat, 4),
        "p_value": round(p_val, 6),
        "rank_biserial_r": round(rank_biserial_r, 4),
    }
