"""Unit and integration tests for relationship discovery and entity diversity evaluation."""

import pytest
import math
from src.evaluation.relationship_diversity import (
    compute_shannon_entropy,
    compute_simpson_diversity,
)


def test_shannon_entropy_calculation():
    # Empty list
    assert compute_shannon_entropy([]) == 0.0

    # All identical elements (zero entropy)
    assert compute_shannon_entropy(["UoN", "UoN", "UoN"]) == 0.0

    # Two equally distributed elements (1 bit of entropy)
    assert compute_shannon_entropy(["UoN", "KEMRI"]) == 1.0

    # 4 equally distributed elements (2 bits of entropy)
    items = ["A", "B", "C", "D"]
    assert compute_shannon_entropy(items) == 2.0


def test_simpson_diversity_calculation():
    # Empty or single element
    assert compute_simpson_diversity([]) == 0.0
    assert compute_simpson_diversity(["UoN"]) == 0.0

    # All identical
    assert compute_simpson_diversity(["KEMRI", "KEMRI", "KEMRI"]) == 0.0

    # Two equal classes -> 1 - (0.5^2 + 0.5^2) = 0.5
    assert compute_simpson_diversity(["A", "B"]) == 0.5
