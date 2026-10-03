"""Tests for metric calculations and profiling logic."""

import pytest
from src.ingestion.profiling import compute_field_completeness, calculate_summary_stats, profile_dataset


def test_calculate_summary_stats():
    values = [10, 20, 30, 40, 50]
    stats = calculate_summary_stats(values)
    assert stats["min"] == 10
    assert stats["max"] == 50
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0


def test_compute_field_completeness():
    records = [
        {"id": "1", "title": "Paper 1", "doi": "10.1/a"},
        {"id": "2", "title": "Paper 2", "doi": None},
        {"id": "3", "title": None, "doi": "10.1/c"},
        {"id": "4", "title": "Paper 4", "doi": "10.1/d"}
    ]

    extractors = {
        "Title": lambda r: bool(r.get("title")),
        "DOI": lambda r: bool(r.get("doi"))
    }

    completeness = compute_field_completeness(records, extractors)

    title_metric = next(m for m in completeness if m["field"] == "Title")
    assert title_metric["present"] == 3
    assert title_metric["missing"] == 1
    assert title_metric["present_pct"] == 75.0

    doi_metric = next(m for m in completeness if m["field"] == "DOI")
    assert doi_metric["present"] == 3
    assert doi_metric["missing"] == 1
    assert doi_metric["present_pct"] == 75.0


def test_profile_dataset_edge_cases():
    records = [
        {
            "id": "W1",
            "doi": "10.1000/1",
            "title": "Duplicate Title",
            "publication_year": 2020,
            "abstract_available": True,
            "abstract_word_count": 100,
            "authors_count": 2,
            "authors": [{"id": "A1"}],
            "institutions_count": 1,
            "institutions": [{"name": "UoN", "country_code": "KE"}],
            "has_ke_institution": True,
            "has_international_institution": False,
            "topics_count": 1,
            "topics": [{"name": "AI"}],
            "concepts_count": 0,
            "concepts": [],
            "cited_by_count": 5,
            "oa_status": "gold",
            "type": "article"
        },
        {
            "id": "W2",
            "doi": "10.1000/1",  # Duplicate DOI
            "title": "Duplicate Title",  # Duplicate Title
            "publication_year": 1950,  # Anomalous year (<1960)
            "abstract_available": False,
            "abstract_word_count": 0,
            "authors_count": 1,
            "authors": [],
            "institutions_count": 0,
            "institutions": [],
            "has_ke_institution": False,
            "has_international_institution": False,
            "topics_count": 0,
            "topics": [],
            "concepts_count": 0,
            "concepts": [],
            "cited_by_count": 0,
            "oa_status": "closed",
            "type": "article"
        }
    ]

    metrics = profile_dataset(records)

    assert metrics["total_records"] == 2
    assert metrics["edge_cases"]["duplicate_dois_count"] == 1
    assert metrics["edge_cases"]["duplicate_titles_count"] == 1
    assert metrics["edge_cases"]["anomalous_years_count"] == 1
    assert metrics["edge_cases"]["records_without_identified_ke_institution"] == 1
