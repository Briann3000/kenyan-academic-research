"""Tests for abstract reconstruction and record normalization."""

import pytest
from src.ingestion.normalizer import reconstruct_abstract, normalize_work


def test_reconstruct_abstract_basic():
    inverted_index = {
        "Mobile": [0],
        "money": [1],
        "in": [2],
        "Kenya": [3]
    }
    result = reconstruct_abstract(inverted_index)
    assert result == "Mobile money in Kenya"


def test_reconstruct_abstract_empty_and_null():
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None
    assert reconstruct_abstract({"test": []}) is None


def test_normalize_work_full():
    raw = {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.1000/182",
        "title": "Fintech and Financial Inclusion in Kenya",
        "publication_year": 2023,
        "publication_date": "2023-05-12",
        "type": "article",
        "cited_by_count": 15,
        "primary_location": {
            "source": {"display_name": "Journal of African Economies", "issn_l": "0963-8024"}
        },
        "open_access": {
            "is_oa": True,
            "oa_status": "gold",
            "oa_url": "https://example.com/oa"
        },
        "abstract_inverted_index": {
            "This": [0],
            "study": [1],
            "examines": [2],
            "M-Pesa.": [3]
        },
        "authorships": [
            {
                "author": {"id": "https://openalex.org/A999", "display_name": "Jane Mwangi"},
                "institutions": [
                    {
                        "id": "https://openalex.org/I888",
                        "display_name": "University of Nairobi",
                        "country_code": "KE",
                        "ror": "https://ror.org/03rh1gc68"
                    }
                ]
            }
        ],
        "primary_topic": {
            "display_name": "Mobile Banking and Financial Inclusion",
            "subfield": {"display_name": "Finance"},
            "field": {"display_name": "Economics"},
            "domain": {"display_name": "Social Sciences"}
        },
        "topics": [
            {"id": "https://openalex.org/T1", "display_name": "Mobile Banking and Financial Inclusion", "score": 0.99}
        ],
        "concepts": [
            {"id": "https://openalex.org/C1", "display_name": "Economics", "level": 0, "score": 0.8}
        ]
    }

    norm = normalize_work(raw)

    assert norm["id"] == "W12345"
    assert norm["doi"] == "https://doi.org/10.1000/182"
    assert norm["title"] == "Fintech and Financial Inclusion in Kenya"
    assert norm["abstract_available"] is True
    assert norm["abstract_text"] == "This study examines M-Pesa."
    assert norm["has_ke_institution"] is True
    assert norm["has_international_institution"] is False
    assert norm["authors_count"] == 1
    assert norm["institutions_count"] == 1
    assert norm["institutions"][0]["ror"] == "https://ror.org/03rh1gc68"
    assert "Fintech and Financial Inclusion in Kenya" in norm["search_text"]
    assert "M-Pesa." in norm["search_text"]


def test_normalize_work_missing_abstract():
    raw = {
        "id": "https://openalex.org/W54321",
        "title": "Agricultural Productivity in Kenya",
        "publication_year": 2021,
        "abstract_inverted_index": None,
        "authorships": [],
        "topics": [{"id": "https://openalex.org/T2", "display_name": "Maize Yield"}]
    }

    norm = normalize_work(raw)
    assert norm["abstract_available"] is False
    assert norm["abstract_text"] is None
    assert norm["search_text"] == "Agricultural Productivity in Kenya. Maize Yield"
