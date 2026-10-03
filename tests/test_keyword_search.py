"""Automated test suite for BM25 keyword search baseline."""

import pytest
from src.search.models import RetrievalDocument, SearchResult, SearchResponse
from src.search.corpus import RetrievalCorpus
from src.search.keyword import BM25SearchEngine, tokenize_text


@pytest.fixture
def sample_corpus(tmp_path):
    jsonl_file = tmp_path / "test_works.jsonl"
    data = [
        {
            "id": "W1",
            "title": "Mobile Money Adoption in Rural Kenya",
            "search_text": "Mobile Money Adoption in Rural Kenya. Mobile Banking. M-Pesa has expanded financial inclusion for rural businesses.",
            "abstract_text": "M-Pesa has expanded financial inclusion for rural businesses.",
            "topics": [{"name": "Mobile Banking"}],
            "publication_year": 2021,
            "doi": "10.1000/1",
            "type": "article",
            "is_oa": True,
            "oa_status": "gold"
        },
        {
            "id": "W2",
            "title": "Climate Change and Maize Crop Yields in Sub-Saharan Africa",
            "search_text": "Climate Change and Maize Crop Yields in Sub-Saharan Africa. Agricultural Climatology. Elevated temperatures reduce maize yield significantly.",
            "abstract_text": "Elevated temperatures reduce maize yield significantly.",
            "topics": [{"name": "Agricultural Climatology"}],
            "publication_year": 2022,
            "doi": "10.1000/2",
            "type": "article",
            "is_oa": False,
            "oa_status": "closed"
        },
        {
            "id": "W3",
            "title": "Malaria Transmission Dynamics and Child Health in Western Kenya",
            "search_text": "Malaria Transmission Dynamics and Child Health in Western Kenya. Vector Biology. Long-lasting insecticidal nets reduce malaria incidence in infants.",
            "abstract_text": "Long-lasting insecticidal nets reduce malaria incidence in infants.",
            "topics": [{"name": "Vector Biology"}],
            "publication_year": 2023,
            "doi": "10.1000/3",
            "type": "article",
            "is_oa": True,
            "oa_status": "green"
        }
    ]
    import json
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return RetrievalCorpus(corpus_path=jsonl_file)


def test_tokenize_text():
    tokens = tokenize_text("Mobile Money in Kenya! What about 2024?", remove_stopwords=True)
    assert tokens == ["mobile", "money", "kenya", "2024"]


def test_bm25_search_basic(sample_corpus):
    engine = BM25SearchEngine(corpus=sample_corpus)
    resp = engine.search("mobile money Kenya", top_k=2)

    assert isinstance(resp, SearchResponse)
    assert resp.total_hits >= 1
    # W1 matches 'mobile', 'money', 'kenya' -> highest score
    assert resp.results[0].paper_id == "W1"
    assert resp.results[0].rank == 1
    assert resp.results[0].score > 0.0
    assert resp.results[0].retrieval_method == "bm25"



def test_bm25_empty_and_whitespace_query(sample_corpus):
    engine = BM25SearchEngine(corpus=sample_corpus)
    resp_empty = engine.search("", top_k=5)
    assert resp_empty.total_hits == 0
    assert resp_empty.results == []

    resp_ws = engine.search("   \t\n  ", top_k=5)
    assert resp_ws.total_hits == 0
    assert resp_ws.results == []


def test_bm25_unknown_terms(sample_corpus):
    engine = BM25SearchEngine(corpus=sample_corpus)
    resp = engine.search("quantum teleportation astrophysics", top_k=5)
    assert resp.total_hits == 0
    assert resp.results == []


def test_bm25_stopwords_only(sample_corpus):
    engine = BM25SearchEngine(corpus=sample_corpus)
    resp = engine.search("what is the and or if", top_k=5)
    assert resp.total_hits == 0
    assert resp.results == []


def test_bm25_top_k_boundaries(sample_corpus):
    engine = BM25SearchEngine(corpus=sample_corpus)
    resp_zero = engine.search("Kenya", top_k=0)
    assert resp_zero.total_hits == 0

    resp_large = engine.search("Kenya", top_k=100)
    assert resp_large.total_hits <= len(sample_corpus)


def test_bm25_deterministic_ranking(sample_corpus):
    engine = BM25SearchEngine(corpus=sample_corpus)
    resp1 = engine.search("Kenya malaria", top_k=3)
    resp2 = engine.search("Kenya malaria", top_k=3)

    assert [r.paper_id for r in resp1.results] == [r.paper_id for r in resp2.results]
    assert [r.score for r in resp1.results] == [r.score for r in resp2.results]


def test_full_corpus_load():
    corpus = RetrievalCorpus()
    assert len(corpus) == 1000
    assert len(set(corpus.doc_map.keys())) == 1000
