"""Automated test suite for Dense Semantic Search (BAAI/bge-small-en-v1.5)."""

import pytest
import numpy as np
from src.search.models import SearchResponse, SearchResult
from src.search.corpus import RetrievalCorpus
from src.search.semantic import SemanticSearchEngine


@pytest.fixture
def mini_corpus(tmp_path):
    jsonl_file = tmp_path / "mini_works.jsonl"
    data = [
        {
            "id": "W101",
            "title": "Mobile Money Adoption in Small Retail Businesses in Nairobi",
            "search_text": "Mobile Money Adoption in Small Retail Businesses in Nairobi. Mobile Banking, Financial Inclusion. M-Pesa transactions enhance liquidity for small enterprises in Kenya.",
            "abstract_text": "M-Pesa transactions enhance liquidity for small enterprises in Kenya.",
            "topics": [{"name": "Mobile Banking"}, {"name": "Financial Inclusion"}],
            "publication_year": 2021,
            "doi": "10.1000/101",
            "type": "article",
            "is_oa": True,
            "oa_status": "gold"
        },
        {
            "id": "W102",
            "title": "Climate Variability and Maize Yield Reductions across Sub-Saharan Africa",
            "search_text": "Climate Variability and Maize Yield Reductions across Sub-Saharan Africa. Agricultural Climatology. Rising temperatures correlate with reduced crop productivity.",
            "abstract_text": "Rising temperatures correlate with reduced crop productivity.",
            "topics": [{"name": "Agricultural Climatology"}],
            "publication_year": 2022,
            "doi": "10.1000/102",
            "type": "article",
            "is_oa": False,
            "oa_status": "closed"
        },
        {
            "id": "W103",
            "title": "Efficacy of Malaria Vaccines in Infants in Western Kenya",
            "search_text": "Efficacy of Malaria Vaccines in Infants in Western Kenya. Vector Biology, Vaccine Immunology. Clinical trials demonstrate substantial protection against Plasmodium falciparum.",
            "abstract_text": "Clinical trials demonstrate substantial protection against Plasmodium falciparum.",
            "topics": [{"name": "Vector Biology"}, {"name": "Vaccine Immunology"}],
            "publication_year": 2023,
            "doi": "10.1000/103",
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


def test_semantic_engine_embedding_dimension_and_norm(mini_corpus, tmp_path):
    cache_file = tmp_path / "test_emb.npy"
    meta_file = tmp_path / "test_meta.json"

    engine = SemanticSearchEngine(
        corpus=mini_corpus,
        embeddings_cache_file=cache_file,
        metadata_cache_file=meta_file
    )

    # 1. Check dimension (N=3, D=384)
    assert engine.doc_embeddings.shape == (3, 384)

    # 2. Check unit L2 normalization (norm should be ~1.0 for each vector)
    norms = np.linalg.norm(engine.doc_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_semantic_search_retrieval(mini_corpus, tmp_path):
    engine = SemanticSearchEngine(
        corpus=mini_corpus,
        embeddings_cache_file=tmp_path / "test_emb.npy",
        metadata_cache_file=tmp_path / "test_meta.json"
    )

    # Query without exact keyword matching, testing semantic concept
    resp = engine.search("digital payment solutions for micro-enterprises", top_k=2)

    assert isinstance(resp, SearchResponse)
    assert resp.total_hits == 2
    assert resp.retrieval_method == "semantic"
    # W101 (Mobile Money in Small Retail) should rank #1
    assert resp.results[0].paper_id == "W101"
    assert resp.results[0].rank == 1
    assert -1.0 <= resp.results[0].score <= 1.0


def test_semantic_search_empty_and_invalid_queries(mini_corpus, tmp_path):
    engine = SemanticSearchEngine(
        corpus=mini_corpus,
        embeddings_cache_file=tmp_path / "test_emb.npy",
        metadata_cache_file=tmp_path / "test_meta.json"
    )

    # Empty query
    resp_empty = engine.search("", top_k=5)
    assert resp_empty.total_hits == 0
    assert resp_empty.results == []

    # Whitespace query
    resp_ws = engine.search("    \n\t  ", top_k=5)
    assert resp_ws.total_hits == 0
    assert resp_ws.results == []

    # Top-k zero
    resp_k0 = engine.search("malaria", top_k=0)
    assert resp_k0.total_hits == 0


def test_semantic_search_deterministic_ranking(mini_corpus, tmp_path):
    engine = SemanticSearchEngine(
        corpus=mini_corpus,
        embeddings_cache_file=tmp_path / "test_emb.npy",
        metadata_cache_file=tmp_path / "test_meta.json"
    )

    resp1 = engine.search("agricultural weather shocks and harvest", top_k=3)
    resp2 = engine.search("agricultural weather shocks and harvest", top_k=3)

    assert [r.paper_id for r in resp1.results] == [r.paper_id for r in resp2.results]
    assert [r.score for r in resp1.results] == [r.score for r in resp2.results]


def test_semantic_cache_consistency(mini_corpus, tmp_path):
    cache_file = tmp_path / "cached_emb.npy"
    meta_file = tmp_path / "cached_meta.json"

    # 1. First run generates and caches
    engine1 = SemanticSearchEngine(
        corpus=mini_corpus,
        embeddings_cache_file=cache_file,
        metadata_cache_file=meta_file
    )
    vec1 = engine1.doc_embeddings.copy()

    # 2. Second run loads from cache
    engine2 = SemanticSearchEngine(
        corpus=mini_corpus,
        embeddings_cache_file=cache_file,
        metadata_cache_file=meta_file
    )
    vec2 = engine2.doc_embeddings

    assert np.array_equal(vec1, vec2)
