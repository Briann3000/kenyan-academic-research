"""Unit and integration tests for GraphHybridSearchEngine (System C).

Verifies candidate restriction, alpha bounds, graph overlap scoring, absence of popularity bias,
determinism, Neo4j fallback resilience, and response format compatibility.
"""

from typing import List, Dict, Any
from unittest.mock import MagicMock
import pytest

from src.search.models import SearchResult, SearchResponse
from src.search.graph_hybrid import GraphHybridSearchEngine, DEFAULT_GRAPH_WEIGHTS


def create_mock_semantic_result(paper_id: str, title: str, score: float, rank: int) -> SearchResult:
    """Helper to construct a mock SearchResult from semantic search."""
    return SearchResult(
        paper_id=paper_id,
        title=title,
        score=score,
        rank=rank,
        publication_year=2023,
        doi=f"https://doi.org/10.1000/{paper_id}",
        retrieval_method="semantic",
        snippet=f"Snippet for {paper_id}",
        metadata={"orig_semantic_score": score},
    )


class MockNeo4jClient:
    """Mock Neo4jClient returning pre-configured Cypher query records."""

    def __init__(self, records: List[Dict[str, Any]], should_fail: bool = False):
        self.records = records
        self.should_fail = should_fail

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if self.should_fail:
            raise ConnectionError("Simulated Neo4j connection failure.")
        
        # If parameters contain paper_ids, filter to those paper_ids
        if parameters and "paper_ids" in parameters:
            p_ids = set(parameters["paper_ids"])
            return [r for r in self.records if r["paper_id"] in p_ids]
        return self.records


@pytest.fixture
def sample_candidates() -> List[SearchResult]:
    """Generates 5 sample candidates from semantic search."""
    return [
        create_mock_semantic_result("W101", "Malaria Prevention in Kenya", 0.85, 1),
        create_mock_semantic_result("W102", "Vector Control in Western Kenya", 0.82, 2),
        create_mock_semantic_result("W103", "Maize Yield Optimization", 0.78, 3),
        create_mock_semantic_result("W104", "Pediatric Febrile Illness", 0.75, 4),
        create_mock_semantic_result("W105", "Financial Inclusion and M-Pesa", 0.70, 5),
    ]


@pytest.fixture
def mock_subgraph_records() -> List[Dict[str, Any]]:
    """Generates controlled graph relationships for the sample candidates:
    - W101 & W102 & W104 share Topic 'T_Health' and Researcher 'A_Kemri' and Inst 'I_UoN'.
    - W103 is in Agriculture (Topic 'T_Agri', Inst 'I_KALRO').
    - W105 has high degree with unrelated nodes (topics 'T_Fin1', 'T_Fin2', 'T_Fin3', author 'A_Fin').
    """
    return [
        {
            "paper_id": "W101",
            "topic_ids": ["T_Health", "T_Vector"],
            "author_ids": ["A_Kemri", "A_John"],
            "institution_ids": ["I_UoN"],
        },
        {
            "paper_id": "W102",
            "topic_ids": ["T_Health", "T_Vector"],
            "author_ids": ["A_Kemri", "A_Alice"],
            "institution_ids": ["I_UoN"],
        },
        {
            "paper_id": "W103",
            "topic_ids": ["T_Agri"],
            "author_ids": ["A_Bob"],
            "institution_ids": ["I_KALRO"],
        },
        {
            "paper_id": "W104",
            "topic_ids": ["T_Health"],
            "author_ids": ["A_Kemri"],
            "institution_ids": ["I_UoN"],
        },
        {
            "paper_id": "W105",
            # High degree of non-overlapping nodes (tests popularity vs relational overlap)
            "topic_ids": ["T_Fin1", "T_Fin2", "T_Fin3", "T_Fin4", "T_Fin5"],
            "author_ids": ["A_Econ1", "A_Econ2", "A_Econ3"],
            "institution_ids": ["I_Econ1", "I_Econ2"],
        },
    ]


def test_candidate_restriction(sample_candidates, mock_subgraph_records):
    """Verifies System C reranks ONLY the candidates returned by System B."""
    mock_semantic = MagicMock()
    mock_semantic.search.return_value = SearchResponse(
        query="malaria in kenya",
        retrieval_method="semantic",
        total_hits=len(sample_candidates),
        latency_ms=5.0,
        results=sample_candidates,
    )
    mock_neo4j = MockNeo4jClient(mock_subgraph_records)

    engine = GraphHybridSearchEngine(
        semantic_engine=mock_semantic,
        neo4j_client=mock_neo4j,
        candidate_k=5,
    )

    response = engine.search("malaria in kenya", top_k=5)
    returned_ids = [r.paper_id for r in response.results]
    expected_ids = {"W101", "W102", "W103", "W104", "W105"}

    assert set(returned_ids) == expected_ids
    assert len(returned_ids) == 5
    # Verify semantic search was called with candidate_k
    mock_semantic.search.assert_called_once_with("malaria in kenya", top_k=5)


def test_alpha_pure_semantic(sample_candidates, mock_subgraph_records):
    """Verifies that alpha=1.0 strictly reproduces the original semantic ranking."""
    mock_semantic = MagicMock()
    mock_semantic.search.return_value = SearchResponse(
        query="test query",
        retrieval_method="semantic",
        total_hits=len(sample_candidates),
        latency_ms=5.0,
        results=sample_candidates,
    )
    mock_neo4j = MockNeo4jClient(mock_subgraph_records)

    engine = GraphHybridSearchEngine(
        semantic_engine=mock_semantic,
        neo4j_client=mock_neo4j,
        alpha=1.0,
    )

    response = engine.search("test query", top_k=5, alpha=1.0)
    # Ranks should exactly match original semantic candidate order
    for idx, res in enumerate(response.results, start=1):
        assert res.rank == idx
        assert res.paper_id == sample_candidates[idx - 1].paper_id
        assert res.score == pytest.approx(sample_candidates[idx - 1].score, abs=1e-5)
        assert res.metadata["rank_change"] == 0


def test_alpha_pure_graph(sample_candidates, mock_subgraph_records):
    """Verifies that alpha=0.0 ranks purely by graph affinity score."""
    mock_semantic = MagicMock()
    mock_semantic.search.return_value = SearchResponse(
        query="malaria health",
        retrieval_method="semantic",
        total_hits=len(sample_candidates),
        latency_ms=5.0,
        results=sample_candidates,
    )
    mock_neo4j = MockNeo4jClient(mock_subgraph_records)

    engine = GraphHybridSearchEngine(
        semantic_engine=mock_semantic,
        neo4j_client=mock_neo4j,
        alpha=0.0,
    )

    response = engine.search("malaria health", top_k=5, alpha=0.0)
    # In graph records: W101 and W102 have highest overlap (shared topics, author A_Kemri, inst I_UoN)
    # W103 and W105 have 0 overlap with others in candidate set.
    top_ids = [r.paper_id for r in response.results[:2]]
    assert set(top_ids) == {"W101", "W102"}
    
    # W103 and W105 should have graph score of 0.0
    for res in response.results:
        if res.paper_id in ["W103", "W105"]:
            assert res.metadata["graph_score"] == 0.0


def test_no_popularity_bias(sample_candidates, mock_subgraph_records):
    """Verifies that high node degree without candidate overlap does not give high graph score."""
    mock_semantic = MagicMock()
    mock_semantic.search.return_value = SearchResponse(
        query="health query",
        retrieval_method="semantic",
        total_hits=len(sample_candidates),
        latency_ms=5.0,
        results=sample_candidates,
    )
    mock_neo4j = MockNeo4jClient(mock_subgraph_records)

    engine = GraphHybridSearchEngine(
        semantic_engine=mock_semantic,
        neo4j_client=mock_neo4j,
    )

    response = engine.search("health query", top_k=5)
    w105_res = next(r for r in response.results if r.paper_id == "W105")

    # W105 had 5 topics, 3 authors, 2 institutions (highest raw degree), but ZERO shared with candidates
    assert w105_res.metadata["raw_topic_overlap"] == 0
    assert w105_res.metadata["raw_author_overlap"] == 0
    assert w105_res.metadata["raw_institution_overlap"] == 0
    assert w105_res.metadata["graph_score"] == 0.0


def test_graph_overlap_calculation_accuracy():
    """Verifies exact manual calculation of candidate-relative overlap."""
    paper_ids = ["P1", "P2", "P3"]
    subgraph_data = {
        "P1": {"topics": {"T1", "T2"}, "authors": {"A1"}, "institutions": {"I1"}},
        "P2": {"topics": {"T2", "T3"}, "authors": {"A1"}, "institutions": {"I1"}},
        "P3": {"topics": {"T1"}, "authors": {"A2"}, "institutions": {"I2"}},
    }

    overlaps = GraphHybridSearchEngine._calculate_candidate_relative_overlaps(paper_ids, subgraph_data)

    # Topics:
    # P1 & P2 share {T2} (1)
    # P1 & P3 share {T1} (1) -> raw_topic(P1) = 2
    # P2 & P3 share {} (0)   -> raw_topic(P2) = 1, raw_topic(P3) = 1
    # Max topic raw = 2
    assert overlaps["P1"]["raw_topic"] == 2
    assert overlaps["P1"]["o_topic"] == 1.0
    assert overlaps["P2"]["raw_topic"] == 1
    assert overlaps["P2"]["o_topic"] == 0.5
    assert overlaps["P3"]["raw_topic"] == 1
    assert overlaps["P3"]["o_topic"] == 0.5

    # Authors:
    # P1 & P2 share {A1} (1) -> raw_author(P1) = 1, P2 = 1, P3 = 0
    assert overlaps["P1"]["raw_author"] == 1
    assert overlaps["P1"]["o_author"] == 1.0
    assert overlaps["P2"]["raw_author"] == 1
    assert overlaps["P2"]["o_author"] == 1.0
    assert overlaps["P3"]["raw_author"] == 0
    assert overlaps["P3"]["o_author"] == 0.0


def test_neo4j_fallback_handling(sample_candidates):
    """Verifies that if Neo4j is unavailable or fails, fallback is clean and transparently flagged."""
    mock_semantic = MagicMock()
    mock_semantic.search.return_value = SearchResponse(
        query="malaria query",
        retrieval_method="semantic",
        total_hits=len(sample_candidates),
        latency_ms=5.0,
        results=sample_candidates,
    )
    # Simulated failure client
    failing_client = MockNeo4jClient([], should_fail=True)

    engine = GraphHybridSearchEngine(
        semantic_engine=mock_semantic,
        neo4j_client=failing_client,
    )

    response = engine.search("malaria query", top_k=5)
    assert response.total_hits == 5
    for res in response.results:
        assert res.metadata["fallback_used"] is True
        assert res.metadata["graph_available"] is False
        assert res.metadata["graph_score"] == 0.0
        # Hybrid score gracefully matches semantic score
        assert res.score == pytest.approx(res.metadata["semantic_score"], abs=1e-5)


def test_empty_and_whitespace_queries():
    """Verifies empty and whitespace queries return empty SearchResponse without error."""
    mock_semantic = MagicMock()
    engine = GraphHybridSearchEngine(semantic_engine=mock_semantic)
    for empty_q in ["", "   ", "\t\n"]:
        resp = engine.search(empty_q)
        assert resp.total_hits == 0
        assert resp.results == []
        assert resp.retrieval_method == "graph_hybrid"
    mock_semantic.search.assert_not_called()


def test_deterministic_ranking(sample_candidates, mock_subgraph_records):
    """Verifies repeated queries with identical inputs yield strictly deterministic results."""
    mock_semantic = MagicMock()
    mock_semantic.search.return_value = SearchResponse(
        query="malaria kenya",
        retrieval_method="semantic",
        total_hits=len(sample_candidates),
        latency_ms=5.0,
        results=sample_candidates,
    )
    mock_neo4j = MockNeo4jClient(mock_subgraph_records)

    engine = GraphHybridSearchEngine(
        semantic_engine=mock_semantic,
        neo4j_client=mock_neo4j,
        candidate_k=5,
        alpha=0.7,
    )

    res1 = engine.search("malaria kenya", top_k=5)
    res2 = engine.search("malaria kenya", top_k=5)

    assert [r.paper_id for r in res1.results] == [r.paper_id for r in res2.results]
    assert [r.score for r in res1.results] == [r.score for r in res2.results]


def test_response_structure_compatibility(sample_candidates, mock_subgraph_records):
    """Verifies all SearchResponse and SearchResult schema requirements."""
    mock_semantic = MagicMock()
    mock_semantic.search.return_value = SearchResponse(
        query="malaria",
        retrieval_method="semantic",
        total_hits=len(sample_candidates),
        latency_ms=5.0,
        results=sample_candidates,
    )
    mock_neo4j = MockNeo4jClient(mock_subgraph_records)

    engine = GraphHybridSearchEngine(semantic_engine=mock_semantic, neo4j_client=mock_neo4j)
    resp = engine.search("malaria", top_k=3)

    assert isinstance(resp, SearchResponse)
    assert resp.retrieval_method == "graph_hybrid"
    assert resp.total_hits == 3
    assert len(resp.results) == 3

    for item in resp.results:
        assert isinstance(item, SearchResult)
        assert item.retrieval_method == "graph_hybrid"
        assert "semantic_score" in item.metadata
        assert "graph_score" in item.metadata
        assert "hybrid_score" in item.metadata
        assert "semantic_rank" in item.metadata
        assert "hybrid_rank" in item.metadata
        assert "rank_change" in item.metadata
        assert "topic_overlap" in item.metadata
        assert "author_overlap" in item.metadata
        assert "institution_overlap" in item.metadata
