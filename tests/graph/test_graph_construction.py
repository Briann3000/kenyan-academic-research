"""Automated test suite for graph data modeling, entity resolution, and normalization."""

import pytest
from src.graph.models import PaperNode, ResearcherNode, InstitutionNode, TopicNode, YearNode
from src.graph.entity_resolution import EntityResolver, generate_synthetic_id
from src.graph.normalize import build_canonical_graph_elements
from src.graph.loader import GraphLoader
from src.graph.topology import profile_graph_topology


def test_entity_resolver_identity_and_variants():
    resolver = EntityResolver()

    # Same ID, different name spellings
    id1 = resolver.register_researcher("A100", "John Doe")
    id2 = resolver.register_researcher("A100", "J. Doe")
    assert id1 == id2 == "A100"

    # Different IDs, same name (homonyms)
    id3 = resolver.register_researcher("A200", "John Doe")
    assert id3 == "A200"

    # Missing author ID gets synthetic ID
    id_synth = resolver.register_researcher(None, "Peter Jones")
    assert id_synth.startswith("AUTH_SYNTH_")

    audit = resolver.audit_resolution()
    assert audit["researchers"]["total_unique_ids"] == 3
    assert audit["researchers"]["ids_with_multiple_name_variants"] == 1
    assert audit["researchers"]["names_shared_by_multiple_ids"] == 1
    assert audit["researchers"]["missing_source_ids_resolved_synthetically"] == 1


def test_build_canonical_elements_and_loader():
    sample_records = [
        {
            "id": "W001",
            "title": "Malaria Research in Western Kenya",
            "doi": "10.1000/1",
            "publication_year": 2022,
            "publication_date": "2022-04-01",
            "type": "article",
            "source_name": "Lancet",
            "is_oa": True,
            "oa_status": "gold",
            "abstract_text": "This paper analyzes malaria interventions in Kisumu.",
            "search_text": "Malaria Research in Western Kenya. Malaria. This paper analyzes malaria interventions in Kisumu.",
            "cited_by_count": 10,
            "authors": [
                {"id": "A1", "name": "Alice Ochieng"},
                {"id": "A2", "name": "Bob Smith"}
            ],
            "institutions": [
                {"id": "I1", "name": "KEMRI", "country_code": "KE", "ror": "https://ror.org/kemri"},
                {"id": "I2", "name": "Oxford", "country_code": "GB", "ror": "https://ror.org/oxford"}
            ],
            "topics": [
                {"id": "T1", "name": "Malaria Control", "score": 0.95}
            ]
        }
    ]

    elements, resolver = build_canonical_graph_elements(sample_records)
    nodes = elements["nodes"]
    relationships = elements["relationships"]

    assert len(nodes["papers"]) == 1
    assert len(nodes["researchers"]) == 2
    assert len(nodes["institutions"]) == 2
    assert len(nodes["topics"]) == 1
    assert len(nodes["years"]) == 1

    # Check relationship counts:
    # 2 AUTHORED + 2 AFFILIATED_WITH + 1 ABOUT + 1 PUBLISHED_IN = 6 edges
    assert len(relationships) == 6

    # Test NetworkX builder and topology
    loader = GraphLoader()
    nx_g = loader.build_networkx_graph(elements)

    assert nx_g.number_of_nodes() == 7  # 1 paper + 2 researchers + 2 insts + 1 topic + 1 year
    assert nx_g.number_of_edges() == 6

    topo = profile_graph_topology(nx_g, elements)
    assert topo["connectivity"]["total_connected_components"] == 1
    assert topo["connectivity"]["isolated_nodes_count"] == 0
    assert topo["integrity_checks"]["papers_without_authors"] == 0
    assert topo["integrity_checks"]["papers_without_institutions"] == 0
    assert topo["integrity_checks"]["papers_without_topics"] == 0
