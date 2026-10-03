"""Extracts and builds canonical node and relationship collections from normalized records."""

import logging
from typing import Dict, Any, List, Set, Tuple
from src.graph.models import (
    PaperNode,
    ResearcherNode,
    InstitutionNode,
    TopicNode,
    YearNode,
    GraphRelationship
)
from src.graph.entity_resolution import EntityResolver

logger = logging.getLogger(__name__)


def build_canonical_graph_elements(
    normalized_records: List[Dict[str, Any]],
    resolver: EntityResolver | None = None
) -> Tuple[Dict[str, Any], EntityResolver]:
    """Transforms normalized work records into distinct sets of canonical nodes and edges."""
    resolver = resolver or EntityResolver()

    papers: Dict[str, PaperNode] = {}
    researchers: Dict[str, ResearcherNode] = {}
    institutions: Dict[str, InstitutionNode] = {}
    topics: Dict[str, TopicNode] = {}
    years: Dict[int, YearNode] = {}

    relationships: List[GraphRelationship] = []
    seen_rel_keys: Set[Tuple[str, str, str, str, str]] = set()

    def add_relationship(src_type: str, src_id: str, tgt_type: str, tgt_id: str, rel_type: str, props: Dict[str, Any] | None = None):
        key = (src_type, src_id, tgt_type, tgt_id, rel_type)
        if key not in seen_rel_keys:
            seen_rel_keys.add(key)
            relationships.append(GraphRelationship(
                source_type=src_type,
                source_id=src_id,
                target_type=tgt_type,
                target_id=tgt_id,
                rel_type=rel_type,
                properties=props or {}
            ))

    for rec in normalized_records:
        paper_id = rec["id"]
        pub_year = rec.get("publication_year")

        # 1. Year Node
        if pub_year and pub_year not in years:
            years[pub_year] = YearNode(value=pub_year)

        # 2. Paper Node
        if paper_id not in papers:
            papers[paper_id] = PaperNode(
                id=paper_id,
                title=rec.get("title") or "Untitled",
                doi=rec.get("doi"),
                publication_year=pub_year,
                publication_date=rec.get("publication_date"),
                work_type=rec.get("type"),
                source_name=rec.get("source_name"),
                is_oa=rec.get("is_oa", False),
                oa_status=rec.get("oa_status"),
                abstract=rec.get("abstract_text"),
                search_text=rec.get("search_text") or rec.get("title", ""),
                cited_by_count=rec.get("cited_by_count", 0),
                openalex_url=rec.get("raw_id") or f"https://openalex.org/{paper_id}"
            )

        # Connect Paper -> PUBLISHED_IN -> Year
        if pub_year:
            add_relationship("Paper", paper_id, "Year", str(pub_year), "PUBLISHED_IN")

        # 3. Researchers & AUTHORED relationships
        for auth in rec.get("authors", []):
            raw_author_id = auth.get("id")
            author_name = auth.get("name") or "Unknown Author"
            canonical_author_id = resolver.register_researcher(raw_author_id, author_name)

            if canonical_author_id not in researchers:
                researchers[canonical_author_id] = ResearcherNode(
                    id=canonical_author_id,
                    display_name=author_name,
                    orcid=None,
                    openalex_url=f"https://openalex.org/{raw_author_id}" if raw_author_id else None
                )

            # Connect Researcher -> AUTHORED -> Paper
            add_relationship("Researcher", canonical_author_id, "Paper", paper_id, "AUTHORED")

        # 4. Institutions & AFFILIATED_WITH relationships
        for inst in rec.get("institutions", []):
            raw_inst_id = inst.get("id")
            inst_name = inst.get("name") or "Unknown Institution"
            inst_ror = inst.get("ror")
            canonical_inst_id = resolver.register_institution(raw_inst_id, inst_name, inst_ror)

            if canonical_inst_id not in institutions:
                institutions[canonical_inst_id] = InstitutionNode(
                    id=canonical_inst_id,
                    display_name=inst_name,
                    ror=inst_ror,
                    country_code=inst.get("country_code"),
                    type=inst.get("type"),
                    openalex_url=f"https://openalex.org/{raw_inst_id}" if raw_inst_id else None
                )

            # Connect Paper -> AFFILIATED_WITH -> Institution
            add_relationship("Paper", paper_id, "Institution", canonical_inst_id, "AFFILIATED_WITH")

        # 5. Topics & ABOUT relationships
        for top in rec.get("topics", []):
            topic_id = top.get("id")
            topic_name = top.get("name")
            if not topic_id or not topic_name:
                continue

            resolver.register_topic(topic_id, topic_name)

            if topic_id not in topics:
                topics[topic_id] = TopicNode(
                    id=topic_id,
                    display_name=topic_name,
                    subfield=rec.get("primary_topic_subfield") if topic_name == rec.get("primary_topic") else None,
                    field=rec.get("primary_topic_field") if topic_name == rec.get("primary_topic") else None,
                    domain=rec.get("primary_topic_domain") if topic_name == rec.get("primary_topic") else None,
                    openalex_url=f"https://openalex.org/{topic_id}"
                )

            # Connect Paper -> ABOUT -> Topic
            score = top.get("score")
            props = {"score": score} if score is not None else {}
            add_relationship("Paper", paper_id, "Topic", topic_id, "ABOUT", props)

    nodes = {
        "papers": papers,
        "researchers": researchers,
        "institutions": institutions,
        "topics": topics,
        "years": years
    }

    return {"nodes": nodes, "relationships": relationships}, resolver
