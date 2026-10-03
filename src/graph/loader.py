"""Graph loader supporting both direct Neo4j Cypher ingestion and in-memory NetworkX analysis."""

import logging
from typing import Dict, Any, List, Optional
import networkx as nx

from src.graph.models import (
    PaperNode,
    ResearcherNode,
    InstitutionNode,
    TopicNode,
    YearNode,
    GraphRelationship
)
from src.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphLoader:
    """Loads normalized graph elements into Neo4j and/or builds a NetworkX representation."""

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.neo4j_client = neo4j_client
        self.nx_graph = nx.MultiDiGraph()

    def build_networkx_graph(self, graph_elements: Dict[str, Any]) -> nx.MultiDiGraph:
        """Constructs an in-memory NetworkX graph from canonical nodes and relationships."""
        self.nx_graph.clear()
        nodes = graph_elements["nodes"]
        relationships: List[GraphRelationship] = graph_elements["relationships"]

        # 1. Add Year nodes
        for y_val, y_node in nodes["years"].items():
            self.nx_graph.add_node(f"Year:{y_val}", label="Year", value=y_val)

        # 2. Add Paper nodes
        for p_id, p_node in nodes["papers"].items():
            self.nx_graph.add_node(
                f"Paper:{p_id}",
                label="Paper",
                id=p_id,
                title=p_node.title,
                doi=p_node.doi,
                publication_year=p_node.publication_year,
                is_oa=p_node.is_oa,
                oa_status=p_node.oa_status,
                abstract=p_node.abstract,
                search_text=p_node.search_text,
                cited_by_count=p_node.cited_by_count,
                openalex_url=p_node.openalex_url
            )

        # 3. Add Researcher nodes
        for r_id, r_node in nodes["researchers"].items():
            self.nx_graph.add_node(
                f"Researcher:{r_id}",
                label="Researcher",
                id=r_id,
                display_name=r_node.display_name,
                orcid=r_node.orcid,
                openalex_url=r_node.openalex_url
            )

        # 4. Add Institution nodes
        for i_id, i_node in nodes["institutions"].items():
            self.nx_graph.add_node(
                f"Institution:{i_id}",
                label="Institution",
                id=i_id,
                display_name=i_node.display_name,
                ror=i_node.ror,
                country_code=i_node.country_code,
                type=i_node.type,
                openalex_url=i_node.openalex_url
            )

        # 5. Add Topic nodes
        for t_id, t_node in nodes["topics"].items():
            self.nx_graph.add_node(
                f"Topic:{t_id}",
                label="Topic",
                id=t_id,
                display_name=t_node.display_name,
                subfield=t_node.subfield,
                field=t_node.field,
                domain=t_node.domain,
                openalex_url=t_node.openalex_url
            )

        # 6. Add Relationships
        for rel in relationships:
            src_node_id = f"{rel.source_type}:{rel.source_id}"
            tgt_node_id = f"{rel.target_type}:{rel.target_id}"
            self.nx_graph.add_edge(src_node_id, tgt_node_id, key=rel.rel_type, type=rel.rel_type, **rel.properties)

        logger.info(
            f"Built NetworkX graph: {self.nx_graph.number_of_nodes()} nodes, "
            f"{self.nx_graph.number_of_edges()} edges."
        )
        return self.nx_graph

    def load_to_neo4j(self, graph_elements: Dict[str, Any], batch_size: int = 500):
        """Loads canonical elements into Neo4j using idempotent Cypher UNWIND batches."""
        if not self.neo4j_client or not self.neo4j_client.driver:
            raise ConnectionError("Neo4j client is not connected.")

        nodes = graph_elements["nodes"]
        relationships: List[GraphRelationship] = graph_elements["relationships"]

        # Ensure constraints exist
        self.neo4j_client.apply_constraints()

        # 1. Load Years
        year_rows = [{"value": y.value} for y in nodes["years"].values()]
        logger.info(f"Loading {len(year_rows)} Year nodes into Neo4j...")
        self.neo4j_client.execute_query(
            "UNWIND $batch AS row MERGE (y:Year {value: row.value})",
            {"batch": year_rows}
        )

        # 2. Load Institutions
        inst_rows = [i.model_dump() for i in nodes["institutions"].values()]
        logger.info(f"Loading {len(inst_rows)} Institution nodes into Neo4j...")
        self._batch_unwind(
            "UNWIND $batch AS row MERGE (i:Institution {id: row.id}) SET i += row",
            inst_rows,
            batch_size
        )

        # 3. Load Researchers
        researcher_rows = [r.model_dump() for r in nodes["researchers"].values()]
        logger.info(f"Loading {len(researcher_rows)} Researcher nodes into Neo4j...")
        self._batch_unwind(
            "UNWIND $batch AS row MERGE (r:Researcher {id: row.id}) SET r += row",
            researcher_rows,
            batch_size
        )

        # 4. Load Topics
        topic_rows = [t.model_dump() for t in nodes["topics"].values()]
        logger.info(f"Loading {len(topic_rows)} Topic nodes into Neo4j...")
        self._batch_unwind(
            "UNWIND $batch AS row MERGE (t:Topic {id: row.id}) SET t += row",
            topic_rows,
            batch_size
        )

        # 5. Load Papers
        paper_rows = [p.model_dump() for p in nodes["papers"].values()]
        logger.info(f"Loading {len(paper_rows)} Paper nodes into Neo4j...")
        self._batch_unwind(
            "UNWIND $batch AS row MERGE (p:Paper {id: row.id}) SET p += row",
            paper_rows,
            batch_size
        )

        # 6. Load Relationships
        logger.info(f"Loading {len(relationships)} relationships into Neo4j...")
        # Group by relationship type
        rel_groups: Dict[str, List[Dict[str, Any]]] = {}
        for rel in relationships:
            rel_groups.setdefault(rel.rel_type, []).append({
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "props": rel.properties
            })

        if "PUBLISHED_IN" in rel_groups:
            self._batch_unwind(
                """
                UNWIND $batch AS row
                MATCH (p:Paper {id: row.source_id})
                MATCH (y:Year {value: toInteger(row.target_id)})
                MERGE (p)-[:PUBLISHED_IN]->(y)
                """,
                rel_groups["PUBLISHED_IN"],
                batch_size
            )

        if "AUTHORED" in rel_groups:
            self._batch_unwind(
                """
                UNWIND $batch AS row
                MATCH (r:Researcher {id: row.source_id})
                MATCH (p:Paper {id: row.target_id})
                MERGE (r)-[:AUTHORED]->(p)
                """,
                rel_groups["AUTHORED"],
                batch_size
            )

        if "AFFILIATED_WITH" in rel_groups:
            self._batch_unwind(
                """
                UNWIND $batch AS row
                MATCH (p:Paper {id: row.source_id})
                MATCH (i:Institution {id: row.target_id})
                MERGE (p)-[:AFFILIATED_WITH]->(i)
                """,
                rel_groups["AFFILIATED_WITH"],
                batch_size
            )

        if "ABOUT" in rel_groups:
            self._batch_unwind(
                """
                UNWIND $batch AS row
                MATCH (p:Paper {id: row.source_id})
                MATCH (t:Topic {id: row.target_id})
                MERGE (p)-[rel:ABOUT]->(t)
                SET rel += row.props
                """,
                rel_groups["ABOUT"],
                batch_size
            )

        logger.info("Neo4j loading completed successfully.")

    def _batch_unwind(self, cypher_template: str, items: List[Dict[str, Any]], batch_size: int):
        """Helper to run parameterized UNWIND batches in Neo4j."""
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            self.neo4j_client.execute_query(cypher_template, {"batch": chunk})
