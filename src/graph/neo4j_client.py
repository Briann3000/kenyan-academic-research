"""Neo4j driver connection manager, in-memory graph client, and transaction runner."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from neo4j import GraphDatabase, Driver, Session

from src.graph.constraints import NEO4J_CONSTRAINTS, NEO4J_INDEXES
from src.ingestion.config import NORMALIZED_WORKS_FILE

logger = logging.getLogger(__name__)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")


class Neo4jClient:
    """Manages connection, session lifecycle, and Cypher execution."""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Optional[Driver] = None

    def connect(self) -> bool:
        """Attempts to establish connection to the Neo4j instance."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            return True
        except Exception as exc:
            logger.warning(f"Could not connect to Neo4j at {self.uri}: {exc}")
            self.driver = None
            return False

    def close(self):
        """Closes the active Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            self.driver = None

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Executes a read/write Cypher query and returns the list of result records as dicts."""
        if not self.driver:
            raise ConnectionError("Neo4j driver is not connected.")
        parameters = parameters or {}
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def apply_constraints(self):
        """Applies uniqueness constraints and indexes to the connected database."""
        logger.info("Applying Neo4j schema constraints and indexes...")
        for stmt in NEO4J_CONSTRAINTS + NEO4J_INDEXES:
            try:
                self.execute_query(stmt)
            except Exception as exc:
                logger.warning(f"Constraint application note: {exc}")


class InMemoryGraphClient:
    """In-memory graph client backed by the Phase 2 canonical knowledge graph data.

    Provides a fast, zero-dependency graph querying interface executing the identical
    candidate subgraph retrieval schema as the live Neo4j Cypher engine.
    """

    def __init__(self, normalized_works_path: Optional[Path] = None):
        self.normalized_works_path = Path(normalized_works_path or NORMALIZED_WORKS_FILE)
        self.paper_topics: Dict[str, Set[str]] = {}
        self.paper_authors: Dict[str, Set[str]] = {}
        self.paper_institutions: Dict[str, Set[str]] = {}
        self._load_graph()

    def _load_graph(self):
        """Builds in-memory relational indexes directly from normalized works."""
        if not self.normalized_works_path.exists():
            logger.warning(f"Normalized works file not found at {self.normalized_works_path}")
            return

        with open(self.normalized_works_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                paper_id = doc["id"]

                # Extract Topics: (Paper)-[:ABOUT]->(Topic)
                topics = {t["id"] for t in doc.get("topics", []) if t.get("id")}
                self.paper_topics[paper_id] = topics

                # Extract Authors & Institutions
                # (Researcher)-[:AUTHORED]->(Paper) and (Paper)-[:AFFILIATED_WITH]->(Institution)
                authors = {a["id"] for a in doc.get("authors", []) if a and a.get("id")}
                institutions = {i["id"] for i in doc.get("institutions", []) if i and i.get("id")}

                self.paper_authors[paper_id] = authors
                self.paper_institutions[paper_id] = institutions

        logger.info(
            f"Loaded InMemoryGraphClient: {len(self.paper_topics)} papers indexed with topics, authors, and institutions."
        )

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Emulates the batched candidate subgraph Cypher query."""
        parameters = parameters or {}
        paper_ids = parameters.get("paper_ids", [])
        records = []
        for p_id in paper_ids:
            records.append(
                {
                    "paper_id": p_id,
                    "topic_ids": list(self.paper_topics.get(p_id, set())),
                    "author_ids": list(self.paper_authors.get(p_id, set())),
                    "institution_ids": list(self.paper_institutions.get(p_id, set())),
                }
            )
        return records
