"""Neo4j driver connection manager and transaction runner."""

import os
import logging
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase, Driver, Session

from src.graph.constraints import NEO4J_CONSTRAINTS, NEO4J_INDEXES

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
