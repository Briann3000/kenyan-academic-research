"""Neo4j schema constraints and index definitions."""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Uniqueness constraints for Neo4j 5+ / 2025+
NEO4J_CONSTRAINTS: List[str] = [
    "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT researcher_id_unique IF NOT EXISTS FOR (r:Researcher) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT institution_id_unique IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT year_value_unique IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE"
]

# Additional performance indexes
NEO4J_INDEXES: List[str] = [
    "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)",
    "CREATE INDEX researcher_name_index IF NOT EXISTS FOR (r:Researcher) ON (r.display_name)",
    "CREATE INDEX institution_country_index IF NOT EXISTS FOR (i:Institution) ON (i.country_code)",
    "CREATE INDEX topic_name_index IF NOT EXISTS FOR (t:Topic) ON (t.display_name)"
]
