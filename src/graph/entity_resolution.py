"""Entity resolution and deduplication analysis module."""

import hashlib
import logging
from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


def generate_synthetic_id(prefix: str, text: str) -> str:
    """Generates a deterministic synthetic ID when OpenAlex ID is missing."""
    hash_val = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_SYNTH_{hash_val}"


class EntityResolver:
    """Audits entity identity, tracks name variants, and handles disambiguation."""

    def __init__(self):
        # Maps canonical ID -> List of display names observed
        self.researcher_names_by_id: Dict[str, Set[str]] = defaultdict(set)
        self.researcher_ids_by_name: Dict[str, Set[str]] = defaultdict(set)

        self.institution_names_by_id: Dict[str, Set[str]] = defaultdict(set)
        self.institution_ids_by_name: Dict[str, Set[str]] = defaultdict(set)
        self.institution_rors_by_id: Dict[str, Set[str]] = defaultdict(set)

        self.topic_names_by_id: Dict[str, Set[str]] = defaultdict(set)
        self.topic_ids_by_name: Dict[str, Set[str]] = defaultdict(set)

        self.missing_author_ids_count = 0
        self.missing_institution_ids_count = 0

    def register_researcher(self, author_id: str | None, name: str) -> str:
        """Registers a researcher observation and returns a resolved canonical ID."""
        clean_name = (name or "").strip()
        if not author_id:
            self.missing_author_ids_count += 1
            canonical_id = generate_synthetic_id("AUTH", clean_name)
        else:
            canonical_id = author_id

        if clean_name:
            self.researcher_names_by_id[canonical_id].add(clean_name)
            self.researcher_ids_by_name[clean_name.lower()].add(canonical_id)

        return canonical_id

    def register_institution(self, inst_id: str | None, name: str, ror: str | None = None) -> str:
        """Registers an institution observation and returns a resolved canonical ID."""
        clean_name = (name or "").strip()
        if not inst_id:
            self.missing_institution_ids_count += 1
            canonical_id = generate_synthetic_id("INST", clean_name)
        else:
            canonical_id = inst_id

        if clean_name:
            self.institution_names_by_id[canonical_id].add(clean_name)
            self.institution_ids_by_name[clean_name.lower()].add(canonical_id)
        if ror:
            self.institution_rors_by_id[canonical_id].add(ror)

        return canonical_id

    def register_topic(self, topic_id: str, name: str) -> str:
        """Registers a topic observation and returns the canonical ID."""
        clean_name = (name or "").strip()
        self.topic_names_by_id[topic_id].add(clean_name)
        self.topic_ids_by_name[clean_name.lower()].add(topic_id)
        return topic_id

    def audit_resolution(self) -> Dict[str, Any]:
        """Calculates entity resolution metrics: polysemy, synonymy, missing IDs."""
        # 1. Researchers with multiple name spellings (Synonymy / Name variants)
        researchers_multiple_names = {
            r_id: list(names)
            for r_id, names in self.researcher_names_by_id.items()
            if len(names) > 1
        }

        # 2. Distinct Researcher IDs sharing the exact same string name (Homonyms / Polysemy)
        researchers_shared_name = {
            name: list(ids)
            for name, ids in self.researcher_ids_by_name.items()
            if len(ids) > 1
        }

        # 3. Institutions with multiple names
        institutions_multiple_names = {
            i_id: list(names)
            for i_id, names in self.institution_names_by_id.items()
            if len(names) > 1
        }

        # 4. Distinct Institution IDs sharing exact string name
        institutions_shared_name = {
            name: list(ids)
            for name, ids in self.institution_ids_by_name.items()
            if len(ids) > 1
        }

        # 5. Topics with multiple names or shared names
        topics_multiple_names = {
            t_id: list(names)
            for t_id, names in self.topic_names_by_id.items()
            if len(names) > 1
        }

        return {
            "researchers": {
                "total_unique_ids": len(self.researcher_names_by_id),
                "ids_with_multiple_name_variants": len(researchers_multiple_names),
                "sample_name_variants": list(researchers_multiple_names.items())[:5],
                "names_shared_by_multiple_ids": len(researchers_shared_name),
                "sample_homonyms": list(researchers_shared_name.items())[:5],
                "missing_source_ids_resolved_synthetically": self.missing_author_ids_count
            },
            "institutions": {
                "total_unique_ids": len(self.institution_names_by_id),
                "ids_with_multiple_name_variants": len(institutions_multiple_names),
                "sample_name_variants": list(institutions_multiple_names.items())[:5],
                "names_shared_by_multiple_ids": len(institutions_shared_name),
                "sample_shared_names": list(institutions_shared_name.items())[:5],
                "missing_source_ids_resolved_synthetically": self.missing_institution_ids_count
            },
            "topics": {
                "total_unique_ids": len(self.topic_names_by_id),
                "ids_with_multiple_name_variants": len(topics_multiple_names)
            }
        }
