"""Registry module for discovering, querying, and storing Kenyan research institutions."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.ingestion.config import INSTITUTIONS_FILE
from src.ingestion.openalex_client import OpenAlexClient

logger = logging.getLogger(__name__)


class InstitutionRegistry:
    """Manages the catalog of Kenyan academic and research institutions."""

    def __init__(self, registry_file: Path = INSTITUTIONS_FILE):
        self.registry_file = Path(registry_file)
        self.institutions: List[Dict[str, Any]] = []
        if self.registry_file.exists():
            self.load()

    def discover_from_openalex(self, client: Optional[OpenAlexClient] = None) -> List[Dict[str, Any]]:
        """Queries OpenAlex for all institutions located in Kenya (country_code:KE)."""
        client = client or OpenAlexClient()
        logger.info("Discovering Kenyan institutions from OpenAlex (country_code:KE)...")

        discovered = []
        # OpenAlex institution endpoint
        for item in client.paginate("institutions", params={"filter": "country_code:KE"}, per_page=100):
            inst = {
                "id": item.get("id"),
                "openalex_id": item.get("id", "").replace("https://openalex.org/", ""),
                "ror": item.get("ror"),
                "display_name": item.get("display_name"),
                "country_code": item.get("country_code"),
                "type": item.get("type"),
                "homepage_url": item.get("homepage_url"),
                "works_count": item.get("works_count", 0),
                "cited_by_count": item.get("cited_by_count", 0),
                "geo": item.get("geo", {}),
                "lineage": item.get("lineage", [])
            }
            discovered.append(inst)

        # Sort by works_count descending
        discovered.sort(key=lambda x: x.get("works_count", 0), reverse=True)
        self.institutions = discovered
        logger.info(f"Discovered {len(self.institutions)} Kenyan institutions in OpenAlex.")
        return self.institutions

    def save(self, filepath: Optional[Path] = None):
        """Persists the institution registry to a JSON file."""
        target_path = filepath or self.registry_file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.institutions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.institutions)} institutions to {target_path}")

    def load(self, filepath: Optional[Path] = None):
        """Loads institutions from the JSON file."""
        source_path = filepath or self.registry_file
        if not source_path.exists():
            raise FileNotFoundError(f"Registry file not found: {source_path}")
        with open(source_path, "r", encoding="utf-8") as f:
            self.institutions = json.load(f)
        logger.info(f"Loaded {len(self.institutions)} institutions from {source_path}")

    def get_openalex_ids(self) -> List[str]:
        """Returns the list of OpenAlex IDs (e.g., I12345) for all registered institutions."""
        return [inst["openalex_id"] for inst in self.institutions if inst.get("openalex_id")]

    def get_ror_ids(self) -> List[str]:
        """Returns the list of ROR IDs."""
        return [inst["ror"] for inst in self.institutions if inst.get("ror")]
