"""Local caching and resumption utilities for raw ingestion and metadata."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
from datetime import datetime, timezone

from src.ingestion.config import RAW_WORKS_FILE, INGESTION_METADATA_FILE

logger = logging.getLogger(__name__)


class RawDataCache:
    """Manages appending, streaming, and metadata tracking for raw OpenAlex responses."""

    def __init__(self, cache_file: Path = RAW_WORKS_FILE, metadata_file: Path = INGESTION_METADATA_FILE):
        self.cache_file = Path(cache_file)
        self.metadata_file = Path(metadata_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def append_work(self, raw_work: Dict[str, Any]):
        """Appends a single raw work record as a line in a JSONL file."""
        with open(self.cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(raw_work, ensure_ascii=False) + "\n")

    def append_batch(self, raw_works: List[Dict[str, Any]]):
        """Appends a batch of raw work records."""
        with open(self.cache_file, "a", encoding="utf-8") as f:
            for work in raw_works:
                f.write(json.dumps(work, ensure_ascii=False) + "\n")

    def clear(self):
        """Clears the existing raw cache."""
        if self.cache_file.exists():
            self.cache_file.unlink()

    def stream_records(self) -> Generator[Dict[str, Any], None, None]:
        """Yields raw work dictionaries from the JSONL cache."""
        if not self.cache_file.exists():
            logger.warning(f"Cache file {self.cache_file} does not exist.")
            return
        with open(self.cache_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.error(f"Malformed JSON at line {line_no} in {self.cache_file}: {exc}")

    def count_records(self) -> int:
        """Returns the total number of cached records."""
        if not self.cache_file.exists():
            return 0
        count = 0
        with open(self.cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def save_metadata(self, metadata: Dict[str, Any]):
        """Saves metadata concerning the ingestion run."""
        metadata["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved ingestion metadata to {self.metadata_file}")

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Loads ingestion metadata if available."""
        if not self.metadata_file.exists():
            return None
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
