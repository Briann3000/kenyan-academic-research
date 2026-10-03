"""Retrieval corpus loader and document collection manager."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from src.ingestion.config import NORMALIZED_WORKS_FILE
from src.search.models import RetrievalDocument

logger = logging.getLogger(__name__)


class RetrievalCorpus:
    """Manages the in-memory retrieval corpus of 1,000 normalized research papers."""

    def __init__(self, corpus_path: Optional[Path] = None):
        self.corpus_path = Path(corpus_path or NORMALIZED_WORKS_FILE)
        self.documents: List[RetrievalDocument] = []
        self.doc_map: Dict[str, RetrievalDocument] = {}
        self.load()

    def load(self):
        """Loads and parses documents from normalized_works.jsonl."""
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        docs = []
        doc_map = {}

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                paper_id = data["id"]
                if paper_id in doc_map:
                    logger.warning(f"Duplicate paper ID {paper_id} at line {line_no}")
                    continue

                topic_names = [t["name"] for t in data.get("topics", []) if t.get("name")]
                doc = RetrievalDocument(
                    paper_id=paper_id,
                    title=data.get("title") or "Untitled",
                    search_text=data.get("search_text") or data.get("title") or "",
                    abstract=data.get("abstract_text"),
                    topics=topic_names,
                    publication_year=data.get("publication_year", 0),
                    doi=data.get("doi"),
                    work_type=data.get("type"),
                    is_oa=data.get("is_oa", False),
                    oa_status=data.get("oa_status")
                )
                docs.append(doc)
                doc_map[paper_id] = doc

        self.documents = docs
        self.doc_map = doc_map
        logger.info(f"Loaded {len(self.documents)} retrieval documents from {self.corpus_path}")

    def __len__(self) -> int:
        return len(self.documents)

    def get_by_id(self, paper_id: str) -> Optional[RetrievalDocument]:
        return self.doc_map.get(paper_id)
