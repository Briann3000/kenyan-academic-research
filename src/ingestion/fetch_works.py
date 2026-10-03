"""Orchestrator module for fetching Kenyan academic research works from OpenAlex."""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from src.ingestion.config import (
    DEFAULT_MAX_WORKS,
    DEFAULT_PAGE_SIZE,
    RAW_WORKS_FILE,
    INGESTION_METADATA_FILE
)
from src.ingestion.openalex_client import OpenAlexClient
from src.ingestion.institution_registry import InstitutionRegistry
from src.ingestion.cache import RawDataCache

logger = logging.getLogger(__name__)


def fetch_kenyan_works(
    max_works: int = DEFAULT_MAX_WORKS,
    per_page: int = DEFAULT_PAGE_SIZE,
    filter_query: str = "institutions.country_code:KE",
    client: Optional[OpenAlexClient] = None,
    cache: Optional[RawDataCache] = None,
    overwrite_cache: bool = True
) -> int:
    """Fetches Kenyan-affiliated works from OpenAlex and streams them into local raw cache.
    
    Returns the total number of records successfully cached.
    """
    client = client or OpenAlexClient()
    cache = cache or RawDataCache(RAW_WORKS_FILE, INGESTION_METADATA_FILE)

    if overwrite_cache:
        logger.info("Overwriting existing raw cache...")
        cache.clear()

    logger.info(f"Starting OpenAlex fetch: filter='{filter_query}', max_works={max_works}")
    start_time = datetime.now(timezone.utc)

    count = 0
    batch = []
    batch_size = 50

    try:
        for work in client.paginate("works", params={"filter": filter_query}, max_records=max_works, per_page=per_page):
            batch.append(work)
            count += 1

            if len(batch) >= batch_size:
                cache.append_batch(batch)
                batch = []
                logger.info(f"Ingested {count}/{max_works} works...")

        if batch:
            cache.append_batch(batch)

    except Exception as exc:
        logger.error(f"Ingestion interrupted at record {count}: {exc}")
        if batch:
            cache.append_batch(batch)
        raise

    end_time = datetime.now(timezone.utc)
    duration_secs = (end_time - start_time).total_seconds()

    metadata = {
        "source": "OpenAlex",
        "filter_query": filter_query,
        "requested_max_works": max_works,
        "per_page": per_page,
        "records_ingested": count,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": round(duration_secs, 2)
    }
    cache.save_metadata(metadata)
    logger.info(f"Ingestion finished: {count} works cached in {duration_secs:.2f}s.")
    return count
