"""OpenAlex HTTP client supporting rate limiting, retries, exponential backoff, and pagination."""

import time
import logging
import random
from typing import Dict, Any, Optional, Generator
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.ingestion.config import (
    OPENALEX_BASE_URL,
    USER_AGENT,
    MAX_REQUESTS_PER_SECOND,
    REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES,
    BACKOFF_FACTOR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class OpenAlexClient:
    """Polite and resilient API client for OpenAlex."""

    def __init__(
        self,
        base_url: str = OPENALEX_BASE_URL,
        user_agent: str = USER_AGENT,
        max_req_per_sec: float = MAX_REQUESTS_PER_SECOND,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        backoff_factor: float = BACKOFF_FACTOR
    ):
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.min_interval = 1.0 / max_req_per_sec if max_req_per_sec > 0 else 0
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.last_request_time = 0.0

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json"
        })

    def _wait_for_rate_limit(self):
        """Enforces minimum delay between consecutive HTTP calls."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            sleep_duration = self.min_interval - elapsed
            time.sleep(sleep_duration)
        self.last_request_time = time.time()

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Performs a GET request with rate limiting and exponential backoff retry."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}

        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                logger.debug(f"GET {url} params={params} (attempt {attempt})")
                response = self.session.get(url, params=params, timeout=self.timeout)

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 429 or response.status_code >= 500:
                    wait_time = (self.backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                    logger.warning(
                        f"HTTP {response.status_code} received from OpenAlex. "
                        f"Retrying in {wait_time:.2f}s (Attempt {attempt}/{self.max_retries})..."
                    )
                    time.sleep(wait_time)
                    continue

                # Non-recoverable 4xx client errors
                response.raise_for_status()

            except (requests.RequestException, requests.Timeout) as exc:
                wait_time = (self.backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    f"Network error on GET {url}: {exc}. "
                    f"Retrying in {wait_time:.2f}s (Attempt {attempt}/{self.max_retries})..."
                )
                if attempt == self.max_retries:
                    logger.error(f"Exceeded max retries for {url}.")
                    raise
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts.")

    def paginate(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_records: Optional[int] = None,
        per_page: int = 100
    ) -> Generator[Dict[str, Any], None, None]:
        """Paginates using OpenAlex cursor or standard page-based pagination.
        
        Yields individual work/entity records until max_records is reached or no more pages exist.
        """
        params = dict(params or {})
        params["per-page"] = min(per_page, 200)
        
        # OpenAlex cursor pagination is recommended for large sets
        params["cursor"] = "*"
        records_yielded = 0

        while True:
            data = self.get(endpoint, params=params)
            meta = data.get("meta", {})
            results = data.get("results", [])

            if not results:
                logger.info("Pagination reached end of results.")
                break

            for record in results:
                yield record
                records_yielded += 1
                if max_records and records_yielded >= max_records:
                    logger.info(f"Reached requested limit of {max_records} records.")
                    return

            next_cursor = meta.get("next_cursor")
            if not next_cursor:
                # No next page
                break
            params["cursor"] = next_cursor
