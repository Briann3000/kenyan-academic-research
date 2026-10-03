"""Configuration settings for OpenAlex ingestion and data directories."""

from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR]:
    os.makedirs(path, exist_ok=True)

# OpenAlex API configuration
OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", "researcher@example.com")
USER_AGENT = f"KenyanAcademicResearchKG/1.0 (mailto:{OPENALEX_EMAIL})"

# Polite pool rate limiting (OpenAlex allows 10 req/s with email)
MAX_REQUESTS_PER_SECOND = 5.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.5

# Default query parameters
DEFAULT_PAGE_SIZE = 100  # Max per-page in OpenAlex is 200 (100 is stable)
DEFAULT_MAX_WORKS = 1000

# File locations
INSTITUTIONS_FILE = RAW_DATA_DIR / "kenyan_institutions.json"
RAW_WORKS_FILE = RAW_DATA_DIR / "openalex_raw_works.jsonl"
INGESTION_METADATA_FILE = RAW_DATA_DIR / "ingestion_metadata.json"
NORMALIZED_WORKS_FILE = PROCESSED_DATA_DIR / "normalized_works.jsonl"
PROFILING_METRICS_FILE = REPORTS_DIR / "profiling_metrics.json"
DATA_QUALITY_REPORT_FILE = PROJECT_ROOT / "DATA_QUALITY_REPORT.md"
