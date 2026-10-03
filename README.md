# Kenyan Academic Research Knowledge Graph

A final-year Computer Science project investigating whether combining a **Knowledge Graph** with **Semantic Search** improves the discovery and exploration of relationships within Kenyan academic research compared with conventional keyword-based search.

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── kenyan_institutions.json        # Discovered Kenyan institution registry (303 institutions)
│   │   ├── openalex_raw_works.jsonl        # Raw OpenAlex responses (seed corpus)
│   │   └── ingestion_metadata.json         # Run metadata & parameters
│   ├── processed/
│   │   └── normalized_works.jsonl          # Cleaned, normalized, tabular-ready records
│   └── reports/
│       └── profiling_metrics.json          # Machine-readable profiling results
│
├── src/
│   └── ingestion/
│       ├── __init__.py
│       ├── config.py                       # Configuration paths, polite pool settings, timeouts
│       ├── openalex_client.py              # Resilient HTTP client (rate limits, backoff, cursor pagination)
│       ├── institution_registry.py         # Ingests & catalogues Kenyan institutions & RORs
│       ├── fetch_works.py                  # Orchestrates retrieval of Kenyan-affiliated works
│       ├── cache.py                        # Raw JSONL streaming cache and metadata
│       ├── normalizer.py                   # Inverted index abstract reconstructor & record normalizer
│       └── profiling.py                    # Empirical data-quality and completeness auditor
│
├── scripts/
│   ├── discover_institutions.py            # CLI script to discover & save Kenyan institutions
│   └── run_ingestion.py                    # Main pipeline CLI (Fetch -> Normalize -> Profile -> Report)
│
├── tests/
│   └── ingestion/
│       ├── test_client.py                  # Client HTTP retry, 429 backoff & pagination tests
│       ├── test_normalizer.py              # Abstract reconstruction & record normalization tests
│       └── test_profiling.py               # Completeness calculation & edge-case tests
│
├── DATA_QUALITY_REPORT.md                  # Comprehensive empirical quality report
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Installation & Setup

1. **Clone/open project directory**:
   ```bash
   cd project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Phase 1 Ingestion & Profiling Pipeline

### 1. Run Automated Test Suite
```bash
pytest
```

### 2. Discover Kenyan Institutions
Queries the OpenAlex API to identify all indexed institutions located in Kenya (`country_code:KE`) with their OpenAlex IDs, ROR IDs, coordinates, and publication counts:
```bash
python scripts/discover_institutions.py
```

### 3. Run Ingestion, Normalization & Profiling
Fetches a seed corpus of Kenyan-affiliated works, streams the raw responses to disk, normalizes fields, reconstructs abstracts from the inverted index, calculates data quality metrics, and updates `DATA_QUALITY_REPORT.md`:
```bash
python scripts/run_ingestion.py --max-works 1000 --page-size 100
```

To re-profile the cached data without refetching from OpenAlex:
```bash
python scripts/run_ingestion.py --skip-fetch
```

---

## Key Phase 1 Empirical Findings

See [DATA_QUALITY_REPORT.md](DATA_QUALITY_REPORT.md) for full details.

* **Abstract Availability:** Abstracts are present in **64.8%** of records. Reconstructed from OpenAlex inverted index.
* **Fallback Search Text:** Due to missing abstracts, search indexing will use composite text: `Title + Topic labels + Abstract (when available)`.
* **International Collaboration Footprint:** **92.9%** of papers feature co-authors from international institutions (primarily US, UK, Germany, France, South Africa).
* **Topic Coverage:** **100%** of records have identified OpenAlex topic models and subfields.
