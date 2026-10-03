# Data Quality & Metadata Profiling Report (Phase 1)

> **Project:** Kenyan Academic Research Knowledge Graph  
> **Report Generated:** 2026-10-03 07:42:49 UTC  
> **Status:** Completed (Seed Corpus Ingestion & Audit)

---

## 1. Dataset Overview & Ingestion Parameters

- **Data Source:** OpenAlex API (`https://api.openalex.org`)
- **Filter Query Used:** `institutions.country_code:KE`
- **Total Records Ingested:** **1000**
- **Ingestion Duration:** 61.15 seconds
- **Registered Kenyan Institutions:** 303 institutions in registry
- **Polite Rate Limiting:** Enforced with email identification

---

## 2. Metadata Field Completeness Matrix

| Field Name | Present Count | Missing Count | Completeness % | Missing % |
| :--- | :---: | :---: | :---: | :---: |
| **OpenAlex ID** | 1000 | 0 | 100.0% | 0.0% |
| **DOI** | 997 | 3 | 99.7% | 0.3% |
| **Title** | 1000 | 0 | 100.0% | 0.0% |
| **Publication Year** | 1000 | 0 | 100.0% | 0.0% |
| **Publication Date** | 1000 | 0 | 100.0% | 0.0% |
| **Work Type** | 1000 | 0 | 100.0% | 0.0% |
| **Source / Journal** | 998 | 2 | 99.8% | 0.2% |
| **Abstract (Reconstructed)** | 648 | 352 | 64.8% | 35.2% |
| **Authors** | 1000 | 0 | 100.0% | 0.0% |
| **Author Identifiers** | 996 | 4 | 99.6% | 0.4% |
| **Institutions / Affiliations** | 1000 | 0 | 100.0% | 0.0% |
| **Kenyan Institution Affiliation** | 985 | 15 | 98.5% | 1.5% |
| **Institution ROR ID** | 1000 | 0 | 100.0% | 0.0% |
| **Primary Topic** | 1000 | 0 | 100.0% | 0.0% |
| **Topics List** | 1000 | 0 | 100.0% | 0.0% |
| **Concepts List** | 1000 | 0 | 100.0% | 0.0% |
| **Open Access Status** | 1000 | 0 | 100.0% | 0.0% |
| **Citations Count** | 1000 | 0 | 100.0% | 0.0% |

---

## 3. Key Distribution Metrics

### A. Entity Cardinalities per Paper

| Dimension | Min | Median | Mean | Max |
| :--- | :---: | :---: | :---: | :---: |
| **Authors per Paper** | 1 | 9.0 | 19.7 | 100 |
| **Institutions per Paper** | 1 | 14.0 | 29.4 | 364 |
| **Topics per Paper** | 1 | 3.0 | 2.96 | 3 |
| **Concepts per Paper** | 1 | 16.0 | 15.85 | 35 |
| **Citations per Paper** | 326 | 505.5 | 805.49 | 20342 |
| **Abstract Word Length (when present)** | 2 | 242.5 | 262.41 | 3561 |

### B. Work Type Distribution

| Work Type | Count | Percentage |
| :--- | :---: | :---: |
| `article` | 932 | 93.2% |
| `review` | 58 | 5.8% |
| `book-chapter` | 9 | 0.9% |
| `book` | 1 | 0.1% |

### C. Open Access (OA) Status Breakdown

| OA Status | Count | Percentage |
| :--- | :---: | :---: |
| `closed` | 300 | 30.0% |
| `green` | 208 | 20.8% |
| `hybrid` | 163 | 16.3% |
| `gold` | 162 | 16.2% |
| `bronze` | 157 | 15.7% |
| `diamond` | 10 | 1.0% |

### D. Top Primary Research Domains & Fields

| Domain / Field | Work Count |
| :--- | :---: |
| **Domain:** Health Sciences | 356 |
| **Domain:** Life Sciences | 310 |
| **Domain:** Physical Sciences | 253 |
| **Domain:** Social Sciences | 81 |
| Field: Medicine | 324 |
| Field: Environmental Science | 211 |
| Field: Agricultural and Biological Sciences | 209 |
| Field: Biochemistry, Genetics and Molecular Biology | 64 |
| Field: Social Sciences | 35 |
| Field: Immunology and Microbiology | 31 |
| Field: Psychology | 24 |
| Field: Earth and Planetary Sciences | 17 |

---

## 4. Affiliation & Institutional Collaboration Analysis

- **International Collaboration Rate:** 92.9% of papers (929 papers) involve co-authors from institutions outside Kenya.

### Top Domestic (Kenyan) Institutions:

| Institution Name | Indexed Works Count |
| :--- | :---: |
| Kenya Medical Research Institute | 587 |
| Centers for Disease Control and Prevention | 362 |
| International Livestock Research Institute | 223 |
| World Agroforestry Centre | 212 |
| KEMRI-Wellcome Trust Research Programme | 205 |
| University of Nairobi | 180 |
| International Maize and Wheat Improvement Center | 133 |
| International Center for Tropical Agriculture | 98 |
| United Nations Environment Programme | 85 |
| International Centre of Insect Physiology and Ecology | 81 |
| BirdLife International | 79 |
| Ministry of Health | 65 |

### Top Collaborating Partner Countries:

| Country Code | Co-authored Papers |
| :--- | :---: |
| `US` | 7337 |
| `GB` | 3517 |
| `DE` | 1346 |
| `FR` | 1111 |
| `CN` | 961 |
| `AU` | 937 |
| `ZA` | 800 |
| `NL` | 615 |
| `CA` | 583 |
| `IN` | 566 |

---

## 5. Identified Edge Cases & Data Anomalies

- **Duplicate DOIs:** 0
- **Duplicate Titles:** 3
- **Duplicate OpenAlex IDs:** 0
- **Records with Country=KE but Unresolved Domestic Institution:** 15
- **Anomalous Publication Years (<1960 or >2027):** 0

---

## 6. Engineering Decisions & Recommendations for Phase 2

1. **Abstract Incompleteness & Semantic Fallback:**
   - Observation: Abstracts are present in only **64.8%** of works.
   - Engineering Decision: Semantic embeddings cannot rely strictly on abstracts. The system MUST construct a composite search text: `Title + Topic labels + Abstract (when present)` to prevent vector index gaps.

2. **High International Collaboration Density:**
   - Observation: Over **92.9%** of works contain co-authors from international institutions (US, GB, ZA, UG, etc.).
   - Engineering Decision: In Phase 2, graph modeling should distinguish domestic parent affiliations (`AFFILIATED_WITH`) from cross-border institutional collaborations (`COLLABORATES_WITH`).

3. **Topic Modeling Consistency:**
   - Observation: Primary topics and OpenAlex topics are present in **100.0%** of records.
   - Engineering Decision: `(:Paper)-[:ABOUT]->(:Topic)` will serve as a highly dense and reliable spine for graph traversals and hybrid retrieval.
