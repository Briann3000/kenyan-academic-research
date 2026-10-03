# Graph Topology & Entity Modeling Report (Phase 2)

> **Project:** Kenyan Academic Research Knowledge Graph  
> **Validation Timestamp:** 2026-10-03 09:05:31 UTC  
> **Reconciliation Script:** `scripts/reconcile_phase2.py`  
> **Corpus Size:** $N = 1000$ Normalized Works  
> **Raw Corpus SHA256:** `716d19429ebf275a936d525471f677c77dec56c4aa7d66b092f8e51144362e5d`  
> **Normalized Corpus SHA256:** `580cb7583725f41e89f60ac25e1a28930d6961509142048b77c733f8c223cb6d`  
> **Status:** Formally Validated & Reconciled

---

## 1. Dataset & Graph Scope

This report documents the definitive, programmatically verified topological metrics, entity disambiguation findings, and retrieval constraints for the knowledge graph constructed from the $N = 1,000$ Kenyan-affiliated OpenAlex seed corpus.

### Core Graph Schema
```text
(:Researcher)-[:AUTHORED]->(:Paper)
(:Paper)-[:ABOUT]->(:Topic)
(:Paper)-[:AFFILIATED_WITH]->(:Institution)
(:Paper)-[:PUBLISHED_IN]->(:Year)
```

---

## 2. Graph Node & Relationship Census

### A. Authoritative Node Census
| Node Label | Distinct Count | Primary Identifier Strategy | Missing ID Fallback Mechanism |
| :--- | :---: | :--- | :--- |
| **`Paper`** | **1000** | OpenAlex Work ID (e.g. `W2112776483`) | None (100% complete) |
| **`Researcher`** | **14860** | OpenAlex Author ID (e.g. `A5014066641`) | Deterministic name hash fallback (`AUTH_SYNTH_<hash>`) |
| **`Institution`** | **4125** | OpenAlex Institution ID (e.g. `I2841861`) | None (100% complete) |
| **`Topic`** | **743** | OpenAlex Topic ID (e.g. `T10029`) | None (100% complete) |
| **`Year`** | **44** | Gregorian Publication Year (e.g. `2024`) | None (100% complete) |
| **Total Nodes** | **20772** | — | — |

### B. Authoritative Relationship Edge Census
| Relationship Type | Directed Semantics | Distinct Edge Count |
| :--- | :--- | :---: |
| **`:AUTHORED`** | `(:Researcher)-[:AUTHORED]->(:Paper)` | **19553** |
| **`:AFFILIATED_WITH`** | `(:Paper)-[:AFFILIATED_WITH]->(:Institution)` | **14616** |
| **`:ABOUT`** | `(:Paper)-[:ABOUT]->(:Topic)` | **2955** |
| **`:PUBLISHED_IN`** | `(:Paper)-[:PUBLISHED_IN]->(:Year)` | **1000** |
| **Total Edges** | — | **38124** |

---

## 3. Node Degree Distributions

| Dimension | Min | $P_{25}$ | Median | $P_{75}$ | $P_{95}$ | Max | Mean | Std Dev ($\sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Researchers per Paper (In-degree)** | 1 | 5.0 | 9.0 | 21.25 | 93.0 | 100 | 19.55 | 24.84 |
| **Institutions per Paper (Out-degree)** | 1 | 4.0 | 7.0 | 16.0 | 59.1 | 134 | 14.62 | 19.42 |
| **Topics per Paper (Out-degree)** | 1 | 3.0 | 3.0 | 3.0 | 3.0 | 3 | 2.96 | 0.27 |
| **Papers per Researcher (Degree)** | 1 | 1.0 | 1.0 | 1.0 | 3.0 | 35 | 1.32 | 1.06 |
| **Papers per Institution (Degree)** | 1 | 1.0 | 1.0 | 3.0 | 13.0 | 188 | 3.54 | 8.0 |
| **Papers per Topic (Degree)** | 1 | 1.0 | 1.0 | 4.0 | 14.0 | 85 | 3.98 | 6.96 |

---

## 4. Connectivity Analysis: Structural vs. Meaningful Research Connectivity

### Subgraph Connectivity Breakdown

| Subgraph Configuration | Node Types Included | Relationships Included | Total Nodes | Total Edges | Total CCs | Largest CC Size | LCC % | Isolated Nodes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Full Graph** | Paper, Researcher, Institution, Topic, Year | `AUTHORED`, `ABOUT`, `AFFILIATED_WITH`, `PUBLISHED_IN` | 20772 | 38124 | **1** | **20772** | **100.0%** | **0** |
| **2. Graph Without Year** | Paper, Researcher, Institution, Topic | `AUTHORED`, `ABOUT`, `AFFILIATED_WITH` | 20728 | 37124 | **2** | **20722** | **99.97%** | **0** |
| **3. Core Research Only** | Paper, Researcher, Institution | `AUTHORED`, `AFFILIATED_WITH` | 19985 | 34169 | **5** | **19965** | **99.9%** | **0** |
| **4. Direct Co-Authorship** | Researcher (projected) | Co-authorship clique edges | 14860 | 448956 | **169** | **13246** | **89.14%** | **10** |

### Interpretation: Structural vs. Meaningful Research Connectivity
- **Structural Connectivity**: In the full graph, **100.00% (20772 nodes)** form **1 single connected component**. Removing Year nodes separates the graph into **2 components** (LCC: **99.97%**). In the core bipartite research graph (AUTHORED + AFFILIATED_WITH), there are **5 components** (LCC: **99.9%**).
- **Meaningful Research Connectivity**: When projecting strictly onto direct researcher co-authorship, the graph separates into **169 distinct research communities**, with **10 solo authors** and an LCC containing **89.14%** of researchers.
- **Implication for Phase 3**: High structural connectivity is driven by large international consortia and institutional hubs. Multi-hop traversals must be strictly bounded (1-to-2 hops) and topic-gated to avoid precision degradation.

---

## 5. Entity Disambiguation & Missing Identifier Handling

### A. Missing Researcher Source Identifiers
- **Total Raw Author Instances:** 19703
- **Author Instances Lacking OpenAlex Author ID:** **527 (2.67%)**
- **Distinct Author Name Strings Involved:** **482**
- **Author Instances with ORCID Available:** **16**
- **Treatment:** Fallback identifier generation (`AUTH_SYNTH_<SHA256(canonical_name)>`) is applied strictly to satisfy graph referential integrity. **No automated entity merging was performed solely based on name strings.**

### B. Institution Name Collisions (Shared Display Names)
- **Total Distinct Institution Display Names:** 4026
- **Display Names Associated with Multiple OpenAlex IDs:** **43**

| Display Name | Distinct OpenAlex IDs | Country Codes | Disambiguation Rationale |
| :--- | :---: | :--- | :--- |
| **Ministry of Health** | 24 | `BW, KE, MZ, SG, CL, OM (+18 more)` | Distinct national branches or sovereign ministries. |
| **International Institute of Tropical Agriculture** | 8 | `KE, NG, CM, UG, NG, BJ (+2 more)` | Distinct national branches or sovereign ministries. |
| **Stockholm Environment Institute** | 7 | `SE, GB, US, TH, GB, KE (+1 more)` | Distinct national branches or sovereign ministries. |
| **Ministry of Public Health** | 6 | `LB, TH, QA, AF, CD, CM` | Distinct national branches or sovereign ministries. |
| **Médecins Sans Frontières** | 6 | `CH, BE, UG, NL, LU, FR` | Distinct national branches or sovereign ministries. |
| **Institut de Recherche pour le Développement** | 5 | `FR, NC, BJ, CG, SN` | Distinct national branches or sovereign ministries. |
| **International Crops Research Institute for the Semi-Arid Tropics** | 4 | `KE, IN, ML, ET` | Distinct national branches or sovereign ministries. |
| **International Maize and Wheat Improvement Center** | 4 | `KE, BD, ET, IN` | Distinct national branches or sovereign ministries. |

---

## 6. Graph Data Integrity Verification

- **Primary Key Uniqueness:** **100% PASS** (0 duplicate IDs across Paper, Researcher, Institution, Topic, Year)
- **Referential Integrity:** **100% PASS** (all 38124 edges point to valid nodes)
- **Isolated / Orphan Nodes:** **0**
- **Papers Missing Authors:** **0**
- **Papers Missing Institutions:** **0**
- **Papers Missing Topics:** **0**
- **Papers Missing Publication Year:** **0**

---

## 7. Recommendations for Phase 3 (Retrieval Baselines & Hybrid Search)

1. **Semantic Search as Precision Anchor**: Dense vector similarity over composite `search_text` (`Title + Topic labels + Abstract`) must generate initial candidate sets ($k pprox 20	ext{–}50$).
2. **Bounded Graph Expansion**: Use `[:ABOUT]->(:Topic)` and `[:AUTHORED]->(:Researcher)` within a strict 1-to-2 hop radius to discover related papers and experts without traversing full institutional hubs.
3. **Geographic Filtering**: Distinguish domestic capacity from international partners by filtering `(:Institution {country_code: 'KE'})`.