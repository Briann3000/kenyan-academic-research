# Phase 3A: Keyword Retrieval Baseline Report (BM25)

> **Project:** Kenyan Academic Research Knowledge Graph  
> **Evaluation Date:** 2026-10-03 09:21:06 UTC  
> **Status:** Completed & Tested (Control Baseline for Information Retrieval)

---

## 1. Objective & Scope

The goal of Phase 3A is to implement a strict, reproducible **conventional keyword-based retrieval baseline** using standard Okapi BM25. This engine acts as the formal experimental control against which dense semantic search (Phase 3B) and graph-enhanced semantic search (Phase 3C) will be compared.

---

## 2. Retrieval Corpus & Search Text Specification

- **Corpus Source:** `data/processed/normalized_works.jsonl`
- **Document Count:** **1000** distinct academic works
- **Searchable Field (`search_text`):**
  ```text
  search_text = Title + Topic labels + Abstract (when available)
  ```
- **Missing Abstract Fallback:** As established in Phase 1, works without abstracts rely on `Title + Topic labels`, preventing omission from the inverted index.

---

## 3. BM25 Algorithm & Technical Configuration

| Parameter / Component | Specification | Description / Standard Defaults |
| :--- | :--- | :--- |
| **Library / Engine** | `rank-bm25 0.2.2` | Standard pure-Python Okapi BM25 implementation |
| **Term Frequency Saturation ($k_1$)** | `1.5` | Standard default governing term frequency scaling |
| **Document Length Normalization ($b$)** | `0.75` | Standard default scaling document length penalty |
| **Tokenizer** | Regex `\b\w+\b` | Alphanumeric word boundary tokenizer |
| **Case Sensitivity** | Lowercase | All tokens lowercased before indexing |
| **Stopword Removal** | Standard English Set | Standard 127-stopword list filtered during tokenization |
| **Tie-Breaking Rule** | Score DESC, Paper ID ASC | Deterministic ranking for equal scores |

---

## 4. Benchmark Performance Metrics

| Metric | Measured Value |
| :--- | :---: |
| **Corpus Loading Time** | 387.03 ms |
| **Index Construction Time** | 330.55 ms |
| **Average Query Latency** | 3.05 ms |
| **Min Query Latency** | 2.42 ms |
| **Max Query Latency** | 4.62 ms |

---

## 5. Test Query Results (Top-10 per Query)

### Query 1: `mobile money and small businesses in Kenya`
- **Execution Latency:** 4.62 ms | **Hits Returned:** 10

| Rank | Score | Paper ID | Year | Title |
| :---: | :---: | :--- | :---: | :--- |
| 1 | 12.80 | `W2148958761` | 2010 | Effects of a mobile phone short message service on antiretroviral treatment adhe |
| 2 | 11.29 | `W1994682066` | 2012 | Quantifying the Impact of Human Mobility on Malaria |
| 3 | 9.38 | `W2160800204` | 2011 | The effect of mobile phone text-message reminders on Kenyan health workers' adhe |
| 4 | 8.99 | `W2153360349` | 2011 | Mobile phone technologies improve adherence to antiretroviral treatment in a res |
| 5 | 8.08 | `W3087706624` | 2020 | COVID-19 implications on household income and food security in Kenya and Uganda: |
| 6 | 7.62 | `W1517154901` | 2015 | Correlates of Total Sedentary Time and Screen Time in 9–11 Year-Old Children aro |
| 7 | 6.62 | `W2747399005` | 2017 | Estimates of burden and consequences of infants born small for gestational age i |
| 8 | 6.39 | `W2176197350` | 2001 | Characteristics of Larval Anopheline (Diptera: Culicidae) Habitats in Western Ke |
| 9 | 6.27 | `W2156747165` | 2013 | National and regional estimates of term and preterm babies born small for gestat |
| 10 | 6.17 | `W2156247188` | 2012 | Capital Is Not Enough: Innovation in Developing Economies |

### Query 2: `climate change effects on agriculture`
- **Execution Latency:** 2.75 ms | **Hits Returned:** 10

| Rank | Score | Paper ID | Year | Title |
| :---: | :---: | :--- | :---: | :--- |
| 1 | 11.46 | `W1989465135` | 2009 | The impacts of climate change on livestock and livestock systems in developing c |
| 2 | 10.99 | `W2117807705` | 2006 | Will African Agriculture Survive Climate Change? |
| 3 | 9.80 | `W3120805833` | 2021 | Impacts of climate change on the livestock food supply chain; a review of the ev |
| 4 | 9.02 | `W2119727696` | 2011 | Options for support to agriculture and food security under climate change |
| 5 | 8.99 | `W2144059688` | 2010 | Radically Rethinking Agriculture for the 21st Century |
| 6 | 8.83 | `W3210895846` | 2021 | A systematic global stocktake of evidence on human adaptation to climate change |
| 7 | 8.81 | `W2010751289` | 2012 | Adapting agriculture to climate change in Kenya: Household strategies and determ |
| 8 | 8.80 | `W2899855266` | 2018 | Increasing resilience of smallholder farmers to climate change through multiple  |
| 9 | 8.78 | `W2078771477` | 2014 | Climate-smart agriculture for food security |
| 10 | 8.72 | `W2024616545` | 2013 | Climate change and Ecosystem-based Adaptation: a new pragmatic approach to buffe |

### Query 3: `malaria prevention among children`
- **Execution Latency:** 2.42 ms | **Hits Returned:** 10

| Rank | Score | Paper ID | Year | Title |
| :---: | :---: | :--- | :---: | :--- |
| 1 | 11.39 | `W2124892357` | 2008 | Efficacy of RTS,S/AS01E Vaccine against Malaria in Children 5 to 17 Months of Ag |
| 2 | 11.39 | `W2472994826` | 2016 | Seven-Year Efficacy of RTS,S/AS01 Malaria Vaccine among Young African Children |
| 3 | 10.65 | `W1996218267` | 1996 | Insecticide‐treated bednets reduce mortality and severe morbidity from malaria a |
| 4 | 10.33 | `W2115179234` | 2005 | Bacteremia among Children Admitted to a Rural Hospital in Kenya |
| 5 | 10.31 | `W2322213143` | 1995 | Indicators of Life-Threatening Malaria in African Children |
| 6 | 9.82 | `W3047978089` | 2020 | Hospitalization Rates and Characteristics of Children Aged <18 Years Hospitalize |
| 7 | 9.11 | `W2116112113` | 2005 | Malaria Infection Increases Attractiveness of Humans to Mosquitoes |
| 8 | 8.74 | `W3135682442` | 2021 | Global Tuberculosis Report 2020 – Reflections on the Global TB burden, treatment |
| 9 | 8.74 | `W1817614869` | 2015 | Immunogenicity of the RTS,S/AS01 malaria vaccine and implications for duration o |
| 10 | 8.44 | `W2289977218` | 2015 | Genetic Diversity and Protective Efficacy of the RTS,S/AS01 Malaria Vaccine |

### Query 4: `maternal healthcare access in rural Kenya`
- **Execution Latency:** 2.98 ms | **Hits Returned:** 10

| Rank | Score | Paper ID | Year | Title |
| :---: | :---: | :--- | :---: | :--- |
| 1 | 16.20 | `W3062442779` | 2020 | Impact of the societal response to COVID-19 on access to healthcare for non-COVI |
| 2 | 11.43 | `W1989089335` | 2000 | Frequency and timing of antenatal care in Kenya: explaining the variations betwe |
| 3 | 10.89 | `W2896867593` | 2018 | Global epidemiology of use of and disparities in caesarean sections |
| 4 | 10.80 | `W2896447320` | 2018 | Interventions to reduce unnecessary caesarean sections in healthy women and babi |
| 5 | 10.79 | `W1985781493` | 2013 | Moving beyond essential interventions for reduction of maternal mortality (the W |
| 6 | 10.49 | `W2057131920` | 2011 | Utilization of maternal health services among young women in Kenya: Insights fro |
| 7 | 10.48 | `W2043449557` | 2014 | Global, regional, and national levels and causes of maternal mortality during 19 |
| 8 | 10.24 | `W3204415617` | 2021 | The Lancet Commission on diagnostics: transforming access to diagnostics |
| 9 | 10.13 | `W2052458288` | 2012 | Population Distribution, Settlement Patterns and Accessibility across Africa in  |
| 10 | 9.92 | `W2103818838` | 2012 | Effect of mother’s education on child’s nutritional status in the slums of Nairo |

### Query 5: `digital financial services and informal businesses`
- **Execution Latency:** 2.50 ms | **Hits Returned:** 10

| Rank | Score | Paper ID | Year | Title |
| :---: | :---: | :--- | :---: | :--- |
| 1 | 8.71 | `W3020814017` | 2020 | Slum Health: Arresting COVID-19 and Improving Well-Being in Urban Informal Settl |
| 2 | 7.66 | `W3087706624` | 2020 | COVID-19 implications on household income and food security in Kenya and Uganda: |
| 3 | 7.44 | `W2097929497` | 2012 | Comparative analysis of solid waste management in 20 cities |
| 4 | 7.15 | `W2149979947` | 2014 | The African Financial Development and Financial Inclusion Gaps |
| 5 | 7.06 | `W2066626803` | 2009 | Digital Soil Map of the World |
| 6 | 6.75 | `W2139905170` | 2015 | Food Safety in Low and Middle Income Countries |
| 7 | 6.59 | `W2785232091` | 2018 | Access to emergency hospital care provided by the public sector in sub-Saharan A |
| 8 | 6.57 | `W2102348230` | 2010 | The dominant Anopheles vectors of human malaria in the Americas: occurrence data |
| 9 | 5.59 | `W2080310378` | 2014 | Analysis of Adoption and Impacts of Improved Maize Varieties in Eastern Zambia |
| 10 | 5.56 | `W2042094287` | 2000 | Annual Plasmodium falciparum entomological inoculation rates (EIR) across Africa |

---

## 6. Input Validation & Edge Case Sanity Checks

Automated unit tests in `tests/test_keyword_search.py` verified the following behavioral sanity checks:
- **Empty Query (`''`)**: Returns 0 hits with 0.0ms latency without crashing.
- **Whitespace-Only Query (`'   \t\n '`)**: Returns 0 hits safely.
- **Out-of-Vocabulary Terms (`'quantum teleportation'`)**: Returns 0 hits.
- **Stopwords-Only Query (`'what is the and or'`)**: Filters out all tokens and returns 0 hits safely.
- **Boundary $k$ Values ($k=0$)**: Returns empty list.
- **Boundary $k$ Values ($k > 1000$)**: Safely bounded to corpus size.
- **Deterministic Output**: Identical queries against the corpus produce identical rankings and scores.

---

## 7. Observed Limitations of the Keyword Baseline

1. **Vocabulary Mismatch (Synonym Blindness):**
   - BM25 cannot match terms that do not share exact lexical roots. For example, a paper on *'financial inclusion via M-Pesa'* will receive a lower score if the query specifically uses the phrasing *'digital financial services'* and that exact phrase does not appear in the title/abstract.
2. **Word Sense & Semantic Context:**
   - BM25 scores independently across terms based on inverse document frequency, lacking awareness of conceptual relations (e.g., that *'malaria'* relates inherently to *'Anopheles mosquitoes'* and *'artemisinin'*).
3. **Absence of Relational Context:**
   - BM25 scores papers solely on document text, ignoring author expertise, institutional research clusters, and conceptual topic graphs.

---

## 8. Requirements for Phase 3B (Dense Semantic Retrieval)

To address keyword retrieval limitations, Phase 3B must:
1. Evaluate and select an appropriate sentence embedding model (e.g., `all-MiniLM-L6-v2` or `BGE-small-en-v1.5`).
2. Generate dense vector embeddings for all 1,000 papers over the same standardized `search_text`.
3. Implement a vector similarity search engine matching the exact same `SearchResult` and `SearchResponse` interface.
