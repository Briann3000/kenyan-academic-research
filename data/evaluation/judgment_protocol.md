# Relevance Judgment Protocol & Annotation Guidelines (Phase 3D)

## 1. Overview & Evaluation Scope
This protocol establishes the standardized guidelines for assigning graded relevance judgments to candidate research papers pooled from the three retrieval systems (System A: BM25 Keyword, System B: BAAI/bge-small Dense Semantic, and System C: Candidate-Relative Graph Hybrid) on the frozen 1,000-paper Kenyan academic research corpus.

---

## 2. Assessment Blinding & Objectivity
To eliminate confirmation bias and system favoritism during judgment:
1. **System Identity Masking**: The retrieval system(s) that retrieved a candidate paper (`system_a`, `system_b`, `system_c`) are masked and hidden from the assessor during document scoring.
2. **Rank Masking**: The retrieval rank (1–10) and numerical retrieval score assigned by any engine are completely withheld from the judging view.
3. **Randomized Candidate Order**: Candidate documents within each query pool are presented in pseudo-randomized order (or sorted by neutral paper ID) rather than in retrieval score order.

---

## 3. Graded Relevance Judgment Scale (0, 1, 2)

Assessors evaluate each document based on the query text and description against the paper's title and abstract:

| Relevance Level | Label | Description & Inclusion Criteria |
| :---: | :--- | :--- |
| **0** | **Irrelevant / Off-topic** | The document does not address the query's information need. It belongs to a completely different domain, or only matches superficial keywords in an unrelated context (e.g. mentions "Kenya" or "models" in an unrelated physics context). |
| **1** | **Partially Relevant** | The document is situated in the broader domain or addresses a related sub-topic, but does not directly or fully answer the specific user information need (e.g., query asks for *push-pull stemborer control* and paper discusses general maize agronomy or chemical pesticides). |
| **2** | **Highly Relevant** | The document directly, specifically, and substantively addresses the user's primary information need in the Kenyan / East African context (e.g., query asks for *antiretroviral therapy adherence in Kenya* and paper is an empirical trial on SMS reminders for HIV ART adherence in Kenyan clinics). |

---

## 4. Evaluator Limitation & Single-Judge Disclosure
* **Assessor Profile**: Domain-literate academic evaluator assessing relevance against published abstracts.
* **Single-Judge Limitation**: Acknowledged as a single-judge evaluation. Because multiple independent judges are not available in this iteration, inter-rater reliability statistics (e.g., Cohen's Kappa, Fleiss' Kappa) are not applicable. All judgment rationale and qrels are serialized openly in `data/evaluation/qrels.json` for full auditability and reproducibility.

---

## 5. Qrels Serialization Format (`data/evaluation/qrels.json`)
The relevance judgments (`qrels`) are stored as a JSON object mapping each `query_id` to a dictionary of `paper_id: relevance_score`:

```json
{
  "Q01": {
    "W2322213143": 2,
    "W2797631732": 2,
    "W1985149691": 1,
    "W2008641344": 0
  },
  "Q02": {
    "W1989089335": 2,
    "W2057131920": 2
  }
}
```

---

## 6. Critical Evaluation Safeguards
1. **No Automatic Ground Truth**: Relevance judgments are never inferred from BM25 scores, cosine similarity, graph scores, or an automated LLM generator.
2. **Unjudged Document Rule**: Documents not present in the judged candidate pool are treated as *unjudged*, not as zero-relevance documents.
3. **Recall Formulation**: Recall is strictly reported as *Recall@10 against judged relevant documents in the pooled qrels*.
