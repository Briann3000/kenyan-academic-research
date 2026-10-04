"""Graph-Enhanced Semantic Search Engine (System C).

Reranks top-K dense semantic candidates using bounded Neo4j Knowledge Graph relational evidence
(shared topics, co-authoring researchers, and affiliated institutions) with linear hybrid scoring.
"""

import time
import logging
from typing import List, Optional, Dict, Any, Tuple

from src.search.models import SearchResult, SearchResponse
from src.search.corpus import RetrievalCorpus
from src.search.semantic import SemanticSearchEngine, SemanticSearchEngine as DenseSemanticSearchEngine
from src.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_K = 20
DEFAULT_ALPHA = 0.7
DEFAULT_GRAPH_WEIGHTS = {
    "topic": 0.50,
    "author": 0.25,
    "institution": 0.25,
}


class GraphHybridSearchEngine:
    """Hybrid search engine combining dense semantic retrieval with knowledge graph relational affinity."""

    def __init__(
        self,
        semantic_engine: Optional[DenseSemanticSearchEngine] = None,
        corpus: Optional[RetrievalCorpus] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        alpha: float = DEFAULT_ALPHA,
        graph_weights: Optional[Dict[str, float]] = None,
    ):
        """Initializes the Graph-Enhanced Semantic Search Engine.

        Args:
            semantic_engine: Initialized DenseSemanticSearchEngine (System B).
            corpus: Optional RetrievalCorpus instance if semantic_engine is not provided.
            neo4j_client: Initialized or connected Neo4jClient instance.
            candidate_k: Number of semantic candidates to generate and rerank (default: 20).
            alpha: Interpolation weight between semantic score (alpha) and graph score (1 - alpha).
            graph_weights: Weights for topic, author, and institution overlap components.
        """
        if semantic_engine is not None:
            self.semantic_engine = semantic_engine
        else:
            loaded_corpus = corpus or RetrievalCorpus()
            self.semantic_engine = DenseSemanticSearchEngine(corpus=loaded_corpus)

        self.neo4j_client = neo4j_client
        self.candidate_k = candidate_k
        self.alpha = alpha
        self.graph_weights = self._validate_weights(graph_weights or DEFAULT_GRAPH_WEIGHTS)

    @staticmethod
    def _validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Validates and normalizes graph affinity weights so they sum to 1.0."""
        required = {"topic", "author", "institution"}
        if not required.issubset(weights.keys()):
            missing = required - set(weights.keys())
            raise ValueError(f"Missing required graph weights: {missing}")

        for k, v in weights.items():
            if v < 0.0:
                raise ValueError(f"Graph weight '{k}' must be non-negative, got {v}")

        total = sum(weights[k] for k in required)
        if total <= 0:
            raise ValueError("Sum of graph weights must be strictly positive.")

        # Normalize to exactly 1.0
        return {k: round(weights[k] / total, 6) for k in required}

    def _fetch_candidate_subgraph(
        self, paper_ids: List[str]
    ) -> Tuple[Dict[str, Dict[str, set]], bool]:
        """Fetches 1-hop relational neighborhood for the candidate set in a single batched Cypher query.

        Returns:
            Tuple of (candidate_entities_dict, success_boolean).
            candidate_entities_dict maps paper_id -> {'topics': set(), 'authors': set(), 'institutions': set()}
        """
        if not self.neo4j_client:
            logger.info("Neo4j client not provided. Operating in fallback mode.")
            return {}, False

        query = """
        MATCH (p:Paper)
        WHERE p.id IN $paper_ids
        OPTIONAL MATCH (p)-[:ABOUT]->(t:Topic)
        OPTIONAL MATCH (a:Researcher)-[:AUTHORED]->(p)
        OPTIONAL MATCH (p)-[:AFFILIATED_WITH]->(i:Institution)
        RETURN p.id AS paper_id,
               collect(DISTINCT t.id) AS topic_ids,
               collect(DISTINCT a.id) AS author_ids,
               collect(DISTINCT i.id) AS institution_ids
        """
        try:
            records = self.neo4j_client.execute_query(query, {"paper_ids": paper_ids})
            subgraph_data: Dict[str, Dict[str, set]] = {}
            for rec in records:
                p_id = rec["paper_id"]
                subgraph_data[p_id] = {
                    "topics": {t for t in rec.get("topic_ids", []) if t is not None},
                    "authors": {a for a in rec.get("author_ids", []) if a is not None},
                    "institutions": {i for i in rec.get("institution_ids", []) if i is not None},
                }

            # Ensure all candidates are present in mapping
            for p_id in paper_ids:
                if p_id not in subgraph_data:
                    subgraph_data[p_id] = {"topics": set(), "authors": set(), "institutions": set()}

            return subgraph_data, True
        except Exception as exc:
            logger.warning(f"Neo4j query execution failed: {exc}. Gracefully falling back to semantic ranking.")
            return {}, False

    @staticmethod
    def _calculate_candidate_relative_overlaps(
        paper_ids: List[str], subgraph_data: Dict[str, Dict[str, set]]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculates candidate-relative overlap scores strictly within the retrieved candidate set.

        For each candidate paper p_i, raw overlap with all other candidates p_j in C (j != i) is computed:
          raw_topic(p_i) = sum_{j != i} |T_i ∩ T_j|
          raw_author(p_i) = sum_{j != i} |A_i ∩ A_j|
          raw_institution(p_i) = sum_{j != i} |I_i ∩ I_j|

        Normalized overlaps O_x(p_i) = raw_x(p_i) / max_{k}(raw_x(p_k)) if max > 0 else 0.0.
        """
        raw_scores: Dict[str, Dict[str, int]] = {
            p_id: {"topic": 0, "author": 0, "institution": 0} for p_id in paper_ids
        }

        n = len(paper_ids)
        if n > 1:
            for i in range(n):
                p_i = paper_ids[i]
                data_i = subgraph_data.get(p_i, {"topics": set(), "authors": set(), "institutions": set()})
                for j in range(i + 1, n):
                    p_j = paper_ids[j]
                    data_j = subgraph_data.get(p_j, {"topics": set(), "authors": set(), "institutions": set()})

                    shared_topics = len(data_i["topics"] & data_j["topics"])
                    shared_authors = len(data_i["authors"] & data_j["authors"])
                    shared_insts = len(data_i["institutions"] & data_j["institutions"])

                    raw_scores[p_i]["topic"] += shared_topics
                    raw_scores[p_j]["topic"] += shared_topics

                    raw_scores[p_i]["author"] += shared_authors
                    raw_scores[p_j]["author"] += shared_authors

                    raw_scores[p_i]["institution"] += shared_insts
                    raw_scores[p_j]["institution"] += shared_insts

        max_topic = max((s["topic"] for s in raw_scores.values()), default=0)
        max_author = max((s["author"] for s in raw_scores.values()), default=0)
        max_inst = max((s["institution"] for s in raw_scores.values()), default=0)

        normalized_scores: Dict[str, Dict[str, Any]] = {}
        for p_id in paper_ids:
            raw_t = raw_scores[p_id]["topic"]
            raw_a = raw_scores[p_id]["author"]
            raw_i = raw_scores[p_id]["institution"]

            o_topic = (raw_t / max_topic) if max_topic > 0 else 0.0
            o_author = (raw_a / max_author) if max_author > 0 else 0.0
            o_inst = (raw_i / max_inst) if max_inst > 0 else 0.0

            normalized_scores[p_id] = {
                "raw_topic": raw_t,
                "raw_author": raw_a,
                "raw_institution": raw_i,
                "o_topic": round(o_topic, 6),
                "o_author": round(o_author, 6),
                "o_institution": round(o_inst, 6),
            }

        return normalized_scores

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: Optional[int] = None,
        alpha: Optional[float] = None,
        graph_weights: Optional[Dict[str, float]] = None,
    ) -> SearchResponse:
        """Executes Graph-Enhanced Semantic Search for a user query.

        Args:
            query: Input user query string.
            top_k: Number of final ranked results to return.
            candidate_k: Candidate set size K (overrides instance default if specified).
            alpha: Hybrid interpolation parameter alpha in [0.0, 1.0].
            graph_weights: Optional overriding graph weights dict.

        Returns:
            Standardized SearchResponse envelope with detailed rank-shift and overlap metadata.
        """
        start_time = time.perf_counter()

        if not query or not query.strip():
            return SearchResponse(
                query=query or "",
                retrieval_method="graph_hybrid",
                total_hits=0,
                latency_ms=0.0,
                results=[],
            )

        eff_candidate_k = candidate_k if candidate_k is not None else self.candidate_k
        eff_alpha = alpha if alpha is not None else self.alpha
        eff_weights = self._validate_weights(graph_weights) if graph_weights else self.graph_weights

        if not (0.0 <= eff_alpha <= 1.0):
            raise ValueError(f"Alpha must be in range [0.0, 1.0], got {eff_alpha}")

        # Step 1: Semantic Candidate Generation (System B)
        sem_start = time.perf_counter()
        semantic_response = self.semantic_engine.search(query, top_k=eff_candidate_k)
        sem_latency_ms = (time.perf_counter() - sem_start) * 1000.0

        candidates = semantic_response.results
        if not candidates:
            return SearchResponse(
                query=query,
                retrieval_method="graph_hybrid",
                total_hits=0,
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                results=[],
            )

        paper_ids = [c.paper_id for c in candidates]

        # Step 2: Bounded Subgraph Retrieval & Fallback Check
        graph_start = time.perf_counter()
        subgraph_data, graph_available = self._fetch_candidate_subgraph(paper_ids)
        graph_latency_ms = (time.perf_counter() - graph_start) * 1000.0

        fallback_used = not graph_available

        # Step 3: Graph Affinity and Hybrid Scoring
        if graph_available and len(candidates) > 1:
            overlap_scores = self._calculate_candidate_relative_overlaps(paper_ids, subgraph_data)
        else:
            overlap_scores = {
                p_id: {
                    "raw_topic": 0,
                    "raw_author": 0,
                    "raw_institution": 0,
                    "o_topic": 0.0,
                    "o_author": 0.0,
                    "o_institution": 0.0,
                }
                for p_id in paper_ids
            }

        scored_items: List[Dict[str, Any]] = []
        for orig_idx, cand in enumerate(candidates, start=1):
            p_id = cand.paper_id
            overlaps = overlap_scores.get(
                p_id,
                {
                    "raw_topic": 0,
                    "raw_author": 0,
                    "raw_institution": 0,
                    "o_topic": 0.0,
                    "o_author": 0.0,
                    "o_institution": 0.0,
                },
            )

            if graph_available and len(candidates) > 1:
                s_graph = (
                    eff_weights["topic"] * overlaps["o_topic"]
                    + eff_weights["author"] * overlaps["o_author"]
                    + eff_weights["institution"] * overlaps["o_institution"]
                )
            else:
                s_graph = 0.0

            s_semantic = cand.score
            if fallback_used:
                s_hybrid = s_semantic
            else:
                s_hybrid = eff_alpha * s_semantic + (1.0 - eff_alpha) * s_graph

            scored_items.append(
                {
                    "candidate": cand,
                    "paper_id": p_id,
                    "semantic_score": round(s_semantic, 6),
                    "graph_score": round(s_graph, 6),
                    "hybrid_score": round(s_hybrid, 6),
                    "semantic_rank": cand.rank,
                    "overlaps": overlaps,
                }
            )

        # Step 4: Deterministic Reranking (hybrid_score DESC, paper_id ASC)
        scored_items.sort(key=lambda item: (-round(item["hybrid_score"], 8), item["paper_id"]))

        # Step 5: Format Standardized Search Results
        final_results: List[SearchResult] = []
        for new_rank, item in enumerate(scored_items[:top_k], start=1):
            cand: SearchResult = item["candidate"]
            sem_rank = item["semantic_rank"]
            rank_change = sem_rank - new_rank  # Positive = moved up, Negative = moved down

            res_metadata = dict(cand.metadata)
            res_metadata.update(
                {
                    "semantic_score": item["semantic_score"],
                    "graph_score": item["graph_score"],
                    "hybrid_score": item["hybrid_score"],
                    "semantic_rank": sem_rank,
                    "hybrid_rank": new_rank,
                    "rank_change": rank_change,
                    "topic_overlap": item["overlaps"]["o_topic"],
                    "author_overlap": item["overlaps"]["o_author"],
                    "institution_overlap": item["overlaps"]["o_institution"],
                    "raw_topic_overlap": item["overlaps"]["raw_topic"],
                    "raw_author_overlap": item["overlaps"]["raw_author"],
                    "raw_institution_overlap": item["overlaps"]["raw_institution"],
                    "alpha": eff_alpha,
                    "graph_weights": eff_weights,
                    "candidate_k": eff_candidate_k,
                    "graph_available": graph_available,
                    "fallback_used": fallback_used,
                    "sem_latency_ms": round(sem_latency_ms, 2),
                    "graph_latency_ms": round(graph_latency_ms, 2),
                }
            )

            final_results.append(
                SearchResult(
                    paper_id=cand.paper_id,
                    title=cand.title,
                    score=item["hybrid_score"],
                    rank=new_rank,
                    publication_year=cand.publication_year,
                    doi=cand.doi,
                    retrieval_method="graph_hybrid",
                    snippet=cand.snippet,
                    metadata=res_metadata,
                )
            )

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SearchResponse(
            query=query,
            retrieval_method="graph_hybrid",
            total_hits=len(final_results),
            latency_ms=round(total_latency_ms, 2),
            results=final_results,
        )
