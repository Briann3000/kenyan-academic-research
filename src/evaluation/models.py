"""Data models for Information Retrieval (IR) evaluation benchmark, pooling, qrels, and metrics."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class BenchmarkQuery(BaseModel):
    """Normalized representation of an evaluation benchmark query."""
    query_id: str = Field(..., description="Unique query identifier, e.g., Q01, Q02")
    domain: str = Field(..., description="Thematic domain (e.g. Health & Epidemiology)")
    query_text: str = Field(..., description="Literal query text submitted to retrieval systems")
    description: Optional[str] = Field(None, description="Detailed description of the user's information need")


class PooledCandidate(BaseModel):
    """Candidate document in the depth-10 pooled candidate set for a query."""
    paper_id: str = Field(..., description="OpenAlex Work ID")
    title: str = Field(..., description="Paper title")
    abstract: Optional[str] = Field(None, description="Reconstructed abstract text if available")
    retrieved_by: List[str] = Field(default_factory=list, description="Systems that retrieved this paper (e.g. ['system_a', 'system_b'])")
    source_ranks: Dict[str, int] = Field(default_factory=dict, description="Rank at which each system retrieved this paper")


class QueryPool(BaseModel):
    """Relevance assessment pool for a single query across evaluated systems."""
    query_id: str = Field(..., description="Query ID")
    domain: str = Field(..., description="Thematic domain")
    query_text: str = Field(..., description="Query text")
    total_candidates: int = Field(0, description="Total unique candidate documents in pool")
    candidates: List[PooledCandidate] = Field(default_factory=list, description="De-duplicated candidate pool")


class RelevanceJudgment(BaseModel):
    """Single graded relevance assessment."""
    query_id: str = Field(..., description="Query ID")
    paper_id: str = Field(..., description="Work ID")
    relevance: int = Field(..., ge=0, le=2, description="Graded relevance: 0=Irrelevant, 1=Partially Relevant, 2=Highly Relevant")
    judge_id: Optional[str] = Field(default="judge_1", description="Identifier of the assessor")
    notes: Optional[str] = Field(default=None, description="Assessment rationale or notes")


class QueryEvaluationMetrics(BaseModel):
    """Computed IR metrics for a single query and retrieval system."""
    query_id: str = Field(..., description="Query ID")
    system_name: str = Field(..., description="System identifier: system_a, system_b, or system_c")
    p_at_5: float = Field(..., description="Precision at rank 5")
    p_at_10: float = Field(..., description="Precision at rank 10")
    recall_at_10: float = Field(..., description="Recall at rank 10 against judged relevant documents in pool")
    mrr: float = Field(..., description="Mean Reciprocal Rank (first result with relevance >= 1)")
    ndcg_at_5: float = Field(..., description="nDCG at rank 5 using graded 0/1/2 relevance")
    ndcg_at_10: float = Field(..., description="nDCG at rank 10 using graded 0/1/2 relevance")
    latency_ms: float = Field(..., description="Query execution latency in milliseconds")
    judged_hits_at_10: int = Field(..., description="Number of top-10 retrieved papers that had judgments in qrels")


class DomainSummaryMetrics(BaseModel):
    """Aggregated evaluation metrics for a specific thematic domain."""
    domain: str = Field(..., description="Domain name")
    num_queries: int = Field(..., description="Number of queries in domain")
    mean_p_at_5: float
    mean_p_at_10: float
    mean_recall_at_10: float
    mean_mrr: float
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    mean_latency_ms: float


class SystemSummaryMetrics(BaseModel):
    """Overall aggregated evaluation metrics for a retrieval system."""
    system_name: str
    num_queries: int
    mean_p_at_5: float
    mean_p_at_10: float
    mean_recall_at_10: float
    mean_mrr: float
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    mean_latency_ms: float
    domain_breakdowns: Dict[str, DomainSummaryMetrics] = Field(default_factory=dict)
