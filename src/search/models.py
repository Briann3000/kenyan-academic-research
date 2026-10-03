"""Data models for retrieval documents, query configurations, and standardized search results."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RetrievalDocument(BaseModel):
    """Normalized document representation for the retrieval corpus."""
    paper_id: str = Field(..., description="OpenAlex Work ID (e.g., W2112776483)")
    title: str = Field(..., description="Paper title")
    search_text: str = Field(..., description="Primary searchable composite text: Title + Topics + Abstract")
    abstract: Optional[str] = Field(None, description="Reconstructed abstract text if present")
    topics: List[str] = Field(default_factory=list, description="List of topic display names")
    publication_year: int = Field(..., description="Year of publication")
    doi: Optional[str] = Field(None, description="DOI URL or identifier")
    work_type: Optional[str] = Field(None, description="Work type, e.g., article, review")
    is_oa: bool = Field(False, description="Open Access boolean")
    oa_status: Optional[str] = Field(None, description="OA status")


class SearchResult(BaseModel):
    """Standardized search result item returned across all retrieval paradigms."""
    paper_id: str = Field(..., description="OpenAlex Work ID")
    title: str = Field(..., description="Paper title")
    score: float = Field(..., description="Retrieval score (BM25 score, cosine similarity, or hybrid score)")
    rank: int = Field(..., description="1-indexed rank in result list")
    publication_year: int = Field(..., description="Publication year")
    doi: Optional[str] = Field(None, description="DOI")
    retrieval_method: str = Field(..., description="Retrieval method label: 'bm25', 'semantic', 'graph_hybrid'")
    snippet: Optional[str] = Field(None, description="Brief snippet or abstract excerpt for inspectability")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional debugging metadata")


class SearchResponse(BaseModel):
    """Standardized top-level response envelope for search queries."""
    query: str = Field(..., description="Input query string")
    retrieval_method: str = Field(..., description="Method used")
    total_hits: int = Field(..., description="Number of results returned")
    latency_ms: float = Field(..., description="Query execution latency in milliseconds")
    results: List[SearchResult] = Field(default_factory=list, description="Ranked search results")
