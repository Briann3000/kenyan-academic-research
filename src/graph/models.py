"""Pydantic data models for the core Knowledge Graph entities and relationships."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PaperNode(BaseModel):
    """Paper entity node model."""
    id: str = Field(..., description="OpenAlex Work ID without URL prefix, e.g. W2112776483")
    title: str = Field(..., description="Full title of the paper")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    publication_year: int = Field(..., description="Year of publication")
    publication_date: Optional[str] = Field(None, description="ISO publication date YYYY-MM-DD")
    work_type: Optional[str] = Field(None, description="Work type, e.g. article, review")
    source_name: Optional[str] = Field(None, description="Journal / venue display name")
    is_oa: bool = Field(False, description="Open Access boolean flag")
    oa_status: Optional[str] = Field(None, description="OA status: closed, green, gold, hybrid, bronze, diamond")
    abstract: Optional[str] = Field(None, description="Reconstructed abstract text (or None)")
    search_text: str = Field(..., description="Composite text: Title + Topic labels + Abstract")
    cited_by_count: int = Field(0, description="Total citation count in OpenAlex")
    openalex_url: str = Field(..., description="Canonical OpenAlex URI")


class ResearcherNode(BaseModel):
    """Researcher (Author) entity node model."""
    id: str = Field(..., description="OpenAlex Author ID, e.g. A5044124578, or synthetic fallback")
    display_name: str = Field(..., description="Canonical author display name")
    orcid: Optional[str] = Field(None, description="ORCID identifier if available")
    openalex_url: Optional[str] = Field(None, description="Canonical OpenAlex Author URI")


class InstitutionNode(BaseModel):
    """Academic / Research Institution entity node model."""
    id: str = Field(..., description="OpenAlex Institution ID, e.g. I1316194761, or synthetic fallback")
    display_name: str = Field(..., description="Canonical institution display name")
    ror: Optional[str] = Field(None, description="Research Organization Registry URI")
    country_code: Optional[str] = Field(None, description="Two-letter ISO country code")
    type: Optional[str] = Field(None, description="Institution type, e.g. education, facility, government")
    openalex_url: Optional[str] = Field(None, description="Canonical OpenAlex Institution URI")


class TopicNode(BaseModel):
    """Academic Topic entity node model."""
    id: str = Field(..., description="OpenAlex Topic ID, e.g. T10029")
    display_name: str = Field(..., description="Topic label")
    subfield: Optional[str] = Field(None, description="Academic subfield name")
    field: Optional[str] = Field(None, description="Academic field name")
    domain: Optional[str] = Field(None, description="Academic domain name")
    openalex_url: Optional[str] = Field(None, description="Canonical OpenAlex Topic URI")


class YearNode(BaseModel):
    """Temporal Year entity node model."""
    value: int = Field(..., description="Gregorian publication year, e.g. 2024")


class GraphRelationship(BaseModel):
    """Canonical edge model connecting two nodes."""
    source_type: str = Field(..., description="Source node label, e.g. Researcher, Paper")
    source_id: str = Field(..., description="Source node unique ID")
    target_type: str = Field(..., description="Target node label, e.g. Paper, Topic, Institution, Year")
    target_id: str = Field(..., description="Target node unique ID")
    rel_type: str = Field(..., description="Relationship label: AUTHORED, ABOUT, AFFILIATED_WITH, PUBLISHED_IN")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Edge properties if any")
