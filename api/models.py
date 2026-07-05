"""
SEM-Swarm — Pydantic Models (Request/Response Schemas)
Defines the data contracts for the Epistemic Memory API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────

class ObservationStatus(str, Enum):
    """Status lifecycle of an observation in the epistemic pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    REJECTED = "rejected"
    VERIFIED = "verified"


# ── Observation Schemas ──────────────────────────────────────

class ObservationCreate(BaseModel):
    """Payload sent by Scout agents to deposit a raw observation."""
    raw_content: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Raw text content of the observation",
    )
    source_agent: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Identifier of the agent that produced this observation",
        examples=["scout-01", "scout-web-scraper"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (e.g., source URL, context tags)",
    )


class ObservationResponse(BaseModel):
    """Response after creating an observation."""
    id: int
    created_at: datetime
    source_agent: str
    status: ObservationStatus
    raw_content_preview: str = Field(
        ...,
        description="First 200 chars of raw_content for confirmation",
    )

    model_config = {"from_attributes": True}


class ObservationPending(BaseModel):
    """A pending observation ready for the Filter agent to process."""
    id: int
    created_at: datetime
    source_agent: str
    raw_content: str
    metadata: dict[str, Any]

    model_config = {"from_attributes": True}


# ── Fact Verification Schemas ────────────────────────────────

class ObservationReject(BaseModel):
    """Payload sent by Filter agents to reject a low-quality observation."""
    observation_id: int = Field(
        ...,
        description="ID of the raw observation being rejected",
    )
    reason: str = Field(
        ...,
        description="Reason for rejection provided by the Filter agent",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the rejection",
    )


class FactVerify(BaseModel):
    """Payload sent by Filter agents to verify an observation and promote it to a fact."""
    observation_id: int = Field(
        ...,
        description="ID of the raw observation being verified",
    )
    fact_text: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Cleaned, validated fact extracted from the observation",
    )
    embedding: list[float] = Field(
        ...,
        description="Semantic embedding vector (must match EMBEDDING_DIM)",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score assigned by the Filter agent (0.0 to 1.0)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the verification process",
    )


class FactResponse(BaseModel):
    """Response after creating a verified fact."""
    id: int
    created_at: datetime
    fact_text_preview: str
    confidence_score: float
    source_observation_id: int | None

    model_config = {"from_attributes": True}


# ── Semantic Search Schemas ──────────────────────────────────

class SemanticSearchRequest(BaseModel):
    """Request for vector similarity search in the epistemic memory."""
    query_embedding: list[float] = Field(
        ...,
        description="Query embedding vector for similarity search",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top results to return",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for results",
    )


class SemanticSearchResult(BaseModel):
    """A single result from semantic search."""
    id: int
    fact_text: str
    confidence_score: float
    similarity: float = Field(
        ...,
        description="Cosine similarity to the query (higher = more similar)",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="The semantic embedding vector (returned for exact re-ranking locally)",
    )
    created_at: datetime
    metadata: dict[str, Any]


class SemanticSearchResponse(BaseModel):
    """Response containing ranked semantic search results."""
    query_dim: int
    results_count: int
    results: list[SemanticSearchResult]


# ── Swarm State Schemas ──────────────────────────────────────

class SwarmStateResponse(BaseModel):
    """Current state of the swarm coordination."""
    current_task: str
    active_agents_count: int
    total_observations: int
    total_verified_facts: int
    last_consensus_at: datetime | None
    last_dreaming_loop_at: datetime | None
    metadata: dict[str, Any]

    model_config = {"from_attributes": True}
