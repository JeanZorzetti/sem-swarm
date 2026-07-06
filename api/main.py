"""
SEM-Swarm — Epistemic Memory Substrate API
FastAPI application exposing REST endpoints for the Shared Epistemic Memory.

Endpoints:
    POST /memory/observe   — Scout deposits raw observation
    GET  /memory/pending   — Filter fetches pending observations
    POST /memory/verify    — Filter verifies and promotes to fact
    POST /memory/search    — Synthesizer performs semantic search
    GET  /swarm/state      — Query swarm coordination state
    GET  /health           — Health check
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import engine, get_db
from models import (
    FactResponse,
    FactVerify,
    ObservationCreate,
    ObservationPending,
    ObservationReject,
    ObservationResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResult,
    SwarmStateResponse,
)

logger = logging.getLogger("sem-swarm.api")


# ── Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: verify database connection on startup."""
    logger.info("🧠 SEM-Swarm API starting up...")
    # Verify database connectivity
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("✅ Database connection verified")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise

    yield

    # Shutdown
    await engine.dispose()
    logger.info("🛑 SEM-Swarm API shut down")


# ── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="SEM-Swarm Epistemic Memory API",
    description=(
        "Substrate API for the Shared Epistemic Memory — "
        "the stigmergic communication layer of the SEM-Swarm system. "
        "Agents deposit observations, verify facts, and query collective knowledge."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── UI (read-only memory inspector) ──────────────────────────

@app.get("/", include_in_schema=False)
async def ui_index():
    """Serve the single-file read-only memory inspector."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# ── Health Check ─────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """Check API and database health."""
    try:
        result = await db.execute(text("SELECT 1"))
        result.fetchone()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {e}")


# ── POST /memory/observe ─────────────────────────────────────

@app.post(
    "/memory/observe",
    response_model=ObservationResponse,
    status_code=201,
    tags=["memory"],
    summary="Deposit a raw observation",
)
async def create_observation(
    payload: ObservationCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Scout agent deposits a raw observation into the epistemic memory.
    The observation starts with status='pending' and awaits processing
    by a Filter agent.
    """
    result = await db.execute(
        text("""
            INSERT INTO env_observations (source_agent, raw_content, metadata)
            VALUES (:source_agent, :raw_content, CAST(:metadata AS jsonb))
            RETURNING id, created_at, source_agent, status
        """),
        {
            "source_agent": payload.source_agent,
            "raw_content": payload.raw_content,
            "metadata": _jsonb_str(payload.metadata),
        },
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create observation")

    # Update swarm state counters
    await db.execute(
        text("UPDATE swarm_state SET total_observations = total_observations + 1 WHERE id = 1")
    )

    logger.info(f"📥 New observation #{row.id} from '{row.source_agent}'")

    return ObservationResponse(
        id=row.id,
        created_at=row.created_at,
        source_agent=row.source_agent,
        status=row.status,
        raw_content_preview=payload.raw_content[:200],
    )


# ── GET /memory/pending ──────────────────────────────────────

@app.get(
    "/memory/pending",
    response_model=list[ObservationPending],
    tags=["memory"],
    summary="Fetch pending observations",
)
async def get_pending_observations(
    limit: int = Query(default=10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Filter agent polls for pending observations ready for verification.
    Returns observations in FIFO order and atomically marks them as 'processing'
    to prevent double-processing by concurrent Filter agents.
    """
    # Atomic: SELECT + UPDATE in one statement using CTE
    result = await db.execute(
        text("""
            WITH pending AS (
                SELECT id FROM env_observations
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT :limit
                FOR UPDATE SKIP LOCKED
            )
            UPDATE env_observations
            SET status = 'processing'
            WHERE id IN (SELECT id FROM pending)
            RETURNING id, created_at, source_agent, raw_content, metadata
        """),
        {"limit": limit},
    )
    rows = result.fetchall()

    return [
        ObservationPending(
            id=row.id,
            created_at=row.created_at,
            source_agent=row.source_agent,
            raw_content=row.raw_content,
            metadata=row.metadata if row.metadata else {},
        )
        for row in rows
    ]


# ── POST /memory/verify ──────────────────────────────────────

@app.post(
    "/memory/verify",
    response_model=FactResponse,
    status_code=201,
    tags=["memory"],
    summary="Verify an observation and promote to fact",
)
async def verify_observation(
    payload: FactVerify,
    db: AsyncSession = Depends(get_db),
):
    """
    Filter agent verifies a raw observation, cleans it, generates an embedding,
    and promotes it to a verified fact in the epistemic memory.
    """
    # Validate embedding dimension
    if len(payload.embedding) != settings.embedding_dim:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Embedding dimension mismatch: got {len(payload.embedding)}, "
                f"expected {settings.embedding_dim}"
            ),
        )

    # Verify the observation exists and is in 'processing' status
    obs_result = await db.execute(
        text("SELECT id, status FROM env_observations WHERE id = :id"),
        {"id": payload.observation_id},
    )
    obs = obs_result.fetchone()

    if not obs:
        raise HTTPException(status_code=404, detail=f"Observation #{payload.observation_id} not found")

    if obs.status not in ("processing", "pending"):
        raise HTTPException(
            status_code=409,
            detail=f"Observation #{payload.observation_id} has status '{obs.status}', expected 'processing'",
        )

    # Insert verified fact with embedding
    embedding_str = f"[{','.join(str(v) for v in payload.embedding)}]"
    result = await db.execute(
        text("""
            INSERT INTO epistemic_memory
                (fact_text, embedding, confidence_score, source_observation_id, metadata)
            VALUES
                (:fact_text, CAST(:embedding AS halfvec), :confidence, :obs_id, CAST(:metadata AS jsonb))
            RETURNING id, created_at, confidence_score, source_observation_id
        """),
        {
            "fact_text": payload.fact_text,
            "embedding": embedding_str,
            "confidence": payload.confidence_score,
            "obs_id": payload.observation_id,
            "metadata": _jsonb_str(payload.metadata),
        },
    )
    fact_row = result.fetchone()

    # Mark observation as verified
    await db.execute(
        text("UPDATE env_observations SET status = 'verified' WHERE id = :id"),
        {"id": payload.observation_id},
    )

    # Update swarm state counters
    await db.execute(
        text("UPDATE swarm_state SET total_verified_facts = total_verified_facts + 1 WHERE id = 1")
    )

    logger.info(
        f"✅ Observation #{payload.observation_id} verified → Fact #{fact_row.id} "
        f"(confidence: {fact_row.confidence_score:.2f})"
    )

    return FactResponse(
        id=fact_row.id,
        created_at=fact_row.created_at,
        fact_text_preview=payload.fact_text[:200],
        confidence_score=fact_row.confidence_score,
        source_observation_id=fact_row.source_observation_id,
    )


# ── POST /memory/reject ──────────────────────────────────────

@app.post(
    "/memory/reject",
    status_code=200,
    tags=["memory"],
    summary="Reject an observation",
)
async def reject_observation(
    payload: ObservationReject,
    db: AsyncSession = Depends(get_db),
):
    """
    Filter agent rejects a raw observation as useless or hallucinated.
    """
    # Verify the observation exists
    obs_result = await db.execute(
        text("SELECT id, status FROM env_observations WHERE id = :id FOR UPDATE"),
        {"id": payload.observation_id},
    )
    obs = obs_result.fetchone()

    if not obs:
        raise HTTPException(status_code=404, detail=f"Observation #{payload.observation_id} not found")

    if obs.status not in ("processing", "pending"):
        raise HTTPException(
            status_code=409,
            detail=f"Observation #{payload.observation_id} has status '{obs.status}'",
        )

    # Mark observation as rejected
    await db.execute(
        text("UPDATE env_observations SET status = 'rejected', metadata = jsonb_set(metadata, '{rejection_reason}', to_jsonb(CAST(:reason AS text))) WHERE id = :id"),
        {"id": payload.observation_id, "reason": payload.reason},
    )

    logger.info(f"🗑️ Observation #{payload.observation_id} rejected. Reason: {payload.reason}")
    return {"status": "rejected", "observation_id": payload.observation_id}



# ── POST /memory/search ──────────────────────────────────────

@app.post(
    "/memory/search",
    response_model=SemanticSearchResponse,
    tags=["memory"],
    summary="Semantic similarity search in epistemic memory",
)
async def semantic_search(
    payload: SemanticSearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Synthesizer agent queries the epistemic memory using vector similarity search.
    Returns the top-K most semantically similar verified facts.
    Uses cosine distance via pgvector's <=> operator.
    """
    if len(payload.query_embedding) != settings.embedding_dim:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Query embedding dimension mismatch: got {len(payload.query_embedding)}, "
                f"expected {settings.embedding_dim}"
            ),
        )

    embedding_str = f"[{','.join(str(v) for v in payload.query_embedding)}]"
    result = await db.execute(
        text("""
            SELECT
                id,
                fact_text,
                confidence_score,
                1 - (embedding <=> CAST(:query AS halfvec)) AS similarity,
                created_at,
                metadata,
                CAST(embedding AS text) AS embedding_str
            FROM epistemic_memory
            WHERE is_active = TRUE
              AND confidence_score >= :min_confidence
              AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:query AS halfvec) ASC
            LIMIT :top_k
        """),
        {
            "query": embedding_str,
            "min_confidence": payload.min_confidence,
            "top_k": payload.top_k,
        },
    )
    rows = result.fetchall()

    import json
    
    results = []
    for row in rows:
        emb_list = None
        if row.embedding_str:
            try:
                emb_list = json.loads(row.embedding_str)
            except json.JSONDecodeError:
                pass

        results.append(
            SemanticSearchResult(
                id=row.id,
                fact_text=row.fact_text,
                confidence_score=row.confidence_score,
                similarity=float(row.similarity),
                created_at=row.created_at,
                metadata=row.metadata if row.metadata else {},
                embedding=emb_list,
            )
        )

    return SemanticSearchResponse(
        query_dim=len(payload.query_embedding),
        results_count=len(results),
        results=results,
    )


# ── GET /memory/facts (read-only, UI inspector) ──────────────

@app.get(
    "/memory/facts",
    tags=["memory"],
    summary="List recent facts (read-only)",
)
async def list_facts(
    limit: int = Query(default=50, ge=1, le=200),
    include_inactive: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
):
    """Read-only listing of recent facts for inspection. No side effects."""
    where = "" if include_inactive else "WHERE is_active = TRUE"
    result = await db.execute(
        text(f"""
            SELECT id, created_at, fact_text, confidence_score, is_active,
                   superseded_by, source_observation_id, metadata
            FROM epistemic_memory
            {where}
            ORDER BY id DESC
            LIMIT :limit
        """),
        {"limit": limit},
    )
    return [
        {
            "id": r.id,
            "created_at": r.created_at,
            "fact_text": r.fact_text,
            "confidence_score": r.confidence_score,
            "is_active": r.is_active,
            "superseded_by": r.superseded_by,
            "source_observation_id": r.source_observation_id,
            "metadata": r.metadata or {},
        }
        for r in result.fetchall()
    ]


# ── GET /memory/observations (read-only, UI inspector) ───────

@app.get(
    "/memory/observations",
    tags=["memory"],
    summary="List recent observations with status (read-only)",
)
async def list_observations(
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """
    Read-only listing of recent observations for inspection.
    Unlike /memory/pending, this does NOT mark anything as 'processing'.
    """
    result = await db.execute(
        text("""
            SELECT id, created_at, source_agent, raw_content, status, metadata
            FROM env_observations
            ORDER BY id DESC
            LIMIT :limit
        """),
        {"limit": limit},
    )
    return [
        {
            "id": r.id,
            "created_at": r.created_at,
            "source_agent": r.source_agent,
            "raw_content": r.raw_content,
            "status": r.status,
            "metadata": r.metadata or {},
        }
        for r in result.fetchall()
    ]


# ── GET /swarm/state ──────────────────────────────────────────

@app.get(
    "/swarm/state",
    response_model=SwarmStateResponse,
    tags=["swarm"],
    summary="Get current swarm coordination state",
)
async def get_swarm_state(db: AsyncSession = Depends(get_db)):
    """Returns the current coordination state of the swarm."""
    result = await db.execute(text("SELECT * FROM swarm_state WHERE id = 1"))
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=500, detail="Swarm state not initialized")

    return SwarmStateResponse(
        current_task=row.current_task,
        active_agents_count=row.active_agents_count,
        total_observations=row.total_observations,
        total_verified_facts=row.total_verified_facts,
        last_consensus_at=row.last_consensus_at,
        last_dreaming_loop_at=row.last_dreaming_loop_at,
        metadata=row.metadata if row.metadata else {},
    )


# ── Helpers ───────────────────────────────────────────────────

def _jsonb_str(data: dict) -> str:
    """Convert a dict to a JSON string for PostgreSQL JSONB casting."""
    import json
    return json.dumps(data, default=str)


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.api_log_level,
        reload=True,
    )
