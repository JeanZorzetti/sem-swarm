"""
SEM-Swarm — Memory Client
REST client for the Epistemic Memory API (VPS or local).
Used by local agents to communicate with the shared memory substrate.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("sem-swarm.memory_client")

DEFAULT_API_URL = os.getenv("SEM_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 30.0


class MemoryClient:
    """Client for the SEM-Swarm Epistemic Memory API."""

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    # ── Health ────────────────────────────────────────────────

    async def is_healthy(self) -> bool:
        """Check if the Memory API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.api_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    # ── Heartbeat (presence, best-effort) ─────────────────────

    async def heartbeat(self, agent_id: str, role: str) -> dict[str, Any]:
        """
        Announce agent presence to the swarm. Best-effort: a failed
        heartbeat never breaks the agent's actual work.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.api_url}/swarm/heartbeat",
                    json={"agent_id": agent_id, "role": role},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(f"Heartbeat failed (ignored): {e}")
            return {}

    # ── Observe (Scout → Memory) ──────────────────────────────

    async def observe(
        self,
        raw_content: str,
        source_agent: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Deposit a raw observation into the epistemic memory.

        Args:
            raw_content: Raw text observation
            source_agent: Agent identifier
            metadata: Optional metadata dict

        Returns:
            API response with observation ID and status
        """
        payload = {
            "raw_content": raw_content,
            "source_agent": source_agent,
            "metadata": metadata or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_url}/memory/observe",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()

        logger.info(f"📤 Observation #{result['id']} deposited (agent: {source_agent})")
        return result

    # ── Pending (Filter polls) ────────────────────────────────

    async def get_pending(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Fetch pending observations for processing.

        Args:
            limit: Maximum number of observations to fetch

        Returns:
            List of pending observation dicts
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.api_url}/memory/pending",
                params={"limit": limit},
            )
            resp.raise_for_status()
            return resp.json()

    # ── Verify (Filter → Memory) ──────────────────────────────

    async def verify(
        self,
        observation_id: int,
        fact_text: str,
        embedding: list[float],
        confidence_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Verify an observation and promote it to a verified fact.

        Args:
            observation_id: ID of the observation to verify
            fact_text: Cleaned fact text
            embedding: Semantic embedding vector
            confidence_score: Confidence score (0.0 to 1.0)
            metadata: Optional metadata

        Returns:
            API response with fact ID
        """
        payload = {
            "observation_id": observation_id,
            "fact_text": fact_text,
            "embedding": embedding,
            "confidence_score": confidence_score,
            "metadata": metadata or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_url}/memory/verify",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Corroborate (Filter → Memory, consensus) ──────────────

    async def corroborate(
        self,
        observation_id: int,
        fact_id: int,
        similarity: float,
        confidence_score: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Reinforce an existing fact with a new matching observation (consensus).

        Args:
            observation_id: ID of the corroborating observation
            fact_id: ID of the existing fact
            similarity: Cosine similarity between evidence and fact
            confidence_score: Confidence of the corroborating evidence
            metadata: Optional metadata

        Returns:
            API response with updated corroborations and confidence
        """
        payload = {
            "observation_id": observation_id,
            "fact_id": fact_id,
            "similarity": similarity,
            "confidence_score": confidence_score,
            "metadata": metadata or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_url}/memory/corroborate",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()

        logger.info(
            f"🤝 Observation #{observation_id} corroborates fact #{fact_id} "
            f"(corroborations={result.get('corroborations')})"
        )
        return result

    # ── Reject (Filter → Memory) ──────────────────────────────

    async def reject(
        self,
        observation_id: int,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Reject an observation as hallucination or useless noise.

        Args:
            observation_id: ID of the observation to reject
            reason: Reason for rejection
            metadata: Optional metadata

        Returns:
            API response
        """
        payload = {
            "observation_id": observation_id,
            "reason": reason,
            "metadata": metadata or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_url}/memory/reject",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Search (Synthesizer queries) ──────────────────────────

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> dict[str, Any]:
        """
        Perform semantic similarity search in the epistemic memory.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            min_confidence: Minimum confidence threshold

        Returns:
            Search results with similarities
        """
        payload = {
            "query_embedding": query_embedding,
            "top_k": top_k,
            "min_confidence": min_confidence,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_url}/memory/search",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Swarm State ───────────────────────────────────────────

    async def get_swarm_state(self) -> dict[str, Any]:
        """Get the current swarm coordination state."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.api_url}/swarm/state")
            resp.raise_for_status()
            return resp.json()
