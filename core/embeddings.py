"""
SEM-Swarm — Embedding Generator (Dual-Endpoint)
Generates semantic embedding vectors via Ollama's embedding API.

Architecture (per paper analysis):
    - Primary: qwen3-embedding (8B) on VPS — 100+ languages, 32k context
    - Fallback: nomic-embed-text on local Ollama if VPS is unreachable

The VPS handles embeddings to free notebook RAM for interactive models
(phi4-mini, nuextract, qwen2.5-coder:1.5b).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("sem-swarm.embeddings")

DEFAULT_LOCAL_URL = "http://localhost:11434"
DEFAULT_VPS_MODEL = "qwen3-embedding"
FALLBACK_LOCAL_MODEL = "nomic-embed-text"


class EmbeddingGenerator:
    """
    Generates embeddings using Ollama's /api/embed endpoint.
    Supports dual-endpoint: VPS (primary) and local (fallback).
    """

    def __init__(
        self,
        model: str = DEFAULT_VPS_MODEL,
        local_url: str = DEFAULT_LOCAL_URL,
        vps_url: str | None = None,
        fallback_model: str = FALLBACK_LOCAL_MODEL,
    ):
        self.model = model
        self.local_url = local_url.rstrip("/")
        self.vps_url = vps_url.rstrip("/") if vps_url else None
        self.fallback_model = fallback_model

    def _get_endpoint_and_model(self) -> tuple[str, str]:
        """
        Resolve which endpoint and model to use.
        Prefers VPS for embeddings (frees local RAM).
        """
        if self.vps_url:
            return self.vps_url, self.model
        return self.local_url, self.fallback_model

    async def _try_embed(self, url: str, model: str, text: str) -> list[float] | None:
        """Attempt to generate an embedding from a specific endpoint."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{url}/api/embed",
                    json={"model": model, "input": text},
                )
                resp.raise_for_status()
                data = resp.json()

            embeddings = data.get("embeddings", [])
            if embeddings and embeddings[0]:
                return embeddings[0]
            return None
        except Exception as e:
            logger.warning(f"Embedding failed on {url} with {model}: {e}")
            return None

    async def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single text input.
        Tries VPS first, falls back to local if unavailable.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as a list of floats
        """
        # Try primary endpoint (VPS with qwen3-embedding)
        if self.vps_url:
            vector = await self._try_embed(self.vps_url, self.model, text)
            if vector:
                logger.debug(
                    f"Embedded via VPS [{self.model}] ({len(vector)} dims) "
                    f"for {len(text)} chars"
                )
                return vector
            logger.warning(
                f"VPS embedding unavailable, falling back to local [{self.fallback_model}]"
            )

        # Fallback to local endpoint
        vector = await self._try_embed(self.local_url, self.fallback_model, text)
        if vector:
            logger.debug(
                f"Embedded via local [{self.fallback_model}] ({len(vector)} dims) "
                f"for {len(text)} chars"
            )
            return vector

        raise ConnectionError(
            f"All embedding endpoints failed. "
            f"VPS: {self.vps_url or 'not configured'}, "
            f"Local: {self.local_url}"
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        Processes sequentially to avoid overloading CPU inference.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        results = []
        for i, text in enumerate(texts):
            vector = await self.embed(text)
            results.append(vector)
            if (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{len(texts)} texts")

        return results

    async def get_dimension(self) -> int:
        """Get the embedding dimension by embedding a test string."""
        test_vec = await self.embed("dimension test")
        return len(test_vec)

    async def check_health(self) -> dict[str, Any]:
        """Check which embedding endpoints are available."""
        result: dict[str, Any] = {"primary_endpoint": "local"}
        
        if self.vps_url:
            vps_vec = await self._try_embed(self.vps_url, self.model, "health check")
            result["vps"] = {
                "url": self.vps_url,
                "model": self.model,
                "healthy": vps_vec is not None,
                "dimension": len(vps_vec) if vps_vec else None,
            }
            if vps_vec:
                result["primary_endpoint"] = "vps"

        local_vec = await self._try_embed(self.local_url, self.fallback_model, "health check")
        result["local"] = {
            "url": self.local_url,
            "model": self.fallback_model,
            "healthy": local_vec is not None,
            "dimension": len(local_vec) if local_vec else None,
        }

        return result
