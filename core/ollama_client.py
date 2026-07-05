"""
SEM-Swarm — Ollama REST Client (Dual-Endpoint)
Wraps the Ollama local API for text generation and structured output.
Supports both local (notebook) and remote (VPS) Ollama instances.

Architecture:
    Local  (http://localhost:11434) → phi4-mini, nuextract, qwen2.5-coder:1.5b
    Remote (http://VPS_IP:11434)   → qwen3-embedding, deepseek-r1:14b
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger("sem-swarm.ollama")

DEFAULT_LOCAL_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120.0  # SLMs on CPU can take a while


class OllamaClient:
    """
    Client for the Ollama REST API.

    Supports dual-endpoint architecture: a local Ollama instance
    for interactive models and a remote VPS instance for heavy
    async workloads (embeddings, deep reasoning).
    """

    def __init__(
        self,
        local_url: str = DEFAULT_LOCAL_URL,
        remote_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.local_url = local_url.rstrip("/")
        self.remote_url = remote_url.rstrip("/") if remote_url else None
        self.timeout = timeout

    def _resolve_endpoint(self, model: str) -> str:
        """
        Route model to the correct Ollama endpoint based on the
        Cognitive Minimum Privilege topology.

        VPS models: qwen3-embedding, deepseek-r1:14b
        Everything else: local notebook
        """
        VPS_MODELS = {"qwen3-embedding", "deepseek-r1:14b", "deepseek-r1"}
        if self.remote_url and model in VPS_MODELS:
            return self.remote_url
        return self.local_url

    # ── Health Check ─────────────────────────────────────────

    async def is_healthy(self, endpoint: str | None = None) -> bool:
        """Check if an Ollama instance is running and responsive."""
        url = endpoint or self.local_url
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def check_all_endpoints(self) -> dict[str, bool]:
        """Check health of all configured Ollama endpoints."""
        result = {"local": await self.is_healthy(self.local_url)}
        if self.remote_url:
            result["remote"] = await self.is_healthy(self.remote_url)
        return result

    async def list_models(self, endpoint: str | None = None) -> list[str]:
        """List all locally available models on a given endpoint."""
        url = endpoint or self.local_url
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]

    # ── Text Generation ──────────────────────────────────────

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate text using Ollama's /api/generate endpoint.
        Automatically routes to the correct endpoint (local/VPS).

        Args:
            model: Model name (e.g., 'phi4-mini')
            prompt: User prompt
            system: Optional system prompt
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text string
        """
        endpoint = self._resolve_endpoint(model)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{endpoint}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        response_text = data.get("response", "")
        duration_s = data.get("total_duration", 0) / 1e9
        logger.debug(
            f"Ollama [{model}@{endpoint}] generated {len(response_text)} chars in {duration_s:.1f}s"
        )
        return response_text

    # ── Structured JSON Output ───────────────────────────────

    async def generate_structured(
        self,
        model: str,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Generate structured JSON output from the model.
        Uses Ollama's `format` parameter for constrained JSON generation.

        Args:
            model: Model name
            prompt: User prompt
            json_schema: Optional JSON schema to enforce structure
            system: Optional system prompt
            temperature: Low temperature for deterministic output
            max_tokens: Maximum tokens

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If the model output is not valid JSON
        """
        endpoint = self._resolve_endpoint(model)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": json_schema if json_schema else "json",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{endpoint}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        response_text = data.get("response", "").strip()

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Ollama [{model}] returned invalid JSON: {response_text[:500]}")
            raise ValueError(f"Model output is not valid JSON: {e}") from e

        return parsed

    # ── Chat (multi-turn) ────────────────────────────────────

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Multi-turn chat using Ollama's /api/chat endpoint.

        Args:
            model: Model name
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Assistant response text
        """
        endpoint = self._resolve_endpoint(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{endpoint}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data.get("message", {}).get("content", "")
