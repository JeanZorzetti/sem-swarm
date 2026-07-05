"""
SEM-Swarm — NuExtract Deterministic JSON Extractor
═══════════════════════════════════════════════════
Dedicated client for the NuExtract model (3.8B), a specialized LLM
fine-tuned exclusively for structured data extraction.

Why NuExtract instead of a generalist SLM?
    - Generalist models <10B achieve only 28-33% exact-match accuracy
      on pure JSON extraction tasks under positional token pressure.
    - NuExtract was microscopically tuned on massive synthetic corpora
      for *structured information extraction only*.
    - Combined with llama.cpp grammar enforcement, it delivers
      corporate-grade stability from a model weighing ~2-3 GB.

Reference: Paper §"A Engenharia de Extração Determinística e a
Supremacia do NuExtract" — Análise Arquitetural SLMs 2026.

Usage in the SEM-Swarm pipeline:
    Scout (raw text) → Phi-4-Mini (reasoning) → NuExtract (JSON) → Memory API
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger("sem-swarm.extractor")

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "nuextract"
DEFAULT_TIMEOUT = 60.0


class NuExtractClient:
    """
    Client for the NuExtract model via Ollama.
    Converts raw text into deterministic, schema-compliant JSON.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def extract(
        self,
        text: str,
        schema: dict[str, Any],
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Extract structured data from text using NuExtract.

        The model is given a JSON schema template and the input text,
        and returns a populated JSON object matching the schema.

        NuExtract uses a specific prompt format:
            <|input|>\n{schema}\n{text}\n<|output|>

        Args:
            text: Raw text to extract data from
            schema: JSON schema defining the expected output structure.
                    Pass the schema as a dict with keys and empty/default values.
                    Example: {"name": "", "age": 0, "facts": []}
            max_tokens: Maximum tokens for the output

        Returns:
            Populated JSON dict matching the schema

        Raises:
            ValueError: If the model output is not valid JSON
            httpx.HTTPStatusError: If Ollama returns an error
        """
        # NuExtract prompt format
        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
        prompt = f"<|input|>\n{schema_str}\n{text}\n<|output|>"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,  # Fully deterministic for extraction
                "num_predict": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        response_text = data.get("response", "").strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"NuExtract returned invalid JSON: {response_text[:500]}")
            raise ValueError(f"NuExtract output is not valid JSON: {e}") from e

        logger.debug(
            f"NuExtract extracted {len(result)} keys from {len(text)} chars "
            f"in {data.get('total_duration', 0) / 1e9:.1f}s"
        )
        return result

    async def extract_observations(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        Extract salient observations from text using the SEM-Swarm
        observation schema. This is the standard extraction pipeline
        used by the Scout agent.

        Args:
            text: Raw text to extract observations from

        Returns:
            Dict with observations array matching the SEM-Swarm schema
        """
        schema = {
            "observations": [
                {
                    "content": "",
                    "relevance": 0.0,
                    "category": "",
                }
            ],
            "input_summary": "",
            "observation_count": 0,
        }
        return await self.extract(text, schema)

    async def extract_with_retry(
        self,
        text: str,
        schema: dict[str, Any],
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """
        Extract with automatic retry on parse failure.
        NuExtract has very high first-pass accuracy, so retries
        are rarely needed but provide an extra safety net.

        Args:
            text: Raw text to extract from
            schema: Expected JSON schema
            max_retries: Maximum retry attempts

        Returns:
            Parsed JSON dict
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self.extract(text, schema)
            except ValueError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"NuExtract parse failed (attempt {attempt + 1}/{max_retries + 1}), retrying..."
                    )

        raise ValueError(
            f"NuExtract failed after {max_retries + 1} attempts: {last_error}"
        )

    async def is_healthy(self) -> bool:
        """Check if NuExtract model is available on the Ollama instance."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                return any(self.model in m.get("name", "") for m in models)
        except Exception:
            return False
