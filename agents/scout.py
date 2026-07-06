"""
SEM-Swarm — Agente Scout (Batedor)
═══════════════════════════════════
O primeiro agente do enxame. Recebe input bruto (textos, logs, perguntas)
e extrai observações salientes em formato JSON determinístico.

Architecture (post-paper revision):
    ┌──────────┐      ┌───────────┐      ┌───────────┐      ┌──────────┐
    │ Raw Text │ ──►  │ Phi-4-Mini│ ──►  │ NuExtract │ ──►  │ Memory   │
    │ (input)  │      │ (reason)  │      │ (extract) │      │ API      │
    └──────────┘      └───────────┘      └───────────┘      └──────────┘

    - Phi-4-Mini (3.8B): Orchestrator — reasons about the text, identifies
      salient observations. Strong tool calling, 128k context, PT-BR fluent.
    - NuExtract (3.8B): Extractor — converts reasoning output into
      deterministic, schema-compliant JSON. 28-33% → near-100% accuracy.

    Why NOT qwen2.5-coder:1.5b for this?
    → "Attention dilution" causes malformed JSON systematically.
       See paper §"O Colapso Epistêmico em Tool Calling e Saídas Estruturadas"

Usage:
    python -m agents.scout --input "Texto para analisar"
    python -m agents.scout --file documento.txt
    python -m agents.scout --input "Texto" --offline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from core.ollama_client import OllamaClient
from core.extractor import NuExtractClient
from core.memory_client import MemoryClient

logger = logging.getLogger("sem-swarm.scout")

# ── Configuration ─────────────────────────────────────────────

AGENT_ID = "scout-01"

# Phi-4-Mini: orchestrator brain — reasoning + observation identification
# Chosen over qwen2.5-coder:1.5b due to attention dilution on structured output
DEFAULT_ORCHESTRATOR_MODEL = "phi4-mini"

# NuExtract: deterministic JSON extraction — replaces generalist JSON output
# Generalist SLMs <10B achieve only 28-33% exact-match on JSON extraction
DEFAULT_EXTRACTOR_MODEL = "nuextract"

OFFLINE_OUTPUT_DIR = PROJECT_ROOT / "data" / "offline_observations"

# ── System Prompt (Phi-4-Mini) ────────────────────────────────
# Note: This prompt is for the REASONING step only. The Phi-4-Mini
# does NOT produce the final JSON — NuExtract handles that.

SCOUT_REASONING_PROMPT = """You are the Scout Agent in a Swarm Intelligence system.
Your role is to analyze raw text input and identify salient observations.

RULES:
1. Read the input text carefully and identify ALL factual, verifiable observations.
2. Each observation must be a self-contained statement.
3. Do NOT add opinions, interpretations, or hallucinated information.
4. Do NOT repeat or rephrase the same observation.
5. For each observation, assess:
   - How informative and novel it is (relevance: 0.0 to 1.0)
   - Its category: fact, definition, relationship, metric, or event
6. If the input contains no meaningful observations, state that clearly.
7. Respond in the same language as the input text.

Provide your analysis as a structured summary listing each observation with its
relevance score and category. Be thorough but concise."""

# Schema for NuExtract (deterministic JSON extraction)
OBSERVATION_EXTRACTION_SCHEMA = {
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


# ── Scout Agent ───────────────────────────────────────────────

class ScoutAgent:
    """
    The Scout (Batedor) agent — first pillar of the SEM-Swarm.

    Uses a two-stage pipeline:
      1. Phi-4-Mini reasons about the input and identifies observations
      2. NuExtract converts the reasoning into deterministic JSON

    This architecture applies the "Cognitive Minimum Privilege" principle:
    each model does exactly what it's best at.
    """

    def __init__(
        self,
        agent_id: str = AGENT_ID,
        orchestrator_model: str = DEFAULT_ORCHESTRATOR_MODEL,
        extractor_model: str = DEFAULT_EXTRACTOR_MODEL,
        ollama_url: str = "http://localhost:11434",
        api_url: str = "http://localhost:8000",
    ):
        self.agent_id = agent_id
        self.orchestrator_model = orchestrator_model
        self.extractor_model = extractor_model
        self.ollama = OllamaClient(local_url=ollama_url)
        self.extractor = NuExtractClient(model=extractor_model, base_url=ollama_url)
        self.memory = MemoryClient(api_url=api_url)

    async def check_readiness(self) -> dict[str, bool]:
        """Check if all dependencies are available."""
        ollama_ok = await self.ollama.is_healthy()
        extractor_ok = await self.extractor.is_healthy()
        memory_ok = await self.memory.is_healthy()
        return {
            "ollama": ollama_ok,
            "extractor_available": extractor_ok,
            "memory_api": memory_ok,
            "ready": ollama_ok,  # Memory + extractor are optional (degraded mode)
        }

    async def _reason_about_input(self, raw_input: str) -> str:
        """
        Stage 1: Use Phi-4-Mini to reason about the input text
        and identify salient observations in natural language.

        This stage leverages the orchestrator's superior reasoning,
        128k context window, and PT-BR fluency.
        """
        logger.info(
            f"🧠 Stage 1: Reasoning via {self.orchestrator_model} "
            f"({len(raw_input)} chars)..."
        )

        user_prompt = (
            f"Analyze the following text and identify all salient observations.\n\n"
            f"--- INPUT TEXT ---\n{raw_input}\n--- END INPUT ---\n\n"
            f"List each observation with its relevance score (0.0-1.0) and "
            f"category (fact/definition/relationship/metric/event)."
        )

        reasoning_output = await self.ollama.generate(
            model=self.orchestrator_model,
            prompt=user_prompt,
            system=SCOUT_REASONING_PROMPT,
            temperature=0.3,
            max_tokens=2048,
        )

        logger.info(
            f"✅ Stage 1 complete: {len(reasoning_output)} chars of reasoning"
        )
        return reasoning_output

    async def _extract_structured(self, reasoning_text: str) -> dict[str, Any]:
        """
        Stage 2: Use NuExtract to convert the reasoning output
        into deterministic, schema-compliant JSON.

        NuExtract achieves near-100% structural accuracy vs 28-33%
        for generalist SLMs on pure extraction tasks.
        """
        logger.info(f"🔧 Stage 2: Extracting JSON via {self.extractor_model}...")

        result = await self.extractor.extract_with_retry(
            text=reasoning_text,
            schema=OBSERVATION_EXTRACTION_SCHEMA,
            max_retries=2,
        )

        obs_count = len(result.get("observations", []))
        logger.info(f"✅ Stage 2 complete: {obs_count} observations extracted")
        return result

    async def _extract_fallback(self, raw_input: str) -> dict[str, Any]:
        """
        Fallback: If NuExtract is unavailable, use Phi-4-Mini directly
        with Ollama's JSON format constraint.

        This is less reliable than the two-stage pipeline but still
        far better than using qwen2.5-coder:1.5b.
        """
        logger.warning(
            "⚠️ NuExtract unavailable, falling back to direct JSON via "
            f"{self.orchestrator_model} (degraded accuracy)"
        )

        user_prompt = (
            f"Analyze the following text and extract all salient observations.\n\n"
            f"--- INPUT TEXT ---\n{raw_input}\n--- END INPUT ---"
        )

        json_schema = {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "relevance": {"type": "number"},
                            "category": {
                                "type": "string",
                                "enum": ["fact", "definition", "relationship", "metric", "event"],
                            },
                        },
                        "required": ["content", "relevance", "category"],
                    },
                },
                "input_summary": {"type": "string"},
                "observation_count": {"type": "integer"},
            },
            "required": ["observations", "input_summary", "observation_count"],
        }

        return await self.ollama.generate_structured(
            model=self.orchestrator_model,
            prompt=user_prompt,
            json_schema=json_schema,
            system=SCOUT_REASONING_PROMPT,
            temperature=0.1,
        )

    async def extract_observations(self, raw_input: str) -> dict[str, Any]:
        """
        Full extraction pipeline:
          1. Phi-4-Mini reasons about the input (identifies observations)
          2. NuExtract converts reasoning to deterministic JSON
          3. If NuExtract unavailable: fallback to Phi-4-Mini JSON directly

        Args:
            raw_input: Raw text input to analyze

        Returns:
            Parsed JSON with observations array
        """
        logger.info(f"🔍 Scout processing input ({len(raw_input)} chars)...")

        # Check if NuExtract is available for the two-stage pipeline
        extractor_available = await self.extractor.is_healthy()

        if extractor_available:
            # Two-stage pipeline: Phi-4-Mini → NuExtract
            reasoning = await self._reason_about_input(raw_input)
            observations = await self._extract_structured(reasoning)
        else:
            # Fallback: Phi-4-Mini with JSON constraint
            observations = await self._extract_fallback(raw_input)

        obs_count = len(observations.get("observations", []))
        logger.info(f"📋 Scout extracted {obs_count} observations total")

        return observations

    async def deposit_to_memory(
        self,
        observations: dict[str, Any],
        raw_input: str,
    ) -> list[dict[str, Any]]:
        """
        Deposit each extracted observation into the Epistemic Memory API.

        Args:
            observations: Parsed observation data from extract_observations
            raw_input: Original raw input (stored as metadata reference)

        Returns:
            List of API responses for each deposited observation
        """
        results = []
        for obs in observations.get("observations", []):
            try:
                response = await self.memory.observe(
                    raw_content=obs["content"],
                    source_agent=self.agent_id,
                    metadata={
                        "relevance": obs.get("relevance", 0.0),
                        "category": obs.get("category", "unknown"),
                        "input_summary": observations.get("input_summary", ""),
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                        "pipeline": "phi4-mini+nuextract",
                    },
                )
                results.append(response)
                logger.info(
                    f"📤 Deposited observation #{response['id']}: "
                    f"{obs['content'][:80]}..."
                )
            except Exception as e:
                logger.error(f"❌ Failed to deposit observation: {e}")
                results.append({"error": str(e), "content": obs["content"][:100]})

        return results

    def save_offline(self, observations: dict[str, Any], raw_input: str) -> Path:
        """
        Save observations to a local JSON file (offline mode).

        Args:
            observations: Parsed observation data
            raw_input: Original raw input

        Returns:
            Path to the saved file
        """
        OFFLINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"scout_{timestamp}.json"
        filepath = OFFLINE_OUTPUT_DIR / filename

        output = {
            "agent_id": self.agent_id,
            "orchestrator_model": self.orchestrator_model,
            "extractor_model": self.extractor_model,
            "pipeline": "phi4-mini+nuextract",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_input_length": len(raw_input),
            "raw_input_preview": raw_input[:500],
            **observations,
        }

        filepath.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"💾 Saved offline: {filepath}")
        return filepath

    async def run(
        self,
        raw_input: str,
        offline: bool = False,
    ) -> dict[str, Any]:
        """
        Full Scout pipeline: extract → deposit (or save offline).

        Args:
            raw_input: Text input to process
            offline: If True, save to file instead of sending to API

        Returns:
            Result summary dict
        """
        # Step 1: Extract observations via two-stage pipeline
        observations = await self.extract_observations(raw_input)

        if not observations.get("observations"):
            logger.warning("⚠️ No observations extracted from input")
            return {"status": "empty", "observations": []}

        # Step 2: Deposit or save
        if offline:
            filepath = self.save_offline(observations, raw_input)
            return {
                "status": "saved_offline",
                "filepath": str(filepath),
                "observation_count": len(observations["observations"]),
                "observations": observations,
            }
        else:
            # Check memory API connectivity
            if not await self.memory.is_healthy():
                logger.warning(
                    "⚠️ Memory API unreachable, falling back to offline mode"
                )
                filepath = self.save_offline(observations, raw_input)
                return {
                    "status": "fallback_offline",
                    "filepath": str(filepath),
                    "observation_count": len(observations["observations"]),
                    "observations": observations,
                }

            await self.memory.heartbeat(self.agent_id, "scout")

            deposit_results = await self.deposit_to_memory(observations, raw_input)
            return {
                "status": "deposited",
                "observation_count": len(observations["observations"]),
                "deposit_results": deposit_results,
                "observations": observations,
            }


# ── CLI Entry Point ───────────────────────────────────────────

async def main():
    """CLI entry point for the Scout agent."""
    parser = argparse.ArgumentParser(
        description="SEM-Swarm Scout Agent — Extract observations from text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Architecture:\n"
            "  Phi-4-Mini (3.8B) → reasoning/identification\n"
            "  NuExtract (3.8B)  → deterministic JSON extraction\n"
            "\n"
            "Examples:\n"
            '  python -m agents.scout -i "O Brasil e o maior pais da America do Sul"\n'
            "  python -m agents.scout -f documento.txt --offline\n"
            "  python -m agents.scout -i \"Test text\" --orchestrator qwen3:8b\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Direct text input to analyze",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a text file to analyze",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Save to local file instead of sending to Memory API",
    )
    parser.add_argument(
        "--orchestrator",
        type=str,
        default=DEFAULT_ORCHESTRATOR_MODEL,
        help=f"Orchestrator model for reasoning (default: {DEFAULT_ORCHESTRATOR_MODEL})",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=DEFAULT_EXTRACTOR_MODEL,
        help=f"Extractor model for JSON output (default: {DEFAULT_EXTRACTOR_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("SEM_API_URL", "http://localhost:8000"),
        help="Memory API URL (env: SEM_API_URL)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Get input text
    if args.input:
        raw_input = args.input
    elif args.file:
        raw_input = Path(args.file).read_text(encoding="utf-8")
    else:
        print("Error: Please provide --input or --file")
        print("\nExample:")
        print(
            '  python -m agents.scout --input '
            '"O Brasil e o maior pais da America do Sul"'
        )
        print("  python -m agents.scout --file documento.txt --offline")
        sys.exit(1)

    # Create and run scout
    scout = ScoutAgent(
        orchestrator_model=args.orchestrator,
        extractor_model=args.extractor,
        ollama_url=args.ollama_url,
        api_url=args.api_url,
    )

    # Check readiness
    status = await scout.check_readiness()
    if not status["ollama"]:
        print("Error: Ollama is not running. Start it with: ollama serve")
        print(f"  Tried to connect to: {args.ollama_url}")
        sys.exit(1)

    if not status["extractor_available"]:
        print(
            f"Warning: NuExtract ({args.extractor}) not found. "
            f"Using {args.orchestrator} directly (degraded JSON accuracy)."
        )
        print(f"  Install with: ollama pull {args.extractor}")

    if not status["memory_api"] and not args.offline:
        print("Warning: Memory API unreachable. Running in offline mode.")
        args.offline = True

    # Run the pipeline
    result = await scout.run(raw_input, offline=args.offline)

    # Output results
    print("\n" + "=" * 60)
    print("SEM-Swarm Scout - Results")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Observations: {result.get('observation_count', 0)}")
    print(f"Pipeline: {args.orchestrator} + {args.extractor}")

    if result.get("filepath"):
        print(f"Saved to: {result['filepath']}")

    if result.get("observations", {}).get("observations"):
        print("\nExtracted Observations:")
        for i, obs in enumerate(result["observations"]["observations"], 1):
            cat = obs.get("category", "?")
            rel = obs.get("relevance", 0)
            print(f"  {i}. [{cat}] (relevance: {rel:.2f})")
            print(f"     {obs['content']}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
