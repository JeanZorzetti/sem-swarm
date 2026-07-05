"""
SEM-Swarm — Scout Agent Local Test
Tests the Scout agent in offline mode (no VPS dependency).
Mocks Ollama (Phi-4-Mini) and NuExtract responses to validate
the two-stage pipeline logic.

Architecture under test:
    Phi-4-Mini (reasoning) → NuExtract (JSON extraction) → Output
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, PropertyMock

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.scout import ScoutAgent


# ── Mock Data ─────────────────────────────────────────────────

MOCK_REASONING_OUTPUT = """Based on the input text, I identified 3 salient observations:

1. [fact] (relevance: 0.95) Brazil is the largest country in South America by territorial extension.
2. [fact] (relevance: 0.85) Brazil borders ten of the twelve South American countries.
3. [fact] (relevance: 0.80) The Amazon rainforest covers a large part of Brazilian territory."""

MOCK_NUEXTRACT_OUTPUT = {
    "observations": [
        {
            "content": "O Brasil e o maior pais da America do Sul em extensao territorial",
            "relevance": 0.95,
            "category": "fact",
        },
        {
            "content": "O Brasil possui fronteiras com dez dos doze paises sul-americanos",
            "relevance": 0.85,
            "category": "fact",
        },
        {
            "content": "A floresta amazonica cobre uma grande parte do territorio brasileiro",
            "relevance": 0.80,
            "category": "fact",
        },
    ],
    "input_summary": "Informacoes geograficas sobre o Brasil na America do Sul",
    "observation_count": 3,
}

# Fallback mock (when NuExtract is unavailable, Phi-4-Mini does JSON directly)
MOCK_FALLBACK_OUTPUT = MOCK_NUEXTRACT_OUTPUT.copy()

TEST_INPUT = (
    "O Brasil e o maior pais da America do Sul, com uma area de 8.5 milhoes de km2. "
    "Faz fronteira com quase todos os paises do continente, exceto Chile e Equador. "
    "A floresta amazonica, a maior floresta tropical do mundo, esta localizada em grande "
    "parte no territorio brasileiro."
)


# ── Tests ─────────────────────────────────────────────────────

async def test_two_stage_pipeline_offline():
    """Test the full two-stage pipeline: Phi-4-Mini → NuExtract → offline save."""
    scout = ScoutAgent(agent_id="test-scout")

    with patch.object(
        scout.ollama, "generate",
        new_callable=AsyncMock,
        return_value=MOCK_REASONING_OUTPUT,
    ), patch.object(
        scout.extractor, "is_healthy",
        new_callable=AsyncMock,
        return_value=True,
    ), patch.object(
        scout.extractor, "extract_with_retry",
        new_callable=AsyncMock,
        return_value=MOCK_NUEXTRACT_OUTPUT,
    ):
        result = await scout.run(raw_input=TEST_INPUT, offline=True)

    assert result["status"] == "saved_offline", f"Expected 'saved_offline', got '{result['status']}'"
    assert result["observation_count"] == 3, f"Expected 3 observations, got {result['observation_count']}"
    assert "filepath" in result, "Expected filepath in result"

    filepath = Path(result["filepath"])
    assert filepath.exists(), f"Output file does not exist: {filepath}"

    # Verify file contents include pipeline metadata
    saved_data = json.loads(filepath.read_text(encoding="utf-8"))
    assert len(saved_data["observations"]) == 3
    assert saved_data["agent_id"] == "test-scout"
    assert saved_data["pipeline"] == "phi4-mini+nuextract"
    assert saved_data["orchestrator_model"] == "phi4-mini"
    assert saved_data["extractor_model"] == "nuextract"

    # Cleanup
    filepath.unlink()
    print(f"  OK: Two-stage pipeline test passed (saved and cleaned up: {filepath.name})")
    return True


async def test_fallback_without_nuextract():
    """Test fallback to Phi-4-Mini direct JSON when NuExtract is unavailable."""
    scout = ScoutAgent(agent_id="test-scout")

    with patch.object(
        scout.extractor, "is_healthy",
        new_callable=AsyncMock,
        return_value=False,  # NuExtract NOT available
    ), patch.object(
        scout.ollama, "generate_structured",
        new_callable=AsyncMock,
        return_value=MOCK_FALLBACK_OUTPUT,
    ):
        result = await scout.run(raw_input=TEST_INPUT, offline=True)

    assert result["status"] == "saved_offline"
    assert result["observation_count"] == 3
    print("  OK: Fallback without NuExtract test passed")

    # Cleanup
    if result.get("filepath"):
        Path(result["filepath"]).unlink(missing_ok=True)

    return True


async def test_empty_input():
    """Test Scout with input that yields no observations."""
    scout = ScoutAgent(agent_id="test-scout")

    empty_response = {
        "observations": [],
        "input_summary": "Empty or meaningless input",
        "observation_count": 0,
    }

    with patch.object(
        scout.extractor, "is_healthy",
        new_callable=AsyncMock,
        return_value=True,
    ), patch.object(
        scout.ollama, "generate",
        new_callable=AsyncMock,
        return_value="No meaningful observations found in the input.",
    ), patch.object(
        scout.extractor, "extract_with_retry",
        new_callable=AsyncMock,
        return_value=empty_response,
    ):
        result = await scout.run(raw_input="asdfghjkl", offline=True)

    assert result["status"] == "empty", f"Expected 'empty', got '{result['status']}'"
    print("  OK: Empty input test passed")
    return True


async def test_memory_fallback():
    """Test Scout falls back to offline when Memory API is unreachable."""
    scout = ScoutAgent(agent_id="test-scout")

    with patch.object(
        scout.extractor, "is_healthy",
        new_callable=AsyncMock,
        return_value=True,
    ), patch.object(
        scout.ollama, "generate",
        new_callable=AsyncMock,
        return_value=MOCK_REASONING_OUTPUT,
    ), patch.object(
        scout.extractor, "extract_with_retry",
        new_callable=AsyncMock,
        return_value=MOCK_NUEXTRACT_OUTPUT,
    ), patch.object(
        scout.memory, "is_healthy",
        new_callable=AsyncMock,
        return_value=False,
    ):
        result = await scout.run(raw_input=TEST_INPUT, offline=False)

    assert result["status"] == "fallback_offline", f"Expected 'fallback_offline', got '{result['status']}'"
    assert result["observation_count"] == 3

    if result.get("filepath"):
        Path(result["filepath"]).unlink(missing_ok=True)

    print("  OK: Memory fallback test passed")
    return True


async def test_readiness_check():
    """Test the readiness check with different component states."""
    scout = ScoutAgent(agent_id="test-scout")

    with patch.object(
        scout.ollama, "is_healthy", new_callable=AsyncMock, return_value=True,
    ), patch.object(
        scout.extractor, "is_healthy", new_callable=AsyncMock, return_value=True,
    ), patch.object(
        scout.memory, "is_healthy", new_callable=AsyncMock, return_value=False,
    ):
        status = await scout.check_readiness()

    assert status["ollama"] is True
    assert status["extractor_available"] is True
    assert status["memory_api"] is False
    assert status["ready"] is True  # Only Ollama is strictly required
    print("  OK: Readiness check test passed")
    return True


# ── Runner ────────────────────────────────────────────────────

async def run_all_tests():
    """Run all Scout tests."""
    print("\nSEM-Swarm Scout Agent - Local Tests (Post-Paper Architecture)")
    print("Pipeline: Phi-4-Mini (reasoning) -> NuExtract (JSON extraction)")
    print("=" * 60)

    tests = [
        ("Two-stage pipeline (Phi-4-Mini + NuExtract)", test_two_stage_pipeline_offline),
        ("Fallback without NuExtract", test_fallback_without_nuextract),
        ("Empty input", test_empty_input),
        ("Memory API fallback", test_memory_fallback),
        ("Readiness check", test_readiness_check),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"\nTest: {name}")
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
