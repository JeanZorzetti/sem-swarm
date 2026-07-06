"""
SEM-Swarm — Pipeline Benchmark (Sprint 3)
═════════════════════════════════════════
Mede o pipeline scout→filter de ponta a ponta contra uma API real:

  - latência por estágio (scout, filter)
  - desfecho das observações (verified / rejected / pendente)
  - fidelidade numérica: os números do input sobrevivem até o fato?
    (métrica do gap 5 — nuextract perde/altera trechos na extração)

Usage:
    python -m tests.benchmark_pipeline [--api-url URL] [--inputs N]

O alvo default é http://localhost:8000 (stack local). Os dados criados são
aditivos (observações/fatos de benchmark ficam no banco alvo).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.filter import EMBEDDING_DIM, process_observation
from agents.scout import ScoutAgent
from core.embeddings import EmbeddingGenerator
from core.extractor import NuExtractClient
from core.memory_client import MemoryClient
from core.ollama_client import OllamaClient

logger = logging.getLogger("sem-swarm.benchmark")

# Frases factuais com números — o benchmark verifica se os números sobrevivem.
DATASET = [
    "O porcelanato retificado de 90x90 cm pesa cerca de 21 kg por caixa com 3 pecas.",
    "O rejunte epoxi cura em 24 horas e atinge resistencia total em 7 dias.",
    "A argamassa ACIII suporta temperaturas de ate 90 graus e e indicada para fachadas.",
    "Uma caixa de porcelanato 60x60 cobre 1,44 metros quadrados com 4 pecas.",
    "O piso vinilico tem espessura entre 2 e 5 milimetros e instalacao 3 vezes mais rapida.",
]

NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_of(text: str) -> set[str]:
    """Extract numbers normalized (decimal comma → dot)."""
    return {n.replace(",", ".") for n in NUM_RE.findall(text)}


async def bench_one(raw_input: str, api_url: str, scout, ollama, extractor, memory, embedder) -> dict:
    input_numbers = numbers_of(raw_input)

    t0 = time.monotonic()
    scout_result = await scout.run(raw_input)
    scout_s = time.monotonic() - t0

    if scout_result.get("status") != "deposited":
        return {
            "input": raw_input,
            "scout_seconds": round(scout_s, 1),
            "status": scout_result.get("status"),
            "error": "scout não depositou",
        }

    t1 = time.monotonic()
    pending = await memory.get_pending(limit=20)
    for obs in pending:
        await process_observation(obs, ollama, extractor, memory, embedder)
    filter_s = time.monotonic() - t1

    # Desfechos e fidelidade: o que virou fato (novo ou corroborado)?
    obs_ids = {o["id"] for o in pending}
    async_client_facts = await memory_list(api_url, "/memory/facts?limit=100&include_inactive=true")
    observations = await memory_list(api_url, "/memory/observations?limit=50")

    outcomes = {"verified": 0, "rejected": 0, "other": 0}
    fact_texts: list[str] = []
    for o in observations:
        if o["id"] not in obs_ids:
            continue
        outcomes[o["status"]] = outcomes.get(o["status"], 0) + 1
        corroborated = (o.get("metadata") or {}).get("corroborated_fact_id")
        if corroborated:
            fact_texts += [f["fact_text"] for f in async_client_facts if f["id"] == corroborated]
    fact_texts += [
        f["fact_text"] for f in async_client_facts
        if f.get("source_observation_id") in obs_ids
    ]

    surviving = numbers_of(" ".join(fact_texts))
    fidelity = (
        len(input_numbers & surviving) / len(input_numbers) if input_numbers else 1.0
    )

    return {
        "input": raw_input,
        "scout_seconds": round(scout_s, 1),
        "filter_seconds": round(filter_s, 1),
        "observations": len(pending),
        "outcomes": outcomes,
        "input_numbers": sorted(input_numbers),
        "surviving_numbers": sorted(input_numbers & surviving),
        "numeric_fidelity": round(fidelity, 2),
    }


async def memory_list(api_url: str, path: str) -> list[dict]:
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{api_url.rstrip('/')}{path}")
        resp.raise_for_status()
        return resp.json()


async def main():
    parser = argparse.ArgumentParser(description="SEM-Swarm Pipeline Benchmark")
    parser.add_argument("--api-url", default=os.getenv("SEM_API_URL", "http://localhost:8000"))
    parser.add_argument("--inputs", type=int, default=len(DATASET), help="Quantos inputs do dataset rodar")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    scout = ScoutAgent(agent_id="scout-bench", api_url=args.api_url)
    ollama = OllamaClient()
    extractor = NuExtractClient()
    memory = MemoryClient(api_url=args.api_url)
    embedder = EmbeddingGenerator(
        model=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding"),
        vps_url=os.getenv("OLLAMA_VPS_URL"),
        dimensions=EMBEDDING_DIM,
    )

    if not await memory.is_healthy():
        print(f"API {args.api_url} inacessível.")
        sys.exit(1)
    await memory.heartbeat("benchmark-01", "benchmark")

    results = []
    for i, raw in enumerate(DATASET[: args.inputs], 1):
        print(f"[{i}/{args.inputs}] {raw[:60]}...")
        r = await bench_one(raw, args.api_url, scout, ollama, extractor, memory, embedder)
        results.append(r)
        print(f"    scout {r.get('scout_seconds')}s · filter {r.get('filter_seconds', '-')}s · "
              f"fidelidade numérica {r.get('numeric_fidelity', '-')}")

    ok = [r for r in results if "numeric_fidelity" in r]
    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "inputs": len(results),
        "avg_scout_seconds": round(sum(r["scout_seconds"] for r in ok) / len(ok), 1) if ok else None,
        "avg_filter_seconds": round(sum(r["filter_seconds"] for r in ok) / len(ok), 1) if ok else None,
        "avg_numeric_fidelity": round(sum(r["numeric_fidelity"] for r in ok) / len(ok), 2) if ok else None,
        "results": results,
    }

    out_dir = Path(__file__).parent / "benchmark-results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n== Resumo ==")
    print(f"scout médio:  {summary['avg_scout_seconds']}s")
    print(f"filter médio: {summary['avg_filter_seconds']}s")
    print(f"fidelidade numérica média: {summary['avg_numeric_fidelity']}")
    print(f"relatório: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
