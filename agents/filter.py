"""
SEM-Swarm — Agente Filter (Filtro Epistêmico)
══════════════════════════════════════════════
Busca observações pendentes no banco de dados e as julga.
Rejeita ruído/alucinação e promove observações válidas a
Fatos Epistêmicos (com embeddings).

Usage:
    python -m agents.filter
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from core.ollama_client import OllamaClient
from core.extractor import NuExtractClient
from core.memory_client import MemoryClient
from core.embeddings import EmbeddingGenerator
from core.vector_ops import batch_cosine

logger = logging.getLogger("sem-swarm.filter")

# ── Configuration ─────────────────────────────────────────────

# O modelo roteador/avaliador local
REASONING_MODEL = "phi4-mini:latest"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))


def _parse_verdict(data: dict) -> tuple[bool, str, float, str]:
    """Normalize the extracted judgment fields (NuExtract or fallback)."""
    is_valid = data.get("is_valid", False)
    if isinstance(is_valid, str):
        is_valid = is_valid.lower() in ["true", "1", "yes", "sim", "válida", "valida"]
    is_valid = bool(is_valid)
    reasoning = str(data.get("reasoning_summary", "") or "").strip()
    conf_raw = data.get("confidence_score", 0.0)
    try:
        confidence = float(conf_raw) if str(conf_raw).strip() else 0.0
    except (ValueError, TypeError):
        confidence = 0.0
    fact = str(data.get("clean_fact", "") or "").strip()
    return is_valid, reasoning, confidence, fact


def _inconclusive(is_valid: bool, reasoning: str, fact: str) -> bool:
    """
    Um veredito precisa se sustentar: rejeição sem justificativa não é
    rejeição (flub do extrator já rejeitou fato verdadeiro com motivo em
    branco), e aprovação sem fato limpo não tem o que promover.
    """
    return (not is_valid and not reasoning) or (is_valid and not fact)


async def process_observation(
    obs: dict,
    ollama: OllamaClient,
    extractor: NuExtractClient,
    memory: MemoryClient,
    embedder: EmbeddingGenerator,
):
    """
    Avalia uma observação bruta e decide se a aprova ou rejeita.
    """
    raw_content = obs["raw_content"]
    obs_id = obs["id"]
    logger.info(f"🔍 Avaliando observação #{obs_id} ({len(raw_content)} chars)...")

    # 1. Raciocínio (Phi-4-Mini)
    prompt = f"""
Você é o Agente Filtro do SEM-Swarm. Sua função é avaliar observações do ambiente e decidir se elas devem entrar na Memória Epistêmica Permanente.

Regras de aprovação:
- A observação deve conter um fato claro, objetivo e útil.
- Fofocas, opiniões subjetivas ou informações pela metade devem ser REJEITADAS.

Observação recebida:
"{raw_content}"

Responda com o seu raciocínio detalhado. Avalie a objetividade e utilidade. Ao final, decida se é Válida ou Inválida. Se for válida, formule o fato limpo e conciso.
"""
    logger.info(f"🧠 Stage 1: Raciocínio via {REASONING_MODEL}...")
    reasoning_text = await ollama.generate(
        prompt=prompt,
        model=REASONING_MODEL,
        temperature=0.3,
    )
    logger.info(f"✅ Stage 1 concluído: {len(reasoning_text)} chars de raciocínio.")

    # 2. Extração Determinística (NuExtract)
    template = {
        "is_valid": False,
        "reasoning_summary": "",
        "clean_fact": "",
        "confidence_score": 0.0
    }

    logger.info("🔧 Stage 2: Extraindo decisão via NuExtract...")
    extraction_text = f"Observação Original: {raw_content}\nRaciocínio: {reasoning_text}"
    extracted_json = await extractor.extract_with_retry(
        text=extraction_text,
        schema=template,
        max_retries=2,
    )

    is_valid, reasoning, confidence, fact = _parse_verdict(extracted_json)

    if _inconclusive(is_valid, reasoning, fact):
        # NuExtract determinístico flubba SEMPRE no mesmo input (~1/3 das
        # obs de catálogo) — retry não resolve. Fallback: decisão via JSON
        # estruturado do modelo de raciocínio (mesmo padrão do Scout).
        logger.warning(
            f"⚠️ Extração inconclusiva p/ obs #{obs_id} — fallback JSON via {REASONING_MODEL}"
        )
        extracted_json = await ollama.generate_structured(
            model=REASONING_MODEL,
            prompt=(
                "Extraia a decisão final do julgamento abaixo.\n\n"
                f"{extraction_text}"
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "is_valid": {"type": "boolean"},
                    "reasoning_summary": {"type": "string"},
                    "clean_fact": {"type": "string"},
                    "confidence_score": {"type": "number"},
                },
                "required": ["is_valid", "reasoning_summary", "clean_fact", "confidence_score"],
            },
            temperature=0.1,
        )
        is_valid, reasoning, confidence, fact = _parse_verdict(extracted_json)
        if _inconclusive(is_valid, reasoning, fact):
            raise RuntimeError("extração inconclusiva mesmo após fallback estruturado")

    # 3. Decisão e Deduplicação Semântica (Rust Accelerated)
    if is_valid and fact:
        logger.info(f"🌟 Observação #{obs_id} APROVADA! (confiança: {confidence})")
        logger.info(f"   Fato: {fact}")
        
        # Embedding real (qwen3-embedding na VPS, truncado a 2048 via MRL)
        try:
            embedding = await embedder.embed(fact)
        except ConnectionError as e:
            logger.error(f"❌ Embedding indisponível ({e}). Observação #{obs_id} fica pendente para retry.")
            return

        # Deduplicação Exata local via Rust (sem-vector)
        logger.info("🔍 Buscando top candidatos no banco para deduplicação exata...")
        search_result = await memory.search(embedding, top_k=5)
        
        duplicate_fact_id = None
        duplicate_sim = 0.0
        related_fact_id = None

        if search_result.get("results"):
            # Extrai os embeddings retornados pela busca para verificação exata
            candidates = search_result["results"]
            targets = [c.get("embedding", []) for c in candidates if c.get("embedding")]

            if targets:
                logger.info(f"⚡ Calculando similaridade exata (SIMD/Rust) contra {len(targets)} fatos...")
                exact_scores = batch_cosine(embedding, targets)

                # O batch_cosine retorna list[tuple[int, float]] => [(idx_targets, similarity), ...]
                for idx, sim in exact_scores:
                    target_fact_id = candidates[idx]["id"]

                    if sim >= 0.95:
                        logger.info(f"🤝 Fato equivalente já existe (sim {sim:.4f} com #{target_fact_id}) — corroborando (consenso).")
                        duplicate_fact_id = target_fact_id
                        duplicate_sim = sim
                        break
                    elif sim >= 0.70 and not related_fact_id:
                        logger.info(f"🔗 Tópico Relacionado identificado: Similaridade {sim:.4f} com Fato #{target_fact_id}")
                        related_fact_id = target_fact_id

        if duplicate_fact_id:
            # Consenso: evidência independente reforça o fato em vez de virar lixo
            await memory.corroborate(
                observation_id=obs_id,
                fact_id=duplicate_fact_id,
                similarity=duplicate_sim,
                confidence_score=confidence,
                metadata={"filter_model": REASONING_MODEL},
            )
        else:
            metadata = {"reasoning": reasoning, "filter_model": REASONING_MODEL}
            if related_fact_id:
                metadata["related_fact_id"] = related_fact_id

            await memory.verify(
                observation_id=obs_id,
                fact_text=fact,
                embedding=embedding,
                confidence_score=confidence,
                metadata=metadata
            )
    else:
        logger.info(f"🗑️ Observação #{obs_id} REJEITADA. Motivo: {reasoning}")
        await memory.reject(
            observation_id=obs_id,
            reason=reasoning,
            metadata={"filter_model": REASONING_MODEL}
        )


async def reprocess_stuck(memory: MemoryClient, ollama, extractor, embedder, limit: int):
    """
    Recover observations stranded in 'processing' (a filter crash mid-batch
    leaves them there; /memory/pending never returns them again).
    /memory/verify and /memory/reject accept status='processing', so we can
    judge them directly from the read-only listing.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{memory.api_url}/memory/observations", params={"limit": 200}
        )
        resp.raise_for_status()
        stuck = [o for o in resp.json() if o["status"] == "processing"][:limit]

    if not stuck:
        logger.info("📭 Nenhuma observação presa em 'processing'.")
        return

    logger.info(f"♻️ Reprocessando {len(stuck)} observações presas em 'processing'...")
    for obs in stuck:
        try:
            await process_observation(obs, ollama, extractor, memory, embedder)
        except Exception as e:
            logger.error(f"❌ Falha na observação #{obs['id']} (seguindo): {e}")


async def main():
    parser = argparse.ArgumentParser(description="SEM-Swarm Filter Agent")
    parser.add_argument("--limit", type=int, default=10, help="Max observations to process")
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Daemon mode: poll continuously every N seconds (0 = run once)",
    )
    parser.add_argument(
        "--reprocess-stuck",
        action="store_true",
        help="Recover observations stranded in 'processing' before normal polling",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silenciar httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ollama = OllamaClient()
    extractor = NuExtractClient()
    memory = MemoryClient()
    embedder = EmbeddingGenerator(
        model=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding"),
        vps_url=os.getenv("OLLAMA_VPS_URL"),
        dimensions=EMBEDDING_DIM,
    )

    # Check dependencies
    if not await memory.is_healthy():
        logger.error("❌ Memory API is unreachable. Start it with: docker compose up -d api")
        sys.exit(1)

    agent_id = "filter-01"

    if args.reprocess_stuck:
        await memory.heartbeat(agent_id, "filter")
        await reprocess_stuck(memory, ollama, extractor, embedder, args.limit)
        if not args.loop:
            return

    while True:
        try:
            await memory.heartbeat(agent_id, "filter")

            logger.info(f"📡 Buscando até {args.limit} observações pendentes...")
            pending = await memory.get_pending(limit=args.limit)

            if pending:
                logger.info(f"📥 Encontradas {len(pending)} observações para julgar.")
                for obs in pending:
                    print("\n" + "="*60)
                    try:
                        await process_observation(obs, ollama, extractor, memory, embedder)
                    except Exception as e:
                        # One bad observation must not strand the rest of the
                        # fetched batch (they are already marked 'processing').
                        # The failed obs stays 'processing'; recover later with
                        # --reprocess-stuck.
                        logger.error(f"❌ Falha na observação #{obs['id']} (seguindo o lote): {e}")
                    print("="*60 + "\n")
                logger.info("✅ Processamento finalizado.")
            else:
                logger.info("📭 Nenhuma observação pendente encontrada. Dormindo...")
        except Exception as e:
            # Daemon mode survives transient API/Ollama failures
            if not args.loop:
                raise
            logger.error(f"❌ Erro no ciclo (seguindo em frente): {e}")

        if not args.loop:
            return
        await asyncio.sleep(args.loop)


if __name__ == "__main__":
    asyncio.run(main())
