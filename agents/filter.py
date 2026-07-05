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
import random
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from core.ollama_client import OllamaClient
from core.extractor import NuExtractClient
from core.memory_client import MemoryClient
from core.vector_ops import batch_cosine

logger = logging.getLogger("sem-swarm.filter")

# ── Configuration ─────────────────────────────────────────────

# O modelo roteador/avaliador local
REASONING_MODEL = "phi4-mini:latest"
EMBEDDING_DIM = 2048


import hashlib

def get_mock_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Generates a deterministic mock embedding for testing based on the text."""
    # Seed the random number generator with the text hash so it's deterministic
    h = hashlib.sha256(text.encode('utf-8')).digest()
    seed = int.from_bytes(h[:4], 'little')
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


async def process_observation(
    obs: dict,
    ollama: OllamaClient,
    extractor: NuExtractClient,
    memory: MemoryClient,
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
    extracted_json = await extractor.extract(
        text=extraction_text,
        schema=template,
    )

    try:
        # extracted_json is already a dict, NuExtractClient.extract() returns a dict
        data = extracted_json
        is_valid = bool(data.get("is_valid", False))
        reasoning = str(data.get("reasoning_summary", "Sem justificativa"))
        conf_raw = data.get("confidence_score", 0.0)
        try:
            confidence = float(conf_raw) if str(conf_raw).strip() else 0.0
        except ValueError:
            confidence = 0.0
        fact = str(data.get("clean_fact", "")).strip()
        
        # Additional safety check for is_valid if NuExtract returned a string
        if isinstance(data.get("is_valid"), str):
            is_valid = data["is_valid"].lower() in ["true", "1", "yes", "sim", "válida", "valida"]
            
    except Exception as e:
        logger.error(f"❌ Falha no parse do NuExtract: {e}. Rejeitando por segurança.")
        await memory.reject(obs_id, f"Falha de extração JSON: {e}")
        return

    # 3. Decisão e Deduplicação Semântica (Rust Accelerated)
    if is_valid and fact:
        logger.info(f"🌟 Observação #{obs_id} APROVADA! (confiança: {confidence})")
        logger.info(f"   Fato: {fact}")
        
        # Gerar mock embedding (até termos a VPS com qwen3-embedding)
        embedding = get_mock_embedding(fact)

        # Deduplicação Exata local via Rust (sem-vector)
        logger.info("🔍 Buscando top candidatos no banco para deduplicação exata...")
        search_result = await memory.search(embedding, top_k=5)
        
        is_duplicate = False
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
                        logger.warning(f"⚠️ Fato Duplicado detectado! Similaridade {sim:.4f} com Fato #{target_fact_id}")
                        is_duplicate = True
                        reasoning = f"Fato rejeitado por deduplicação. Muito similar ({sim:.4f}) ao fato #{target_fact_id}."
                        break
                    elif sim >= 0.70 and not related_fact_id:
                        logger.info(f"🔗 Tópico Relacionado identificado: Similaridade {sim:.4f} com Fato #{target_fact_id}")
                        related_fact_id = target_fact_id
        
        if is_duplicate:
            await memory.reject(
                observation_id=obs_id,
                reason=reasoning,
                metadata={"filter_model": REASONING_MODEL, "dedup": True}
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


async def main():
    parser = argparse.ArgumentParser(description="SEM-Swarm Filter Agent")
    parser.add_argument("--limit", type=int, default=10, help="Max observations to process")
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

    # Check dependencies
    if not await memory.is_healthy():
        logger.error("❌ Memory API is unreachable. Start it with: docker compose up -d api")
        sys.exit(1)

    # Poll pending observations
    logger.info(f"📡 Buscando até {args.limit} observações pendentes...")
    pending = await memory.get_pending(limit=args.limit)

    if not pending:
        logger.info("📭 Nenhuma observação pendente encontrada. Dormindo...")
        return

    logger.info(f"📥 Encontradas {len(pending)} observações para julgar.")

    for obs in pending:
        print("\n" + "="*60)
        await process_observation(obs, ollama, extractor, memory)
        print("="*60 + "\n")

    logger.info("✅ Processamento finalizado.")


if __name__ == "__main__":
    asyncio.run(main())
