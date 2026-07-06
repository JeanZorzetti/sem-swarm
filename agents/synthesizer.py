"""
SEM-Swarm — Agente Sintetizador (Synthesizer/Validator)
═══════════════════════════════════════════════════════
Stub para Sprint 1. O Synthesizer consulta fatos verificados usando
busca vetorial de similaridade, cruza com conhecimentos anteriores
e gera respostas/planos de ação.

Architecture (post-paper revision):
    Local reasoning: qwen3:8b (8B) — deep logic, complex refactoring
    Vector search:   qwen3-embedding (8B, VPS) — multilingual similarity
    Extraction:      nuextract (3.8B) — deterministic output formatting

    Why qwen3:8b for the Synthesizer?
    → Parametric depth rivals prior-gen giants. Excels at fuzzy logic,
      cross-referencing, and architectural synthesis. Operates within
      the ~6-8 GB RAM budget for on-demand activation.

    Dreaming Loop (async, VPS):
    → deepseek-r1:14b (~9 GB RAM) — runs overnight via nohup/cron
      for memory consolidation, contradiction resolution, re-indexing.

Pipeline (Sprint 1):
    1. Receive query/problem
    2. Generate query embedding via qwen3-embedding (VPS)
    3. POST /memory/search → retrieve similar verified facts
    4. Compose context with retrieved facts (confidence-weighted)
    5. Generate synthesized response via qwen3:8b
    6. (Dreaming Loop) deepseek-r1:14b consolidates overnight
"""

import argparse
import asyncio
import os
import logging

from core.ollama_client import OllamaClient
from core.memory_client import MemoryClient
from core.embeddings import EmbeddingGenerator

logger = logging.getLogger("sem-swarm.synthesizer")

# Use qwen3:8b as default, but fallback to phi4-mini for local tests if qwen3 is not present.
REASONING_MODEL = os.getenv("OLLAMA_REASONING_MODEL", "phi4-mini")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))


class SynthesizerAgent:
    """
    The Synthesizer (Validator) agent — third pillar of the SEM-Swarm.
    Queries collective knowledge to answer complex problems.
    """

    def __init__(self, agent_id: str = "synthesizer-01"):
        self.agent_id = agent_id
        self.ollama = OllamaClient()
        self.memory = MemoryClient()
        self.embedder = EmbeddingGenerator(
            model=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding"),
            vps_url=os.getenv("OLLAMA_VPS_URL"),
            dimensions=EMBEDDING_DIM,
        )
        logger.info(f"SynthesizerAgent '{agent_id}' initialized with model {REASONING_MODEL}")

    async def run(self, query: str):
        logger.info(f"🔍 Recebida a query: '{query}'")

        # 1. Gerar o embedding da query (qwen3-embedding na VPS, 2048d via MRL)
        query_embedding = await self.embedder.embed(query)
        
        # 2. Consultar o banco via MemoryClient
        logger.info("🔎 Buscando fatos relevantes na Epistemic Memory...")
        search_result = await self.memory.search(
            query_embedding=query_embedding,
            top_k=5,
            min_confidence=0.5
        )
        
        facts = search_result.get("results", [])
        if not facts:
            logger.warning("⚠️ Nenhum fato relevante encontrado na base.")
            print("\n🤖 [Synthesizer]: Não encontrei nenhuma informação verificada sobre isso na minha base epistêmica.")
            return

        logger.info(f"✅ Encontrados {len(facts)} fatos na base.")
        
        # 3. Montar o contexto para o prompt
        context_blocks = []
        for i, f in enumerate(facts, 1):
            text = f.get('fact_text', '')
            conf = f.get('confidence_score', 0.0)
            context_blocks.append(f"Fato {i} (Confiança {conf:.2f}): {text}")
            
        context_str = "\n".join(context_blocks)
        
        prompt = f"""Você é o Agente Sintetizador (Synthesizer) de um enxame de inteligência artificial epistêmica.
Sua missão é responder à pergunta do usuário usando APENAS os fatos verificados listados abaixo.
Você deve relacionar os fatos e montar uma resposta coesa e direta.
Se os fatos não responderem completamente à pergunta, diga o que sabe e o que não sabe.

[FATOS VERIFICADOS]
{context_str}

[PERGUNTA DO USUÁRIO]
{query}

[RESPOSTA]
"""
        
        # 4. Gerar resposta
        logger.info(f"🧠 Raciocinando com {REASONING_MODEL}...")
        response = await self.ollama.generate(
            model=REASONING_MODEL,
            prompt=prompt,
            system="Você é o Synthesizer. Baseie sua resposta apenas nos fatos fornecidos.",
            temperature=0.3
        )
        
        print(f"\n[Synthesizer]: {response.strip()}\n")


async def main():
    parser = argparse.ArgumentParser(description="SEM-Swarm Synthesizer Agent")
    parser.add_argument("--query", type=str, required=True, help="A pergunta ou problema para sintetizar")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    agent = SynthesizerAgent()
    await agent.run(args.query)

if __name__ == "__main__":
    asyncio.run(main())
