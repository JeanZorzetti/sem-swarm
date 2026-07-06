"""
SEM-Swarm — Ingestão de fontes reais (primeiro uso: ROI Labs Growth Partner)
════════════════════════════════════════════════════════════════════════════
Deposita conhecimento de negócio na Memória Epistêmica em três modos:

  --catalog FILE.json   Catálogo de produtos (porcelanatos.json do roilabs):
                        1 observação determinística por produto, SEM LLM
                        (dado já estruturado — scout seria retrabalho com
                        risco de typo do nuextract).
  --files PATH...       Arquivos .md/.astro/.txt: limpa markup, fatia em
                        chunks (~4000 chars) e roda o pipeline completo do
                        Scout (phi4-mini → nuextract) em cada chunk.
  --rank-csv FILE.csv   rank-tracking.csv (data,keyword,posicao,url):
                        1 observação por linha COM posição; linhas sem
                        posição são ignoradas.

As observações caem como 'pending' na API — quem promove a fato é o Filter
(`python -m agents.filter --loop 60`), como em qualquer outra entrada.

Usage (na raiz do sem-swarm, .env é carregado automaticamente):
    python -m scripts.ingest --catalog "../ROI Labs/site-goiania/porcelanatos.json"
    python -m scripts.ingest --files "../ROI Labs/Docs/Obsidian/00-tese/tese.md"
    python -m scripts.ingest --rank-csv "../ROI Labs/Docs/Obsidian/90-medicao/rank-tracking.csv"
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("sem-swarm.ingest")

CHUNK_CHARS = 4000


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from .env without overriding existing env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_dotenv(PROJECT_ROOT / ".env")

from agents.scout import ScoutAgent  # noqa: E402 (needs .env loaded first)
from core.memory_client import MemoryClient  # noqa: E402


# ── Text cleaning / chunking ──────────────────────────────────

def strip_frontmatter(src: str) -> str:
    if src.startswith("---"):
        end = src.find("\n---", 3)
        if end != -1:
            return src[end + 4:]
    return src


def astro_to_text(src: str) -> str:
    """Crude prose extraction from an .astro page. Leftover artifacts are
    tolerated: the Scout only extracts factual statements anyway."""
    src = strip_frontmatter(src)
    src = re.sub(r"<script[\s\S]*?</script>", " ", src, flags=re.I)
    src = re.sub(r"<style[\s\S]*?</style>", " ", src, flags=re.I)
    for _ in range(6):  # nested template expressions
        new = re.sub(r"\{[^{}]*\}", " ", src)
        if new == src:
            break
        src = new
    src = re.sub(r"<[^>]+>", " ", src)
    src = re.sub(r"[ \t]+", " ", src)
    src = re.sub(r"\n\s*\n+", "\n\n", src)
    return src.strip()


def file_to_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".astro":
        return astro_to_text(raw)
    return strip_frontmatter(raw).strip()


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries, packing up to max_chars per chunk."""
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ── Deterministic observations (structured sources, no LLM) ───

def nome_produto(slug: str, marca: str) -> str:
    """Mirror of nomeProduto() in site-goiania/src/data/produtos.ts."""
    marca_l = marca.lower()
    tokens = [
        t for t in slug.removeprefix("porcelanato-").split("-")
        if t not in (marca_l, "porcelanato")
    ]
    return " ".join(tokens).title()


def catalog_observations(catalog_path: Path) -> list[dict]:
    produtos = json.loads(catalog_path.read_text(encoding="utf-8"))
    obs = []
    for p in produtos:
        a = p["atributos"]
        if not a.get("preco"):
            continue  # sem preço = não é fato de catálogo publicável
        nome = nome_produto(p["slug"], a["marca"])
        partes = [
            f"O porcelanato {nome} da marca {a['marca'].title()}, "
            f"{a['dimensao']} com acabamento {a['acabamento']}, "
            f"custa R$ {a['preco']:.2f} por m² na ROI Labs em Goiânia",
            f"cada caixa cobre {a['m2_caixa']} m²",
        ]
        if a.get("retificado"):
            partes.append("é retificado")
        if a.get("classe_ad"):
            partes.append(f"tem classe de abrasão (PEI) {a['classe_ad']}")
        obs.append({
            "text": ", ".join(partes) + ".",
            "metadata": {"source": "catalogo-roilabs", "slug": p["slug"], "category": "metric"},
        })
    return obs


def rank_observations(csv_path: Path) -> list[dict]:
    obs = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pos = (row.get("posicao") or "").strip()
            if not pos:
                continue
            obs.append({
                "text": (
                    f"Em {row['data']}, o site goiania.roilabs.com.br ocupava a "
                    f"posição {pos} do Google para a busca \"{row['keyword']}\"."
                ),
                "metadata": {"source": "rank-tracking", "category": "metric", "date": row["data"]},
            })
    return obs


async def deposit_direct(observations: list[dict], memory: MemoryClient, agent_id: str) -> int:
    await memory.heartbeat(agent_id, "scout")
    ok = 0
    for o in observations:
        try:
            resp = await memory.observe(
                raw_content=o["text"], source_agent=agent_id, metadata=o["metadata"]
            )
            ok += 1
            logger.info(f"📤 obs #{resp['id']}: {o['text'][:90]}...")
        except Exception as e:
            logger.error(f"❌ Falha ao depositar: {e}")
    return ok


# ── Main ──────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="SEM-Swarm — ingestão de fontes reais")
    parser.add_argument("--catalog", type=Path, help="porcelanatos.json (1 obs/produto, sem LLM)")
    parser.add_argument("--files", type=Path, nargs="+", help="arquivos .md/.astro/.txt via Scout")
    parser.add_argument("--rank-csv", type=Path, help="rank-tracking.csv (só linhas com posição)")
    parser.add_argument("--api-url", default=os.getenv("SEM_API_URL", "http://localhost:8000"))
    args = parser.parse_args()

    if not (args.catalog or args.files or args.rank_csv):
        parser.error("informe ao menos uma fonte: --catalog, --files ou --rank-csv")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    memory = MemoryClient(api_url=args.api_url)
    if not await memory.is_healthy():
        logger.error(f"❌ Memory API inalcançável em {args.api_url}")
        sys.exit(1)

    total = 0

    if args.catalog:
        obs = catalog_observations(args.catalog)
        logger.info(f"📦 Catálogo: {len(obs)} produtos com preço → depositando direto...")
        total += await deposit_direct(obs, memory, "ingest-catalog")

    if args.rank_csv:
        obs = rank_observations(args.rank_csv)
        logger.info(f"📈 Rank tracking: {len(obs)} linhas com posição → depositando direto...")
        if obs:
            total += await deposit_direct(obs, memory, "ingest-rank")

    if args.files:
        scout = ScoutAgent(agent_id="scout-roilabs", api_url=args.api_url)
        # SLM em CPU sob concorrência (ex.: filter rodando junto) estoura os
        # 120s default — a fila do Ollama serializa os generates.
        scout.ollama.timeout = 480.0
        scout.extractor.timeout = 300.0
        for path in args.files:
            text = file_to_text(path)
            chunks = chunk_text(text)
            logger.info(f"📄 {path.name}: {len(text)} chars → {len(chunks)} chunk(s) via Scout")
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"   chunk {i}/{len(chunks)}...")
                for attempt in (1, 2):
                    try:
                        result = await scout.run(chunk, offline=False)
                        n = result.get("observation_count", 0)
                        total += n
                        logger.info(f"   → {n} observações ({result['status']})")
                        break
                    except Exception as e:
                        if attempt == 1:
                            logger.warning(f"   ⚠️ chunk {i} falhou ({e}); retry...")
                        else:
                            logger.error(f"   ❌ chunk {i} falhou de novo, pulando: {e}")

    logger.info(f"✅ Ingestão concluída: {total} observações depositadas em {args.api_url}")
    logger.info("   Próximo passo: python -m agents.filter --loop 60  (promove a fatos)")


if __name__ == "__main__":
    asyncio.run(main())
