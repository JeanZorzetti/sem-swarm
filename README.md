# SEM-Swarm

**Self-Distilling Swarm Intelligence with Epistemic Memory**

Uma arquitetura de Inteligência de Enxame Heterogênea operada por Small Language Models (SLMs), sem modelo centralizador. A coordenação entre agentes descentralizados ocorre de forma estigmérgica através de uma **Memória Epistêmica Compartilhada**.

## Princípio Arquitetural: Privilégio Cognitivo Mínimo

Cada modelo recebe **exatamente** a capacidade intelectual que sua tarefa exige — nem mais, nem menos. Nenhum modelo generalista tenta fazer tudo sozinho.

```
┌─────────────────────────────────────────────────────────────┐
│                    NOTEBOOK LOCAL (~7-10 GB RAM)             │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ qwen2.5-coder│  │  Phi-4-Mini  │  │  NuExtract   │      │
│  │    1.5B      │  │    3.8B      │  │    3.8B      │      │
│  │  ~1.5 GB     │  │  ~3-4 GB     │  │  ~2-3 GB     │      │
│  │              │  │              │  │              │      │
│  │  FIM/Autocmp │  │ Orquestrador │  │ Extração JSON│      │
│  │  >100 tok/s  │  │ Tool Calling │  │ Determinístico│     │
│  │  IDE only    │  │ MCP, Chat    │  │ Schema-exact │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  Sobra: ~14 GB para OS, Docker, IDE, Node.js               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    VPS BOSTON (24/7)                          │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ qwen3-embedding  │  │  DeepSeek-R1     │                 │
│  │      8B          │  │     14B          │                 │
│  │  Embeddings      │  │  Raciocínio      │                 │
│  │  100+ idiomas    │  │  assíncrono      │                 │
│  │  32k contexto    │  │  Dreaming Loop   │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                             │
│  PostgreSQL + pgvector │ FastAPI Substrate API               │
└─────────────────────────────────────────────────────────────┘
```

## Pipeline do Scout Agent (Dois Estágios)

```
Raw Text → Phi-4-Mini (raciocínio) → NuExtract (JSON determinístico) → Memory API
```

Por que dois estágios?
- Modelos generalistas <10B atingem apenas **28-33%** de acurácia em extração JSON pura
- NuExtract, treinado sinteticamente, atinge **near-100%** de acurácia estrutural
- Phi-4-Mini foca no que faz melhor: raciocínio, tool calling, compreensão de contexto

## Quick Start

### Pré-requisitos

- [Ollama](https://ollama.ai/) instalado e rodando
- Python 3.11+
- Docker & Docker Compose (para a stack de banco + API)

### 1. Configure

```bash
cp .env.example .env
# Edite o .env com suas credenciais
```

### 2. Baixe os modelos

```bash
# Modelos locais (notebook)
ollama pull phi4-mini            # Orquestrador (3.8B, ~3-4 GB)
ollama pull nuextract            # Extrator JSON (3.8B, ~2-3 GB)
ollama pull qwen2.5-coder:1.5b   # FIM/autocomplete IDE (~1.5 GB)

# Modelos VPS (opcional, rode na VPS)
ollama pull qwen3-embedding      # Embeddings multilíngues
ollama pull deepseek-r1:14b      # Raciocínio profundo assíncrono
```

### 3. Suba a stack (PostgreSQL + API)

```bash
# Local (banco + API)
docker compose up -d

# VPS (banco + API + Ollama)
docker compose --profile vps up -d
```

### 4. Rode o Scout

```bash
# Modo online (envia para a API)
python -m agents.scout --input "Texto para analisar"

# Modo offline (salva localmente)
python -m agents.scout --input "Texto para analisar" --offline

# Usar modelo alternativo de orquestração
python -m agents.scout --input "Texto" --orchestrator qwen3:8b

# A partir de arquivo
python -m agents.scout --file documento.txt
```

### 5. Testes

```bash
python tests/test_scout_local.py
```

## API Endpoints

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/memory/observe` | POST | Scout deposita observação bruta |
| `/memory/pending` | GET | Filter busca observações pendentes |
| `/memory/verify` | POST | Filter promove observação a fato verificado |
| `/memory/search` | POST | Synthesizer faz busca vetorial por similaridade |
| `/swarm/state` | GET | Estado de coordenação do enxame |
| `/health` | GET | Health check |

## Modelos e Papéis

| Modelo | Params | RAM | Onde | Papel |
|--------|--------|-----|------|-------|
| **qwen2.5-coder:1.5b** | 1.5B | ~1.5 GB | Local | FIM/autocomplete IDE — **NÃO** orquestrando |
| **phi4-mini** | 3.8B | ~3-4 GB | Local | Orquestrador: tool calling, MCP, chat |
| **nuextract** | 3.8B | ~2-3 GB | Local | Extrator JSON determinístico |
| **qwen3:8b** | 8B | ~6-8 GB | Local | Conselheiro para lógica complexa |
| **qwen3-embedding** | 8B | — | VPS | Embeddings multilíngues (100+ idiomas) |
| **deepseek-r1:14b** | 14B | ~9 GB | VPS | Raciocínio assíncrono (Dreaming Loop) |

## Roadmap

- [x] Sprint 0 — Bootstrap (Scout + API + Banco)
- [x] Sprint 0.5 — Revisão arquitetural (paper-driven: Phi-4-Mini + NuExtract)
- [ ] Sprint 1 — Filter + Synthesizer + Dreaming Loop
- [ ] Sprint 2 — Self-Distillation + Consensus
- [ ] Sprint 3 — Multi-agent coordination + Benchmarks

---

*Projeto de pesquisa — ROI Labs*
