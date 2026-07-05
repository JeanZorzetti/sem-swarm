# ADR-001: Componentes Rust no SEM-Swarm

**Status:** Aceito  
**Data:** 2026-07-05  
**Contexto:** Projeto SEM-Swarm (Self-Distilling Swarm Intelligence with Epistemic Memory)

---

## Contexto e Problema

O SEM-Swarm opera em hardware restrito (i7-1255U, 24 GB RAM, sem GPU) com uma VPS
auxiliar (4 vCPUs, 16 GB RAM). A implementação inicial em Python é adequada para
prototipagem e lógica de orquestração (agentes são I/O-bound — 99% do tempo esperando
Ollama), mas três componentes enfrentam gargalos de performance que Python não resolve:

1. **Dreaming Loop (VPS):** Consolidação de memória CPU-intensiva em ambiente com RAM
   escassa (~3-4 GB livres). Python consome ~120 MB só para iniciar + GIL impede
   paralelismo real nos 4 vCPUs.

2. **Operações vetoriais em escala:** Quando a Memória Epistêmica crescer para dezenas
   de milhares de fatos, batch cosine similarity, clustering e deduplicação em Python/
   numpy atingem um teto de throughput.

3. **Daemon do enxame:** Processo always-on para coordenação de agentes precisa de
   footprint mínimo e concorrência sem GIL.

## Decisão

Adotar uma **arquitetura híbrida Python + Rust** seguindo o princípio:
> *"Prove que funciona em Python, otimize com Rust onde dói."*

### Componentes Rust

| Crate | Tipo | Função | Quando |
|-------|------|--------|--------|
| `sem-vector` | PyO3 lib (`.pyd`/`.so`) | Operações vetoriais: cosine similarity, batch dedup, clustering | Sprint 1 |
| `sem-dreaming` | Binário standalone | Dreaming Loop: consolidação noturna de memórias | Sprint 2 |
| `sem-swarmd` | Binário daemon | Coordenador do enxame (lifecycle, heartbeat, routing) | Sprint 3 |

### Componentes que permanecem em Python

| Componente | Razão |
|------------|-------|
| Agentes (Scout, Filter, Synthesizer) | I/O-bound (esperam Ollama). Iteração rápida em prompt engineering. |
| FastAPI (API de Substrato) | Ecossistema maduro, async performante, Pydantic. |
| Clients (Ollama, NuExtract, Memory) | Wrappers HTTP finos. httpx async = reqwest em I/O. |

### Padrão de Integração

```
Rust (sem-vector via PyO3)  ←→  Python (agentes, API)
Rust (sem-dreaming)         ←→  PostgreSQL (direto via sqlx)
Rust (sem-swarmd)           ←→  Memory API (REST), Ollama (REST)
```

A biblioteca `sem-vector` é importável diretamente em Python:
```python
from sem_vector import batch_cosine, find_duplicates, cluster_facts
```

Os binários `sem-dreaming` e `sem-swarmd` são processos independentes que se comunicam
via PostgreSQL e REST API — sem acoplamento direto com Python.

## Consequências

### Positivas
- Dreaming Loop: ~8 MB RAM (vs ~120 MB), ~15x mais rápido em batch vector ops
- Paralelismo real via Rayon nos 4 vCPUs da VPS (sem GIL)
- Binários estáticos para deploy na VPS sem dependências Python
- Agentes mantêm flexibilidade Python para iteração rápida

### Negativas
- Complexidade de build (precisa compilar Rust + maturin para PyO3)
- Dois toolchains para manter (cargo + pip)
- Curva de aprendizado para contribuidores não-familiarizados com Rust

### Mitigações
- Workspace Cargo unificado em `rust/`
- CI/CD compila e publica wheel automaticamente
- Fallback Python puro em `core/vector_ops.py` caso a lib Rust não esteja compilada

## Referências
- Paper: "Análise Arquitetural e Orquestração de SLMs para Ecossistemas Locais" §Topologia Híbrida
- Handoff: SEM-Swarm §Dreaming Loop, §Topologia do Enxame
