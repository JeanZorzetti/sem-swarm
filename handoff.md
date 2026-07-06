# Handoff — Sprints 1–3 + primeiro uso real (2026-07-06)

## 2026-07-06 — Primeiro uso real: ingestão do ROI Labs Growth Partner

- **`scripts/ingest.py` (novo)**: ingestão de fontes reais em 3 modos — `--catalog` (porcelanatos.json → 1 obs determinística/produto, sem LLM), `--files` (md/astro/txt → limpa markup, chunks ~4k chars → pipeline Scout completo), `--rank-csv` (só linhas com posição). Carrega o `.env` do projeto automaticamente (sem override de env já setada).
- **Primeira carga real em PROD**: 30 obs do catálogo (obs #11–#40, UTF-8 validado no banco) + tese.md + mercado.md + fornecedores-goiania.md + 7 guias AEO via Scout. Filter daemon (`--loop 45`) promovendo a fatos.
- **Docs novos**: `docs/VISAO-GERAL.md` + `docs/DOCUMENTO-EXECUTIVO.md` (didáticos, PT-BR). Runbook da ingestão no vault do roilabs: `Docs/Obsidian/80-dev/sem-swarm-ingestao.md`.
- **Gotcha**: rank-tracking.csv de 2026-07-03 não tinha NENHUMA posição → 0 obs dessa fonte; re-rodar quando o site rankear.
- Re-rodar ingestão é seguro por design: duplicata ≥0.95 vira corroboração (consenso), não lixo.

---

# Handoff anterior — Sprints 1, 2 e 3 (2026-07-05)

## Sprint 3 SHIPPED — Coordination + Benchmarks

- **Presença de agentes (coordenação estigmérgica)**: `POST /swarm/heartbeat` grava `{agent_id, role, last_seen}` em `swarm_state.metadata.agents` (JSONB, sem migração); `active_agents_count` = vistos nos últimos 5 min; prune >24h. Scout/filter/synthesizer chamam via `MemoryClient.heartbeat()` (best-effort — falha nunca quebra o agente). UI ganhou card "Agentes" com lista viva.
- **Filter daemon**: `python -m agents.filter --loop N` — poll contínuo a cada N segundos, sobrevive a falhas transitórias (loga e segue). Com o `FOR UPDATE SKIP LOCKED` do `/memory/pending`, múltiplos filters concorrentes já são seguros.
- **Benchmark** (`python -m tests.benchmark_pipeline [--api-url] [--inputs N]`): mede latência por estágio, desfecho das observações e **fidelidade numérica** input→fato (a métrica do gap 5). Baseline 1º run (notebook i7-1255U, prod DB): scout 97s, filter 151s, fidelidade 1.0. Relatórios em `tests/benchmark-results/`.

## OPS RESOLVIDO (2026-07-05, com Jean)
- Volume `ollama` montado em `/root/.ollama` — modelos sobreviveram ao redeploy (validado).
- `OLLAMA_MAX_LOADED_MODELS=1` no env — validado: após generate, `/api/ps` mostra só o 7B residente (o 8B foi descarregado antes; cenário de OOM eliminado).
- Modelos re-pulled DENTRO do volume: `qwen3-embedding` + `qwen2.5:7b-instruct` (~9,4 GB).

## Sprint 2 SHIPPED (`cfc8713`) — Consensus + Self-Distillation

### Consensus (corroboração)
- `POST /memory/corroborate`: observação quase-idêntica a um fato existente (sim ≥0.95) deixa de virar lixo — reforça o fato (contador `corroborations`, confiança fecha metade da incerteza restante ponderada pela evidência), marca a obs `verified` com `corroborated_fact_id` e grava `swarm_state.last_consensus_at`.
- Filter: o branch de dedup agora corrobora em vez de rejeitar. Validado em prod: obs #6 → fato #1 (sim 0.9793, corroborations=1).
- Gotcha asyncpg: `to_jsonb(:param)` precisa de `CAST` explícito (`to_jsonb(CAST(:x AS bigint))`), senão `DatatypeMismatchError` — mesmo padrão do `/memory/reject`.

### Self-Distillation (dreaming loop)
- Pares de contradição do `sem-vector` (banda 0.70–0.95) agora são **julgados pelo LLM da VPS** (CONTRADITORIOS/COMPATIVEIS); perdedor é superseded com `metadata.resolution='contradiction_resolved'`. Vence quem tem mais corroborações; empate → mais recente (regra do doc-fonte). Cap de 10 julgamentos/tick.
- Fato consolidado agora recebe **embedding real do texto unificado** (MRL 2048) em vez de centroide.
- **E2E em prod (demonstração completa)**: fato falso semeado ("porcelanato absorve >8%") → dreaming julgou: vs #2 (0×0 corroborações, empate) → recência venceu, #2 superseded; vs #1 (**1 corroboração**) → **#1 venceu e expurgou o fato falso**. O consenso derrotou a desinformação. Bônus: o filter (phi4-mini) tinha REJEITADO sozinho a observação falsa — o fato falso só entrou porque foi semeado direto via API.
- ⚠️ Artefato de ordenação observado: um perdedor julgado cedo no tick pode arrastar um fato verdadeiro (#2 caiu pro #5 antes do #5 cair pro #1; cadeia #2→#5→#1). Mitigação candidata pro Sprint 3: ordenar julgamentos por corroborações/recência ou passe duplo.

### 🔴 Achados de OPS (ação sua no EasyPanel, serviço ollama)
1. **SEM volume persistente**: o OOM recriou o container e **apagou os dois modelos** (re-pull de ~9,4 GB foi feito via API pra restaurar). Montar volume em `/root/.ollama`.
2. **OOM real**: embedding 8B + qwen2.5 7B carregados juntos estouram a RAM do serviço → setar `OLLAMA_MAX_LOADED_MODELS=1` no env.
3. O container recriado veio com Ollama 0.31.1 (era 0.30.2) — imagem `:latest` sem pin.

## Rodada 3 — Gap 3 FECHADO: embeddings reais (qwen3-embedding, 2048d via MRL)
- `EmbeddingGenerator` ganhou `dimensions` (truncamento MRL no `/api/embed`); `filter.py` e `synthesizer.py` trocaram o mock pelo embed real na VPS (`OLLAMA_VPS_URL`). Falha de embedding deixa a observação **pendente** (retry), nunca insere ruído.
- Os 4 fatos seed foram re-embedados in-place no DB de prod (script one-shot, zero LLM local).
- **Validação semântica real**: observação quase-idêntica ao fato #1 foi REJEITADA por dedup com similaridade **0.9888** (com mock, dedup semântica era impossível); busca "quintal que pega chuva" rankeou o fato certo (#4) no topo com 0.78 e ordenação coerente; synthesizer respondeu fundamentado no fato correto.
- Com embeddings reais, o dreaming loop passa a poder clusterizar/consolidar de verdade.

## Feito

### Rodada 1 — E2E local→prod (API prod-parity rodada local apontada pro DB de prod)
Pipeline validado ponta-a-ponta pela primeira vez: Scout (phi4-mini+nuextract) → `/memory/observe` → Filter (dedup Rust SIMD) → `/memory/verify` → Synthesizer → `/memory/search`. Obs/fatos #1–#2 no banco, `swarm_state` atualizado pela API, UTF-8 íntegro.

### Rodada 2 — E2E 100% REMOTO (após exposição dos domínios no EasyPanel)
- **Gap 1 FECHADO (ops)**: API pública no ar — `GET /health` = healthy. Ollama da VPS também exposto, com `qwen3-embedding` e `qwen2.5:7b-instruct` pulled.
- **Gap 2 FECHADO (código)**: `MemoryClient` e o `--api-url` do scout agora leem `SEM_API_URL` do env. Validado: scout depositou obs #3–#4, filter aprovou e promoveu, synthesizer respondeu **corretamente** (0,5% vs 3–10% de absorção) — tudo pela URL pública, sem API local.
- **Gap 4 FECHADO (código)**: dreaming loop grava heartbeat em `swarm_state.last_dreaming_loop_at` a cada tick, mesmo sem nada a consolidar. Validado com a imagem Docker de prod-parity contra o DB de prod (tick registrado; ciclo completo em 2.8s; o wrapper do Dockerfile dorme 3h entre ticks).

## Decisões
- Dados de teste (obs/fatos #1–#4 sobre porcelanato/cerâmica) ficaram no banco como seed.
- Embeddings reais (gap 3) ficaram de fora desta rodada de propósito: trocar agora misturaria vetores reais com os 4 mocks já persistidos; fechar junto com uma limpeza/re-embed do seed.

## Descoberta importante (gap 3 está a uma flag de distância)
`qwen3-embedding` retorna **4096 dims** por padrão, mas a coluna é `halfvec(2048)` — e HNSW em halfvec só indexa até ~4000 dims, então migrar a coluna não é opção. Porém o Ollama aceita `{"dimensions": 2048}` no `/api/embed` (MRL) e retorna 2048 certinho. O fix do gap 3 é: chamar o embed real com `dimensions: 2048` em `filter.py`/`synthesizer.py` (substituindo `get_mock_embedding`) + re-embedar/limpar os 4 fatos seed.

## Rodada 4 — UI inspetora NO AR (`6380dc0`)
- `GET /` da API serve inspetor read-only single-file (dark, vanilla JS, auto-refresh 30s): cards de swarm state (observações, fatos, último tick do dreaming com alerta de staleness >4h), tabela de fatos (confiança, ativo/consolidado) e observações recentes com badge de status.
- Novos endpoints read-only `GET /memory/facts` e `GET /memory/observations` — a UI **não pode** usar `/memory/pending` (ele marca as observações como `processing`, tem efeito colateral).
- Deploy automático por push confirmado (API e dreaming). Verificado em prod com screenshot: https://sofia-sem-swarm-api.7c17iw.easypanel.host/ — e o heartbeat do dreaming está **ticking em produção** (serviço redeployou sozinho e registrou tick).

## Pendências
1. **Ops — env do dreaming**: setar `OLLAMA_DEEP_REASONING_MODEL=qwen2.5:7b-instruct` no serviço (nem `deepseek-r1:14b` nem o default `phi4-mini` estão pulled no Ollama da VPS; só importa quando houver cluster a consolidar).
2. **Gap 5** (qualidade): nuextract gera typos e perde trechos na extração — métrica pros benchmarks do Sprint 3.
3. Sprint 2 — Self-Distillation + Consensus.
4. (Opcional, se o notebook continuar sofrendo) fallback pra rodar o raciocínio do filter na VPS (`qwen2.5:7b-instruct`) — a iGPU Iris Xe não ajuda: "11,8 GB" é RAM compartilhada, mesma banda da CPU. `OLLAMA_MAX_LOADED_MODELS=1` no Ollama local também alivia ~3 GB de RAM.

## Gotchas
- Console Windows mostra mojibake nos logs dos agentes; os dados chegam UTF-8 corretos no banco.
- Toolchain Rust local (Windows/GNU) falha com `dlltool not found` em deps — validar o crate sempre via docker build do `rust/sem-dreaming/Dockerfile` (mesmo caminho do EasyPanel).
- O container do dreaming nunca "termina": o Dockerfile envolve o binário em loop com sleep de 3h — é o design do serviço, não travamento.
