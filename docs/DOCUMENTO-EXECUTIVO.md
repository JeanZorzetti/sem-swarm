# SEM-Swarm — Documento Executivo (parte a parte)

> Este documento explica **cada componente** do projeto em quatro dimensões: **o que faz**, **por que faz (assim)**, **como faz** e **qual a utilidade**. Para o panorama do sistema, leia antes a [Visão Geral](VISAO-GERAL.md).

## Índice

1. [Os agentes (`agents/`)](#1-os-agentes-agents)
   - [1.1 Scout — o batedor](#11-scout--o-batedor-agentsscoutpy)
   - [1.2 Filter — o filtro epistêmico](#12-filter--o-filtro-epistêmico-agentsfilterpy)
   - [1.3 Synthesizer — o sintetizador](#13-synthesizer--o-sintetizador-agentssynthesizerpy)
2. [A Memory API (`api/`)](#2-a-memory-api-api)
3. [O banco de dados (`infra/init.sql`)](#3-o-banco-de-dados-infrainitsql)
4. [A caixa de ferramentas (`core/`)](#4-a-caixa-de-ferramentas-core)
5. [Os componentes Rust (`rust/`)](#5-os-componentes-rust-rust)
   - [5.1 sem-vector — matemática vetorial](#51-sem-vector--matemática-vetorial-simd)
   - [5.2 sem-dreaming — o Dreaming Loop](#52-sem-dreaming--o-dreaming-loop)
6. [Infraestrutura e deploy (`docker-compose.yml`)](#6-infraestrutura-e-deploy-docker-composeyml)
7. [Testes e benchmark (`tests/`)](#7-testes-e-benchmark-tests)
8. [Apêndice: limiares e números que governam o sistema](#8-apêndice-limiares-e-números-que-governam-o-sistema)

---

## 1. Os agentes (`agents/`)

Os três agentes são programas Python independentes, rodados via linha de comando (ou daemon). Eles **nunca conversam entre si** — toda comunicação passa pela Memory API (coordenação estigmérgica). Cada um anuncia presença via *heartbeat* ao trabalhar, para que o enxame saiba quem está vivo.

### 1.1 Scout — o batedor (`agents/scout.py`)

**O que faz.** Recebe texto bruto (frase, documento, log) e o transforma numa lista de **observações** — afirmações factuais, autocontidas, cada uma com nota de relevância (0–1) e categoria (`fact`, `definition`, `relationship`, `metric`, `event`). Deposita cada observação na Memory API com status `pending`.

**Por que faz.** Todo conhecimento do enxame nasce aqui. O mundo entrega informação em formato caótico; o Scout é a porta de entrada que padroniza esse caos em unidades pequenas e julgáveis. Ele **não decide o que é verdade** — só observa e reporta (a decisão é do Filter; separar os papéis evita que um único modelo seja juiz e parte).

**Como faz.** Pipeline de **dois estágios**, a decisão arquitetural mais importante do projeto:

```
texto bruto ──► ESTÁGIO 1: phi4-mini raciocina         ──► ESTÁGIO 2: nuextract converte
               (identifica observações, em linguagem       o raciocínio em JSON exato
                natural, temperature 0.3)                   conforme o schema
```

1. **Estágio 1 (raciocínio)**: o `phi4-mini` recebe um prompt de sistema com regras rígidas (só fatos verificáveis, sem opinião, sem repetição, responder no idioma do input) e produz uma análise em texto livre.
2. **Estágio 2 (extração)**: o `nuextract` — um modelo treinado exclusivamente para extração estruturada — recebe esse texto e um template JSON (`observations[]` com `content`, `relevance`, `category`) e devolve JSON válido, com até 2 retries.
3. **Fallbacks em cascata**: sem NuExtract instalado → o phi4-mini gera JSON diretamente com *format constraint* do Ollama (precisão degradada, mas funciona); sem API alcançável → salva em `data/offline_observations/*.json` para envio posterior.

*Por que dois estágios e não um?* Modelos generalistas <10B acertam só 28–33% das extrações JSON puras ("diluição de atenção" — o modelo se perde entre raciocinar e formatar ao mesmo tempo). Separando, cada modelo opera na sua zona de quase-100%.

**Utilidade.** É o único ponto do sistema que ingere o mundo externo. Uso: `python -m agents.scout --input "texto"` (ou `--file doc.txt`, `--offline`, `--orchestrator qwen3:8b` para trocar o cérebro).

---

### 1.2 Filter — o filtro epistêmico (`agents/filter.py`)

**O que faz.** Busca observações `pending` na API e **julga cada uma**: rejeita ruído/opinião/alucinação; promove as válidas a **fatos verificados** com embedding semântico; e, quando a observação repete um fato já existente, **corrobora** o fato em vez de criar duplicata.

**Por que faz.** É o guardião da memória: sem ele, a base viraria um depósito de boatos e o Synthesizer responderia com base em lixo. A qualidade de todo o sistema depende deste gargalo — por design, **nada entra na memória permanente sem passar por ele**. Foi validado na prática: quando um fato falso foi injetado por fora (direto na API), o Filter já tinha rejeitado a mesma alegação sozinho.

**Como faz.** Para cada observação:

1. **Raciocínio** (`phi4-mini`): avalia objetividade e utilidade com regras explícitas ("fofocas, opiniões subjetivas ou informações pela metade devem ser REJEITADAS") e, se válida, formula o **fato limpo** (frase concisa e definitiva).
2. **Extração da decisão** (`nuextract`): converte o veredito em JSON `{is_valid, reasoning_summary, clean_fact, confidence_score}`. Falha de parse → rejeita por segurança (política *fail-closed*).
3. **Embedding real**: gera o vetor do fato via `qwen3-embedding` na VPS (2048 dimensões via MRL). Se o serviço de embedding estiver fora, a observação **fica pendente para retry** — o sistema prefere atrasar a inserir ruído.
4. **Deduplicação em duas fases**: busca os top-5 fatos mais parecidos via índice aproximado do banco (HNSW), depois recalcula a similaridade **exata** localmente com o `sem-vector` (Rust/SIMD). Três desfechos:
   - **sim ≥ 0.95** → `POST /memory/corroborate`: o fato existente ganha +1 corroboração e mais confiança (consenso);
   - **0.70 ≤ sim < 0.95** → fato novo é criado, anotando `related_fact_id` (pista para o Dreaming Loop investigar contradição);
   - **sim < 0.70** → fato novo e independente.
5. **Modo daemon**: `--loop N` faz poll contínuo a cada N segundos, sobrevivendo a falhas transitórias. Como a API entrega observações com `FOR UPDATE SKIP LOCKED`, **vários Filters concorrentes são seguros** — nunca julgam a mesma observação duas vezes.

**Utilidade.** Transforma volume em confiança. Uso: `python -m agents.filter` (uma passada) ou `python -m agents.filter --loop 60` (daemon).

---

### 1.3 Synthesizer — o sintetizador (`agents/synthesizer.py`)

**O que faz.** Recebe uma pergunta, busca os fatos verificados mais relevantes na memória (busca vetorial) e compõe uma resposta **fundamentada exclusivamente nesses fatos** — citando confiança e admitindo o que não sabe.

**Por que faz.** É o retorno do investimento do enxame: o momento em que a memória coletiva vira resposta útil. A restrição "responda APENAS com os fatos fornecidos" existe para **eliminar alucinação por construção** — se a base não cobre o assunto, a resposta honesta é "não encontrei nenhuma informação verificada sobre isso".

**Como faz.** Padrão RAG (Retrieval-Augmented Generation) sobre a memória própria:

1. Gera o embedding da pergunta (`qwen3-embedding`, VPS, 2048d) — pergunta e fatos vivem no mesmo espaço vetorial, então proximidade geométrica = relevância semântica.
2. `POST /memory/search`: top-5 fatos com `confidence ≥ 0.5`, ordenados por similaridade de cosseno.
3. Monta o contexto rotulando cada fato com sua confiança (`Fato 1 (Confiança 0.95): ...`) e pergunta ao modelo de raciocínio (`phi4-mini` por padrão; `qwen3:8b` via env var para lógica mais profunda).

Validação em produção: perguntado sobre absorção de água de porcelanato, respondeu corretamente "0,5% contra 3–10% da cerâmica" — números que vieram dos fatos, não da imaginação do modelo.

**Utilidade.** Interface de consulta do conhecimento coletivo. Uso: `python -m agents.synthesizer --query "sua pergunta"`.

---

## 2. A Memory API (`api/`)

**O que faz.** Serviço FastAPI que expõe a Memória Epistêmica via REST. É o **único caminho** de leitura/escrita dos agentes ao banco — o "quadro de avisos" do enxame. Serve também uma UI de inspeção em `GET /`.

**Por que faz.** Centralizar o acesso num serviço (em vez de cada agente abrir conexão SQL) dá três coisas: agentes rodam de qualquer lugar com uma URL (`SEM_API_URL`); as regras de transição de estado ficam num único lugar auditável; e o contrato REST permite trocar a implementação de qualquer agente (Python, Rust, o que for) sem tocar no resto.

**Como faz.** FastAPI + SQLAlchemy async + SQL explícito (sem ORM mágico). Endpoints e seus papéis:

| Endpoint | Quem usa | O que garante |
|---|---|---|
| `POST /memory/observe` | Scout | Insere observação `pending` + incrementa contador global |
| `GET /memory/pending` | Filter | Entrega lote FIFO e marca `processing` **atomicamente** (CTE com `FOR UPDATE SKIP LOCKED` — dois Filters nunca pegam a mesma obs) |
| `POST /memory/verify` | Filter | Valida dimensão do embedding (2048), confere status, insere fato e marca a obs `verified` |
| `POST /memory/corroborate` | Filter | Consenso: `confidence += (1 − confidence) × evidência × 0.5` (fecha metade da incerteza restante, nunca passa de 1.0), incrementa `corroborations`, marca `last_consensus_at` |
| `POST /memory/reject` | Filter | Marca `rejected` gravando o motivo no metadata |
| `POST /memory/search` | Synthesizer, Filter | Busca vetorial: `1 − (embedding <=> query)` (cosseno via pgvector), filtra `is_active` e confiança mínima |
| `GET /memory/facts` · `GET /memory/observations` | UI | Listagens **read-only, sem efeito colateral** (a UI não pode usar `/memory/pending`, que marca `processing`) |
| `GET /swarm/state` | Todos | Contadores globais + timestamps de consenso e dreaming |
| `POST /swarm/heartbeat` | Todos os agentes | Presença em `swarm_state.metadata.agents` (JSONB, sem migração); ativo = visto em 5 min; poda >24h |
| `GET /health` | Monitoração | API + banco respondendo |

A UI (`api/static/index.html`, arquivo único, vanilla JS, tema escuro, auto-refresh 30s) mostra o estado do enxame: contadores, agentes vivos, tabela de fatos com confiança e badge ativo/consolidado, observações recentes, e alerta se o Dreaming Loop está há mais de 4h sem tick.

**Utilidade.** É o sistema nervoso central. Em produção: `https://sofia-sem-swarm-api.7c17iw.easypanel.host/` (UI na raiz, OpenAPI em `/docs`).

---

## 3. O banco de dados (`infra/init.sql`)

**O que faz.** Define o substrato físico da memória: PostgreSQL + extensão pgvector, três tabelas, índices e triggers.

**Por que faz.** Postgres com pgvector dá, num único serviço battle-tested, tudo que o enxame precisa: transações (estados nunca ficam inconsistentes), locks (`SKIP LOCKED` vira fila de trabalho de graça), JSONB (metadados evoluem sem migração) e busca vetorial indexada. Nenhum banco vetorial dedicado seria mais simples de operar.

**Como faz.** As três tabelas espelham o ciclo de vida do conhecimento:

- **`env_observations`** — a "caixa de entrada": conteúdo bruto, agente de origem, status (`pending → processing → verified | rejected`, um ENUM nativo), metadata JSONB. Índice parcial em status pendente/processando (a consulta quente do Filter).
- **`epistemic_memory`** — o "cérebro da colmeia": `fact_text`, `embedding halfvec(2048)`, `confidence_score` (com CHECK 0–1 no banco, não só no código), contadores de corroboração no metadata, e a linhagem: `source_observation_id` (de onde veio) e `superseded_by` (quem o substituiu). Fatos derrotados ficam `is_active = FALSE` — **nada é deletado**, a história é auditável.
- **`swarm_state`** — uma única linha com o estado global: contadores, `last_consensus_at`, `last_dreaming_loop_at` e o mapa de agentes vivos no metadata.

Detalhes técnicos com propósito:

- **`halfvec(2048)`**: vetores em meia-precisão (float16) — metade do espaço, e o índice HNSW sobre halfvec só suporta até ~4000 dimensões. Como o `qwen3-embedding` emite 4096 por padrão, o sistema trunca para 2048 via MRL **na hora de gerar** o embedding (a coluna não poderia simplesmente crescer).
- **Índice HNSW** (`m=16, ef_construction=64`): busca de vizinhos aproximada em milissegundos mesmo com dezenas de milhares de fatos — é o que mantém `/memory/search` rápido para sempre.
- **Trigger `update_updated_at`**: carimba `updated_at` em qualquer UPDATE, de graça, no banco.

**Utilidade.** Roda automaticamente na primeira subida do Postgres (montado em `/docker-entrypoint-initdb.d/`). É idempotente (`IF NOT EXISTS` em tudo).

---

## 4. A caixa de ferramentas (`core/`)

**O que faz.** Cinco módulos compartilhados por todos os agentes — os "órgãos dos sentidos" que cada agente monta ao nascer.

**Por que faz.** Os três agentes precisam das mesmas capacidades (falar com Ollama, extrair JSON, falar com a API, gerar embeddings, comparar vetores). Centralizar evita triplicação e garante que um fix (ex.: retry de embedding) beneficie todos de uma vez.

**Como faz.** Cada módulo, sua função e sua decisão de projeto:

| Módulo | O que é | Decisão de projeto embutida |
|---|---|---|
| `ollama_client.py` | Client REST do Ollama (generate, chat, structured output) | **Dual-endpoint**: resolve automaticamente se o modelo mora no notebook ou na VPS; timeout de 120s porque SLM em CPU é lento e isso é normal |
| `extractor.py` | Client dedicado do NuExtract | Formaliza o estágio 2 do pipeline: recebe texto + template e devolve `dict` validado, com `extract_with_retry` (até 2 novas tentativas em JSON malformado) |
| `memory_client.py` | Client REST da Memory API | Espelha 1:1 os endpoints; lê `SEM_API_URL` do ambiente (mesma linha de comando funciona local e contra produção); `heartbeat()` é *best-effort* — falha de presença **nunca** derruba um agente |
| `embeddings.py` | Gerador de embeddings | VPS primeiro (`qwen3-embedding`, poupa RAM do notebook), fallback local (`nomic-embed-text`); passa `dimensions: 2048` (truncamento MRL) para caber no `halfvec(2048)`; se tudo falhar, lança `ConnectionError` — quem chama decide (o Filter deixa pendente para retry) |
| `vector_ops.py` | Fachada de matemática vetorial | Tenta importar a extensão Rust `sem_vector`; se não existir, usa fallback em **Python puro com a mesma assinatura**. O chamador nunca sabe qual implementação rodou |

**Utilidade.** É o que faz um agente novo custar ~100 linhas: toda a infraestrutura já está pronta, o agente só escreve seu raciocínio.

---

## 5. Os componentes Rust (`rust/`)

Contexto (formalizado no [ADR-001](architecture-decision-rust.md)): o hardware é restrito (notebook i7 sem GPU; VPS de 4 vCPUs com ~3–4 GB livres). Python resolve tudo que é I/O-bound, mas dois pontos são CPU-bound de verdade — e ali entra Rust. Gotcha de build: no Windows local o toolchain GNU falha (`dlltool not found`); os crates são validados **sempre via docker build** (mesmo caminho do deploy).

### 5.1 sem-vector — matemática vetorial (SIMD)

**O que faz.** Biblioteca Rust exposta a Python via PyO3 (`import sem_vector`) com cinco operações: similaridade de cosseno, batch cosine (1×N ordenado), detecção de duplicatas (N×N), detecção de candidatos a contradição e clustering de duplicatas via union-find.

**Por que faz.** Comparar um vetor contra um lote é O(N×2048) multiplicações — em Python puro, isso vira o gargalo do Filter e inviabiliza o Dreaming Loop quando a memória crescer para dezenas de milhares de fatos. Rust com SIMD e Rayon (paralelismo real, sem GIL) entrega ~10–15× a velocidade.

**Como faz.** `src/ops.rs` (operações base), `src/batch.rs` (lotes paralelos), `src/dedup.rs` (duplicatas/contradições/clusters), `src/python.rs` (bindings PyO3). A heurística de contradição é geométrica e barata: dois fatos com similaridade na **banda 0.70–0.95** falam do *mesmo assunto* (≥0.70) *sem serem a mesma frase* (<0.95) — exatamente o perfil de "porcelanato absorve 0,5%" vs "porcelanato absorve >8%". A banda só **seleciona candidatos**; quem decide se há contradição real é um LLM (ver 5.2).

**Utilidade.** Usada pelo Filter (dedup exata pós-busca) e pelo Dreaming Loop (clustering da memória inteira). Com o fallback de `vector_ops.py`, sua ausência nunca quebra nada — só deixa mais lento.

### 5.2 sem-dreaming — o Dreaming Loop

**O que faz.** Binário Rust que roda na VPS em ciclos (sono de 3h entre ticks, via wrapper do Dockerfile — o container "nunca terminar" é design, não travamento). A cada tick: registra heartbeat, carrega todos os fatos ativos, **funde redundâncias** e **resolve contradições**.

**Por que faz.** É a inspiração no sono biológico: durante o dia o enxame acumula depressa (Scout/Filter otimizam latência); de madrugada, um processo sem pressa reorganiza a memória — comprime redundância e elimina inconsistência. Sem ele, a memória cresceria para sempre com fatos repetidos e contraditórios coexistindo. É também o mecanismo de **auto-destilação** que dá nome ao projeto: o sistema refina o próprio conhecimento sem intervenção humana.

**Como faz.** Conecta direto no Postgres via sqlx (sem passar pela API — é processo interno de manutenção):

1. **Heartbeat primeiro**: grava `last_dreaming_loop_at` mesmo sem trabalho a fazer, para o mundo externo saber que está vivo (a UI alerta staleness >4h).
2. **Consolidação**: `cluster_duplicates(embeddings, 0.96)` agrupa fatos quase idênticos; o LLM da VPS funde cada grupo numa única frase sem perda de informação; o texto unificado recebe **embedding real próprio** (semântica de verdade vence centroide); numa transação, o fato novo entra com `source: dreaming_loop` e os antigos viram `superseded_by → novo`.
3. **Julgamento de contradições**: para cada par candidato da banda 0.70–0.95 (cap de 10 julgamentos/tick — o loop volta em 3h), pergunta ao LLM um veredito de uma palavra: `CONTRADITORIOS` ou `COMPATIVEIS`. Se contraditórios, aplica a regra de consenso: **vence quem tem mais corroborações; empate → o mais recente**. O perdedor é desativado com `resolution: contradiction_resolved`.

Prova em produção: um fato falso semeado deliberadamente foi julgado contra um fato verdadeiro que tinha 1 corroboração — **o corroborado venceu e o falso foi expurgado**. O consenso estatístico do enxame derrotou a desinformação, exatamente a hipótese central do projeto.

**Utilidade.** Qualidade de longo prazo da memória. É a razão de a base ficar *melhor* com o tempo, e não só maior.

---

## 6. Infraestrutura e deploy (`docker-compose.yml`)

**O que faz.** Sobe a stack com um comando: Postgres (imagem `pgvector/pgvector:pg16`) + Memory API. Com `--profile vps`, adiciona o serviço Ollama (para embeddings e o juiz do dreaming).

**Por que faz.** O mesmo arquivo serve o notebook (só banco + API; o Ollama local já roda nativo) e a VPS (tudo). O profile evita dois composes divergindo.

**Como faz.**

- Postgres monta `infra/init.sql` no diretório de init (schema automático na primeira subida) e persiste em volume `pgdata`.
- A API só sobe após o healthcheck do banco passar (`depends_on: condition: service_healthy`).
- O Ollama da VPS tem **volume em `/root/.ollama`** (lição aprendida em produção: sem volume, um OOM recriou o container e apagou ~9,4 GB de modelos) e roda com `OLLAMA_MAX_LOADED_MODELS=1` no ambiente (impede o embedding 8B e o juiz 7B de coexistirem na RAM — era o cenário do OOM).
- Deploy real: EasyPanel na VPS, **deploy automático por push** para API e dreaming.

**Utilidade.** `docker compose up -d` local; `docker compose --profile vps up -d` na VPS. Config via `.env` (copiar de `.env.example`).

---

## 7. Testes e benchmark (`tests/`)

**O que faz.**

- `test_scout_local.py`: teste funcional do Scout isolado (extração local, sem API).
- `benchmark_pipeline.py`: mede o pipeline scout→filter de ponta a ponta contra uma API real — latência por estágio, desfecho de cada observação (verified/rejected/pendente) e **fidelidade numérica**: os números presentes no input sobrevivem intactos até o fato final?

**Por que faz.** A métrica de fidelidade existe por causa de um risco real observado (gap 5): o NuExtract às vezes gera typos ou perde trechos na extração. Num sistema cuja promessa é "fatos confiáveis", corromper um número silenciosamente ("0,5%" virar "5%") é o pior modo de falha possível — então virou métrica de regressão.

**Como faz.** Roda inputs com números conhecidos pelo pipeline completo, extrai os números do fato resultante via regex e compara com os do input. Baseline registrado (notebook i7-1255U, DB de produção): **scout ~97s, filter ~151s por item, fidelidade 1.0**. Relatórios ficam em `tests/benchmark-results/`.

**Utilidade.** `python -m tests.benchmark_pipeline [--api-url URL] [--inputs N]`. Qualquer mudança de modelo/prompt tem um número objetivo para comparar antes/depois.

---

## 8. Apêndice: limiares e números que governam o sistema

| Valor | Onde | Significado |
|---|---|---|
| **0.95** | Filter | Similaridade ≥ 0.95 = mesmo fato → corrobora em vez de duplicar |
| **0.70–0.95** | Filter + Dreaming | Banda de "mesmo assunto, frase diferente" = candidato a relação/contradição |
| **0.96** | Dreaming | Limiar (mais conservador) para fundir fatos na consolidação |
| **0.5** | Synthesizer | Confiança mínima de um fato para entrar no contexto de resposta |
| **`(1−c)·e·0.5`** | API corroborate | Cada corroboração fecha metade da incerteza restante, ponderada pela evidência — converge a 1.0 sem nunca estourar |
| **2048** | Todo o sistema | Dimensão dos embeddings (MRL trunca os 4096 nativos; HNSW sobre halfvec exige <~4000) |
| **5 min / 24 h** | Heartbeat | Janela de "agente ativo" / poda de agentes sumidos |
| **10** | Dreaming | Cap de julgamentos LLM por tick (custo controlado; o loop volta em 3h) |
| **3 h** | Dreaming | Sono entre ticks do loop na VPS |
| **temperature 0.3 / 0.1** | Agentes | Raciocínio quase determinístico; 0.1 no fallback JSON (formato > criatividade) |

---

*Projeto de pesquisa — ROI Labs · Repositório: [JeanZorzetti/sem-swarm](https://github.com/JeanZorzetti/sem-swarm) · Última atualização: 2026-07-06*
