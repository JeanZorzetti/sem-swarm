# Handoff — E2E Sprint 1 + fechamento de gaps (2026-07-05)

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

## Pendências
1. **Ops — redeploy do serviço `sem_swarm_dreaming` no EasyPanel** pra pegar o commit do heartbeat, e setar no env dele `OLLAMA_DEEP_REASONING_MODEL=qwen2.5:7b-instruct` (nem `deepseek-r1:14b` nem o default `phi4-mini` estão pulled no Ollama da VPS).
2. ~~Gap 3~~ **FECHADO** (rodada 3 acima).
3. **Gap 5** (qualidade): nuextract gera typos e perde trechos na extração (ex.: perdeu "adequado para áreas externas" na rodada 1) — métrica pros benchmarks do Sprint 3.
4. UI: inspetor read-only mínimo (pending/fatos/swarm state) consumindo a API pública.
5. (Opcional, se o notebook continuar sofrendo) fallback pra rodar o raciocínio do filter na VPS (`qwen2.5:7b-instruct`) — a iGPU Iris Xe não ajuda: "11,8 GB" é RAM compartilhada, mesma banda da CPU.

## Gotchas
- Console Windows mostra mojibake nos logs dos agentes; os dados chegam UTF-8 corretos no banco.
- Toolchain Rust local (Windows/GNU) falha com `dlltool not found` em deps — validar o crate sempre via docker build do `rust/sem-dreaming/Dockerfile` (mesmo caminho do EasyPanel).
- O container do dreaming nunca "termina": o Dockerfile envolve o binário em loop com sleep de 3h — é o design do serviço, não travamento.
