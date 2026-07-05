# Handoff — E2E Sprint 1 (2026-07-05)

## Feito
Primeira validação ponta-a-ponta do pipeline contra o banco de PRODUÇÃO:

1. **Scout** (phi4-mini + nuextract locais) → extraiu 2 observações de 1 input e depositou via `POST /memory/observe` (obs #1 e #2).
2. **Filter** (phi4-mini + nuextract) → julgou as 2 pendentes, aprovou ambas, rodou dedup exata via Rust `sem_vector` (batch_cosine SIMD) e promoveu a fatos via `POST /memory/verify` (fatos #1 e #2, confiança 1.0).
3. **Synthesizer** → `POST /memory/search` recuperou os 2 fatos e gerou resposta com phi4-mini.
4. **DB prod**: observações `verified`, fatos ativos com FK correta, `swarm_state` atualizado pela API (`total_observations=2`, `total_verified_facts=2`). Encoding UTF-8 íntegro no banco.

A API usada foi a imagem do **Dockerfile raiz (mesma de prod)**, rodando local apontada pro DB de prod — porque a API deployada não tem rota pública (ver pendências).

## Decisões
- Dados de teste (obs/fatos #1 e #2 sobre porcelanato técnico) ficaram no banco — são inofensivos e servem de seed.

## Pendências / Gaps encontrados (em ordem de impacto)
1. **API sem domínio público no EasyPanel** — o serviço está deployado mas nenhum host responde (probes deram 404 no proxy). Sem isso, os agentes locais não alcançam a memória compartilhada de fora. → atribuir domínio no EasyPanel.
2. **`MemoryClient()` hard-coded `localhost:8000` no filter e synthesizer** — só o scout tem `--api-url`. Mesmo com domínio público, filter/synthesizer não conseguem apontar pra VPS. → ler `SEM_API_URL` do env no `MemoryClient`.
3. **Embeddings ainda são mock** (hash→random 2048d em `filter.py` e `synthesizer.py`) — busca semântica retorna ruído; dedup só pega texto idêntico; dreaming loop nunca vai formar cluster ≥0.96. → expor endpoint de embedding (Ollama VPS não responde externamente na porta padrão).
4. **Dreaming loop sem heartbeat** — nunca escreve `swarm_state.last_dreaming_loop_at` (coluna existe e está NULL); impossível confirmar de fora que o serviço deployado está vivo. → gravar timestamp a cada tick, mesmo quando não consolida.
5. **Qualidade**: nuextract gerou typos ("absorcição", "caracterição") e perdeu o trecho "adequado para áreas externas" na extração; o synthesizer então respondeu errado à pergunta sobre área externa (alucinou "não serve" apesar do prompt mandar usar só os fatos). Esperado com SLMs + contexto incompleto, mas é métrica a acompanhar no Sprint 3 (benchmarks).

## Gotchas
- Console Windows mostra mojibake nos logs dos agentes (emoji/acentos), mas os dados chegam UTF-8 corretos no banco — é só o encoding do terminal.
- Stack local do compose (`sem-api` + `sem-postgres`) ocupa a porta 8000; foi parada durante o E2E e restaurada ao final.

## Próximos
- Fechar gaps 1–2 (rota pública + `SEM_API_URL`) → daí o E2E roda 100% remoto.
- Trocar mock embedding por real quando houver endpoint de embedding acessível.
- UI: inspetor read-only mínimo (pending/fatos/swarm state) consumindo a API — depois dos gaps acima.
