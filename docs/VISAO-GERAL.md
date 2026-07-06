# SEM-Swarm — Visão Geral

> **Documento geral do projeto.** Para o detalhamento parte a parte (o que cada arquivo/módulo faz, por que, como e sua utilidade), veja o [Documento Executivo](DOCUMENTO-EXECUTIVO.md).

---

## 1. O que é o SEM-Swarm em uma frase

O SEM-Swarm (**S**elf-Distilling Swarm Intelligence with **E**pistemic **M**emory) é um **enxame de pequenas IAs locais** que colaboram sem um "chefe" central: cada agente lê o mundo, deposita o que viu numa **memória compartilhada**, e essa memória — não um modelo gigante — é o que coordena o grupo e acumula conhecimento verificado ao longo do tempo.

## 2. O problema que ele resolve

Modelos de linguagem grandes (GPT, Claude, Gemini) são caros, dependem de nuvem e "esquecem" tudo entre conversas. Modelos pequenos (SLMs, de 1 a 14 bilhões de parâmetros) rodam de graça num notebook, mas individualmente são fracos: alucinam, erram JSON, têm raciocínio raso.

O SEM-Swarm aposta que a solução não é um modelo pequeno melhor, e sim uma **arquitetura melhor em volta de vários modelos pequenos**:

1. **Nenhum modelo faz tudo** — cada um recebe só a tarefa em que é comprovadamente bom (princípio do *Privilégio Cognitivo Mínimo*).
2. **O conhecimento não mora nos modelos** — mora num banco de dados com "fatos verificados", que sobrevive a reinícios, pode ser inspecionado e corrigido.
3. **O sistema se autocorrige** — evidências repetidas reforçam fatos (consenso), fatos contraditórios são julgados e o perdedor é expurgado (auto-destilação).

O resultado é um sistema de conhecimento que **fica mais confiável quanto mais é usado**, rodando inteiramente em hardware próprio (um notebook + uma VPS barata), sem nenhuma API paga.

## 3. As três ideias-chave (didático)

### 3.1 Estigmergia — coordenação pelo ambiente, não por ordens

Formigas não têm gerente. Elas se coordenam deixando feromônios no chão: uma formiga altera o ambiente, e as outras reagem a essa alteração. Isso se chama **estigmergia**.

No SEM-Swarm, o "chão" é o banco de dados (a Memória Epistêmica). O Scout deposita observações; o Filter as encontra lá e as julga; o Synthesizer consulta os fatos aprovados. **Nenhum agente fala diretamente com outro** — todos leem e escrevem no mesmo substrato. Isso torna o sistema tolerante a falhas (um agente pode morrer e outro assume) e trivialmente escalável (basta ligar mais agentes apontando pro mesmo banco).

### 3.2 Privilégio Cognitivo Mínimo — cada modelo faz só o que faz bem

Assim como segurança usa "privilégio mínimo" (cada processo só acessa o que precisa), aqui cada modelo recebe **exatamente a capacidade intelectual que sua tarefa exige**:

| Modelo | Tamanho | Especialidade | Papel no enxame |
|---|---|---|---|
| `phi4-mini` | 3.8B | Raciocínio, tool calling, PT-BR | Cérebro do Scout e do Filter |
| `nuextract` | 3.8B | **Só** extração de JSON estruturado | Converte raciocínio em JSON confiável |
| `qwen3-embedding` | 8B | Transformar texto em vetores (100+ idiomas) | "Senso de similaridade" do enxame (na VPS) |
| `qwen2.5:7b-instruct` / `deepseek-r1` | 7–14B | Raciocínio profundo, sem pressa | Juiz do Dreaming Loop (na VPS) |
| `qwen2.5-coder:1.5b` | 1.5B | Autocomplete de código | Só IDE — **nunca** orquestra nada |

O dado que motivou essa separação: modelos generalistas com menos de 10B acertam apenas **28–33%** das extrações de JSON puro; o NuExtract, treinado sinteticamente só para isso, chega perto de **100%**. Em vez de brigar com essa limitação, a arquitetura a contorna: quem raciocina não formata, quem formata não raciocina.

### 3.3 Memória Epistêmica — conhecimento com ciclo de vida

"Epistêmica" vem de *episteme* (conhecimento justificado). A memória do enxame não é um amontoado de textos: cada informação tem um **ciclo de vida com estados explícitos**, do boato ao fato consolidado:

```
                 (Scout deposita)
  texto bruto ──► OBSERVAÇÃO [pending]
                        │ (Filter pega o lote)
                        ▼
                  [processing]
                   │         │
        reprovada  │         │  aprovada
                   ▼         ▼
             [rejected]   é quase idêntica (≥0.95) a um fato existente?
                              │                │
                          sim │                │ não
                              ▼                ▼
                    CORROBORAÇÃO          FATO NOVO [verified]
                 (fato ganha +confiança)  (com embedding 2048d)
                              │                │
                              └───────┬────────┘
                                      ▼
                          DREAMING LOOP (madrugada, VPS)
                       ├─ funde fatos redundantes (≥0.96)
                       ├─ julga contradições via LLM
                       └─ perdedor vira superseded (inativo)
```

Cada fato carrega um `confidence_score` (0 a 1), um contador de `corroborations` e ponteiros de linhagem (`source_observation_id`, `superseded_by`). Nada é deletado — fatos derrotados ficam inativos com o registro de quem os substituiu, formando uma trilha auditável.

## 4. Analogia geral: uma redação de jornal

- O **Scout** é o repórter: sai a campo, volta com anotações brutas ("observações").
- O **Filter** é o editor de checagem: lê cada anotação, rejeita fofoca e opinião, e o que sobrevive vira **fato publicado** no arquivo do jornal.
- O **Synthesizer** é o colunista: quando alguém faz uma pergunta, ele consulta **só o arquivo de fatos verificados** e escreve uma resposta fundamentada — se o arquivo não cobre o assunto, ele diz que não sabe (em vez de inventar).
- O **Dreaming Loop** é o arquivista que trabalha de madrugada: funde reportagens repetidas numa só, encontra matérias que se contradizem e decide qual prevalece.

## 5. Topologia física: notebook + VPS

O sistema divide o trabalho entre duas máquinas comuns, segundo a natureza da tarefa:

```
NOTEBOOK (interativo, ~7–10 GB de RAM p/ modelos)     VPS BOSTON (24/7)
┌────────────────────────────────────┐    ┌─────────────────────────────────┐
│ Ollama local:                      │    │ Ollama VPS:                     │
│   phi4-mini (raciocínio)           │    │   qwen3-embedding (vetores)     │
│   nuextract (JSON)                 │    │   qwen2.5:7b (juiz do dreaming) │
│                                    │    │                                 │
│ Agentes Python:                    │    │ PostgreSQL + pgvector           │
│   scout / filter / synthesizer ────┼───►│ FastAPI (Memory API) + UI       │
└────────────────────────────────────┘    │ sem-dreaming (binário Rust)     │
                                          └─────────────────────────────────┘
```

- **Notebook**: tudo que é interativo e iterado com frequência (agentes, prompts).
- **VPS**: tudo que precisa estar sempre de pé (banco, API, embeddings) ou roda sem pressa (Dreaming Loop, que dorme 3h entre ciclos).

Em produção, API e Ollama da VPS estão públicos em `*.7c17iw.easypanel.host`, com **deploy automático por push** no repositório.

## 6. Por que Python **e** Rust?

Regra do projeto (formalizada no [ADR-001](architecture-decision-rust.md)): *"prove que funciona em Python, otimize com Rust onde dói"*.

- **Python** fica com os agentes e a API: são I/O-bound (99% do tempo esperando o Ollama responder), e prompt engineering exige iteração rápida.
- **Rust** fica com o que é CPU-bound e roda em RAM escassa: a matemática vetorial (`sem-vector`, com SIMD e paralelismo real, ~10–15× mais rápido) e o Dreaming Loop (`sem-dreaming`, binário de ~8 MB de RAM contra ~120 MB de um processo Python equivalente).
- Se a extensão Rust não estiver compilada, `core/vector_ops.py` tem **fallback em Python puro** com a mesma interface — o sistema degrada, mas nunca quebra.

## 7. O que já foi provado (estado em julho/2026)

Todos os sprints do roadmap (0 a 3) estão concluídos e validados **em produção**:

| Capacidade | Evidência |
|---|---|
| Pipeline ponta a ponta remoto | Scout → API pública → Filter → Synthesizer respondendo corretamente |
| Embeddings semânticos reais | qwen3-embedding 2048d (via truncamento MRL); dedup pegou quase-duplicata com similaridade 0.9888 |
| Consenso por corroboração | Observação redundante reforçou fato (#6→#1, sim 0.9793) em vez de virar lixo |
| Auto-destilação | Fato falso semeado de propósito ("porcelanato absorve >8%") foi **julgado e expurgado** pelo Dreaming Loop — o fato com corroboração venceu |
| Defesa em profundidade | O Filter já tinha rejeitado sozinho a observação falsa; ela só entrou porque foi injetada direto na API |
| Coordenação/presença | Heartbeat de agentes com janela de 5 min; UI mostra agentes vivos |
| Benchmark reproduzível | `tests/benchmark_pipeline.py`: scout ~97s, filter ~151s por item (CPU i7-1255U), fidelidade numérica 1.0 |

## 8. Mapa do repositório

| Caminho | O que é |
|---|---|
| `agents/` | Os três agentes Python (scout, filter, synthesizer) |
| `api/` | Memory API (FastAPI) + UI inspetora + Dockerfile |
| `core/` | Clients compartilhados: Ollama, NuExtract, Memory API, embeddings, ops vetoriais |
| `infra/init.sql` | Schema do banco (3 tabelas + índices pgvector) |
| `rust/sem-vector` | Extensão PyO3 de matemática vetorial (SIMD) |
| `rust/sem-dreaming` | Binário do Dreaming Loop (consolidação noturna) |
| `tests/` | Teste local do Scout + benchmark do pipeline |
| `data/offline_observations/` | Saída do Scout em modo offline |
| `docs/` | Este documento, o executivo e o ADR do Rust |
| `docker-compose.yml` | Stack: Postgres + API (+ Ollama com `--profile vps`) |

## 9. Utilidade — para que isso serve

1. **Pesquisa arquitetural**: é um laboratório de perguntas caras de responder em sistemas grandes — enxames sem coordenador central funcionam? Consenso estatístico derrota desinformação? SLMs especializados batem um generalista? (Até agora: sim, sim e sim.)
2. **Base de conhecimento autoverificável e soberana**: qualquer domínio que precise acumular fatos confiáveis a partir de texto bruto (documentos internos, atendimento, pesquisa de mercado) pode usar esse pipeline — 100% local, sem custo por token e sem enviar dados a terceiros.
3. **Padrões reutilizáveis**: o pipeline raciocínio→extração determinística, a fila `FOR UPDATE SKIP LOCKED`, o truncamento MRL para caber em `halfvec`, e o híbrido Python/Rust com fallback são receitas aplicáveis a qualquer outro projeto.

## 10. Glossário rápido

| Termo | Significado |
|---|---|
| **SLM** | Small Language Model — modelo de linguagem pequeno (1–14B parâmetros), roda em CPU |
| **Embedding** | Vetor de números que representa o *significado* de um texto; textos parecidos → vetores próximos |
| **Similaridade de cosseno** | Medida (0–1) de quão próximos dois embeddings estão; ~0.95+ = praticamente a mesma frase |
| **MRL** | Matryoshka Representation Learning — permite truncar um embedding (4096→2048 dims) mantendo o significado |
| **pgvector / halfvec / HNSW** | Extensão do Postgres para vetores / tipo de vetor em meia-precisão (economiza 50% de espaço) / índice de busca aproximada rápida |
| **Estigmergia** | Coordenação indireta: agentes se comunicam modificando o ambiente compartilhado |
| **Corroboração** | Evidência independente que confirma um fato existente e aumenta sua confiança |
| **Superseded** | Estado de um fato substituído (por consolidação ou por perder um julgamento de contradição) |
| **Dreaming Loop** | Processo assíncrono de "sono" do sistema: consolida, deduplica e resolve contradições da memória |

---

*Projeto de pesquisa — ROI Labs · Repositório: [JeanZorzetti/sem-swarm](https://github.com/JeanZorzetti/sem-swarm)*
