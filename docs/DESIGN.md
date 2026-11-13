# ComplianceRadar: design document

> **This is the plan the project was built from (Nov 2026), kept as written below §0.** Several planned items were
> not built or were built differently. §0 lists what exists in the repository today; the README is the source of truth
> for results. Where this plan and §0 disagree, §0 is correct.

**One line (as planned):** a real-time compliance and supplier-risk copilot for electronics manufacturers. It watches regulatory and news feeds, works out which of the company's own policies, contracts and suppliers are affected, and drafts an evidence-cited impact brief for the compliance owner to approve, in minutes instead of weeks.

## 0. What was actually built (updated after the 13 Nov 2026 audit)

| Planned | Built |
|---|---|
| Continuous feed polling, real-time | Feeds are polled **when a user presses Poll** (UI) or calls `POST /ingest/poll`. A Redis Streams worker classifies what was published. No scheduler is shipped, and nothing consumes `events:classified`: a person picks the event to brief. |
| Fine-tuned classifier + LLM fallback | Rules -> LoRA (optional) -> LLM. Since commit c5adf9e the worker, API and UI all run the LLM tier (cheap model); before that it ran only in `evaluation/classifier.py`. Measured on 52 hand-written **English** headlines, **topic only** (jurisdiction and urgency are not evaluated). |
| bge-m3 multilingual embeddings, bge-reranker-v2-m3 | Default is English `bge-small-en-v1.5` + MiniLM reranker; the multilingual models are a setting, not evaluated. |
| Critic "checks every claim has a citation" | Deterministic verifier: cited sections must be retrieved; supplier/part/product ids must be in the SQL result (names replaced with the record's); policy/contract entries cite only their own document; suppliers cite only their own contract; every action needs a valid citation; explicit totals in the summary must equal SQL. **Not checked deterministically:** verbatim quotes, whether a passage supports a claim, obligation direction, deadlines. That is left to the LLM critic and the human reviewer. Structural failures block automatic approval whatever the critic says. |
| Right to be forgotten that "proves" a document no longer influences answers | **Retrieval erasure with recorded checks:** a tombstone (respected by every ingest/rebuild path), vector chunks deleted, local source copies removed, SQL row marked deleted, then probe/citation checks recorded in an HMAC-signed certificate. The certificate is demo-signed (public key) unless `RADAR_SIGNING_KEY` is set. Briefs, tasks, checkpoints and chat history that quoted the document are listed or out of scope, not redacted. No model here was trained on the documents. |
| PII redaction before any external call | E-mail addresses and phone numbers only, on the question, history and retrieved passages (chat and briefs). Dates are preserved. |
| RAGAS, DeepEval, nightly real-model evals | Not built. Retrieval hit@k/MRR, classifier accuracy/F1 and a brief evaluation with narrow named checks exist; CI runs mock-model tests only. |
| Langfuse + OpenTelemetry traces | A callback is attached when keys are set; not verified end to end (the `RADAR_LANGFUSE_*` settings are not passed to the SDK handler). Treat as not working. |
| Role-based approval, audit log | Audit log yes. No authentication: reviewer names are typed text. Run locally only. |
| Durable review | SQLite/Postgres checkpointer; if it cannot open, an in-memory saver is used and recorded on each brief. Duplicate briefs per event are possible; task publication is not idempotent across a crash. |
| Pending-message recovery, transactional outbox | Not built (documented limitation). |
| `create_task` write tool, PDF export, HF Hub push, Next.js, Kubernetes, Terraform | Not built. Tasks are created by the graph's `publish` step after human approval. |

Owner: Muskaan Mahadik · Project 2 of 2 · Nov 2026 · Companion to [Factory Copilot](https://github.com/Muskaanmahadik10/factory-copilot)

---

## 1. Why this project (what a hiring manager sees)

| What they look for | Where it shows up |
|---|---|
| A use case that is obviously worth money | Every Taiwanese electronics manufacturer (LITEON, Pegatron, Wistron, Delta) runs RBA, RoHS/REACH, conflict-minerals, EU CSRD and export-control compliance across thousands of suppliers. Today a team reads PDFs and emails. Missing a change means a blocked shipment or a failed customer audit. |
| **Real-time**, not a batch notebook | Feeds stream in continuously (Redis Streams), the UI streams tokens (SSE), and the dashboard updates live. |
| The stack they already use | LangGraph, LangChain, Qdrant, FastAPI, Redis, Postgres, Langfuse, RAGAS, Docker Compose, GitHub Actions, OpenTelemetry, pydantic, MCP. See §4. |
| Production thinking | Tracing on every call, evals in CI, cost per brief, retries, idempotent ingestion, role-based approval, audit log. |
| Depth, not just glue | Hybrid retrieval with a cross-encoder reranker, query decomposition, multi-agent LangGraph with checkpointing and human-in-the-loop interrupts, structured outputs, a fine-tuned classifier (LoRA, Colab), and a **right-to-be-forgotten** module that removes a document and *proves* it no longer influences answers (my thesis topic, applied). |

## 2. Users and the job to be done

| User | Job |
|---|---|
| Compliance officer | "A new EU rule on PFAS dropped this morning. Which of our products, suppliers and contracts are affected, and what do I have to do by when?" |
| Procurement lead | "Supplier X was just added to an entity list / had a factory fire. Which open POs and parts depend on them?" |
| Head of AI / IT | "Show me it's accurate, auditable, cheap, and that confidential documents never leave our environment." |

## 3. Scope

**v1 (two weeks of evenings)**

1. **Ingestion (real-time):** pollers for official sources (EU Official Journal / EUR-Lex RSS, US Federal Register API, Taiwan MOEA/BSMI announcements, GDELT or news RSS) push normalised events into **Redis Streams**; a worker deduplicates (MinHash), language-detects, and classifies each item (topic, jurisdiction, urgency) with a **fine-tuned small classifier** plus an LLM fallback.
2. **Knowledge base:** the company's documents, generated for the demo but realistic: supplier master (200 suppliers, parts, countries, certifications), 40 policies/SOPs, 30 supplier contracts with clauses, 20 product BOMs. Chunked, embedded (`bge-m3`, multilingual: English and Traditional Chinese), stored in **Qdrant** with payload filters; **hybrid search** (dense + BM25 via Qdrant sparse vectors) and a **cross-encoder reranker** (`bge-reranker-v2-m3`).
3. **Agent (LangGraph):** a stateful graph with nodes `triage → decompose → research (parallel retrieval per sub-question) → map impact (which suppliers/parts/contract clauses) → draft brief → critic (checks every claim has a citation; loops back once) → human review (interrupt)`. Tools: `search_policies`, `search_contracts`, `lookup_suppliers` (SQL over Postgres), `lookup_bom`, `web_fetch_source` (allow-listed domains only), `create_task` (write; needs approval).
4. **Brief:** structured output (pydantic) with: summary, affected suppliers/parts/clauses with citations (doc id + page/chunk), required actions with deadlines, confidence, open questions. Rendered in the UI and exportable as PDF/Markdown.
5. **Chat over the knowledge base** with streaming answers, citations, conversation memory (Postgres checkpointer), and the same guardrails.
6. **Observability and evaluation:** **Langfuse** traces for every graph run (latency, tokens, cost per brief); **RAGAS** (faithfulness, answer relevancy, context precision/recall) on a 60-question golden set; retrieval metrics (hit@k, MRR) comparing dense vs hybrid vs hybrid+rerank; classifier F1; all run in **GitHub Actions** on every push with a mock LLM, and nightly with real models.
7. **Right-to-be-forgotten:** `DELETE /documents/{id}` removes chunks from Qdrant and rows from Postgres, invalidates caches, then runs a **forgetting verification**: the 10 golden questions that used that document are re-asked and must no longer cite it; a signed deletion certificate (JSON) is stored in the audit log.
8. **Security basics:** prompt-injection defence for fetched web content (untrusted wrapping + detector), allow-listed domains, least-privilege tools, approval gate, PII redaction before any external LLM call, secrets only via env.
9. **Deployment:** `docker compose up` brings up API, worker, Qdrant, Redis, Postgres, Langfuse, UI. Health checks, OpenTelemetry traces to Langfuse, structured JSON logs.

**v2 (if time):** Next.js UI instead of Streamlit, Slack/Teams notifications, Kubernetes manifests (Helm), a small Terraform module, LlamaIndex ingestion comparison.

**Out of scope:** legal advice, real company data, production SSO.

## 4. Stack and the reason for each choice

| Layer | Choice | Why this one (what to say in the interview) |
|---|---|---|
| Orchestration | **LangGraph** (+ LangChain core) | The most common production agent framework in 2026 job posts; gives explicit state, checkpoints, retries, parallel branches and human-in-the-loop interrupts. |
| Models | **Claude** (Anthropic API) for drafting/critique, **gpt-oss / Qwen via Groq or Ollama** for cheap steps; one `ModelRouter` as in Factory Copilot | Shows multi-provider routing by cost and capability; local model option for confidential data. |
| Embeddings / reranker | **bge-m3** + **bge-reranker-v2-m3** (local, multilingual) | English + Traditional Chinese documents; no data leaves the box; reranking is the single biggest retrieval win and is measurable. |
| Vector store | **Qdrant** | Hybrid dense+sparse in one engine, payload filtering (jurisdiction, supplier, date), snapshots; widely used; runs in Docker. |
| Relational | **PostgreSQL** (+ `pgvector` only for comparison) | Supplier master, BOMs, audit log, LangGraph checkpointer. SQL tool for the agent. |
| Streaming/queue | **Redis Streams** | Real-time ingestion with consumer groups and replay; simpler than Kafka for a portfolio, same concepts. |
| API | **FastAPI** with SSE streaming, pydantic v2 models | Standard; auto docs; async. |
| Observability | **Langfuse** (self-hosted) + **OpenTelemetry** | Trace every node, prompt version, cost; managers love the cost-per-brief chart. |
| Evaluation | **RAGAS**, custom retrieval metrics, **DeepEval**-style assertions in pytest | "LLM evals in CI" is a top interview topic. |
| Fine-tuning | **LoRA** on a small encoder (e.g. `xlm-roberta-base`) for the classifier, trained on **Colab** with Hugging Face `transformers` + `peft`; weights pushed to the **Hugging Face Hub** | Shows training, not just prompting; cheap at inference. |
| Agent interop | **MCP** server exposing the tools | Same pattern as project 1; Claude Desktop can drive it. |
| Packaging | **Docker Compose**, **GitHub Actions**, `ruff`, `mypy`, `pre-commit`, `uv` | Hygiene recruiters' ATS keywords match. |
| UI | **Streamlit** (v1) → Next.js (v2) | Fast first; upgrade shows full-stack. |

## 5. Architecture

```
 feeds (EUR-Lex RSS, Federal Register API, MOEA, news) ─► pollers ─► Redis Streams ─► worker
                                                                                        │ dedupe (MinHash) · lang · classify (LoRA clf → LLM fallback)
                                                                                        ▼
                                                                                 Postgres (events)
                                                                                        │ new relevant event
                                                                                        ▼
   company docs ─► ingest ─► chunk ─► bge-m3 ─► Qdrant (dense+sparse, payloads)   LangGraph run (checkpointed)
   supplier master / BOM / contracts ─► Postgres                                        triage → decompose → research ∥ → map impact → draft → critic ↺ → HUMAN REVIEW → publish
                                                                                        │ tools: search_policies · search_contracts · lookup_suppliers(SQL) · lookup_bom · web_fetch(allow-list) · create_task(write)
                                                                                        ▼
                                                                        Impact Brief (pydantic) + citations + cost ─► UI (SSE) · PDF · audit log
   Langfuse ◄── OpenTelemetry traces from every node and tool ──┘        RAGAS + retrieval evals in CI · forgetting verification on delete
```

## 6. Evaluation plan (numbers for the README)

| What | Metric | Target |
|---|---|---|
| Retrieval | hit@5, MRR on 60 golden Q→chunk pairs: dense vs hybrid vs hybrid+rerank | rerank ≥ +10 pts hit@5 over dense |
| Answers | RAGAS faithfulness, answer relevancy, context precision | faithfulness ≥ 0.9 |
| Briefs | citation coverage (claims with a valid citation / claims), impact recall (affected suppliers found / truth) on 20 scripted events | coverage ≥ 0.95, recall ≥ 0.9 |
| Classifier | macro-F1 on held-out events, LoRA vs zero-shot LLM, cost per 1k events | F1 ≥ 0.85 at <1% of LLM cost |
| Real-time | event → brief p50/p95 latency | p50 < 90 s |
| Cost | USD per brief, per chat answer (from Langfuse) | reported, with the router's savings |
| Forgetting | after deletion, 0/10 answers cite the removed document; chunk count = 0 | 100% |
| Safety | 10 injected web pages: actions unchanged | 10/10 |

## 7. Milestones (vertical slices, each ends runnable and tested)

| # | Slice | Done when |
|---|---|---|
| 1 | Repo, Compose (Qdrant, Redis, Postgres, Langfuse), settings, CI skeleton | `docker compose up` healthy; `pytest` green |
| 2 | Synthetic company data + ingestion into Qdrant/Postgres | 40 policies, 30 contracts, 200 suppliers, 20 BOMs indexed; hybrid search returns citations |
| 3 | Retrieval evaluation harness | hit@5/MRR table for dense / hybrid / rerank |
| 4 | Chat with streaming + citations + Langfuse traces | SSE endpoint, Streamlit chat, traces visible |
| 5 | Feed pollers → Redis Streams → worker (dedupe, classify) | live events appear in UI; classifier F1 reported |
| 6 | LangGraph brief pipeline with critic loop and human interrupt | one event → approved brief end to end |
| 7 | RAGAS + brief evals in CI (mock) and nightly (real) | results JSON + README table |
| 8 | Right-to-be-forgotten with verification certificate | delete → verified → certificate in audit log |
| 9 | LoRA classifier on Colab, pushed to HF Hub, loaded by worker | F1 and cost comparison in README |
| 10 | MCP server, PDF export, polish, demo GIF, README | published |

## 8. Honesty notes to keep in the README

Company documents are synthetic (realistic, generated for the demo); regulatory feeds are real public sources; evaluation sets are built from the generator's ground truth; mock-LLM runs test the pipeline, real-model runs give the headline numbers; this is not legal advice.
