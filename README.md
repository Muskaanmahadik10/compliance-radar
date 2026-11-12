# ComplianceRadar

**A real-time compliance and supplier-risk copilot for electronics manufacturers.** It watches regulatory and news feeds, works out which of the company's suppliers, parts, products and contract clauses a change affects, and drafts an impact brief in which every claim is cited. Numbers come from SQL, reasoning from an LLM, a verifier removes anything the model made up, and a person approves before any task is issued.

![CI](https://github.com/Muskaanmahadik10/compliance-radar/actions/workflows/ci.yml/badge.svg)

Built with **LangGraph · LangChain · Qdrant (hybrid dense + BM25) · cross-encoder reranking · Redis Streams · PostgreSQL · FastAPI (SSE) · MCP · Docker Compose · GitHub Actions**. Runs fully offline on a laptop with no keys (`RADAR_MODE=local`, mock LLM); the same code runs against real services and real models.

## The problem

A manufacturer with thousands of suppliers has to react to a steady stream of changes: a PFAS restriction, a RoHS exemption expiring, a new SVHC on the REACH Candidate List, a supplier added to the US Entity List, a factory fire. Today someone reads the notice, searches policies and contracts, queries the supplier master and BOMs, and writes an assessment. That takes days, and a missed change means a blocked shipment or a failed customer audit.

## What it does

```
 feeds (EUR-Lex, ECHA, Federal Register, news, replay fixture)
        │ pollers
        ▼
   Redis Stream  ──►  worker: dedupe (id + MinHash LSH) · language · classify (rules → LLM fallback → LoRA)
                              │ relevant event
                              ▼
   LangGraph pipeline (durable checkpoints)
   triage ─► decompose ─► research ×N in parallel ─► map impact (SQL) ─► draft ─► verify + critic ─┬─► HUMAN REVIEW ─► publish tasks
              (cheap model)   (hybrid search + rerank)   suppliers·parts·BOMs   (strong model)  ▲        │   (interrupt / resume)
                                                                                                └ revise ┘
   Also: streaming chat over the knowledge base with citations · verified document erasure with a signed certificate · MCP server
```

![Brief review screen: RoHS exemption expiry, 23 suppliers affected, critic feedback, cited actions](docs/img/brief_review.jpg)

**One real run** (the EU removes RoHS exemption 7(a) for lead in high-melting-temperature solder): the brief reports 23 suppliers, 11 parts and 10 products from SQL, lists the priority suppliers by id, and proposes owned actions with deadlines quoted from the cited policy and contract sections. The larger model had hit its daily quota, so the router fell back to the smaller one and recorded that on the brief. The critic did not approve it: one action cited SOP-102 §1 where §2 is the section that supports it. That note goes to the human reviewer, who approved it in the UI. Cost: **$0.0026**.

## Results

All numbers are produced by scripts in `evaluation/` and stored in `artifacts/`.

### Retrieval: does hybrid search and reranking earn its keep?

70 questions against 301 chunks. *Direct* questions reuse the documents' vocabulary; *paraphrased* questions avoid it ("forever chemicals" for PFAS, "blacklisted vendor" for Entity List).

| Method | Direct hit@1 (n=45) | Paraphrased hit@1 (n=25) | Paraphrased hit@5 | Paraphrased MRR | p50 latency |
|---|---|---|---|---|---|
| Dense (bge-small) | 0.98 | 0.64 | 0.76 | 0.69 | 12 ms |
| Sparse (BM25) | 0.96 | 0.40 | 0.60 | 0.47 | 16 ms |
| Hybrid (RRF fusion) | 1.00 | 0.52 | **0.84** | 0.65 | 17 ms |
| **Hybrid + cross-encoder rerank** | 1.00 | **0.76** | **0.84** | **0.79** | 430 ms |

Reading: on easy questions everything works, which is why the harder set exists. Fusion widens what is found (hit@5), reranking puts the right passage first (hit@1 +12 points over dense, +36 over BM25) at the price of latency.

### Event classification: rules vs LLM fallback (52 hand-written headlines, 13 topics)

| Tier | Accuracy | Macro-F1 | Cost for 52 headlines |
|---|---|---|---|
| Keyword rules | 0.60 | 0.62 | $0 |
| **Rules, LLM decides low-confidence items** | **0.94** | **0.94** | $0.0007 (24 LLM calls) |
| LoRA fine-tuned encoder (`colab/train_classifier.ipynb`) | run the notebook | | $0 at inference |

### Impact briefs (11 events, ground truth written by hand in `evaluation/briefs.py`)

| Metric | Full run, 11 events | After verifier and prompt refinements, 3-event sample |
|---|---|---|
| Supplier impact exactly matches ground truth | **11 / 11** | 3 / 3 |
| Governing policy is among the citations | **11 / 11** | 3 / 3 |
| Every action carries a valid citation | **11 / 11** | 3 / 3 |
| Invented supplier/part ids or citations outside the evidence | **0** | 0 |
| Wrong-supplier contract citations removed by the verifier | (check did not exist yet) | 5 across 3 drafts; 0 in a further run after the prompt fix |
| LLM critic approved (after at most one revision) | 7 / 11 | 3 / 3 |
| Fallback briefs (model produced nothing usable) | 0 | 0 |
| Mean cost per brief / median time | $0.0034 / 71 s | $0.0028 / 74 s |

Models: gpt-oss-20b (decomposition, rewriting) and gpt-oss-120b (drafting, critique) on Groq's free tier; `artifacts/brief_eval_env.json` and `brief_eval_env_sample3.json`.

Reading the critic column: in the full run it rejected 4 briefs. Two were real reasoning errors (a clause cited in the wrong direction; one supplier's contract used as evidence about another). Two were over-strict (asking for citations on facts restated from the event or on database ids). I turned the first kind into a deterministic check that removes the citation, told the critic what counts as a source, and tightened the drafting prompt; on the sample that followed, the critic approved all three. The refinements were validated on 3 events rather than all 11 because the free tier's daily token quota (200k) was exhausted by the first run; `python -m evaluation.briefs --provider env` reproduces the full run.

The mock-model run (`artifacts/brief_eval_mock.json`, part of CI) passes 11 / 11 on every row; it checks the pipeline, not model quality.

### Verified erasure ("right to be forgotten")

![Erasure verified in the UI with its signed certificate](docs/img/erasure_certificate.jpg)

`DELETE /documents/{id}` removes a document from the vector store, database, disk and caches, then proves it: zero chunks left, no probe question (built from the document's own wording) retrieves it in dense, sparse or hybrid mode, no answer cites it, a full collection scan is clean. The result is an HMAC-signed certificate in the audit log; tampering with any field invalidates the signature (tested). Scope stated in the certificate: this is retrieval-store erasure; a model fine-tuned on the documents would also need retraining or unlearning.

## Design decisions

| Decision | Why |
|---|---|
| **SQL counts, LLM prose.** The model's output schema has no field for counts or the event id. | A wrong supplier count is the most damaging error a brief can make, so it is not the model's to make. |
| **Verifier before critic.** Cited sections must exist in the retrieved evidence; supplier and part ids must exist in the SQL result; violations are removed and reported. | Deterministic checks are free and exact; the LLM critic then judges only whether passages support the claims. |
| **System-of-record vs evidence** in the prompts. | First real runs hid supplier ids because "everything needs a citation". Separating the two kinds of input fixed it. |
| **Function-calling structured output with retries and a labelled fallback brief.** | Strict JSON-schema mode rejected optional fields; hosted reasoning models occasionally answer in prose. The pipeline must never crash or invent. |
| **LangGraph with `Send`, a revise loop and `interrupt`.** Checkpoints in SQLite (local) or Postgres. | Parallel research, a bounded critic loop, and a human decision that can arrive a day later. |
| **Two runtime modes behind one interface** (embedded Qdrant + fakeredis + SQLite, or real services). | Anyone can run it in a minute; CI runs both, the second against real Qdrant, Redis and Postgres containers. |
| **Local embeddings, BM25 and reranker (ONNX).** | Company documents never leave the machine for retrieval; only selected passages reach the LLM, after PII masking. |
| **Retry policy that reads the provider's answer, plus model fallback.** Per-minute limits are waited out; a daily quota that clears in half an hour is not, and the request goes to the smaller model, which is recorded on the brief. Oversized requests are never retried. | Found the hard way: retrying a daily-quota error three times cost 85 seconds per brief and produced nothing. |
| **Mock chat model implementing the LangChain interface.** | 33 tests and the mock evaluation need no key, so prompt-format and schema bugs fail in CI. |

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate        # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
make data index       # generate the demo company (200 suppliers, 715 parts, 20 products, 52 documents) and index it
make test             # 33 tests, offline
streamlit run ui/app.py
```

In the UI: **Live events → Poll feeds (replay) → Draft brief → Briefs → Approve**. Then try the chat and erase a document under **Documents**.

Real models: copy `.env.example` to `.env`, set `RADAR_LLM_PROVIDER=openai_compatible` (Groq, OpenAI, Ollama, vLLM) or `anthropic` and the key. Real services: `docker compose up --build` (Qdrant, Redis, Postgres, Langfuse, API, worker, UI).

Evaluations: `python -m evaluation.retrieval` · `python -m evaluation.classifier` · `python -m evaluation.briefs --provider env`.

MCP: `python -m radar.mcp_server` exposes search, supplier and BOM lookup, events and brief drafting to Claude Desktop / Claude Code. There is deliberately no tool to approve a brief.

## Repository map

```
radar/
  config.py · infra/        settings; Qdrant / Redis / SQL clients for both modes
  data/                     schemas (pydantic), synthetic company generator, material aliases
  db/                       SQLAlchemy tables: suppliers, parts, BOM, documents, events, briefs, tasks, audit log
  retrieval/                chunking with section citations, Qdrant hybrid store, reranker, ingestion
  ingest/                   feed pollers, Redis Streams worker, MinHash dedupe, tiered classifier
  llm/                      model router, cost ledger (LangChain callback), structured-call retries, mock model
  tools/                    schema-validated, permission-tagged tools with injection redaction
  agent/                    LangGraph pipeline, prompts, service (start / decide / read)
  chat.py · forget.py       streaming RAG chat with citation check · verified erasure with signed certificate
  api.py · mcp_server.py    FastAPI (SSE) · MCP server
ui/app.py                   Streamlit: events, briefs, chat, documents, metrics
evaluation/                 retrieval, classifier and brief evaluations
colab/train_classifier.ipynb   LoRA fine-tuning of the event classifier
docs/                       DESIGN.md (the plan this was built from), INTERVIEW_NOTES.md
tests/                      33 tests
```

## Honest notes

- Company data (suppliers, BOMs, policies, contracts) is synthetic and generated for the demo; regulatory feeds are real public sources, and a replay fixture keeps tests deterministic. Not legal advice.
- Mock-model results check the pipeline, not model quality. Headline numbers are labelled with the models that produced them.
- The default embedding model is English (`bge-small`) for a small download; set `RADAR_EMBEDDING_MODEL=BAAI/bge-m3` for multilingual retrieval. The Traditional Chinese event above was handled by the classifier's aliases and the LLM, not by multilingual retrieval.
- Langfuse tracing is wired (callback added when keys are set) but not exercised in CI.

## Author

Muskaan Mahadik · M.S. Intelligent Manufacturing Technology, Taiwan Tech · thesis on quantization and machine unlearning in LLMs · see also [Factory Copilot](https://github.com/Muskaanmahadik10/factory-copilot) and [DocChat](https://github.com/Muskaanmahadik10/docchat).
