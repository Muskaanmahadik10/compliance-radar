# Keywords this project lets you claim honestly (for CV, LinkedIn, 104/Cake)

Only what is implemented and can be shown in the repository. Items that were planned but not built (RAGAS, DeepEval,
working Langfuse/OpenTelemetry tracing, Kubernetes, Terraform, mypy, pre-commit, uv, pgvector, multilingual evaluation,
machine unlearning) are deliberately left out.

**LLM / GenAI:** LLM applications, RAG (retrieval-augmented generation), hybrid search (dense + BM25), cross-encoder reranking, embeddings, vector database (Qdrant), structured outputs, function/tool calling, LangGraph (parallel fan-out, interrupt/resume, checkpoints), LangChain, MCP (Model Context Protocol), OpenAI-compatible APIs, Groq, model routing and cost tracking, query decomposition, citation checking, human-in-the-loop review, prompt-injection redaction of retrieved text, e-mail/phone masking.

**ML:** fine-tuning, LoRA/PEFT, Hugging Face Transformers, text classification (XLM-RoBERTa), classifier cascades (rules -> fine-tuned encoder -> LLM), evaluation metrics (accuracy, macro-F1, hit@k, MRR).

**Data / backend:** Python, FastAPI, pydantic, SSE streaming, PostgreSQL, SQLite, SQLAlchemy, SQL, Redis Streams (consumer groups), deduplication (MinHash LSH), Streamlit.

**MLOps / DevOps:** Docker, Docker Compose, GitHub Actions CI (two jobs: embedded services and real Qdrant/Redis/Postgres containers), pytest, ruff, mock-model testing.

**Domain:** manufacturing compliance, supplier risk, RoHS/REACH/PFAS, export control, BOM, audit trail, document erasure from retrieval.
