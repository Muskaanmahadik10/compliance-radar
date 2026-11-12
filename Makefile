PY ?= python

.PHONY: data index test lint api worker ui up down eval

data:
	$(PY) -m radar.data.build

index:
	$(PY) -m radar.retrieval.ingest

test:
	$(PY) -m pytest -q

lint:
	ruff check .

api:
	uvicorn radar.api:app --reload

worker:
	$(PY) -m radar.ingest.worker

ui:
	streamlit run ui/app.py

up:
	docker compose up --build -d

down:
	docker compose down

eval:
	$(PY) -m evaluation.retrieval
