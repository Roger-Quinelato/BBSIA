PYTHON ?= .venv/Scripts/python.exe
PIP ?= .venv/Scripts/pip.exe
UVICORN ?= .venv/Scripts/uvicorn.exe

.PHONY: install install-dev test lint format typecheck run reprocess solucoes-embedding

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install ruff mypy

test:
	PYTHONPATH=. $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy api.py rag_engine.py reprocess_worker.py catalogo_solucoes.py

run:
	$(UVICORN) api:app --host 0.0.0.0 --port 8000

reprocess:
	curl -X POST http://localhost:8000/reprocessar

solucoes-embedding:
	$(PYTHON) scripts/gerar_embeddings_solucoes.py
