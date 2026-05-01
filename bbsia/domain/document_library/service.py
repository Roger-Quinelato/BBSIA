from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
BIBLIOTECA_FILE = DATA_DIR / "biblioteca.json"


def empty_document_library() -> dict[str, Any]:
    return {"versao": 1, "atualizado_em": "", "documentos": []}


def load_document_library(path: str | Path | None = None) -> dict[str, Any]:
    library_path = Path(path) if path is not None else BIBLIOTECA_FILE
    if not library_path.is_absolute():
        library_path = REPO_ROOT / library_path
    if not library_path.exists():
        return empty_document_library()

    try:
        data = json.loads(library_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("documentos"), list):
            return data
    except Exception:
        pass
    return empty_document_library()


def save_document_library(biblioteca: dict[str, Any], path: str | Path | None = None) -> None:
    library_path = Path(path) if path is not None else BIBLIOTECA_FILE
    if not library_path.is_absolute():
        library_path = REPO_ROOT / library_path
    library_path.parent.mkdir(parents=True, exist_ok=True)
    biblioteca["atualizado_em"] = datetime.now(timezone.utc).isoformat()
    library_path.write_text(
        json.dumps(biblioteca, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_document_metadata(metadado: Any, path: str | Path | None = None) -> None:
    biblioteca = load_document_library(path)
    docs = biblioteca.get("documentos", [])
    metadata = metadado.model_dump() if hasattr(metadado, "model_dump") else dict(metadado)

    docs = [
        doc
        for doc in docs
        if doc.get("id") != metadata.get("id")
        and doc.get("documento_original") != metadata.get("documento_original")
    ]
    docs.append(metadata)
    biblioteca["documentos"] = docs
    save_document_library(biblioteca, path)


def index_document_library(biblioteca: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    payload = biblioteca if biblioteca is not None else load_document_library()
    index: dict[str, dict[str, Any]] = {}
    for doc in payload.get("documentos", []):
        if not isinstance(doc, dict):
            continue
        original = str(doc.get("documento_original", "")).strip()
        if original:
            index[original] = doc
            index[os.path.basename(original)] = doc
    return index


carregar_biblioteca = load_document_library
salvar_biblioteca = save_document_library
atualizar_biblioteca = upsert_document_metadata
