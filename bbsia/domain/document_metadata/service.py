from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPLOADS_DIR = REPO_ROOT / "uploads"
DEFAULT_UPLOAD_QUARANTINE_DIR = DEFAULT_UPLOADS_DIR / "quarantine"
DEFAULT_UPLOAD_APPROVED_DIR = DEFAULT_UPLOADS_DIR / "approved"
DEFAULT_UPLOAD_METADATA_FILE = DEFAULT_UPLOADS_DIR / "metadata_uploads.json"


def normalize_upload_doc_name(doc_name: str) -> str:
    normalized = (doc_name or "").strip().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return "uploads/desconhecido.pdf"
    if parts[0] == "uploads":
        return "/".join(parts)
    return f"uploads/{parts[-1]}"


def load_upload_metadata(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    metadata_path = Path(path) if path is not None else DEFAULT_UPLOAD_METADATA_FILE
    if not metadata_path.is_absolute():
        metadata_path = REPO_ROOT / metadata_path
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_upload_metadata(data: dict[str, dict[str, Any]], path: str | Path | None = None) -> None:
    metadata_path = Path(path) if path is not None else DEFAULT_UPLOAD_METADATA_FILE
    if not metadata_path.is_absolute():
        metadata_path = REPO_ROOT / metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_upload_metadata_entry(
    doc_name: str,
    area: str,
    assuntos: list[str],
    extra: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> None:
    metadata = load_upload_metadata(path)
    key = normalize_upload_doc_name(doc_name)
    entry: dict[str, Any] = {
        "area": area.strip(),
        "assuntos": [a.strip() for a in assuntos if a.strip()] or ["geral"],
    }
    if extra:
        entry.update(extra)
    metadata[key] = entry
    save_upload_metadata(metadata, path)
