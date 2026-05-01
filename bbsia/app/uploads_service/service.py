from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from bbsia.app.runtime.state import (
    BASE_DIR,
    MAX_PDF_EXTRACTED_CHARS,
    MAX_PDF_PAGES,
    PDF_VALIDATION_TIMEOUT_SEC,
    PROMPT_INJECTION_PATTERNS,
    UPLOAD_APPROVED_DIR,
    UPLOAD_QUARANTINE_DIR,
)

PDF_VALIDATION_SCRIPT = r"""
import json
import sys
import fitz

path = sys.argv[1]
max_pages = int(sys.argv[2])
max_chars = int(sys.argv[3])
patterns = json.loads(sys.argv[4])

extracted_chars = 0
findings = set()

try:
    doc = fitz.open(path)
except Exception as exc:
    print(f"PDF invalido ou corrompido: {exc}", file=sys.stderr)
    sys.exit(2)

try:
    page_count = int(doc.page_count)
    if page_count <= 0:
        print("PDF sem paginas validas.", file=sys.stderr)
        sys.exit(3)
    if page_count > max_pages:
        print(f"PDF excede o limite de {max_pages} paginas.", file=sys.stderr)
        sys.exit(4)

    for page in doc:
        text = page.get_text("text") or ""
        lower = text.lower()
        for pattern in patterns:
            if pattern in lower:
                findings.add(pattern)
        extracted_chars += len(text)
        if extracted_chars > max_chars:
            print(f"PDF excede o limite de {max_chars} caracteres extraidos.", file=sys.stderr)
            sys.exit(5)
finally:
    doc.close()

print(json.dumps({
    "page_count": page_count,
    "extracted_chars": extracted_chars,
    "prompt_injection_findings": sorted(findings),
}, ensure_ascii=False))
"""


@dataclass(frozen=True)
class PdfValidationResult:
    page_count: int
    extracted_chars: int
    prompt_injection_findings: list[str]


def validate_pdf_upload(path: Path, timeout_sec: int = PDF_VALIDATION_TIMEOUT_SEC) -> PdfValidationResult:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                PDF_VALIDATION_SCRIPT,
                str(path),
                str(MAX_PDF_PAGES),
                str(MAX_PDF_EXTRACTED_CHARS),
                json.dumps(PROMPT_INJECTION_PATTERNS, ensure_ascii=False),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Validacao do PDF excedeu {timeout_sec}s.") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Validacao do PDF falhou.").strip()
        raise ValueError(detail)

    try:
        payload = json.loads(result.stdout)
        return PdfValidationResult(
            page_count=int(payload["page_count"]),
            extracted_chars=int(payload["extracted_chars"]),
            prompt_injection_findings=[str(item) for item in payload.get("prompt_injection_findings", [])],
        )
    except Exception as exc:
        raise ValueError("Validacao do PDF retornou saida invalida.") from exc


def _safe_quarantine_path(original_filename: str) -> Path:
    extension = Path(original_filename).suffix.lower()
    if extension != ".pdf":
        extension = ".pdf"
    filename = f"{secrets.token_hex(16)}{extension}"
    UPLOAD_QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    destination = (UPLOAD_QUARANTINE_DIR / filename).resolve()
    quarantine_root = UPLOAD_QUARANTINE_DIR.resolve()
    if quarantine_root not in destination.parents:
        raise HTTPException(status_code=400, detail="Destino de upload invalido.")
    return destination


def _safe_approved_path(stored_filename: str) -> Path:
    name = Path((stored_filename or "").strip()).name
    if not name or Path(name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=422, detail="Nome de arquivo aprovado invalido.")
    UPLOAD_APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    destination = (UPLOAD_APPROVED_DIR / name).resolve()
    approved_root = UPLOAD_APPROVED_DIR.resolve()
    if approved_root not in destination.parents:
        raise HTTPException(status_code=400, detail="Destino de aprovacao invalido.")
    return destination


def _metadata_key_for_stored_filename(filename: str, metadata: dict[str, Any]) -> str | None:
    for key, entry in metadata.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("stored_filename") == filename:
            return key
        if Path(str(key)).name == filename:
            return key
    return None


def _resolve_quarantine_source_path(quarantine_path: str, stored_filename: str) -> Path:
    candidates: list[Path] = []
    if quarantine_path:
        qpath = Path(quarantine_path)
        if qpath.is_absolute():
            candidates.append(qpath)
        else:
            candidates.append(BASE_DIR / qpath)
    candidates.append(UPLOAD_QUARANTINE_DIR / stored_filename)

    quarantine_root = UPLOAD_QUARANTINE_DIR.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if quarantine_root in resolved.parents and resolved.exists():
            return resolved

    return (UPLOAD_QUARANTINE_DIR / stored_filename).resolve()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
