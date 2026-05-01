import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import shutil

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from bbsia.app.core import (
    BASE_DIR,
    MAX_UPLOAD_SIZE_MB,
    UPLOAD_QUARANTINE_DIR,
    UploadMetadataRequest,
    _audit_event,
    _metadata_key_for_stored_filename,
    _reprocess_manager,
    _resolve_quarantine_source_path,
    _safe_approved_path,
    _safe_quarantine_path,
    _sha256_bytes,
    _record_event,
    _raise_http_exception,
    load_upload_metadata,
    normalize_upload_doc_name,
    reload_resources,
    save_upload_metadata,
    update_upload_metadata_entry,
    validate_pdf_upload,
)

router = APIRouter(prefix="", tags=["Admin", "Upload"])


@router.post("/reprocessar")
def reprocessar_base(request: Request, background_tasks: BackgroundTasks) -> dict[str, Any]:
    del background_tasks
    enqueue = _reprocess_manager.enqueue(reason="api_manual")
    if enqueue["status"] == "queued":
        _record_event("reprocess_queued", request, run_id=enqueue["run_id"], queue_size=enqueue["queue_size"])
        return JSONResponse(
            status_code=409,
            content={
                "status": "enfileirado",
                "detail": "Reprocessamento enfileirado para ser executado a seguir.",
                "run_id": enqueue["run_id"],
                "fila_tamanho": enqueue["queue_size"],
            },
        )
    _record_event("reprocess_started", request, run_id=enqueue["run_id"], queue_size=enqueue["queue_size"])
    return {
        "status": "iniciado",
        "mensagem": "Reprocessamento iniciado em background. Consulte /status para acompanhar.",
        "run_id": enqueue["run_id"],
    }


@router.post("/recarregar")
def recarregar_indice(request: Request) -> dict[str, str]:
    try:
        reload_resources()
        _record_event("index_reload", request, status="ok")
        return {"status": "ok", "mensagem": "Indice vetorial (Qdrant) recarregado em memoria."}
    except Exception as exc:
        _record_event("index_reload_failed", request, level=logging.ERROR, error=str(exc))
        _raise_http_exception(exc)
        raise


@router.post("/upload-legacy-disabled", include_in_schema=False)
def upload(
    files: list[UploadFile] = File(...),
    area: str | None = Form(default=None),
    assuntos: str | None = Form(default=None),
) -> dict[str, Any]:
    del files, area, assuntos
    raise HTTPException(status_code=410, detail="Endpoint legado de upload desativado.")


@router.post("/upload")
def upload_hardened(
    request: Request,
    files: list[UploadFile] = File(...),
    area: str | None = Form(default=None),
    assuntos: str | None = Form(default=None),
) -> dict[str, Any]:
    UPLOAD_QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    salvos: list[dict[str, Any]] = []
    parsed_assuntos = [a.strip() for a in assuntos.split(",") if a.strip()] if assuntos else ["geral"]
    metadata_area = (area or "geral").strip() or "geral"

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=422, detail="Nome de arquivo invalido.")
        if not file.filename.lower().endswith(".pdf"):
            _audit_event("upload_rejected", request, reason="invalid_extension", original_filename=file.filename)
            raise HTTPException(status_code=422, detail=f"Arquivo invalido: {file.filename}. Apenas .pdf e permitido.")

        content = file.file.read()
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            _audit_event("upload_rejected", request, reason="file_too_large", original_filename=file.filename, size_bytes=len(content))
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Arquivo '{file.filename}' excede o limite de {MAX_UPLOAD_SIZE_MB} MB "
                    f"({len(content) / (1024 * 1024):.1f} MB recebidos)."
                ),
            )

        if not content.startswith(b"%PDF-"):
            _audit_event("upload_rejected", request, reason="invalid_pdf_magic", original_filename=file.filename)
            raise HTTPException(status_code=422, detail=f"Arquivo invalido: {file.filename}. Assinatura PDF ausente.")

        destino = _safe_quarantine_path(file.filename)
        destino.write_bytes(content)
        try:
            rel_path = destino.relative_to(BASE_DIR).as_posix()
        except ValueError:
            rel_path = f"uploads/quarantine/{destino.name}"

        try:
            validation = validate_pdf_upload(destino)
        except TimeoutError as exc:
            destino.unlink(missing_ok=True)
            _audit_event("upload_rejected", request, reason="pdf_validation_timeout", original_filename=file.filename)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            destino.unlink(missing_ok=True)
            _audit_event("upload_rejected", request, reason="pdf_validation_failed", original_filename=file.filename, error=str(exc))
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        status = "quarantined_prompt_review" if validation.prompt_injection_findings else "quarantined_pending_review"
        metadata_extra = {
            "status": status,
            "original_filename": Path(file.filename).name,
            "stored_filename": destino.name,
            "quarantine_path": rel_path,
            "sha256": _sha256_bytes(content),
            "size_bytes": len(content),
            "page_count": validation.page_count,
            "extracted_chars": validation.extracted_chars,
            "prompt_injection_findings": validation.prompt_injection_findings,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        update_upload_metadata_entry(doc_name=rel_path, area=metadata_area, assuntos=parsed_assuntos, extra=metadata_extra)

        salvos.append(
            {
                "arquivo": destino.name,
                "original_filename": Path(file.filename).name,
                "status": status,
                "page_count": validation.page_count,
                "extracted_chars": validation.extracted_chars,
                "prompt_injection_findings": validation.prompt_injection_findings,
            }
        )
        _audit_event(
            "upload_quarantined",
            request,
            original_filename=file.filename,
            stored_filename=destino.name,
            status=status,
            sha256=metadata_extra["sha256"],
            prompt_injection_findings=validation.prompt_injection_findings,
        )

    return {
        "arquivos_salvos": [item["arquivo"] for item in salvos],
        "arquivos": salvos,
        "total": len(salvos),
        "mensagem": "Arquivos salvos em quarentena. Revise e aprove antes de mover para a base indexavel.",
    }


@router.post("/upload-metadata")
def upload_metadata(request: Request, payload: UploadMetadataRequest) -> dict[str, Any]:
    update_upload_metadata_entry(doc_name=payload.documento, area=payload.area, assuntos=payload.assuntos)
    _audit_event("upload_metadata_updated", request, documento=normalize_upload_doc_name(payload.documento))
    return {
        "status": "ok",
        "mensagem": "Metadados de upload cadastrados com sucesso.",
        "documento": normalize_upload_doc_name(payload.documento),
        "area": payload.area,
        "assuntos": payload.assuntos,
    }


@router.get("/admin/quarantine")
def list_quarantine(request: Request) -> dict[str, Any]:
    metadata = load_upload_metadata()
    itens: list[dict[str, Any]] = []
    for key, entry in metadata.items():
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "")).strip()
        if not status.startswith("quarantined_"):
            continue
        stored_filename = str(entry.get("stored_filename", Path(str(key)).name))
        quarantine_path = str(entry.get("quarantine_path", ""))
        resolved = _resolve_quarantine_source_path(quarantine_path, stored_filename)
        file_exists = resolved.exists() and resolved.is_file()
        itens.append(
            {
                "documento": key,
                "stored_filename": stored_filename,
                "original_filename": entry.get("original_filename", stored_filename),
                "status": status,
                "area": entry.get("area", "geral"),
                "assuntos": entry.get("assuntos", ["geral"]),
                "page_count": entry.get("page_count"),
                "size_bytes": entry.get("size_bytes"),
                "uploaded_at": entry.get("uploaded_at"),
                "prompt_injection_findings": entry.get("prompt_injection_findings", []),
                "quarantine_path": quarantine_path or f"uploads/quarantine/{stored_filename}",
                "file_exists": file_exists,
            }
        )

    itens.sort(key=lambda item: str(item.get("uploaded_at") or ""), reverse=True)
    _audit_event("admin_quarantine_listed", request, total=len(itens))
    return {"total": len(itens), "itens": itens}


@router.post("/admin/quarantine/{stored_filename}/approve")
def approve_quarantine_file(stored_filename: str, request: Request) -> dict[str, Any]:
    clean_name = Path((stored_filename or "").strip()).name
    if not clean_name:
        raise HTTPException(status_code=422, detail="Arquivo de quarentena invalido.")

    metadata = load_upload_metadata()
    source_key = _metadata_key_for_stored_filename(clean_name, metadata)
    if source_key is None:
        raise HTTPException(status_code=404, detail="Arquivo nao encontrado na quarentena.")

    entry = metadata.get(source_key)
    if not isinstance(entry, dict):
        raise HTTPException(status_code=404, detail="Metadado de quarentena invalido.")

    status = str(entry.get("status", "")).strip()
    if not status.startswith("quarantined_"):
        raise HTTPException(status_code=409, detail="Arquivo nao esta pendente em quarentena.")

    quarantine_path = str(entry.get("quarantine_path") or f"uploads/quarantine/{clean_name}")
    source_path = _resolve_quarantine_source_path(quarantine_path, clean_name)
    quarantine_root = UPLOAD_QUARANTINE_DIR.resolve()
    if quarantine_root not in source_path.parents:
        raise HTTPException(status_code=400, detail="Caminho de quarentena invalido.")
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo fisico da quarentena nao encontrado.")

    approved_path = _safe_approved_path(clean_name)
    shutil.move(str(source_path), str(approved_path))

    approved_rel = f"uploads/approved/{clean_name}"
    updated = dict(entry)
    updated["status"] = "approved_pending_index"
    updated["approved_at"] = datetime.now(timezone.utc).isoformat()
    updated["approved_path"] = approved_rel
    updated.pop("quarantine_path", None)

    metadata.pop(source_key, None)
    metadata[approved_rel] = updated
    save_upload_metadata(metadata)

    _audit_event("upload_approved", request, stored_filename=clean_name, source_key=source_key, approved_path=approved_rel)

    return {
        "status": "ok",
        "mensagem": "Arquivo aprovado e movido para uploads/approved.",
        "arquivo": clean_name,
        "approved_path": approved_rel,
    }

