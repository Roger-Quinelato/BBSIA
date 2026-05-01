from __future__ import annotations

import logging
import time
from typing import Any

from bbsia.app.runtime.audit import _record_event
from bbsia.rag.ingestion.chunking import run_chunking
from bbsia.rag.ingestion.embedding import run_embedding
from bbsia.rag.ingestion.extrator import run_extraction
from bbsia.rag.ingestion.worker import ReprocessWorker
from bbsia.rag.public_api.engine import reload_resources


def _run_reprocess_pipeline(mark_step: Any) -> None:
    stages: list[tuple[str, Any]] = [
        ("extracao", run_extraction),
        ("chunking", run_chunking),
        ("embedding", run_embedding),
        ("reload", reload_resources),
    ]

    for stage_name, stage_fn in stages:
        mark_step(stage_name)
        stage_started = time.perf_counter()
        _record_event("reprocess_stage_started", None, stage=stage_name)
        try:
            stage_result = stage_fn()
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - stage_started) * 1000)
            _record_event(
                "reprocess_stage_failed",
                None,
                level=logging.ERROR,
                stage=stage_name,
                duration_ms=elapsed_ms,
                error=str(exc),
            )
            raise

        elapsed_ms = int((time.perf_counter() - stage_started) * 1000)
        metrics: dict[str, Any] = stage_result if isinstance(stage_result, dict) else {}
        _record_event(
            "reprocess_stage_completed",
            None,
            stage=stage_name,
            duration_ms=elapsed_ms,
            metrics=metrics,
        )

        for item in metrics.get("documento_erros", []):
            if not isinstance(item, dict):
                continue
            _record_event(
                "reprocess_document_error",
                None,
                level=logging.ERROR,
                stage=stage_name,
                documento=item.get("documento", "desconhecido"),
                error=item.get("erro", "erro nao informado"),
            )


def _build_reprocess_manager() -> ReprocessWorker:
    def _on_worker_event(event: str, details: dict[str, Any]) -> None:
        _record_event(event, None, **details)

    return ReprocessWorker(pipeline=_run_reprocess_pipeline, on_event=_on_worker_event)


_reprocess_manager = _build_reprocess_manager()
