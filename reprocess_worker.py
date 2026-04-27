from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable


class ReprocessWorker:
    """Gerencia reprocessamentos em fila com execução serial e estado thread-safe."""

    def __init__(
        self,
        pipeline: Callable[[Callable[[str], None]], None],
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._on_event = on_event
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._queue: deque[int] = deque()
        self._worker_thread: threading.Thread | None = None

        self._running = False
        self._current_step: str | None = None
        self._last_step: str | None = None
        self._error: str | None = None
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._current_run_id: int | None = None
        self._next_run_id = 1
        self._completed_runs = 0

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._on_event:
            self._on_event(event, payload)

    def _ensure_worker_locked(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="reprocess-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def enqueue(self, reason: str = "manual") -> dict[str, Any]:
        with self._condition:
            run_id = self._next_run_id
            self._next_run_id += 1
            self._queue.append(run_id)
            queue_len = len(self._queue)
            already_busy = self._running or queue_len > 1
            self._ensure_worker_locked()
            self._condition.notify()

        status = "queued" if already_busy else "started"
        self._emit(
            "reprocess_enqueued",
            {
                "run_id": run_id,
                "reason": reason,
                "status": status,
                "queue_size": queue_len,
            },
        )
        return {
            "status": status,
            "run_id": run_id,
            "queue_size": queue_len,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            queue_size = len(self._queue)
            worker_alive = bool(self._worker_thread and self._worker_thread.is_alive())
            return {
                "rodando": self._running,
                "pendente": queue_size > 0,
                "fila_tamanho": queue_size,
                "worker_ativo": worker_alive,
                "etapa_atual": self._current_step,
                "ultima_etapa": self._last_step,
                "erro": self._error,
                "iniciado_em": self._started_at,
                "concluido_em": self._finished_at,
                "run_id_em_execucao": self._current_run_id,
                "ultimo_run_id": self._next_run_id - 1,
                "execucoes_concluidas": self._completed_runs,
            }

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()

                run_id = self._queue.popleft()
                self._running = True
                self._current_run_id = run_id
                self._current_step = None
                self._last_step = None
                self._error = None
                self._started_at = self._now()

            self._emit("reprocess_started", {"run_id": run_id})

            try:
                def mark_step(step: str) -> None:
                    with self._lock:
                        self._current_step = step
                        self._last_step = step

                self._pipeline(mark_step)

                with self._lock:
                    self._finished_at = self._now()
                    self._completed_runs += 1

                self._emit(
                    "reprocess_completed",
                    {"run_id": run_id, "ultima_etapa": self._last_step},
                )
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                    self._finished_at = self._now()
                self._emit(
                    "reprocess_failed",
                    {
                        "run_id": run_id,
                        "error": str(exc),
                        "ultima_etapa": self._last_step,
                    },
                )
            finally:
                with self._lock:
                    self._running = False
                    self._current_step = None
                    self._current_run_id = None
