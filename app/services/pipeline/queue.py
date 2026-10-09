"""Database-backed task queue for the heavy compute pipeline.

The API enqueues tasks and a standalone worker process claims them, so
pipeline execution is durable: a queued survey survives an API or worker
restart, and the request that uploaded it returns immediately.
"""

import datetime
import logging
import uuid
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.database.session import SessionLocal
from app.models.pipeline import PipelineTask

logger = logging.getLogger("roadessy.pipeline_queue")

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"


def enqueue_pipeline_task(
    db: Session, *, run_id: uuid.UUID, payload: Dict[str, Any]
) -> PipelineTask:
    """Persist a pipeline execution for the worker to claim."""
    task = PipelineTask(run_id=run_id, status=QUEUED, payload=payload)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def claim_next_task(worker_id: str) -> Optional[PipelineTask]:
    """Claim the oldest queued task, skipping rows another worker holds.

    The row lock is taken in the same transaction that flips the status,
    so two workers can never run the same task.
    """
    with SessionLocal() as db:
        task = (
            db.query(PipelineTask)
            .filter(PipelineTask.status == QUEUED)
            .order_by(PipelineTask.created_at.asc(), PipelineTask.id)
            .limit(1)
            .with_for_update(skip_locked=True)
            .one_or_none()
        )
        if task is None:
            return None
        task.status = PROCESSING
        task.worker_id = worker_id
        task.started_at = datetime.datetime.utcnow()
        task.heartbeat_at = task.started_at
        task.attempts = (task.attempts or 0) + 1
        db.commit()
        db.refresh(task)
        return task


def update_heartbeat(task_id: uuid.UUID) -> None:
    """Record that the worker holding this task is still alive."""
    with SessionLocal() as db:
        db.query(PipelineTask).filter(PipelineTask.id == task_id).update(
            {"heartbeat_at": datetime.datetime.utcnow()},
            synchronize_session=False,
        )
        db.commit()


def _finish_task(task_id: uuid.UUID, status: str, error: Optional[str]) -> None:
    with SessionLocal() as db:
        db.query(PipelineTask).filter(PipelineTask.id == task_id).update(
            {
                "status": status,
                "finished_at": datetime.datetime.utcnow(),
                "error": error,
            },
            synchronize_session=False,
        )
        db.commit()


def mark_task_completed(task_id: uuid.UUID) -> None:
    _finish_task(task_id, COMPLETED, None)


def mark_task_failed(task_id: uuid.UUID, error: str) -> None:
    _finish_task(task_id, FAILED, error)


def requeue_stale_tasks(stale_after_seconds: float) -> int:
    """Return dead workers' tasks to the queue.

    A task whose heartbeat has not moved for the grace period was being
    run by a worker that died; the survey has to be processed, so it is
    queued again rather than left "processing" forever.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(
        seconds=stale_after_seconds
    )
    with SessionLocal() as db:
        count = (
            db.query(PipelineTask)
            .filter(
                PipelineTask.status == PROCESSING,
                PipelineTask.heartbeat_at.is_(None)
                | (PipelineTask.heartbeat_at < cutoff),
            )
            .update(
                {"status": QUEUED, "worker_id": None},
                synchronize_session=False,
            )
        )
        db.commit()
    if count:
        logger.info("Requeued %d stale pipeline task(s)", count)
    return count
