# app/models/pipeline.py
import datetime
import uuid

from sqlalchemy import Column, DateTime, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.database.session import Base


class PipelineTask(Base):
    """One heavy-pipeline execution, queued in the database.

    The API only enqueues: a standalone worker process claims tasks from
    this table and runs them, so a survey upload survives an API restart
    and no request is held hostage by a pipeline that can run for hours.
    """

    __tablename__ = "pipeline_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # The survey run this task will produce; the pipeline writes its
    # results keyed by this id when it finishes.
    run_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    status = Column(String, nullable=False, index=True, default="queued")
    # Keyword arguments for run_heavy_pipeline, stored as JSON so the
    # worker can execute the task without the upload request's state.
    payload = Column(JSON, nullable=False)
    attempts = Column(Integer, nullable=False, default=0)
    worker_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    # Updated while the task runs; a stale heartbeat means the worker
    # died mid-pipeline, and the task is requeued.
    heartbeat_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)
