"""Standalone worker process for the Roadessy heavy compute pipeline.

The API enqueues survey processing tasks in the `pipeline_tasks` table
and returns immediately. This process claims queued tasks -- one at a
time, so the box is not thrashed -- runs the pipeline, and records the
outcome. A task left "processing" by a dead worker is requeued once
its heartbeat goes stale, so a survey upload survives a restart of
either process.

Run with the project's virtualenv from the repository root:
    .venv\\Scripts\\python.exe app/worker.py
"""

import logging
import os
import shutil
import signal
import threading
import traceback

# Set temp directory to F: drive to avoid C: drive space issues with
# large uploads, matching app.main.
os.environ["TEMP"] = r"F:\temp"
os.environ["TMP"] = r"F:\temp"

from app.core.env import get_env  # noqa: F401  (loads .env before the DB URL is read)
from app.database.session import SessionLocal, engine
from app.database.sync_schema import sync_schema
from app.services.pipeline.heavy_compute import run_heavy_pipeline
from app.services.pipeline.queue import (
    claim_next_task,
    mark_task_completed,
    mark_task_failed,
    requeue_stale_tasks,
    update_heartbeat,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("roadessy.worker")

WORKER_ID = f"{os.environ.get('COMPUTERNAME', 'worker')}-{os.getpid()}"
POLL_INTERVAL_SECONDS = float(os.getenv("ROADESSY_WORKER_POLL_SECONDS", "2"))
HEARTBEAT_SECONDS = float(os.getenv("ROADESSY_WORKER_HEARTBEAT_SECONDS", "30"))
STALE_AFTER_SECONDS = float(os.getenv("ROADESSY_TASK_STALE_SECONDS", "900"))

_stop = threading.Event()


def _request_stop(signum, frame):
    logger.info("Signal %s received; finishing the current task and stopping", signum)
    _stop.set()


def _heartbeat(task_id, stop: threading.Event) -> None:
    """Tell the database this worker is still alive while a task runs."""
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            update_heartbeat(task_id)
        except Exception as exc:
            logger.warning("Heartbeat failed for task %s: %s", task_id, exc)


def _workspace_dir(payload: dict):
    """Directory holding this task's uploaded inputs, if it can be found."""
    for key in ("vibration_file_path", "video_path", "gps_file_path"):
        path = payload.get(key)
        if path:
            parent = os.path.dirname(str(path))
            if parent:
                return parent
    return None


def _remove_workspace(workspace) -> None:
    if not workspace:
        return
    try:
        shutil.rmtree(workspace, ignore_errors=True)
    except Exception as exc:
        logger.warning("Could not remove workspace %s: %s", workspace, exc)


def _execute_task(task) -> None:
    payload = task.payload or {}
    workspace = _workspace_dir(payload)

    # A previous run consumed these files before it died, so the task
    # cannot be re-run: report that instead of failing with a
    # confusing missing-file error from deep inside the pipeline.
    missing = [
        key
        for key, path in (
            ("vibration_file_path", payload.get("vibration_file_path")),
            ("video_path", payload.get("video_path")),
        )
        if path and not os.path.exists(str(path))
    ]
    if missing:
        mark_task_failed(
            task.id,
            "Workspace files were removed before the worker ran "
            f"(missing: {', '.join(missing)}); the task cannot be retried.",
        )
        _remove_workspace(workspace)
        return

    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat, args=(task.id, heartbeat_stop), daemon=True
    )
    heartbeat.start()
    try:
        logger.info("Running pipeline for run_id=%s", task.run_id)
        run_heavy_pipeline(**payload)
        mark_task_completed(task.id)
        logger.info("Pipeline finished for run_id=%s", task.run_id)
    except Exception as exc:
        logger.exception("Pipeline failed for run_id=%s: %s", task.run_id, exc)
        mark_task_failed(
            task.id,
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )
    finally:
        heartbeat_stop.set()
        # The pipeline deletes the inputs it consumed; removing the
        # directory afterwards clears whatever is left of the workspace.
        _remove_workspace(workspace)


def main() -> None:
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    sync_schema(engine)
    requeue_stale_tasks(STALE_AFTER_SECONDS)
    logger.info("Pipeline worker %s started (poll every %ss)", WORKER_ID, POLL_INTERVAL_SECONDS)
    while not _stop.is_set():
        try:
            task = claim_next_task(WORKER_ID)
        except Exception as exc:
            logger.exception("Could not claim a task: %s", exc)
            _stop.wait(POLL_INTERVAL_SECONDS)
            continue
        if task is None:
            _stop.wait(POLL_INTERVAL_SECONDS)
            continue
        _execute_task(task)
    logger.info("Pipeline worker %s stopped", WORKER_ID)


if __name__ == "__main__":
    main()
