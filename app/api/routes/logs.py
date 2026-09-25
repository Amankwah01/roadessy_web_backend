# app/api/routes/logs.py
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database.session import get_db

router = APIRouter(prefix="/api/v1/logs", tags=["Logging & Monitoring"])

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def _read_tail(path: Path, lines: int = 200) -> List[str]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        return all_lines[-lines:] if lines > 0 else all_lines
    except Exception:
        return []


def _parse_log_line(line: str) -> Dict[str, Any]:
    parts = line.strip().split(" | ", 3)
    if len(parts) < 4:
        return {"raw": line.strip()}
    return {
        "timestamp": parts[0],
        "level": parts[1],
        "logger": parts[2],
        "message": parts[3] if len(parts) > 3 else "",
    }


def _filter_logs(
    log_type: str,
    level: str = "INFO",
    search: str = "",
    limit: int = 200,
) -> List[Dict[str, Any]]:
    file_map = {
        "app": "app.log",
        "errors": "errors.log",
        "pipeline": "pipeline.log",
        "iri": "iri.log",
        "vision": "vision.log",
        "spatial": "spatial.log",
        "fusion": "fusion.log",
    }
    filename = file_map.get(log_type, "app.log")
    path = LOG_DIR / filename
    lines = _read_tail(path, lines=limit * 2)

    parsed: List[Dict[str, Any]] = []
    for line in lines:
        entry = _parse_log_line(line)
        if level != "ALL" and entry.get("level", "") != level:
            continue
        if search and search.lower() not in entry.get("message", "").lower():
            continue
        parsed.append(entry)
        if len(parsed) >= limit:
            break
    return parsed


@router.get("/health", response_class=JSONResponse)
async def logs_health():
    return {
        "status": "healthy",
        "log_directory": str(LOG_DIR.resolve()),
        "log_files": sorted([f.name for f in LOG_DIR.glob("*.log")]),
    }


@router.get("/{log_type}", response_class=JSONResponse)
async def get_logs(
    log_type: str,
    level: str = "INFO",
    search: str = "",
    limit: int = 200,
):
    valid_types = {"app", "errors", "pipeline", "iri", "vision", "spatial", "fusion", "frontend"}
    if log_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid log type. Valid types: {sorted(valid_types)}",
        )

    # Frontend logs are stored in a separate JSON file
    if log_type == "frontend":
        path = LOG_DIR / "frontend_events.jsonl"
        if not path.exists():
            return {"events": [], "count": 0}
        try:
            events = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        events.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            if search:
                events = [e for e in events if search.lower() in json.dumps(e).lower()]
            return {"events": events[-limit:], "count": len(events)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read frontend logs: {exc}")

    logs = _filter_logs(log_type, level, search, limit)
    return {"logs": logs, "count": len(logs)}


@router.post("/frontend", response_class=JSONResponse)
async def ingest_frontend_logs(payload: Dict[str, Any]):
    events = payload.get("events", [])
    if not events:
        return {"status": "skipped", "reason": "no events"}

    path = LOG_DIR / "frontend_events.jsonl"
    try:
        with open(path, "a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, default=str) + "\n")
        return {"status": "accepted", "count": len(events)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store frontend logs: {exc}")
