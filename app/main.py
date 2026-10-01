import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.analysis import router as analysis_router
from app.api.routes.admin import router as admin_router
from app.api.routes.calibration import router as calibration_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.logs import router as logs_router
from app.api.routes.notifications import router as notifications_router
from app.api.routes.organisations import (
    router as organisations_router,
    teams_router,
)
from app.api.routes.projects import roads_router, router as projects_router
from app.api.routes.runs import me_router as runs_me_router, router as runs_router
from app.database.session import engine
from app.database.sync_schema import sync_schema


@asynccontextmanager
async def lifespan(application: FastAPI):
    del application  # Unused; tables are created for the bound engine.
    sync_schema(engine)
    yield


app = FastAPI(title="Roadessy Road Condition Assessment API", lifespan=lifespan)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ROADESSY_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)
app.include_router(calibration_router)
app.include_router(logs_router)
app.include_router(projects_router)
app.include_router(roads_router)
app.include_router(dashboard_router)
app.include_router(runs_router)
app.include_router(runs_me_router)
app.include_router(organisations_router)
app.include_router(teams_router)
app.include_router(notifications_router)
app.include_router(admin_router)


@app.get("/health")
async def health_check():
    """Unblocked endpoint for system monitoring."""
    return {"status": "healthy"}
