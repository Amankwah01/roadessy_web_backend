import asyncio
from contextlib import asynccontextmanager
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from app.services.heavy_compute import run_heavy_pipeline
from app.api.routes.analysis import router as analysis_router
from app.api.routes.calibration import router as calibration_router
from app.api.routes.logs import router as logs_router
from app.database.session import engine, Base
import app.models.survey  # Ensure models are imported so SQLAlchemy registers them

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup (e.g., create tables)
    Base.metadata.create_all(bind=engine)
    yield
    # Code to run on shutdown (if any)
    
app = FastAPI(title="Roadessy Road Condition Assessment API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)
app.include_router(calibration_router)
app.include_router(logs_router)

# Limit concurrent heavy CPU/GPU routines to keep system stable and prevent OOM
PROCESSING_SEMAPHORE = asyncio.Semaphore(1)


class SurveyResponse(BaseModel):
    survey_id: str
    status: str
    metrics: dict


def _save_upload_file_sync(upload_file: UploadFile, destination_path: str) -> None:
    """Synchronous file copy helper to be offloaded to threadpool."""
    with open(destination_path, "wb") as buffer:
        import shutil
        shutil.copyfileobj(upload_file.file, buffer)


@app.get("/health")
async def health_check():
    """Unblocked endpoint for system monitoring."""
    return {"status": "healthy"}


@app.post("/api/v1/process-survey", response_model=SurveyResponse)
async def process_road_survey(
    survey_id: str = Form(...),
    speed_m_s: float = Form(...),
    sample_rate: float = Form(100.0),
    file: UploadFile = File(...),
):
    # -------------------------------------------------------------
    # STAGE 1: Non-blocking I/O (Outside Semaphore)
    # -------------------------------------------------------------
    safe_filename = os.path.basename(file.filename or "upload.bin")
    temp_file_path = os.path.join(tempfile.gettempdir(), f"{survey_id}_{safe_filename}")

    try:
        # Offload synchronous disk write to thread pool to preserve event loop response
        await run_in_threadpool(_save_upload_file_sync, file, temp_file_path)

        # Standard array extraction or parsing step (placeholder raw sample)
        mock_z_accel = [0.12, -0.05, 0.45, 1.2, 0.88, -0.34, 0.02, 0.15] * 200

        # -------------------------------------------------------------
        # STAGE 2: Narrow Semaphore Scope for Heavy Compute
        # -------------------------------------------------------------
        async with PROCESSING_SEMAPHORE:
            pipeline_func = asyncio.to_thread(
                run_heavy_pipeline,
                z_accel_data=mock_z_accel,
                speed=speed_m_s,
                sample_rate=sample_rate,
                video_path=temp_file_path,
            )
            computed_results = await pipeline_func

        return SurveyResponse(
            survey_id=survey_id,
            status="completed",
            metrics=computed_results
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        # Guarantee disk cleanup regardless of pipeline success or failure
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass