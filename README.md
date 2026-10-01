# RoadEssy Road Condition Analysis

RoadEssy is a research/prototype road survey system. It combines forward-facing video, inertial/vibration data, optional external GPS, camera calibration, computer vision defect detection, OpenLayers mapping, IRI segmentation, and a PCI-style condition summary.

This repository is the FastAPI backend. The Next.js frontend is a separate repository at `../roadessy-frontend`.
Its condition result is a prototype fusion score (video distress plus IRI), not an ASTM D6433 PCI, until inspection sample units and ASTM correction curves are implemented and validated.

## Project Structure

```
roadessy_web_backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── analysis.py       # Main pipeline endpoints
│   │   │   ├── calibration.py    # Camera calibration
│   │   │   └── logs.py           # Logging endpoints
│   │   └── storage/              # Working storage
│   ├── core/
│   │   ├── config.py             # Golden QCM constants, PCHIP knots
│   │   ├── logger.py             # Multi-handler logging
│   │   └── security.py           # API key auth
│   ├── database/
│   │   └── session.py            # DB session management
│   ├── models/
│   │   ├── calibration.py        # CalibrationProfile model
│   │   └── survey.py             # SurveyRun, IRISegment, TrackedDefect, MapMatchingRun
│   ├── schemas/
│   │   ├── calibration.py        # Calibration schemas
│   │   └── payloads.py           # Request/response payloads
│   ├── services/
│   │   ├── __init__.py           # Unified exports
│   │   ├── astm/                 # ASTM standards
│   │   │   ├── __init__.py
│   │   │   └── pci.py            # ASTM D6433-18 PCI calculator
│   │   ├── calibration/
│   │   │   ├── __init__.py
│   │   │   └── map_matching.py   # Map matching compatibility wrapper
│   │   ├── fusion/
│   │   │   ├── __init__.py
│   │   │   └── service.py        # PCI fusion (ASTM + Roadessy)
│   │   ├── gps/
│   │   │   ├── __init__.py
│   │   │   └── service.py        # Universal GPS loading
│   │   ├── imu/
│   │   │   ├── __init__.py
│   │   │   └── service.py        # IMU processing (pitch/roll/height)
│   │   ├── inertial/
│   │   │   ├── __init__.py
│   │   │   └── service.py        # IRI computation (simple + QCM)
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── heavy_compute.py  # Main pipeline orchestrator
│   │   │   ├── preprocessing.py  # ST1-ST5b preprocessing
│   │   │   └── preprocessing_service.py
│   │   ├── spatial/
│   │   │   ├── __init__.py
│   │   │   └── service.py        # Map matching, UTM projection
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── homography.py     # Image→world coordinate mapping
│   │   │   └── motion_stability.py
│   │   └── vision/
│   │       ├── __init__.py
│   │       ├── service.py        # YOLO detection
│   │       └── depth/
│   │           ├── __init__.py
│   │           ├── dav2e.py      # DepthAnythingV2 estimator
│   │           ├── geom_anchor.py # Geometry-anchored depth scaling
│   │           ├── mbtp_est.py   # MBTP pothole area estimation
│   │           ├── rgbsegment.py # RGB segmentation fallback
│   │           ├── depth_service.py  # SfM depth estimation
│   │           └── edfe_est.py   # Enhanced dimension fusion
├── alembic/
│   ├── env.py
│   └── versions/                 # 10 migration files
├── data/
│   └── models/                   # YOLO model files (defect.pt only)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_PREPARATION.md
│   ├── INDUSTRY_ASSESSMENT.md
│   └── REBUILD_GUIDE.md
├── scripts/                      # Utility scripts
│   ├── check_*.py
│   ├── diagnose_iri.py
│   ├── frame_counter.py
│   ├── iri_pipeline_v2.py
│   ├── verify_qc.py
│   ├── video_stitcher.py
│   ├── apply_migrations.py
│   ├── setup_postgis.py
│   ├── reset_db.py
│   ├── restore_tables.py
│   ├── yolo_defect_detector.py
│   ├── import_check.py
│   └── opencv_check.py
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/
│   └── fixtures/
├── tools/
│   └── exiftool/                 # ExifTool binaries
├── storage/                      # Output storage
│   ├── workspaces/
│   └── defect_images/
└── Root files: .env, requirements.txt, README.md, alembic.ini, backend_documentation.txt
```

## Dual-Pipeline Architecture

**Pipeline 1: ASTM (International Standard)**
- IRI: Quarter-Car Model (QCM) at 100Hz upsampled (ASTM E1926/E950)
- Vision: MDPI Sensors 2024 paper methodology (AASHTO R85-18 + ASTM D6433-18)
- Output: ASTM PCI (IRI reported separately per ASTM D6433)

**Pipeline 2: Roadessy (Custom/Ghana-aligned)**
- IRI: Simple displacement-to-distance method (Enochs Thesis Eq 2.9/3.9)
- Vision: Ghana Manual 2023 Table 3 ratings + custom DV computation
- Output: Roadessy PCI = 100 - corrected(DV_sum) (IRI reported separately)

## Documentation

- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Rebuild and operations guide](docs/REBUILD_GUIDE.md)
- [Short data preparation guide](docs/DATA_PREPARATION.md)
- [Industry assessment and research basis](docs/INDUSTRY_ASSESSMENT.md)
- [Backend comprehensive documentation](backend_documentation.txt)

## Quick Start on Windows

```powershell
cd F:\Adubofour\roadessy_mphil\roadessy_web_backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m alembic upgrade head
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://127.0.0.1:8000`. OpenAPI is at `/docs`.

PostgreSQL with PostGIS is the default and required runtime database. Configure `DATABASE_URL` before running migrations. SQLite is available only when an isolated test explicitly sets `ROADESSY_ALLOW_SQLITE=true`; it is not a production fallback and should never hold survey evidence.

Copy `.env.example` to `.env` and configure the database URL. In production, set `ROADESSY_ENV=production` and one or more `ROADESSY_API_KEYS`; clients must send the selected key in `X-API-Key`. Do not put keys in source control.

## Main Endpoints

### Full Pipeline Processing
`POST /analysis/process`
- Processes video + GPS + vibration + calibration through both pipelines
- Returns `run_id`, output directory, ASTM + Roadessy results

### Isolated Testing
- `POST /analysis/test-inertial` - Simple IRI only
- `POST /analysis/test-inertial-compare` - Both IRI methods side-by-side
- `POST /analysis/test-vision` - YOLO detection without DB commit
- `POST /analysis/calibrate-checkerboard` - Camera calibration from checkerboard images

### Data Retrieval
- `GET /analysis/runs` - List recent survey runs
- `GET /analysis/runs/{run_id}` - Full run summary with IRI segments
- `GET /analysis/runs/{run_id}/defects` - All tracked defects
- `GET /analysis/runs/{run_id}/map-matching` - Map matching results
- `GET /analysis/runs/geojson` - Survey trajectories as GeoJSON
- `GET /analysis/iri/segments/geojson` - IRI segments with PCI as GeoJSON
- `GET /analysis/roads/gh` - Ghana road network with latest survey metrics

### Camera Calibration
- `POST /calibration/save` - Save calibration profile
- `GET /calibration/list` - List calibration profiles

## Verification

```powershell
python -m pytest -q
python -m py_compile app\main.py app\services\**\*.py
python -m tools.opencv_check
```

The full test suite may require the dependencies in `requirements.txt`, a working PostGIS service for integration paths, and YOLO weights for vision tests.

## Model Files

Exactly one YOLO checkpoint is shipped, at `data/models/defect.pt`, and
`DEFAULT_MODEL_PATH` points at it. Its class order is the one the service rates
against:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Class | Bleeding | Corrugation | Crack | Patching | Pothole | Ravelling | Rutting | Shoving | Subgrade Failure |

Do not add a second checkpoint to `data/models/`. Every downstream stage keys on
class *index*, so another 9-class model with a different order loads without
error and silently mislabels its output — one previously present in this
repository had `Erosion Gully` at index 4. `VisionAnalysisService` now refuses to
run against a model whose class names do not match, and a test pins that the
directory holds only `defect.pt`.

## Important Scope Statement

The current IRI implementation is an engineering prototype. It computes a distance-domain profile from vehicle acceleration, filters stationary samples, simulates a fixed-speed quarter-car model, and writes 50 m segments. It is not yet a certified Class 1/2 road roughness measurement system. See the industry assessment before using results for contractual acceptance, funding decisions, or safety-critical maintenance prioritization.

The 50 m IRI table and the 50 m defect table are indexed by distance, and the two distances come from two loggers started by hand, so their section indices are aligned by projecting each detection onto the measured road rather than by assuming a shared origin. The result reports whether that alignment succeeded. In the KNUST Kumasi-to-Obuasi data the vibration logger's GPS holds a stale fix and jumps (median step between fixes is 0.00 m; up to 13.1 km of a trip's apparent distance comes from those jumps), so the survey chainage those segments are cut from is not yet sound, and the camera's GPS is the only clean source in that data. See `docs/ARCHITECTURE.md`.