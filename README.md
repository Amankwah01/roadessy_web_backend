# RoadEssy Road Condition Analysis

RoadEssy is a research/prototype road survey system. It combines forward-facing video, inertial/vibration data, optional external GPS, camera calibration, computer vision defect detection, OpenLayers mapping, IRI segmentation, and a PCI-style condition summary.

This repository is the FastAPI backend. The Next.js frontend is a separate repository at `../roadessy-frontend`.
Its condition result is a prototype fusion score (video distress plus IRI), not
an ASTM D6433 PCI, until inspection sample units and ASTM correction curves are
implemented and validated.

## Documentation

- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Rebuild and operations guide](docs/REBUILD_GUIDE.md)
- [Short data preparation guide](docs/DATA_PREPARATION.md)
- [Industry assessment and research basis](docs/INDUSTRY_ASSESSMENT.md)

## Quick start on Windows

```powershell
cd F:\roadessy_mhil\roadessy_backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m alembic upgrade head
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://127.0.0.1:8000`. OpenAPI is at `/docs`.

PostgreSQL with PostGIS is the default and required runtime database. Configure
`DATABASE_URL` before running migrations. SQLite is available only when an
isolated test explicitly sets `ROADESSY_ALLOW_SQLITE=true`; it is not a
production fallback and should never hold survey evidence.

Copy `.env.example` to `.env` and configure the database URL. In production,
set `ROADESSY_ENV=production` and one or more `ROADESSY_API_KEYS`; clients must
send the selected key in `X-API-Key`. Do not put keys in source control.

## Main endpoint

`POST /api/v2/survey/process`

Required form fields:

- `video_data`: road survey video unless `preprocess_id` is supplied
- `road_width_meters`
- `calibration_metadata_json`
- `user_id`
- `organisation_id`
- `name`

Optional fields:

- `vibrational_data`: one vibration CSV
- `external_gps_log`: CSV, GPX, SRT, or JSON
- `gps_source`: `auto`, `csv`, `gpx`, `srt`, `json`, `video`, or `vib_csv`
- `preprocess_id`
- `reference_speed_kmh`: IRI quarter-car reference speed, default 80 km/h
- `video_gps_offset_seconds`: explicit video-to-GPS clock offset, default 0

## Verification

```powershell
python -m pytest -q
python -m py_compile app\main.py app\services\*.py
python -m tools.opencv_check
```

The full test suite may require the dependencies in `requirements.txt`, a working PostGIS service for integration paths, and YOLO weights for vision tests.

## Important scope statement

The current IRI implementation is an engineering prototype. It computes a distance-domain profile from vehicle acceleration, filters stationary samples, simulates a fixed-speed quarter-car model, and writes 50 m segments. It is not yet a certified Class 1/2 road roughness measurement system. See the industry assessment before using results for contractual acceptance, funding decisions, or safety-critical maintenance prioritization.
