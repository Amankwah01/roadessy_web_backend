# Rebuild and Operations Guide

This guide is intended for a future developer who has only the repositories, model weights, and survey files.

## 1. Prerequisites

Windows 10/11:

- Python 3.12 recommended.
- Node.js compatible with the pinned Next.js version.
- pnpm.
- Git.
- FFmpeg and FFprobe on PATH for preprocessing and video operations.
- PostgreSQL 14+ with PostGIS for real spatial deployment. SQLite is acceptable only for basic local work.
- ExifTool if extracting embedded video metadata.
- Enough disk space for model weights, videos, temporary files, crops, and database storage.

The backend requirements pin the Python libraries. The frontend `package.json` and `pnpm-lock.yaml` pin JavaScript dependencies.

## 2. Obtain the repositories and weights

Expected layout:

```text
F:\roadessy_mhil\
  roadessy_backend\
  roadessy-frontend\
```

Place the YOLO weights at `data/models/defect.pt` — that single path is what `DEFAULT_MODEL_PATH` resolves to and what both detection scripts default to. Do not add a second checkpoint to `data/models/`: the service keys every downstream stage on class index, so a model with a different class order loads cleanly and mislabels its output, and the service now refuses such a model. Never commit private recordings, credentials, or large replacement weights without an explicit storage policy.

## 3. Backend setup

```powershell
cd F:\roadessy_mhil\roadessy_backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env`:

```dotenv
DATABASE_URL=sqlite:///./roadessy.db
```

For PostGIS use a SQLAlchemy URL, for example:

```dotenv
DATABASE_URL=postgresql+psycopg2://roadessy:password@localhost:5432/roadessy
```

Run migrations:

```powershell
python -m alembic upgrade head
```

Start the API:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.

## 4. Frontend setup

```powershell
cd F:\roadessy_mhil\roadessy-frontend
pnpm install
```

Create `.env.local`:

```dotenv
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Clerk variables are required by the authenticated UI and should be copied from the project’s deployment configuration, never from source control.

Start the UI:

```powershell
pnpm dev
```

Open `http://localhost:3000/dashboard`.

## 5. Database lifecycle

Use Alembic for shared environments:

```powershell
python -m alembic current
python -m alembic history
python -m alembic upgrade head
```

The current application also calls `Base.metadata.create_all` during import. That is convenient for local SQLite but should eventually be removed or restricted so migrations are the sole schema authority.

## 6. First end-to-end run

1. Start the backend.
2. Start the frontend.
3. Open the analysis workflow.
4. Select or create a camera calibration profile.
5. Enter a descriptive survey name and measured road width.
6. Upload a forward-facing video.
7. Upload a vibration CSV if IRI is required.
8. Upload a matching external GPS CSV or choose the correct source for embedded GPS.
9. Submit and wait for the run UUID.
10. Open the dashboard or map view.
11. Click the route for the report; click a defect marker for its image.
12. Inspect backend logs and the run response before treating the result as valid.

## 7. Troubleshooting

### API does not start

Check the active interpreter, install `requirements.txt`, inspect `.env`, and run `python -m py_compile app\main.py app\services\*.py`.

### PostGIS errors

Use PostgreSQL/PostGIS rather than SQLite for `ST_AsGeoJSON`, `ST_AsText`, map matching, buffers, and distance queries. Confirm the database user can create or already has the required extensions.

### No route on the map

Check that GPS has valid numeric latitude/longitude, that the route is not `UNMATCHED_TRACK`, and that the API returns non-null trajectory geometry.

### Defects all appear at the start

Verify video elapsed time is valid, external GPS time is relative and monotonic, and the run was processed after the GPS synchronization fix. Existing database rows are not retroactively corrected.

### IRI fails or is empty

Confirm `RecIntvl`, an acceleration column, and a speed column exist. Confirm the vehicle moved at least 0.5 m/s for enough distance to produce a 0.25 m profile and 50 m segment. Stationary-only data is not a valid IRI survey.

### Defect image is missing

Check the stored `image_path`, the backend `app/storage/defect_images` directory, and the image endpoint URL containing both run ID and track ID.

## 8. Quality checks before release

```powershell
cd F:\roadessy_mhil\roadessy_backend
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m py_compile app\main.py app\services\*.py

cd F:\roadessy_mhil\roadessy-frontend
pnpm exec tsc --noEmit
pnpm lint
pnpm build
```

Also perform a real test with a known route, compare GPS start/end coordinates, inspect several defect points against video time, and compare IRI against a trusted reference instrument before publishing numbers.

## 9. Reproducibility record

For every survey, retain:

- Video filename, codec, FPS, resolution, duration, and camera mounting details.
- Exact vibration and GPS files.
- Calibration JSON and reprojection error.
- Road width measurement and route name.
- Git commit IDs of both repositories.
- Python and Node versions.
- Model filename/checksum.
- GPS source and synchronization offset.
- IRI parameters, moving-speed threshold, reference speed, and segment size.
- Database run ID, logs, output images, route geometry, and validation notes.
