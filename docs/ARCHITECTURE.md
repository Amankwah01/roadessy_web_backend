# Architecture and Data Flow

## System boundary

RoadEssy consists of two separate applications:

- `roadessy_backend`: FastAPI, SQLAlchemy, Alembic, PostGIS/SQLite, OpenCV, SciPy, pandas, Ultralytics YOLO.
- `roadessy-frontend`: Next.js App Router, React, Clerk, OpenLayers, Recharts, shadcn-style UI.

The frontend currently calls a fixed local API URL in several components. For a reusable deployment, standardize on `NEXT_PUBLIC_API_BASE_URL` and configure CORS/reverse proxying.

## Backend startup

`app.main:app` is the ASGI entry point. It imports models and routers, creates database tables during import, registers startup behavior, and exposes:

- `GET /health`
- `GET /api/surveys/latest`
- `POST /api/v2/survey/process`
- `/analysis/*`
- `/api/v1/calibration/*`
- `/preprocessing/*`
- `/logs/*`

There is also a second, older route family in `app/api/routes/analysis.py`. The production frontend workflow uses `/api/v2/survey/process` in `app/main.py`; treat the modular `analysis/process-survey-run` path as legacy until it is consolidated.

## Production processing flow

1. `AnalysisWorkflow` builds a `FormData` request containing survey name, road width, calibration JSON, video, vibration CSV, and optional external GPS CSV.
2. `process_road_survey` creates a UUID and scratch folder under the Windows temporary directory.
3. Uploaded files are copied to temporary paths.
4. `UniversalGPSService.load_gps_universal` loads external GPS, vibration-embedded GPS, video GPS, GPX, SRT, or JSON.
5. `SpatialService` computes chainage, map matches the route, and stores a centerline when PostGIS is available.
6. `InertialRefineryService.process_imu_csv` parses acceleration/speed, reconstructs time from `RecIntvl`, resamples to 100 Hz, filters acceleration, removes stationary samples, integrates into a road profile, and computes 50 m IRI segments.
7. Video frames are extracted and synchronized to GPS by elapsed time. `VisionTrackingService` runs YOLO tracking, projects detections through the calibrated ground-plane transform, estimates dimensions/depth, stores a location, and writes the best defect crop.
8. `PavementConditionFusioneer` computes defect deduct values and combines them with IRI into the current PCI-style score.
9. `SurveyRun`, `IRISegment`, and `TrackedDefect` are committed.
10. The frontend requests run summary, IRI segments, and defects. OpenLayers draws the route and defect points; clicking a defect displays its crop image; clicking a road opens the report dialog.

## Data model

### SurveyRun

Run-level identity and result: UUID, name, user and organisation, computed IRI, final PCI, matched road ID, trajectory geometry, source filenames, and JSON processing metadata.

### IRISegment

Run ID, road ID, start/end chainage in metres, IRI in m/km, and five-rank classification. The target segment size is 50 m; a final shorter remainder may be written.

### TrackedDefect

Run ID, track ID, defect type, measured area/width/depth, confidence/method, deduct value, image filename, road ID, and PostGIS point geometry.

### RoadCenterline

Road ID, owner, name, LineString geometry, width, optional buffer, and creation time.

## GPS formats

### External route CSV

Accepted canonical columns are `time`/`Timestamp`/`RecIntvl`/`GPSTime` plus `lat`/`Latitude` and `lon`/`Longitude`. Headers are trimmed. Numeric `Timestamp` is preferred over human-readable `Global_Second`. Times are normalized relative to the first GPS row.

### Vibration CSV with GPS

The common logger format contains `Timestamp` as a human-readable date, `Latitude`, `Longitude`, `SpeedKmh` or `Speed`, and `RecIntvl`. IRI uses `RecIntvl` for elapsed sensor time. GPS loading uses `Timestamp` only when the selected source is CSV and no higher-priority time field is present; for this logger, use the external GPS path or `vib_csv` explicitly when GPS is embedded in vibration data.

## IRI algorithm as implemented

- Acceleration: prefers `LinAccelZ`, then `AccelZ`, then an acceleration column containing `acc` and `z`.
- Speed: first column containing `speed`.
- `SpeedKmh` and `km/h` values are converted to m/s.
- Time: cumulative `RecIntvl`.
- Resampling: 100 Hz.
- Filter: second-order high-pass at 0.5 Hz followed by fourth-order low-pass at 15 Hz.
- Displacement: two numerical integrations.
- Stationary rule: samples below 0.5 m/s are excluded from moving-road profile construction; an all-stationary recording returns no IRI profile.
- Spatial spacing: 0.25 m.
- Quarter-car reference speed: fixed 80 km/h.
- Segmentation: 50 m.
- Classification: Ghana Road Condition Survey Manual 2023 labels currently represented by the code.

## Frontend behavior

- Dashboard loads survey GeoJSON and defects.
- Survey route color is based on PCI.
- Defect features are parsed from WKT and plotted as points.
- Road selection opens `IriRoadDialog`.
- The dialog contains summary metrics, classified interval map, condition report, IRI line trend, and defect cards/images.
- `run_id` must be attached to defect feature properties for image URLs.

## Known architectural risks

- Duplicate primary and modular pipelines can diverge.
- Import-time `Base.metadata.create_all` competes with Alembic ownership.
- SQLite cannot provide equivalent PostGIS behavior.
- Several frontend components still hard-code `127.0.0.1:8000`.
- Uploaded filename handling should be sanitized everywhere.
- Video GPS extraction from metadata is not a frame-level trajectory source.
- There is no calibration quality gate, survey quality score, or authoritative reference-run validation.
