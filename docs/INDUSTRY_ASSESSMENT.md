# Industry Assessment and Research Basis

## Honest position

RoadEssy is a promising research and field-screening prototype, not an industry-grade certified IRI application yet.

It has several strong product foundations: synchronized video/GPS ingestion, explicit camera calibration, distance-domain segmentation, map visualization, defect evidence images, API persistence, migrations, and automated tests. Those are meaningful capabilities beyond a simple dashboard.

It is not currently comparable to a certified inertial profiler, network-level survey contractor, or mature commercial pavement-management platform for contractual acceptance or asset funding decisions. The main reason is measurement validity and traceability, not the UI.

## Where it is strong

- Clear end-to-end workflow from mobile survey files to mapped results.
- Uses a quarter-car response model rather than a raw acceleration average.
- Produces 50 m interval records suitable for spatial review.
- Connects visual defects, GPS points, route geometry, images, and condition summaries.
- Has calibration and depth-estimation concepts instead of treating pixels as measurements.
- Includes unit tests for GPS, IRI, fusion, mapping, preprocessing, and vision components.
- Now excludes stationary vehicle vibration from IRI profile construction.

## Gaps against professional IRI practice

### 1. Reference standard and class are not demonstrated

Professional systems document a measurement class, speed range, accelerometer/grade sensor characteristics, validation protocol, repeatability, and calibration traceability. RoadEssy does not yet demonstrate compliance with ASTM E1926 or an equivalent national acceptance procedure.

### 2. Fixed reference speed

The quarter-car simulation uses a fixed 80 km/h reference speed even when the vehicle travelled at another speed. That can be a valid standardized calculation choice only when documented and validated; it is not evidence that the mobile measurement is speed-independent.

### 3. Acceleration-to-profile drift

Double integration of noisy acceleration is highly sensitive to bias, orientation, timing, filtering, mounting, and vehicle dynamics. The current approach filters and detrends, but does not provide a traceable bias/orientation correction, uncertainty estimate, or comparison to a reference profile.

### 4. Sparse and irregular source data

The example vibration data is around 1 Hz while the algorithm interpolates it to 100 Hz. Interpolation does not recreate missing high-frequency road response. A professional IRI system normally uses sufficiently high-rate, time-synchronized sensors and documents anti-aliasing.

### 5. GPS synchronization and trajectory quality

The app interpolates external GPS against video elapsed time. It needs explicit clock-offset estimation, quality flags, maximum extrapolation limits, and validation plots. GPS points can repeat during stops, and the first stationary period in the supplied files is real data that must be labelled rather than mistaken for route coverage.

### 6. Stationary and low-speed semantics

The current fix ignores acceleration below 0.5 m/s vehicle speed. That is appropriate for rejecting parked-car shaking, but it also means the result has no valid IRI content until the vehicle moves. The API should expose moving duration, moving distance, excluded duration, and a quality status instead of only returning a numeric IRI.

### 7. PCI is a project-specific heuristic

The fusion code adds IRI directly to visual deduct values and calls the result PCI-style. That is not automatically equivalent to ASTM D6433 PCI calculation, which requires defined distress surveying, severity, density, deduct curves, and correction procedures. Rename/report it as a RoadEssy condition score until formally validated.

### 8. Vision measurement validation

YOLO detections and homography-derived dimensions need precision/recall, localization error, depth error, and calibration repeatability reports. A saved crop is valuable evidence, but it does not by itself prove the measured dimensions.

### 9. Operational maturity

The code currently has duplicate pipelines, hard-coded localhost URLs, import-time schema creation alongside Alembic, incomplete SQLite/PostGIS equivalence, broad exception handling, and known storage/path risks. These are manageable engineering issues but materially affect repeatability and supportability.

## Recommended maturity path

### Prototype / research pilot

Current status, with the stationary handling fix and documentation: suitable for experiments, route screening, demonstrations, and building a labelled dataset.

### Controlled field pilot

Add a run-quality gate, explicit source/time diagnostics, moving distance and excluded-time reporting, input manifests, model checksums, deterministic output versioning, and manual review of at least 20 runs.

### Validation study

Run RoadEssy beside a reference profiler over representative surfaces and speeds. Report bias, RMSE, repeatability, between-run variation, missing-data behavior, and agreement by 50 m segment. Split calibration and evaluation routes.

### Production network screening

Add authenticated deployments, centralized object storage, job queues, retries, observability, database constraints/foreign keys, versioned algorithms, privacy controls, and a documented acceptance threshold.

### Contractual or regulatory use

Obtain formal review against the applicable road authority specification. Demonstrate sensor calibration traceability, measurement class, validation results, uncertainty, audit logs, and change control.

## Research references

These are the standards and institutional references to use when turning this prototype into a defensible measurement system:

- ASTM International, ASTM E1926, _Standard Practice for Computing International Roughness Index of Roads from Longitudinal Profile Measurements_: https://www.astm.org/e1926-08r20.html
- ISO 8608, _Mechanical vibration - Road surface profiles - Reporting of measured data_: https://www.iso.org/standard/71263.html
- World Bank, _HDM-4 Highway Development and Management Model_, road deterioration and roughness context: https://www.hdmglobal.com/
- FHWA, _Long-Term Pavement Performance (LTPP) Information Management System_ and profiler/roughness resources: https://highways.dot.gov/research/long-term-pavement-performance
- PIARC, road asset management and pavement condition guidance: https://www.piarc.org/en/knowledge-base-road-infrastructure
- ASTM D6433, _Standard Practice for Roads and Parking Lots Condition Index Surveys_: https://www.astm.org/d6433-24.html

The references define or inform measurement, profile reporting, pavement condition, and asset-management practice. They do not certify this implementation. Certification requires testing the actual hardware, software version, settings, and operating procedure.
