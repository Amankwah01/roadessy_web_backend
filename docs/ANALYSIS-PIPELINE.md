# Running an analysis: the ASTM track, the camera, and the reports

What a run does by default, what it withholds and why, and how to switch the
withheld parts back on. Every claim here was checked against the code in this
repository, and the log strings quoted are the strings the code emits.

## Starting a run

`POST /api/analysis/process` (`app/api/routes/analysis.py`).

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Label for the run |
| `video` | yes | Forward-facing dashcam video |
| `vibration` | yes | High-frequency accelerometer CSV |
| `gps` | no | Route log. `gps_source=auto` prefers it and falls back to the vibration log |
| `user_id`, `organisation_id` | in practice no | Resolved from the verified Clerk session; the form fields are only trusted for anonymous development callers |
| `calibration_metadata_json` / `calibration_profile_id` | no | Both omitted means the module defaults below |
| `matched_road_id` / `road_id` / `project_id` | no | Map matching is skipped without them |

Camera calibration is optional. Omitting it does not leave the run without a
camera: it falls back to the canonical defaults in `app/core/camera.py`
(1.5 m height, 15.0° downward tilt, `fx=fy=1000.0`, `cx=960.0`, `cy=540.0`).

## The ASTM track is muted in this build

This is the first thing to know before reading a result. The published figures
from a default run are the **Roadessy** ones only.

Three environment flags, all defaulting to `False`:

| Variable | Default | Meaning |
|---|---|---|
| `ASTM_TRACK_ENABLED` | `False` | Master switch. Gates whether any ASTM work is *computed* |
| `ASTM_QCM_IRI_ENABLED` | `False` | Publishes the ASTM QCM IRI |
| `ASTM_PCI_ENABLED` | `False` | Publishes the ASTM D6433 PCI |

`ASTM_TRACK_RUNNING` is **derived**, not configured:

```python
ASTM_TRACK_RUNNING = ASTM_TRACK_ENABLED and (ASTM_PCI_ENABLED or ASTM_QCM_IRI_ENABLED)
```

So `ASTM_QCM_IRI_ENABLED=true` on its own changes nothing — the master switch
has to be set as well. Verified across all four combinations:

```
default (all off)                -> Starting Heavy Compute Pipeline (Roadessy only)
ASTM_TRACK_ENABLED + QCM         -> Starting Heavy Compute Pipeline (ASTM + Roadessy)
ASTM_TRACK_ENABLED + PCI         -> Starting Heavy Compute Pipeline (ASTM + Roadessy)
sub-switch only (no master)      -> Starting Heavy Compute Pipeline (Roadessy only)
```

To publish both halves:

```bash
ASTM_TRACK_ENABLED=true
ASTM_QCM_IRI_ENABLED=true
ASTM_PCI_ENABLED=true
```

Detection is unaffected either way: the Roadessy track reuses the same
detections under a different distress mapping, so switching the ASTM track off
cannot skip `process_video_astm`, and does not try to.

### Why it is off

The calculation code is **retained, not removed**:

- The D6433 deduct-value curves are placeholder fits rather than digitizations
  of the standard's figures, and the corrected-deduct-value curves saturate
  below the TDV of the standard's own worked example.
- The QCM IRI resampling is expensive, and no report currently consumes the
  result.

A muted figure is reported as `null`, never `0.0`. A zero IRI reads as a
measured smooth road, which is the one thing a muted track must not assert;
`null` reads as "not reported".

## What the log says

A muted run opens like this:

```
INFO    Starting Heavy Compute Pipeline (Roadessy only)
WARNING ASTM track is disabled in this build and will not be computed: no ASTM
        PCI, no ASTM QCM IRI, no ASTM segments. The ASTM track is disabled in
        this build, so no ASTM PCI or ASTM QCM IRI is reported for this run.
        Condition figures come from the Roadessy track. The ASTM calculation
        code is retained and disabled, not removed.
WARNING ASTM D6433 PCI is not published: ASTM D6433 PCI is not published in this
        build. The deduct-value curves are placeholder fits ...
```

The banner used to read `(2 pipelines: ASTM + Roadessy)` unconditionally. It was
changed because a fixed string becomes a lie the moment the track is switched
off: a reader looking for the ASTM IRI in the log would take its absence for a
fault rather than for the switch. Historical runs in `logs/` still contain the
old wording; anything logged by the current code does not.

`astm_track_state()` is the single source for this, and both the log line and
the API response read it, so the log and the response cannot disagree about
what the build does.

## Camera geometry

`app/core/camera.py` is the only place camera geometry is decided. Consumers call
`resolve_camera_geometry` and receive one resolved pose; there is no per-service
default that can drift from the others.

Resolution order:

1. **Payload** — what this upload supplied (`calibration_metadata_json`).
2. **Stored profile** — `calibration_profile_id` on the run.
3. **Module defaults** — the constants above.

Two rules that are easy to get wrong and are enforced centrally:

- **Height and tilt resolve independently.** A profile with a height but no tilt
  keeps the default tilt rather than falling back to a default height too.
- **Intrinsics resolve atomically, from one source.** Mixing `fx` from a
  calibration with `cx` from a default produces a camera that has never existed.
  Either the whole intrinsic set comes from one source or none of it does.

`tilt = 0` cannot reach the road. `homography_from_payload` refuses to build
without tilt rather than producing a homography that maps the image to infinity.

Both spellings of each field are accepted, because callers send both:
`camera_height_m` and `camera_height_meters`, `camera_tilt_deg` and
`camera_tilt_degrees`, flat or nested under `intrinsics` or `camera_metadata`.
`normalise_calibration_payload` flattens all of them into one shape and reports
what the payload actually contained — the resolution reason is separate from the
payload contents, because "the payload said nothing about tilt" and "the tilt is
15°" are different facts and a reader diagnosing a bad survey needs both.

## Reports

Report endpoints live under `/api/me` and are scoped to the caller's
organization:

| Endpoint | Purpose |
|---|---|
| `GET /api/me/details/` | The caller's profile |
| `PATCH /api/me/details/` | Update it |
| `GET /api/me/reports/` | List reports, filtered by `org_id` |
| `POST /api/me/reports/generate/` | Generate one for a run |
| `GET /api/me/reports/{report_id}/download/` | Download the PDF |

Download links are signed and expire. `DOWNLOAD_LINK_TTL_SECONDS` defaults to
600. Signing uses `ROADESSY_DOWNLOAD_SECRET`; with no secret configured, tokens
are signed with a random per-process secret and a warning is logged — links then
break across restarts and do not work across workers. Set the secret in any
deployment with more than one worker.

Report-generation failures return
`Could not generate the report. Try again, and report it if it keeps failing.`
The underlying exception is logged server-side and not returned, because the
message reaches the browser.

## Test coverage

The muted default is why the ASTM tests opt in explicitly. Anything asserting on
ASTM output turns the track on with a fixture, because a muted
`compute_astm_pipeline` short-circuits before the alignment logic runs — a test
against the default would pass without exercising the code it names.

```bash
python -m pytest tests/unit -q     # 785 passed, 1 skipped, 1 xfailed
```