# Survey Data: Short Guide

## What to upload

For a normal survey, prepare:

1. **Road video**: forward-facing, stable, readable pavement view.
2. **Vibration CSV**: required for IRI. It must contain `RecIntvl`, an acceleration column such as `LinAccelZ` or `AccelZ`, and a speed column such as `SpeedKmh`, `SpeedGPS`, or `Speed`.
3. **External GPS CSV**: recommended for mapping. It must contain time, latitude, and longitude.
4. **Camera calibration**: checkerboard calibration for the exact camera, resolution, lens, mounting height, and tilt.
5. **Road width**: measured total paved width in metres.

## Accepted GPS CSV shape

```csv
Global_Second,Timestamp,Latitude,Longitude,Speed
2026:08:04 07:55:35,3428,6.53431,-1.573055,0
2026:08:04 07:55:48,3441,6.53431,-1.573055,1
2026:08:04 07:55:55,3448,6.534238,-1.573155,14
```

Note that the two time columns are named for each other's contents:
`Global_Second` carries a wall clock and `Timestamp` carries the elapsed time.
The loader therefore picks the time axis by inspecting column *values*, not by
trusting the header name, and it accepts all three forms these logger exports
use — plain seconds, an `H:MM:SS` elapsed clock, and a full datetime:

| Written as | Example |
| --- | --- |
| Plain seconds | `3428` |
| Elapsed clock | `0:57:08` |
| Wall clock | `2026:08:04 07:55:35` |

The chosen axis is rebased to start at zero, because video time is measured from
the first frame and a logger counter that starts at 8241 s would otherwise place
the whole track outside the video's timeline, resolving every detection to the
first fix. In this dataset the same trip is exported both ways — as
`3428`-style seconds and as `0:00:00`-style clock — and both loaders now produce
an identical track from the two files.

## Accepted vibration CSV shape

```csv
Timestamp,LinAccelZ,SpeedKmh,Latitude,Longitude,RecIntvl
Tue Aug 04 07:04:44 GMT 2026,0.475,29.015997,6.683902,-1.5734836,1.0
Tue Aug 04 07:04:45 GMT 2026,-0.720,29.015997,6.683902,-1.5734836,1.0
```

The human-readable `Timestamp` is not used to reconstruct IRI time. `RecIntvl` is the elapsed sample interval. Header spaces are tolerated.

## Preparation checklist

- Keep video, vibration, and GPS from the same trip.
- Start recording all devices before moving, and keep clocks or elapsed time aligned.
- Drive forward continuously long enough to cover the road; avoid long stationary periods in the survey section.
- Do not use a vehicle that is only shaking while parked as an IRI survey.
- Check that latitude/longitude are decimal degrees and longitude is negative for this Ghana route.
- Check that GPS time increases and does not reset midway. A reset to zero mid-file will be read as a time going backwards.
- Check that the time column is the one you think it is: the loader trusts values over headers, so a CSV whose seconds were written as `0:00:00` will be read correctly while one with a *missing* time column will be read on an invented 1 Hz axis.
- Check that speed units are labelled; `SpeedKmh` is converted automatically.
- Remove empty rows, broken coordinates, duplicated time rows, and accidental spreadsheet formatting.
- Keep the original files unchanged as evidence.

## Camera setup

Mount the camera rigidly and record:

- camera model and resolution
- camera height above pavement
- camera tilt
- lens/FOV
- calibration image set
- whether the video is already undistorted

A poor calibration can make defect dimensions and ground projection wrong even when GPS is perfect.

## Before pressing Submit

- Select the correct GPS source: `csv` for external GPS, `vib_csv` for GPS inside the vibration CSV, or `video` only when the video contains usable GPS metadata.
- Confirm the road width in metres.
- Confirm the saved calibration belongs to this camera.
- Confirm the vehicle will move. Stationary acceleration is ignored for IRI and cannot produce a valid road profile.

## Result interpretation

IRI is reported in metres per kilometre and segmented at 50 m. The map and report are useful for screening and field review. Treat results as provisional until compared with a trusted reference profiler on the same route.
