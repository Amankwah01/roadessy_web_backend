# Curve data — what is real and what is not

`curves_d6433_99.json` holds the deduct-value curves. The lookup around it is
finished and tested. **No value in it is a verified ASTM number.** This file says
exactly which parts are authoritative, which are provisional, and what remains.

## Measured: the supplied D6433-24 PDF is good for text, not for figures

The standard used for this implementation was checked directly, because "it is
the real document" and "its curves can be read off it" are different claims.
Both were tested rather than assumed.

**The text layer is genuine and complete.** All 47 pages carry extractable text
(93,972 characters), from 12 embedded subset fonts — not OCR, not a bitmap.
Clauses 9.5.3, 9.5.4 and 9.6.2, the X1.4 ride-quality rules and the pothole
750 mm rule all extract as readable text, including the worked example
(`m = 1 + (9/98)(100 - 25.1) = 7.9`) and the Fig. 6 deduction list. So the
*procedure* this module implements is verifiable against this file, clause by
clause, and the arithmetic is not in question.

**The figures cannot be digitized from this file.** Measured, not estimated:

| Property | Value |
|---|---|
| Producer | `GPL Ghostscript 10.02.1` (re-encoded, not a native ASTM file) |
| Figures carrying curve data | 205 placements |
| Effective resolution | **150 dpi, uniform** (min 150, median 150, max 300) |
| Figure size | ~500 × 355 px, drawn at 3.35 × 2.35 in |
| Encoding | `/DCTDecode` — lossy JPEG, ~1.8 bits/px |
| Tone | continuous, not bimodal paper/ink |

Three consequences, and they compound:

1. **150 dpi** is roughly a third of what curve tracing needs. A 0–100 DV axis
   across a ~250 px plot frame is about 2.5 px per DV, so one pixel of
   centreline uncertainty is already ±0.4 DV before JPEG artifacts and the
   calibration of the frame.
2. **Lossy JPEG at 1.8 bits/px** puts ringing artifacts along the curve itself,
   which is precisely the signal being measured. The frame lines do not survive:
   no row or column produces a full-length dark run, so the plot rectangle
   cannot be found by structure alone.
3. **Continuous tone, not line art.** A clean print of a curve figure is bimodal
   — paper near 255, ink near 0. This histogram peaks in the midtones, which
   means the figure is a photograph of a printed page, already degraded once
   before this re-encoding.

This is a scan of a print that has been through at least one lossy
recompression. Tracing it would produce numbers that looked digitized but
carried an unquantified error — worse than the current honest
`UNVERIFIED_FIRST_PASS_DIGITIZATION` label, because it would claim a provenance
the file cannot support.

**So the authority question is settled at the file level:** this PDF settles the
text, and does not settle a single curve value. Buying the $113 standard only
helps if it arrives as a **hard copy** (printed, clean line art, traceable at
1200+ dpi) or as a vector PDF. A PDF reprint of this same scanned source would
add nothing.

## Current state

| | Source | Status |
|---|---|---|
| 9.5.1–9.5.5 PCI procedure | standard text, verified against the supplied PDF | **authoritative** |
| 9.6.2 baseline-variant selection | standard text, verified against the supplied PDF | **authoritative** |
| Severity tables | X1.10, X1.13, X1.14, X1.17, X1.19 | **authoritative** |
| Measurement rules (area/length/count) | X1.5, X1.10, X1.14, X1.15, X1.17, X1.19, X1.25 | **authoritative** |
| Pothole→area factor (0.5 m²) | X1.17.1.2 | **authoritative** |
| Log-density interpolation, no extrapolation | implementation choice | **authoritative** |
| Figs. X3.1–X3.25 distress curves | first-pass digitized anchors | **provisional** |
| Log-density interpolation axis | inferred from the anchors | **unvalidated** |
| Fig. X3.26 CDV correction curve | MDPI placeholder fit | **placeholder** |

So: a section with one distress is scored close to ASTM, because the severity,
quantity, density and procedure are all right and only the curve lookup is
approximate. A section with several distresses goes through the X3.26
placeholder, and that correction is currently wrong by tens of points.

## Two unvalidated choices, sized

The package asks for log10 interpolation to be validated before use. It has not
been, so both open choices are measured rather than assumed.

**The axis.** 255 of the 284 supplied anchors sit exactly on a power of ten
(0.1, 1, 10, 100). A digitization taken off a linear grid would land on
arbitrary values; one taken off a logarithmic grid lands on the decades. The 29
that miss are 26 curve endpoints, plus X3.24 swell's 5% (its range starts at 1%
and steps 1 → 5 → 30). So log10 is the well-supported reading — but it is an
inference from the data, not a statement in the standard, which never says
"logarithmic".

**What it costs.** Reading these same anchors on a linear axis instead changes
the deduct value by up to **14.8 points**, worst on potholes. PCI moves 1:1 with
CDV, so that is 14.8 PCI points of unaccounted error, from the axis alone.
`TestDensityAxisIsLogarithmic` holds that ceiling so it cannot quietly grow.

**The anchor count.** Three to four points per curve means most detections are
interpolated rather than read. This is the larger of the two problems and the
reason the package recommends 10–20 points per curve. Potholes run 0.1→1→10→100
with a 3–50 rise in between; the real graph bends inside that span.

Neither is fixed by more code. Both are fixed by a better digitization.

## The distress curves are now read off the figures

`data/first_pass/` holds the supplied package verbatim, with SHA-256 recorded in
the dataset's `source` block:

* `astm_d6433_99_distress_curves_first_pass.json` / `.csv` — 284 anchor points
  across 25 figures, identical in both formats
* `x3_26_cdv_validation_points.csv` — 16 worked-example points, **not** curve data
* `README.md` — the package's own provenance and usage notes

`curves_d6433_99.json` is generated from the first of these by
`scripts/build_astm_curves.py`. Each curve is 3–4 anchor points read off the
published graph, rather than 240 points of a fitted polynomial. That is the
difference between a number that came from the standard and one that came from a
regression, and it is the main reason the values moved.

They stay `provisional: true`. The vendored file labels every point
`UNVERIFIED_FIRST_PASS_DIGITIZATION`: read by eye off a scan, not checked against
the figure by a second pass, and never independently validated. Three to four
points per curve is also coarse — a detection between anchors is interpolated,
not measured.

Two limits worth knowing before trusting a value:

* **Coarse interpolation.** X3.1 low runs 0.1→5, 1→10, 10→25, 100→60. A density
  of 5% is read off a straight line between 10 and 25 in log space. The real graph
  is not a straight line there.
* **Low densities price at zero.** Every curve's lowest anchor is 0.1%, and
  `Curve.interpolate` returns 0 below it. A distress too small to register on the
  published graph is treated as harmless, which is the right default but does mean
  the curve cannot express "a little damage".

### To verify them

1. Compare each anchor against the figure in the standard. The 11 figures the code
   reads are X3.1, X3.2, X3.6, X3.7, X3.15, X3.16, X3.19, X3.21, X3.22, X3.25.
2. Add anchors where the graph curves between the ones supplied. More points is
   the whole improvement; nothing else needs to change.
3. Set `verification` to `verified` per curve, `authoritative` to `true` on the
   dataset, and `provisional` to `false`.
4. `test_the_placeholder_status_is_reported_not_hidden` and
   `test_the_correction_curve_still_fails_the_worked_example` will fail. Both are
   `strict` xfails or explicit assertions on purpose — they are the reminder.

## X3.26 is the remaining gap, and it is a large one

`x3_26_cdv_validation_points.csv` is **not** a digitization of Fig. X3.26 and is
deliberately not loaded as curve data. Every row is labelled `Worked example`:
they are (TDV, q, CDV) triples read out of the standard's own arithmetic example.
They cannot stand in for the curve —

* sixteen rows cannot describe ten curves
* nothing at all for q above five
* the q=1 group is not monotonic in TDV: (17, 17), (28, 10), (47, 46), (54, 26).
  A curve read off a graph cannot do that.

So they are used for what they are good for: measuring. Against them the current
X3.26 placeholder overshoots by up to 66 points, and returns 100 where the
standard's example says 58. That is the single largest error in the PCI pipeline,
and it is only reached when a section has more than one deduct value above 2.0.

`TestCorrectedDeductValueAccuracy` in `tests/unit/test_astm_curves.py` pins this:
one test asserts the file is worked-example data so nobody loads it as curves, one
bounds the current error, and one is a `strict=True` xfail that passes the moment
a real X3.26 digitization lands.

**To close it:** digitize Fig. X3.26 for q=1…10 from a vector or high-resolution
source, tabulate as (TDV, CDV) per q, and put it in `cdv_curves`. The x-axis is
TDV, a linear sum, so those curves interpolate linearly — the log axis applies to
the distress density curves only.

See `FIGURE4_DIGITIZATION_NOTES.md` for a calibrated digitization of five of these
curves from a third-hand reproduction, blocked only on identifying which curve is
which q.

## What happens past the end of the placeholder curves

The placeholder CDV curves are tabulated over TDV 0–100, and they reach CDV 100 at
the end of that range. A section whose total deduct value exceeds 100 therefore has
no published value to read.

It used to raise `CurveRangeExceeded` out of `compute_section_pci`, which took the
whole request down. The standard's own worked example sat at TDV 104.7, so the
pipeline could not finish the one complete calculation the standard publishes.

Now the lookup saturates to the curve's own terminal value and reports that it did:

* `corrected_deduct_value_with_provenance` returns `(cdv, saturated)`.
* The section result carries `cdv_curve_saturated.saturated`.
* The run provenance carries `cdv_saturated_sections`, a list of section indices.
  One saturated section drags `overall_pci` toward 0 and nothing in the mean
  reveals why, so the indices are how a caller finds it.

A saturated PCI is a floor, not a measurement. It never reports a road as better
than the evidence supports, but it is not a value the standard publishes at that
input either, and CDV 100 is indistinguishable from a genuine PCI 0 without the
flag. Compare saturated sections by `total_cdv`, which is a real sum.

`CurveRangeExceeded` now subclasses `CurveNotFound` so the existing
"no usable reading" handler covers it. It was a sibling, so that handler missed it.

Two things deliberately not done:

* **The low-level reader still raises.** `Curve.interpolate` refuses to
  extrapolate, because it is the last guard between placeholder data and a
  reported number. Saturation is decided one layer up, where it can be recorded.
* **No cap at TDV 60.** A proposal to cap CDV at 100 for TDV ≥ 60 was rejected. The
  curves return live, varying values across 60–100 (q=8 gives 52.8 at TDV 60), and
  capping there would report every heavily distressed section as PCI 0. Against the
  digitized Figure X1.1 curves, TDV 60 is roughly PCI 46 and TDV 80 roughly PCI 21.
  The threshold is the curve's domain edge, which is the only point past which
  there is genuinely nothing left to read.

Pinned by `tests/unit/test_astm_cdv_saturation.py`.

## The q=1 curve was a bug, and it is fixed in code rather than in data

Found while checking the saturation path, and it was worse than the domain defect.

`_cdv_from_curve_with_provenance` looked up a **q=1** curve on the final pass of
the 9.5.5.4 iteration. That pass is always q=1 — it is where the iteration is
built to end — and 9.5.1 says that when only one deduct value exceeds 2.0 the
total *is* the maximum corrected deduct value, with no curve consulted.

The fitted q=1 curve therefore contradicted the clause on the standard's own
worked example, and by a lot:

| TDV | q=1 curve returned | 9.5.1 requires | error |
|---|---|---|---|
| 10.0 | 11.05 | 10.00 | +1.05 |
| 30.0 | 40.65 | 30.00 | +10.65 |
| 50.0 | 80.25 | 50.00 | +30.25 |
| 58.5 | 100.00 | 58.50 | +41.50 |
| 39.10 | 57.43 | 39.10 | +18.33 |

Two things about the curve itself said it was never a reading. It was
**byte-identical to the q=2 curve**, and q=1 and q=2 cannot be the same data,
because one applies no correction and the other does. And neither the 1999 nor
the 2024 edition publishes a q=1 correction curve, because there is nothing for
one to say.

**Fix:** `_cdv_from_curve_with_provenance` now returns the total for `q <= 1`
and is not marked saturated, since that is the standard's own answer rather than
a reading off the end of a table. The q=1 entry stays in the dataset, unused —
removing it would hide the evidence, and it is reported by `self_check` below.

This is a code fix rather than a data fix because the clause fully determines
the answer. No digitization is needed, and no curve was invented.

## Structural faults are now reported when the data loads

`CurveRepository.self_check()` checks three properties of the CDV curves that the
standard determines, and logs each as a warning on first load. The point is that
these were invisible for a long time: every lookup returned a plausible number,
so nothing raised.

1. **A q=1 curve should not be in the data at all** (9.5.1).
2. **q=1 must not equal q=2** — different rules, so not the same data.
3. **The curves must reach TDV 104.7**, the total in the standard's own worked
   example (9.5.5.1), read on a q=8 curve (9.5.5.2). All ten stop at 100.

All three currently fire. They are deliberately written to stop firing the
moment a real Fig. X3.27 digitization lands, so they double as the acceptance
test for that work — see `tests/unit/test_astm_cdv_data_check.py`.

## The domain is still short, and that is not fixable here

`cdv_curves` stops at TDV 100. The standard's example needs 104.7. **The data
was not extended**, because there is nothing to extend it with: extending a
placeholder fit past its own end is how a fabricated curve gets mistaken for a
digitized one, which is the specific failure this file exists to prevent.

So the domain defect is *reported* (fault 3 above) rather than patched, and the
PCI stays muted by default via `ASTM_PCI_ENABLED` in `app/core/config.py`. The
two remaining options are both honest:

* digitize Fig. X3.27 from a vector or high-resolution source — blocked, see
  above, and separately blocked on the TDV axis carrying no tick labels in the
  available raster;
* get the CDV values from USACE PAVER / MicroPAVER, which is what the standard
  intends for authoritative values.

`WORKED_EXAMPLE_TDV` and `WORKED_EXAMPLE_Q` in `app/services/astm/curves.py`
name the threshold so the check does not hardcode it.

## Note on the standard's own cross-reference

Clause 9.5.5.3 sends you to "Fig. X4.15 in Appendix X3" for the AC correction
curve. X4.15 is *Railroad Crossing*, in the **PCC** appendix. The AC
corrected-deduct-value curve is **Fig. X3.26**. The cross-reference is wrong in
the 1999 edition; the dataset records X3.26.

## The figures, by name

| Figure | Distress | Figure | Distress |
|---|---|---|---|
| X3.1 | Alligator cracking | X3.15 | Long/transverse cracking (metric) |
| X3.2 | Bleeding | X3.16 | Patching, utility cut patching |
| X3.3 | Block cracking | X3.17 | Polished aggregate (no L/M/H) |
| X3.4 / X3.5 | Bumps and sags (metric) | X3.18 / X3.19 | Potholes (metric) |
| X3.6 | Corrugation | X3.20 | Railroad crossing |
| X3.7 | Depression | X3.21 | Rutting |
| X3.8 / X3.9 | Edge cracking (metric) | X3.22 | Shoving |
| X3.10 / X3.11 | Joint reflection (metric) | X3.23 | Slippage cracking |
| X3.12 / X3.13 | Lane/shoulder drop-off (metric) | X3.24 | Swell |
| X3.14 | Long/transverse cracking | X3.25 | Weathering and raveling |
| | | **X3.26** | **Total Deduct Value (q correction)** |

The standard publishes a US-unit and a metric figure for several distresses, and
they are different graphs with different axes. The code works in metres and reads
the metric ones; `FIGURE_FOR` in `scripts/build_astm_curves.py` makes that
explicit rather than defaulted.

Longitudinal and transverse cracking share Fig. X3.15. The standard gives cracking
one curve, not one per orientation; the two are told apart by the severity band
their width falls in, not by a separate graph.

## The other thing that is not a curve

Only alligator, patching, raveling, corrugation, shoving and bleeding are
**condition-rated** — the standard decides those on condition and ride quality
rather than a measurable extent, so they carry no automatic severity and are
reported as `provisional_severity_detections` rather than being given an invented
band. The curve for them is present; what is missing is the severity judgement,
and no amount of curve data supplies it.
