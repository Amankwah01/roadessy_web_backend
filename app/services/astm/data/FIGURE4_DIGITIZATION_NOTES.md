# Figure 4 (Fig. X1.1) digitization findings

Source: `C:\Users\PC\Downloads\MilanMarinkovi.pdf`, page 4, `/Image54`.
901x461 px, `/FlateDecode` (lossless), reproduces ASTM D6433-24 Figure X1.1.

Reproducing a third-hand raster of the standard. Useful as a cross-check on the
curve family and as a source for digitization, **not** as an authority: the printed
standard or USACE PAVER/MicroPAVER remains the only authoritative source.

## Axis calibration (confirmed by reading, not assumed)

| Axis | Pixel range | Quantity | Scale |
|---|---|---|---|
| Horizontal | x 119..846 | TDV, total deduct value | 0..190, 3.8263 px/unit |
| Vertical | y 38..390 | CDV, corrected deduct value | 0..200, 1.7600 px/unit |

Evidence:

- **Horizontal tick labels.** 20 label components at y 402..412, read directly
  off the glyph bitmaps: `0` at x=115, `10` at x=150, `20` at x=188, then the
  3-digit group `100, 110, ..., 190` at x=491..857. The digit-count transition
  from 2 to 3 glyphs between x=463 and x=501 lands exactly where 100 falls,
  which is an independent confirmation of the scale.
- **Vertical tick labels.** 11 components at x 70..112, reading `200, 180, 160,
  140, 120, 100, 80, 60, 40, 20, 0` top to bottom.
- **`(200)` annotation.** Five glyphs at x 34..60, y 218..278, rendering
  `(`, `2`, `0`, `0`, `)`. This independently fixes the top of the vertical scale.

### A correction made during this work

An earlier pass read the vertical axis as 0..100 and produced values at half
scale. The error was assuming the top gridline was 100 while the 11th gridline at
y=38 is 200. Caught by re-reading the tick labels with the correct right-aligned
window and confirmed by the `(200)` annotation.

## Figure text

- Y-axis title, read as three rotated groups rendered upright:
  `Bars` (y 90..162), `Number` (y 168..260), `of Deducts` (y 268..407).
- Caption at y 439..458: `FIG. X1.1 Deduct Value Correction Factor Curves for
  Different Deduct Values (q)`.

So the family is the deduct-value correction curves, parameterised by q, the
count of retained deduct values above 2.0 (9.5.5).

## Extraction

The curves are colour separated, which removes the obstacle from the earlier
session where grayscale curves were fragmented by the gridlines they cross. Six
distinct hues; five traced cleanly by taking the largest connected component per
hue and the median y per column.

| Trace | rgb | TDV span | CDV range | Monotonicity violations |
|---|---|---|---|---|
| yellow | (255,192,0) | 23..120 | 17.1..149.7 | 0 / 371 |
| green | (0,176,80) | 29..165 | 15.9..170.5 | 0 / 519 |
| orange | (228,108,10) | 26..190 | 17.1..191.8 | 0 / 626 |
| red | (255,0,0) | 60..163 | 55.7..160.2 | 0 / 396 |
| blue | (0,0,255) | 50..190 | 39.8..162.2 | 0 / 537 |

**Monotonicity is the validation that matters.** A correction curve cannot
increase CDV as TDV increases, so a wrong scale, a wrong offset or a flipped axis
all produce increases immediately. All five traces are clean at both the
incorrect 0..100 scale and the corrected 0..200 scale, which is what makes the
scale correction trustworthy rather than a matter of opinion.

The olive hue (148,138,84) is a contaminated match and was excluded: its trace is
non-monotonic (CDV falling to 22.7 near TDV 168, then jumping to 100) and it
matches a non-curve element near the axis rather than a legend key.

## Finding on the existing curve data in `app/services/astm`

The repository's q-curves are wrong in shape, not merely imprecise:

1. **q=1 and q=2 point lists are identical** (201 points each, same values). They
   cannot both be correct.
2. **Built on a 0..100 x-axis.** The figure's horizontal axis is 0..190 TDV.
3. **q=1 saturates at CDV=100 for any TDV >= 60.** So
   `curves.repository().cdv(1).interpolate(120)` raises `CurveRangeExceeded`, and
   at TDV=60 it already returns 100.

Consequence: any section whose TDV exceeds 60 either loses its CDV lookup
entirely or receives a saturated 100. This path is
`ASTMD6433PCI._cdv_from_curve` -> 9.5.5.3 (`app/services/astm/pci.py:599`).

The digitized orange trace, which passes through the origin and rises to CDV ~191
at TDV 190, is what q=1 must look like, since a single deduct value needs no
correction.

### Saturation thresholds in the current placeholder data

The placeholder curves do not fail at TDV 60. They return a live value across
60..100 and reach CDV 100 at these points:

| q | CDV reaches 100 at TDV | q | CDV reaches 100 at TDV |
|---|---|---|---|
| 1 | 58.5 | 6 | 80.5 |
| 2 | 58.5 | 7 | 88.0 |
| 3 | 63.5 | 8 | 97.5 |
| 4 | 68.5 | 9 | never within 0..100 |
| 5 | 74.0 | 10 | never within 0..100 |

Only a total deduct value past 100 has nothing to read at all; the domain edge,
not 60, is the threshold. See `README.md` for how that case is now handled.

q=1 and q=2 saturating at the same point is another symptom of the two curves
being the same curve.

## Open problem: colour-to-q mapping is unresolved

The legend swatches sit at x 848..885 (y 43, 58, 74, 99) and **carry no text**.
There is no in-plot curve labelling either: the apparent text inside the plot
turned out to be the gridline lattice, which disappears once full-length rules are
removed.

So five curves are digitized and the repository needs ten (q=1..10), and no q has
been assigned to any curve. Assigning q by curve shape, or by the vertical order
of the swatches, would be a guess dressed as data, so none was assigned.

Resolving this needs either the printed ASTM figure (whose legend has the labels)
or a labelled reproduction of it.

## Recommendation

Do not swap these in as a q=1..10 set. The usable conclusions are:

- the existing placeholders are unusable past TDV 60, and
- a genuine digitizable source exists for at least five of the curves, pending
  labels.

Both are worth recording; the curves themselves are not yet fit to commit, since
unlabelled curves would be no more trustworthy than the placeholders they replace.

## Methods that failed, and why

Recorded so the same ground is not covered twice.

- **Grayscale curve extraction.** Curves fragment where they cross gridlines;
  curves cross constantly. Superseded by colour separation.
- **Glyph template matching.** Leave-one-out accuracy 0/7 on digits whose identity
  was already known from the bitmap. Unusable; all OCR here is direct reading of
  the rendered glyphs instead.
- **x-label template matching.** Earlier session measured IoU 0.07-0.24. Unusable.
- **Leading-digit crop for vertical labels.** Labels are right-aligned, so the
  leading digit of a single-digit label falls outside the crop; and the gridline
  tick stub intrudes into the label band and splits digits into fragments.
  Superseded by reading whole labels.
- **Connected-component merging for in-plot text.** A 10 px merge window swallowed
  the gridline lattice (one component spans x 114..884) and produced unreadable
  output.
