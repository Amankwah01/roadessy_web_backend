# ASTM D6433-99 Curve Digitization Package

## What is included

- `astm_d6433_99_distress_curves_first_pass.json`
- `astm_d6433_99_distress_curves_first_pass.csv`
- `x3_26_cdv_validation_points.csv`

## Source

The supplied PDF is ASTM D 6433-99, *Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys*. Appendix X3 contains Figures X3.1-X3.26 for asphalt deduct-value curves. The supplied scan is rasterized.

## Critical accuracy status

The X3.1-X3.25 values in this package are **first-pass manually digitized anchor points from the raster figures**. They are NOT an official ASTM machine-readable dataset and are marked `UNVERIFIED_FIRST_PASS_DIGITIZATION`.

Do not present these values in a thesis as exact ASTM tabulated values without independent validation.

## X3.26

X3.26 is the Total Deduct Value / Corrected Deduct Value graph. It is kept separate because the scan has dense horizontal grid lines and multiple correction curves. `x3_26_cdv_validation_points.csv` contains secondary worked-example points that can be used to validate a later X3.26 digitization; it is not a substitute for the full X3.26 curve dataset.

## Recommended Kilo Code use

Kilo Code should:
1. load the JSON as external curve data;
2. preserve `standard`, `figure`, `severity`, and `status`;
3. never silently treat UNVERIFIED data as official;
4. use log10(density) interpolation only after validation;
5. fail clearly if a required curve is missing;
6. keep X3.26 as a separate curve table.

## Why this is structured this way

The ASTM PDF itself publishes the curves as figures rather than numerical equations/tables. Published automated-PCI research likewise describes digitizing the ASTM curves into numerical points before interpolation.

The source standard's PCI procedure requires DV lookup from Appendix X3 and then the CDV procedure; therefore the numerical curve dataset is a data asset, not something the coding agent should invent.
