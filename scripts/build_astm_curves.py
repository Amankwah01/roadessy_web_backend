"""Regenerate the ASTM D6433-99 deduct-value dataset from the vendored first pass.

The point arrays in this dataset were MDPI polynomial fits: curves invented from a
2024 paper, on a linear density axis, with a constant term. This script replaces
them with the anchor points in `data/first_pass/`, which were read off the
published figures. That is a different kind of number and it is recorded as one.

It stays provisional. The vendored file labels every point
`UNVERIFIED_FIRST_PASS_DIGITIZATION`: read by eye off a scan, not checked against
the figure by a second pass, and never independently validated. So a PCI from this
dataset is closer to ASTM than the placeholder was, and is still not an ASTM
measurement. `provisional` stays true and the method name changes to say where the
points came from, so the two are never confused.

Run:  python scripts/build_astm_curves.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "app", "services", "astm", "data")
FIRST_PASS = os.path.join(DATA, "first_pass")
SOURCE_JSON = os.path.join(FIRST_PASS, "astm_d6433_99_distress_curves_first_pass.json")
SOURCE_CSV = os.path.join(FIRST_PASS, "astm_d6433_99_distress_curves_first_pass.csv")
CDV_CSV = os.path.join(FIRST_PASS, "x3_26_cdv_validation_points.csv")
PACKAGE_README = os.path.join(FIRST_PASS, "README.md")
DATASET = os.path.join(DATA, "curves_d6433_99.json")

#: The distress names this project prices, and the figure each is read from.
#:
#: All metric figures. The standard publishes a US-unit and a metric figure for
#: several distresses, and they are different graphs with different axes, so the
#: choice is explicit rather than defaulted. The code works in metres, so it reads
#: the metric ones -- and it did so before this script existed.
#:
#: Longitudinal and transverse cracking share Fig. X3.15: the standard gives
#: cracking one curve, not one per orientation, and the two are distinguished by
#: the severity band their width falls in rather than by a separate graph.
FIGURE_FOR = {
    "alligator_cracking": "X3.1",
    "bleeding": "X3.2",
    "corrugation": "X3.6",
    "depression": "X3.7",
    "longitudinal_cracking": "X3.15",
    "transverse_cracking": "X3.15",
    "patching": "X3.16",
    "potholes": "X3.19",
    "rutting": "X3.21",
    "shoving": "X3.22",
    "weathering_raveling": "X3.25",
}

METHOD = "first_pass_digitization_anchor_points"


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source() -> dict:
    with open(SOURCE_JSON, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_distress_curves(source: dict, axis_note: str) -> list:
    curves = []
    for key, figure in FIGURE_FOR.items():
        figure_data = source["figures"].get(figure)
        if figure_data is None:
            raise SystemExit(
                "The vendored first pass has no figure {}. The dataset cannot be "
                "built from figures that are not there.".format(figure)
            )
        severities = {}
        for severity, points in figure_data["severity_curves"].items():
            if len(points) < 2:
                raise SystemExit(
                    "Figure {} severity {} has {} point(s); a curve needs at least "
                    "two to interpolate between.".format(figure, severity, len(points))
                )
            severities[severity] = [
                {"density": float(p["density_percent"]), "dv": float(p["deduct_value"])}
                for p in points
            ]
        curves.append({
            "distress": key,
            "figure": figure,
            "x_scale": figure_data.get("x_scale", "log10"),
            "x_axis": "distress_density_percent",
            "y_axis": "deduct_value",
            "method": METHOD,
            "provisional": True,
            "verification": "unverified_first_pass",
            "note": (
                "Anchor points read off Fig. {figure} of the supplied ASTM D6433-99 "
                "raster scan, status UNVERIFIED_FIRST_PASS_DIGITIZATION. They are a "
                "read of the published graph rather than a fit to it, so they are "
                "usable, but three to four points per curve is coarse: a detection "
                "between anchors is interpolated, not measured. Confirm against the "
                "figure. ".format(figure=figure) + axis_note
            ),
            "severities": sorted(severities),
            "curves": severities,
        })
    return curves


def build_cdv_curves(previous: dict) -> dict:
    """Carry the existing X3.26 curves forward unchanged, and say why.

    `x3_26_cdv_validation_points.csv` is *not* curve data and is not used as any.
    All sixteen of its rows are labelled `Worked example`: they are (TDV, q, CDV)
    triples read out of the standard's own arithmetic example, not points read off
    Fig. X3.26. They cannot be turned into ten curves -- there are at most five
    points for any q, none at all for q above five, and the q=1 group is not even
    monotonic in TDV ((17, 17), (28, 10), (47, 46), (54, 26)), which a real curve
    read off a graph cannot be.

    So they are kept as a test target for the correction curve, and the CDV curves
    stay on the old placeholder. That is the largest remaining gap in this dataset
    and the reason a multi-distress section is still not an ASTM PCI.
    """
    cdv = previous.get("cdv_curves")
    if not cdv:
        raise SystemExit("No existing cdv_curves to carry forward.")
    return cdv


def main() -> int:
    source = load_source()
    with open(DATASET, "r", encoding="utf-8") as handle:
        previous = json.load(handle)

    cdv_validation_note = (
        "Fig. X3.26 is the AC corrected-deduct-value curve. These are still the "
        "MDPI placeholder fits and are the weakest numbers in this dataset: the "
        "vendored x3_26_cdv_validation_points.csv shows them reading 3 to 66 points "
        "high, and 100 where the standard's example says 58."
    )

    axis_note = (
        "The density axis is treated as logarithmic, which the standard's figures "
        "are drawn on. The vendored package asks for this to be validated first, "
        "and it has not been, so the exposure is measured rather than assumed: "
        "reading these same anchors on a linear axis instead changes the deduct "
        "value by up to 14.8 points, and PCI moves one-for-one with CDV. The "
        "evidence for the axis is in the anchors themselves -- 255 of 284 sit "
        "exactly on a power of ten, which a linear grid would not produce."
    )

    dataset = {
        "standard": "ASTM D6433",
        "edition": "D6433-99",
        "description": (
            "Deduct-value curve data for asphalt concrete pavements. The distress "
            "curves are first-pass anchor points read off the published figures in "
            "the supplied standard; they are PROVISIONAL because that read has not "
            "been verified. The X3.26 correction curves are still placeholder fits."
        ),
        "authoritative": False,
        "provisional": True,
        "source": {
            "document": "ASTM D 6433-99, Standard Practice for Roads and Parking "
                        "Lots Pavement Condition Index Surveys",
            "figures": "Appendix X3, Figs. X3.1-X3.25",
            "digitization_file": os.path.basename(SOURCE_JSON),
            "digitization_csv": os.path.basename(SOURCE_CSV),
            "digitization_sha256": {
                os.path.basename(SOURCE_JSON): sha256(SOURCE_JSON),
                os.path.basename(SOURCE_CSV): sha256(SOURCE_CSV),
            },
            "cdv_validation_file": os.path.basename(CDV_CSV),
            "cdv_validation_sha256": sha256(CDV_CSV),
            "package_readme": os.path.basename(PACKAGE_README),
            "package_readme_sha256": sha256(PACKAGE_README),
            "extraction_method": "manual first-pass read of the raster figures",
            "status": source.get("status", "UNVERIFIED_FIRST_PASS_DIGITIZATION"),
            "verified": False,
        },
        "cdv_figure_note": (
            "The AC corrected-deduct-value curve is Fig. X3.26. The standard's own "
            "cross-reference at 9.5.5.3 says 'Fig. X4.15 in Appendix X3', but X4.15 "
            "is Railroad Crossing in the PCC appendix; the cross-reference is wrong "
            "in the 1999 edition. " + cdv_validation_note
        ),
        "distress_curves": build_distress_curves(source, axis_note),
        "cdv_curves": build_cdv_curves(previous),
    }

    with open(DATASET, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=1)
        handle.write("\n")

    points = sum(
        len(points)
        for entry in dataset["distress_curves"]
        for points in entry["curves"].values()
    )
    print("wrote {path}".format(path=os.path.relpath(DATASET, ROOT)))
    print("  {n} distress curves, {p} anchor points".format(
        n=len(dataset["distress_curves"]), p=points))
    print("  method={m}, provisional=True, authoritative=False".format(m=METHOD))
    print("  cdv_curves: unchanged placeholder (see cdv_figure_note)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
