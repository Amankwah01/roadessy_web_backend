"""Report generation.

A report is a real PDF built from the survey evidence this service holds: the
runs an organisation recorded in a year, the distance each covers, the PCI and
IRI each published, and the defects each found. Nothing in it is estimated at
report time and nothing is filled in with a placeholder.

Two rules govern what may appear:

* A muted number is reported as absent, never as zero. PCI 0 and PCI 100 are
  both strong claims about a road, so an unpublished PCI is written as "not
  published" with the reason, exactly as the API reports it.
* A figure that is absent because it was never measured is labelled as absent,
  not averaged away. A mean IRI over the runs that published one is not the mean
  IRI of the organisation, and the report says how many runs it covers.
"""

import datetime
import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import (
    ASTM_PCI_DISABLED_REASON,
    ASTM_PCI_ENABLED,
    IRI_SEGMENT_LENGTH_M,
    output_base_dir,
)
from app.models.report import KIND_ORGANISATION, ReportAsset
from app.models.survey import IRISegment, SurveyRun, TrackedDefect

logger = logging.getLogger(__name__)

#: Subdirectory of the output root that holds every generated report. Named here
#: rather than derived per call so the download route's confinement check and
#: the writer agree on one directory.
REPORTS_DIRNAME = "reports"

CONTENT_TYPE_PDF = "application/pdf"

#: A filename that is safe on every filesystem the service might write to, and
#: that a browser will not try to interpret as something other than a PDF.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def reports_root() -> Path:
    return Path(output_base_dir()) / REPORTS_DIRNAME


def _safe_filename(value: str, fallback: str = "report") -> str:
    cleaned = _UNSAFE_FILENAME_CHARS.sub("-", (value or "").strip()).strip("-._")
    return cleaned[:80] or fallback


def _finite(value: Any) -> Optional[float]:
    """A float, or None. NaN and infinity are absences, not measurements."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def year_bounds(year: int) -> Tuple[datetime.datetime, datetime.datetime]:
    """The half-open UTC window `[start, end)` covering `year`."""
    start = datetime.datetime(year, 1, 1)
    return start, datetime.datetime(year + 1, 1, 1)


def runs_for_year(db: Session, organisation_id: str, year: int) -> List[SurveyRun]:
    """Every run an organisation recorded in `year`, oldest first.

    `created_at` rather than `started_at`, because `SurveyRun` has no started_at
    column: a run is recorded when the upload lands, which is the only timestamp
    this table owns.
    """
    start, end = year_bounds(year)
    return (
        db.query(SurveyRun)
        .filter(
            SurveyRun.organisation_id == organisation_id,
            SurveyRun.created_at >= start,
            SurveyRun.created_at < end,
            SurveyRun.removed_at.is_(None),
        )
        .order_by(SurveyRun.created_at.asc())
        .all()
    )


def distance_by_run(db: Session, run_ids: List[uuid.UUID]) -> Dict[uuid.UUID, float]:
    """Surveyed distance per run, in metres, as one query.

    A run's distance is the far end of its last IRI segment. Summing segment
    lengths would work and would also be wrong: the segments are consecutive, so
    the sum is the same number times the segment count.
    """
    if not run_ids:
        return {}
    rows = (
        db.query(
            IRISegment.run_id.label("run_id"),
            func.max(IRISegment.segment_end_m).label("furthest"),
        )
        .filter(IRISegment.run_id.in_(run_ids))
        .group_by(IRISegment.run_id)
        .all()
    )
    distances: Dict[uuid.UUID, float] = {}
    for row in rows:
        value = _finite(row.furthest)
        if value is not None:
            distances[row.run_id] = value
    return distances


def defect_counts_by_run(db: Session, run_ids: List[uuid.UUID]) -> Dict[uuid.UUID, int]:
    """Tracked defect count per run, as one query rather than one per run."""
    if not run_ids:
        return {}
    rows = (
        db.query(TrackedDefect.run_id.label("run_id"), func.count(TrackedDefect.id).label("n"))
        .filter(TrackedDefect.run_id.in_(run_ids))
        .group_by(TrackedDefect.run_id)
        .all()
    )
    return {row.run_id: int(row.n or 0) for row in rows}


def defect_modes_by_run(db: Session, run_ids: List[uuid.UUID]) -> Dict[uuid.UUID, Dict[str, int]]:
    """Defect counts per run grouped by failure mode, as one query."""
    if not run_ids:
        return {}
    rows = (
        db.query(
            TrackedDefect.run_id.label("run_id"),
            TrackedDefect.failure_mode.label("mode"),
            func.count(TrackedDefect.id).label("n"),
        )
        .filter(TrackedDefect.run_id.in_(run_ids))
        .group_by(TrackedDefect.run_id, TrackedDefect.failure_mode)
        .all()
    )
    grouped: Dict[uuid.UUID, Dict[str, int]] = {}
    for row in rows:
        mode = row.mode or "Unclassified"
        grouped.setdefault(row.run_id, {})
        grouped[row.run_id][mode] = grouped[row.run_id].get(mode, 0) + int(row.n or 0)
    return grouped


def iri_samples_by_run(db: Session, run_ids: List[uuid.UUID]) -> Dict[uuid.UUID, List[float]]:
    """Per-segment roughness values per run, for the distribution plot.

    Both methods are collected and kept apart by the caller: they are different
    measurements of the same road and averaging them together would report a
    number no method produced.
    """
    if not run_ids:
        return {}
    rows = (
        db.query(
            IRISegment.run_id.label("run_id"),
            IRISegment.iri_value.label("iri"),
            IRISegment.simple_iri.label("simple_iri"),
            IRISegment.astm_qcm_iri.label("qcm_iri"),
        )
        .filter(IRISegment.run_id.in_(run_ids))
        .order_by(IRISegment.segment_start_m)
        .all()
    )
    samples: Dict[uuid.UUID, Dict[str, List[float]]] = {}
    for row in rows:
        bucket = samples.setdefault(row.run_id, {"iri": [], "simple_iri": [], "qcm_iri": []})
        for key, raw in (
            ("iri", row.iri),
            ("simple_iri", row.simple_iri),
            ("qcm_iri", row.qcm_iri),
        ):
            value = _finite(raw)
            if value is not None:
                bucket[key].append(value)
    return {
        run_id: {"iri": b["iri"], "simple_iri": b["simple_iri"], "qcm_iri": b["qcm_iri"]}
        for run_id, b in samples.items()
    }


def list_reports(
    db: Session, organisation_id: str, year: Optional[int] = None, limit: int = 200
) -> List[ReportAsset]:
    """An organization's reports, newest first."""
    query = db.query(ReportAsset).filter(ReportAsset.organisation_id == organisation_id)
    if year is not None:
        query = query.filter(ReportAsset.year == year)
    return query.order_by(ReportAsset.created_at.desc()).limit(limit).all()


def _summarise(rows: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], int]:
    """Mean of the runs that published `key`, and how many published it.

    Both halves are returned because a mean without its denominator reads as a
    measurement of the whole organisation rather than of the runs that reported.
    """
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def build_report_rows(
    db: Session, organisation_id: str, year: int
) -> List[Dict[str, Any]]:
    """Assemble the per-run rows the PDF is drawn from.

    Kept separate from rendering so the numbers can be asserted on without a PDF
    in the way.
    """
    runs = runs_for_year(db, organisation_id, year)
    run_ids = [run.run_id for run in runs]
    distances = distance_by_run(db, run_ids)
    defects = defect_counts_by_run(db, run_ids)
    modes = defect_modes_by_run(db, run_ids)
    samples = iri_samples_by_run(db, run_ids)

    rows: List[Dict[str, Any]] = []
    for run in runs:
        distance = distances.get(run.run_id)
        iri_samples = samples.get(run.run_id, {})
        simple_values = iri_samples.get("simple_iri") or iri_samples.get("iri") or []
        rows.append(
            {
                "run_id": run.run_id,
                "name": run.name or str(run.run_id),
                "created_at": run.created_at,
                "project_name": run.project.name if run.project is not None else None,
                "road_name": run.road.name if run.road is not None else None,
                "distance_m": distance,
                # Published PCI values only. `final_pci` and `roadessy_pci` are
                # this service's own index; `astm_pci` is None whenever the
                # ASTM track is muted, and stays None here rather than falling
                # back to the other track's number.
                "pci": _finite(run.final_pci if run.final_pci is not None else run.roadessy_pci),
                "astm_pci": _finite(run.astm_pci),
                "simple_iri": _finite(run.simple_iri),
                "astm_qcm_iri": _finite(run.astm_qcm_iri),
                "defect_count": defects.get(run.run_id, 0),
                "defect_modes": modes.get(run.run_id, {}),
                "iri_samples": simple_values,
                "astm_iri_samples": iri_samples.get("qcm_iri") or [],
                "removed": run.removed_at is not None,
            }
        )
    return rows


def render_report_pdf(
    path: Path,
    title: str,
    organisation_name: str,
    year: int,
    rows: List[Dict[str, Any]],
    generated_at: Optional[datetime.datetime] = None,
) -> None:
    """Write the report PDF.

    matplotlib is already a dependency (the pipeline draws the IRI profile with
    it) and its Agg backend is safe in a request thread, so the same library
    produces the pages rather than adding a PDF dependency for one route.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    if generated_at is None:
        generated_at = datetime.datetime.utcnow()

    measured = [r for r in rows if not r["removed"]]
    total_distance_m = sum(r["distance_m"] for r in measured if r["distance_m"] is not None)
    total_defects = sum(r["defect_count"] for r in measured)
    mean_pci, pci_n = _summarise(measured, "pci")
    mean_iri, iri_n = _summarise(measured, "simple_iri")

    with PdfPages(path) as pdf:
        # ---- Page 1: what this report covers -----------------------------
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.text(0.07, 0.93, title, fontsize=17, fontweight="bold", va="top")
        fig.text(0.07, 0.895, f"{organisation_name} · {year}", fontsize=11, color="#444444", va="top")
        fig.text(
            0.07,
            0.865,
            f"Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            fontsize=9,
            color="#666666",
            va="top",
        )

        headline = [
            ("Survey runs in year", str(len(measured))),
            ("Distance surveyed", f"{total_distance_m / 1000.0:,.2f} km"),
            ("Road condition index (mean)", f"{mean_pci:.1f}" if mean_pci is not None else "not published"),
            ("Roughness, simple method (mean)", f"{mean_iri:.2f} m/km" if mean_iri is not None else "not published"),
            ("Defects tracked", f"{total_defects:,}"),
        ]
        fig.text(0.07, 0.80, "At a glance", fontsize=12, fontweight="bold", va="top")
        y = 0.765
        for label, value in headline:
            fig.text(0.07, y, label, fontsize=10, va="top", color="#333333")
            fig.text(0.62, y, value, fontsize=10, va="top", fontweight="bold")
            y -= 0.032

        fig.text(0.07, y - 0.02, "Coverage of the figures above", fontsize=12, fontweight="bold", va="top")
        y -= 0.065
        coverage = [
            f"Roughness mean covers {iri_n} of {len(measured)} runs; runs with no published IRI are excluded.",
        ]
        if mean_pci is not None:
            coverage.append(
                f"Road condition index mean covers {pci_n} of {len(measured)} runs."
            )
        else:
            coverage.append("No run published a road condition index for this period.")
        if not ASTM_PCI_ENABLED:
            coverage.append(ASTM_PCI_DISABLED_REASON)
        for line in coverage:
            fig.text(0.07, y, line, fontsize=8.5, va="top", color="#555555", wrap=True)
            y -= 0.03

        fig.text(
            0.07,
            0.06,
            "RoadEssy road condition assessment · research prototype · "
            "not a certified ASTM D6433 pavement condition index",
            fontsize=7.5,
            color="#777777",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2: the runs -------------------------------------------
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.0, 1.0, f"{organisation_name} · {year} · survey runs", fontsize=13, fontweight="bold", va="bottom")
        if not rows:
            ax.text(0.0, 0.95, "No survey runs were recorded for this period.", fontsize=10, va="top")
            pdf.savefig(fig)
            plt.close(fig)
        else:
            headers = ["Run", "Date", "Road", "Distance", "PCI", "IRI", "Defects"]
            widths = [0.24, 0.11, 0.22, 0.11, 0.09, 0.11, 0.08]
            table_rows = []
            for row in rows:
                distance = "-" if row["distance_m"] is None else f"{row['distance_m'] / 1000.0:.2f} km"
                table_rows.append(
                    [
                        (row["name"] or "")[:38],
                        row["created_at"].strftime("%Y-%m-%d") if row["created_at"] else "-",
                        ((row["road_name"] or row["project_name"] or "-") or "-")[:26],
                        distance,
                        "-" if row["pci"] is None else f"{row['pci']:.1f}",
                        "-" if row["simple_iri"] is None else f"{row['simple_iri']:.2f}",
                        str(row["defect_count"]),
                    ]
                )
            table = ax.table(
                cellText=table_rows,
                colLabels=headers,
                colWidths=widths,
                loc="upper left",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.0, 1.35)
            for (r, c), cell in table.get_celld().items():
                cell.set_linewidth(0.4)
                if r == 0:
                    cell.set_text_props(fontweight="bold")
                    cell.set_facecolor("#eeeeee")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ---- Page 3: distribution, only when there is something to plot ----
        pci_values = [r["pci"] for r in measured if r["pci"] is not None]
        iri_values = [v for r in measured for v in (r["iri_samples"] or [])]
        if not pci_values and not iri_values:
            return

        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        for axis in axes:
            axis.set_facecolor("white")

        if pci_values:
            axes[0].bar(range(len(pci_values)), pci_values, color="#3b6ea5")
            axes[0].set_ylim(0, 100)
            axes[0].set_ylabel("Road condition index")
            axes[0].set_xlabel("Run (chronological)")
            axes[0].set_title("Road condition index per run", fontsize=11)
            axes[0].grid(axis="y", color="#e5e5e5", linewidth=0.6)
            axes[0].set_axisbelow(True)
            for spine in ("top", "right"):
                axes[0].spines[spine].set_visible(False)
        else:
            axes[0].axis("off")
            axes[0].text(0.5, 0.5, "No road condition index was published for these runs.", ha="center", va="center", fontsize=10)

        if iri_values:
            axes[1].hist(iri_values, bins=min(30, max(5, len(iri_values) // 4)), color="#5b8c5a")
            axes[1].set_xlabel("IRI (m/km)")
            axes[1].set_ylabel("Segments")
            axes[1].set_title(
                f"Roughness distribution, {len(iri_values)} segments at "
                f"{IRI_SEGMENT_LENGTH_M:.0f} m",
                fontsize=11,
            )
            axes[1].grid(axis="y", color="#e5e5e5", linewidth=0.6)
            axes[1].set_axisbelow(True)
            for spine in ("top", "right"):
                axes[1].spines[spine].set_visible(False)
        else:
            axes[1].axis("off")
            axes[1].text(0.5, 0.5, "No IRI segments were published for these runs.", ha="center", va="center", fontsize=10)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def generate_organisation_report(
    db: Session,
    *,
    organisation_id: str,
    organisation_name: str,
    year: int,
    title: Optional[str],
    created_by: Optional[str],
) -> Tuple[ReportAsset, List[ReportAsset]]:
    """Build an organization's report for `year` and record it.

    Returns the new row and the organization's current report list, which is
    what the dashboard renders straight after generating.
    """
    rows = build_report_rows(db, organisation_id, year)

    report_id = uuid.uuid4()
    safe_title = _safe_filename(title or f"{organisation_name} road condition report {year}")
    file_name = f"{safe_title}_{year}_{report_id.hex[:8]}.pdf"

    target_dir = reports_root() / organisation_id / str(year)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / file_name

    display_title = (title or "").strip() or f"{organisation_name} Road Condition Report {year}"
    render_report_pdf(
        target,
        display_title,
        organisation_name,
        year,
        rows,
    )

    payload = target.read_bytes()
    asset = ReportAsset(
        id=report_id,
        organisation_id=organisation_id,
        year=year,
        title=display_title,
        kind=KIND_ORGANISATION,
        file_name=file_name,
        original_filename=file_name,
        content_type=CONTENT_TYPE_PDF,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        storage_path=str(target),
        bucket=REPORTS_DIRNAME,
        key=f"{organisation_id}/{year}/{file_name}",
        created_by=created_by,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)

    logger.info(
        "Generated report %s for organisation %s (%d runs)",
        asset.id,
        organisation_id,
        len(rows),
    )
    return asset, list_reports(db, organisation_id)


def resolve_report_path(asset: ReportAsset) -> Optional[str]:
    """The report's path, or None when it is gone or escapes the reports root.

    The path is written by this service and stored absolute, but a stored path is
    still just a string in a column: confining it before opening is what stops a
    edited row from becoming arbitrary file read.
    """
    stored = (asset.storage_path or "").strip()
    if not stored:
        return None
    candidate = Path(stored)
    try:
        candidate.resolve().relative_to(reports_root().resolve())
    except (ValueError, OSError):
        logger.warning("Report %s has a storage_path outside the reports root", asset.id)
        return None
    if not candidate.is_file():
        return None
    return str(candidate)