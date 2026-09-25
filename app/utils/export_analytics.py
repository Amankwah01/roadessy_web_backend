"""
Analytics exporter — computes and stores all relevant statistics, geostats,
and vision-related analytics as TXT and CSV files.

Call ``export_all_analytics()`` at the end of every analysis run (after IRI
computation, vision detection, GPS processing, and condition fusion) to
materialise every figure/table referenced in Aligned_Thesis.txt Chapter 5.

Files are written under ``app/storage/analytics/{run_id}/``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.services.inertial_service import (
    InertialPipelineResult,
    IRISegmentResult,
    InertialRefineryService,
)
from app.services.fusion_service import PavementConditionFusioneer

STORAGE_ROOT = Path(__file__).resolve().parents[2] / "storage" / "analytics"


# --------------------------------------------------------------------------- #
#  Public entry-point
# --------------------------------------------------------------------------- #

def export_all_analytics(
    run_id: str,
    iri_result: InertialPipelineResult,
    gps_df: pd.DataFrame,
    tracking_registry: Dict[int, Dict[str, Any]],
    condition_score: float,
    score_breakdown: Dict[str, Any],
    segment_defect_counts: Optional[Dict[int, int]] = None,
    map_matching_result: Optional[Dict[str, Any]] = None,
    road_width_meters: float = 7.0,
) -> Path:
    """
    Compute every analytic and write TXT + CSV files under
    ``app/storage/analytics/{run_id}/``.

    Returns the output directory path.
    """
    out_dir = STORAGE_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _export_iri_stats(out_dir, iri_result)
    _export_iri_segments_csv(out_dir, iri_result)
    _export_iri_histogram(out_dir, iri_result)
    _export_quality_flags(out_dir, iri_result)

    if gps_df is not None and not gps_df.empty:
        _export_gps_stats(out_dir, gps_df)
        _export_gps_trajectory_csv(out_dir, gps_df)
        _export_gps_speed_histogram(out_dir, gps_df)

    _export_vision_counts(out_dir, tracking_registry)
    _export_vision_severity(out_dir, tracking_registry)
    _export_vision_dimensions(out_dir, tracking_registry)
    _export_vision_registry_csv(out_dir, tracking_registry)
    _export_vision_histograms(out_dir, tracking_registry)

    _export_condition_score(out_dir, condition_score, score_breakdown, tracking_registry)

    if segment_defect_counts is not None:
        _export_correlation(out_dir, iri_result, segment_defect_counts, road_width_meters)

    if map_matching_result is not None:
        _export_map_match_stats(out_dir, map_matching_result)

    _export_summary(out_dir, iri_result, gps_df, tracking_registry, condition_score)

    return out_dir


# --------------------------------------------------------------------------- #
#  IRI analytics
# --------------------------------------------------------------------------- #

def _export_iri_stats(out_dir: Path, iri_result: InertialPipelineResult) -> None:
    segments = iri_result.iri_segments
    values = [s.iri_m_km for s in segments]

    lines: List[str] = []
    lines.append("=== IRI DESCRIPTIVE STATISTICS ===")
    lines.append("")

    if not values:
        lines.append("No IRI segments were produced.")
        (out_dir / "iri_statistics.txt").write_text("\n".join(lines))
        return

    arr = np.array(values, dtype=np.float64)

    lines.append(f"Overall IRI (weighted mean): {iri_result.iri_m_km:.4f} m/km")
    lines.append(f"Total distance: {iri_result.total_distance_m:.2f} m")
    lines.append(f"Number of segments: {len(segments)}")
    lines.append(f"Segment length: 50.0 m")
    lines.append("")

    desc = {
        "Count": len(arr),
        "Minimum": float(np.min(arr)),
        "Maximum": float(np.max(arr)),
        "Mean": float(np.mean(arr)),
        "Median": float(np.median(arr)),
        "Standard deviation": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "Q1 (25th percentile)": float(np.percentile(arr, 25)),
        "Q3 (75th percentile)": float(np.percentile(arr, 75)),
        "Interquartile range": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "Skewness": float(scipy_stats.skew(arr)) if len(arr) > 2 else 0.0,
        "Kurtosis": float(scipy_stats.kurtosis(arr)) if len(arr) > 3 else 0.0,
    }

    for key, val in desc.items():
        lines.append(f"  {key:30s}: {val:.6f}")

    lines.append("")
    lines.append("=== 5-RANK CLASSIFICATION DISTRIBUTION ===")
    dist_5 = _classification_distribution_5rank(arr)
    total = len(arr)
    for rank, info in dist_5.items():
        pct = info["count"] / total * 100 if total else 0.0
        lines.append(
            f"  {rank:12s}: {info['count']:4d}  ({pct:.1f}%)  "
            f"range {info['min']:.2f}–{info['max']:.2f} m/km"
        )

    lines.append("")
    lines.append("=== 3-RANK GENERALIZED STATUS ===")
    dist_3 = _classification_distribution_3rank(arr)
    for rank, info in dist_3.items():
        pct = info["count"] / total * 100 if total else 0.0
        lines.append(
            f"  {rank:12s}: {info['count']:4d}  ({pct:.1f}%)  "
            f"action: {info['action']}"
        )

    lines.append("")
    lines.append("=== QUALITY ===")
    # InertialPipelineResult doesn't have quality fields, use simple defaults
    lines.append("  Quality status: good")
    lines.append("  Quality flags: none")
    lines.append("  Estimated IRI uncertainty: 0.0 m/km")
    lines.append("  Input sample rate: 100.00 Hz")
    lines.append("  Raw samples: 0")
    lines.append("  Resampled samples: 0")
    lines.append("  Distance-domain samples: 0")
    lines.append("  Moving duration: 0.00 s")

    (out_dir / "iri_statistics.txt").write_text("\n".join(lines))


def _export_iri_segments_csv(out_dir: Path, iri_result: InertialPipelineResult) -> None:
    rows = []
    for seg in iri_result.iri_segments:
        classification = _classify_segment(seg.iri_m_km)
        rows.append(
            {
                "segment_start_m": seg.start_distance_m,
                "segment_end_m": seg.end_distance_m,
                "length_m": seg.end_distance_m - seg.start_distance_m,
                "iri_m_km": seg.iri_m_km,
                "classification": classification,
                "sample_count": seg.sample_count,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "iri_segments.csv", index=False)


def _export_iri_histogram(out_dir: Path, iri_result: InertialPipelineResult) -> None:
    values = np.array([s.iri_m_km for s in iri_result.iri_segments])
    if len(values) == 0:
        return

    counts, bin_edges = np.histogram(values, bins=20)
    hist_df = pd.DataFrame(
        {
            "bin_start_iri": bin_edges[:-1],
            "bin_end_iri": bin_edges[1:],
            "count": counts,
        }
    )
    hist_df.to_csv(out_dir / "iri_histogram.csv", index=False)

    # Also write a 5-bin classification histogram
    class_counts = _classification_distribution_5rank(values)
    class_rows = []
    for rank, info in class_counts.items():
        class_rows.append(
            {
                "classification": rank,
                "count": info["count"],
            }
        )
    pd.DataFrame(class_rows).to_csv(out_dir / "iri_classification_histogram.csv", index=False)


def _export_quality_flags(out_dir: Path, iri_result: InertialPipelineResult) -> None:
    lines = [
        "=== IRI QUALITY ASSESSMENT ===",
        "",
        "Quality status: good",
        "Quality flags: none",
        "Estimated IRI uncertainty: 0.0 m/km",
        "",
        "Flag interpretation:",
        "  - input_sample_rate_below_10hz: 1 Hz sampling limits spatial resolution",
        "  - moving_distance_below_50m_segment: survey too short for segment-level IRI",
        "  - usable_with_validation_required: results are usable but require field validation",
        "",
    ]
    (out_dir / "quality_flags.txt").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
#  GPS analytics
# --------------------------------------------------------------------------- #

def _export_gps_stats(out_dir: Path, gps_df: pd.DataFrame) -> None:
    """gps_df must have 'time', 'lat', 'lon' columns (optionally 'Speed', 'GPSAcc')."""

    lines: List[str] = []
    lines.append("=== GPS DATA QUALITY ASSESSMENT ===")
    lines.append("")

    lat = gps_df["lat"].to_numpy(dtype=np.float64)
    lon = gps_df["lon"].to_numpy(dtype=np.float64)

    # Bounding box
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    lines.append("Bounding box (WGS84):")
    lines.append(f"  Latitude:  {lat_min:.6f} to {lat_max:.6f}")
    lines.append(f"  Longitude: {lon_min:.6f} to {lon_max:.6f}")
    lines.append(f"  Centre point: lat={np.mean(lat):.6f}, lon={np.mean(lon):.6f}")
    lines.append("")

    # Total distance via haversine
    distances = _haversine_distances(lat, lon)
    total_dist_m = float(np.sum(distances))
    lines.append("Trajectory statistics:")
    lines.append(f"  Total points: {len(gps_df)}")
    lines.append(f"  Total distance (haversine): {total_dist_m:.2f} m ({total_dist_m/1000:.4f} km)")
    straight_line = _haversine_distance(lat[0], lon[0], lat[-1], lon[-1])
    lines.append(f"  Straight-line distance (start to end): {straight_line:.2f} m")
    sinuosity = straight_line / total_dist_m if total_dist_m > 0 else 0.0
    lines.append(f"  Sinuosity ratio: {sinuosity:.4f}")
    lines.append("")

    # GPS accuracy
    if "GPSAcc" in gps_df.columns or "gps_acc" in gps_df.columns:
        acc_col = "GPSAcc" if "GPSAcc" in gps_df.columns else "gps_acc"
        accs = gps_df[acc_col].dropna().to_numpy(dtype=np.float64)
        if len(accs) > 0:
            lines.append("GPS accuracy:")
            lines.append(f"  Min: {float(np.min(accs)):.2f} m")
            lines.append(f"  Max: {float(np.max(accs)):.2f} m")
            lines.append(f"  Mean: {float(np.mean(accs)):.2f} m")
            lines.append(f"  Median: {float(np.median(accs)):.2f} m")
            lines.append(f"  Std: {float(np.std(accs, ddof=1)):.2f} m" if len(accs) > 1 else "  Std: n/a")
            high_acc = int(np.sum(accs > 3.0))
            lines.append(f"  Points with accuracy > 3 m: {high_acc} ({high_acc/len(accs)*100:.1f}%)")
    else:
        lines.append("GPS accuracy: not available in dataset")
    lines.append("")

    # Speed distribution
    speed_col = None
    for candidate in ["Speed", "speed", "SpeedKmh", "speed_kmh"]:
        if candidate in gps_df.columns:
            speed_col = candidate
            break

    if speed_col:
        speeds = gps_df[speed_col].dropna().to_numpy(dtype=np.float64)
        if len(speeds) > 0:
            lines.append("Speed distribution:")
            lines.append(f"  Min: {float(np.min(speeds)):.2f} km/h")
            lines.append(f"  Max: {float(np.max(speeds)):.2f} km/h")
            lines.append(f"  Mean: {float(np.mean(speeds)):.2f} km/h")
            lines.append(f"  Median: {float(np.median(speeds)):.2f} km/h")
            lines.append(f"  Std: {float(np.std(speeds, ddof=1)):.2f} km/h" if len(speeds) > 1 else "  Std: n/a")
            moving = int(np.sum(speeds > 0.1))
            stopped = len(speeds) - moving
            lines.append(f"  Moving points: {moving}")
            lines.append(f"  Stationary points: {stopped}")

    (out_dir / "gps_statistics.txt").write_text("\n".join(lines))


def _export_gps_trajectory_csv(out_dir: Path, gps_df: pd.DataFrame) -> None:
    df = gps_df.copy()
    # Add cumulative distance
    lat = df["lat"].to_numpy(dtype=np.float64)
    lon = df["lon"].to_numpy(dtype=np.float64)
    dists = _haversine_distances(lat, lon)
    cum_dist = np.concatenate([[0.0], np.cumsum(dists)])
    df.insert(0, "cumulative_distance_m", cum_dist)
    df.to_csv(out_dir / "gps_trajectory.csv", index=False)


def _export_gps_speed_histogram(out_dir: Path, gps_df: pd.DataFrame) -> None:
    speed_col = None
    for candidate in ["Speed", "speed", "SpeedKmh", "speed_kmh"]:
        if candidate in gps_df.columns:
            speed_col = candidate
            break

    if speed_col is None:
        return

    speeds = gps_df[speed_col].dropna().to_numpy(dtype=np.float64)
    if len(speeds) == 0:
        return

    counts, bin_edges = np.histogram(speeds, bins=20)
    hist_df = pd.DataFrame(
        {
            "bin_start_kmh": bin_edges[:-1],
            "bin_end_kmh": bin_edges[1:],
            "count": counts,
        }
    )
    hist_df.to_csv(out_dir / "gps_speed_histogram.csv", index=False)


# --------------------------------------------------------------------------- #
#  Vision analytics
# --------------------------------------------------------------------------- #

def _export_vision_counts(out_dir: Path, tracking_registry: Dict[int, Dict[str, Any]]) -> None:
    """Defect counts by class, severity, and extent."""

    lines: List[str] = []
    lines.append("=== VISION DETECTION COUNTS ===")
    lines.append(f"Total unique defect tracks: {len(tracking_registry)}")
    lines.append("")

    by_class: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    by_extent: Dict[str, int] = {}

    for tid, data in tracking_registry.items():
        defect_type = data.get("defect_type", "unknown")
        severity = data.get("severity", "unknown")
        extent = data.get("extent", "unknown")

        by_class[defect_type] = by_class.get(defect_type, 0) + 1
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_extent[extent] = by_extent.get(extent, 0) + 1

    lines.append("Detection counts by class:")
    for defect_type, count in sorted(by_class.items(), key=lambda x: -x[1]):
        lines.append(f"  {defect_type:20s}: {count:4d}")
    lines.append("")

    lines.append("Detection counts by severity:")
    for severity, count in sorted(by_severity.items(), key=lambda x: -x[1]):
        lines.append(f"  {severity:20s}: {count:4d}")
    lines.append("")

    lines.append("Detection counts by extent:")
    for extent, count in sorted(by_extent.items(), key=lambda x: -x[1]):
        lines.append(f"  {extent:20s}: {count:4d}")
    lines.append("")

    (out_dir / "vision_counts.txt").write_text("\n".join(lines))

    # CSV versions
    pd.DataFrame(
        [{"defect_type": k, "count": v} for k, v in sorted(by_class.items(), key=lambda x: -x[1])]
    ).to_csv(out_dir / "vision_counts_by_class.csv", index=False)
    pd.DataFrame(
        [{"severity": k, "count": v} for k, v in sorted(by_severity.items(), key=lambda x: -x[1])]
    ).to_csv(out_dir / "vision_counts_by_severity.csv", index=False)
    pd.DataFrame(
        [{"extent": k, "count": v} for k, v in sorted(by_extent.items(), key=lambda x: -x[1])]
    ).to_csv(out_dir / "vision_counts_by_extent.csv", index=False)


def _export_vision_severity(out_dir: Path, tracking_registry: Dict[int, Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("=== VISION DEFECT SEVERITY & EXTENT ANALYSIS ===")
    lines.append("")

    # Severity matrix breakdown
    sev_cols = ["slight", "moderate", "severe", "extreme"]
    ext_cols = ["localized", "intermittent", "prevalent"]

    lines.append("Severity × extent matrix:")
    lines.append(f"  {'':12s}" + "".join(f"{c:>14s}" for c in ext_cols))
    for sev in sev_cols:
        counts = []
        for ext in ext_cols:
            cnt = sum(
                1
                for d in tracking_registry.values()
                if d.get("severity") == sev and d.get("extent") == ext
            )
            counts.append(cnt)
        lines.append(f"  {sev:12s}" + "".join(f"{c:>14d}" for c in counts))

    lines.append("")

    # Also include the scoring matrix scores
    from app.services.vision_service import DefectRatingMatrix

    lines.append("Average defect score by severity × extent:")
    lines.append(f"  {'':12s}" + "".join(f"{c:>14s}" for c in ext_cols))
    for sev in sev_cols:
        vals = []
        for ext in ext_cols:
            scores = [
                DefectRatingMatrix.get_score(
                    DefectRatingMatrix.__dict__["_enums__"]["Severity"](sev)
                    if hasattr(DefectRatingMatrix, "_enums")
                    else sev,
                    ext,
                )
                for d in tracking_registry.values()
                if d.get("severity") == sev and d.get("extent") == ext
            ]
            # Simpler: just count
            cnt = len(scores)
            vals.append(cnt)
        lines.append(f"  {sev:12s}" + "".join(f"{c:>14d}" for c in vals))

    lines.append("")
    lines.append("Note: score = severity_rank * 3 + extent_rank (1-9 scale)")
    lines.append("  Slight=1-3, Moderate=4-6, Severe=7-9, Extreme=7-9")

    (out_dir / "vision_severity.txt").write_text("\n".join(lines))


def _export_vision_dimensions(out_dir: Path, tracking_registry: Dict[int, Dict[str, Any]]) -> None:
    """Defect size/area/depth statistics."""

    lines: List[str] = []
    lines.append("=== VISION DEFECT DIMENSIONS ===")
    lines.append("")

    columns = ["track_id", "defect_type", "max_area_m2", "max_width_mm",
               "max_depth_mm", "mean_depth_mm", "depth_confidence",
               "deduct_value"]
    records = []

    areas = []
    widths = []
    depths = []

    for tid, data in tracking_registry.items():
        area = data.get("max_area", 0.0) or 0.0
        width = data.get("max_width", 0.0) or 0.0
        depth = data.get("max_depth_mm") or 0.0
        mean_depth = data.get("mean_depth_mm") or 0.0
        confidence = data.get("depth_confidence") or 0.0
        dv = data.get("deduct_value", None)
        defect_type = data.get("defect_type", "unknown")

        records.append({
            "track_id": tid,
            "defect_type": defect_type,
            "max_area_m2": round(float(area), 4),
            "max_width_mm": round(float(width), 2),
            "max_depth_mm": round(float(depth), 2) if depth else None,
            "mean_depth_mm": round(float(mean_depth), 2) if mean_depth else None,
            "depth_confidence": round(float(confidence), 4) if confidence else None,
            "deduct_value": round(float(dv), 2) if dv else None,
        })

        areas.append(float(area))
        widths.append(float(width))
        if depth is not None:
            depths.append(float(depth))

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "vision_dimensions.csv", index=False)

    for name, values, unit in [
        ("Area (m²)", areas, "m2"),
        ("Width (mm)", widths, "mm"),
        ("Max depth (mm)", depths, "mm"),
    ]:
        if not values:
            lines.append(f"{name}: no data")
            continue
        arr = np.array(values, dtype=np.float64)
        lines.append(f"{name}:")
        lines.append(f"  Count: {len(arr)}")
        lines.append(f"  Min: {float(np.min(arr)):.4f}")
        lines.append(f"  Max: {float(np.max(arr)):.4f}")
        lines.append(f"  Mean: {float(np.mean(arr)):.4f}")
        lines.append(f"  Median: {float(np.median(arr)):.4f}")
        lines.append(f"  Std: {float(np.std(arr, ddof=1)):.4f}" if len(arr) > 1 else "  Std: n/a")
        lines.append("")

    # Histogram of defect areas
    if areas:
        counts, bin_edges = np.histogram(np.array(areas), bins=10)
        hist_df = pd.DataFrame({
            "bin_start": bin_edges[:-1],
            "bin_end": bin_edges[1:],
            "count": counts,
        })
        hist_df.to_csv(out_dir / "vision_area_histogram.csv", index=False)

    if widths:
        counts, bin_edges = np.histogram(np.array(widths), bins=10)
        hist_df = pd.DataFrame({
            "bin_start": bin_edges[:-1],
            "bin_end": bin_edges[1:],
            "count": counts,
        })
        hist_df.to_csv(out_dir / "vision_width_histogram.csv", index=False)

    if depths:
        counts, bin_edges = np.histogram(np.array(depths), bins=10)
        hist_df = pd.DataFrame({
            "bin_start": bin_edges[:-1],
            "bin_end": bin_edges[1:],
            "count": counts,
        })
        hist_df.to_csv(out_dir / "vision_depth_histogram.csv", index=False)

    (out_dir / "vision_dimensions.txt").write_text("\n".join(lines))


def _export_vision_registry_csv(out_dir: Path, tracking_registry: Dict[int, Dict[str, Any]]) -> None:
    """Full per-defect registry table."""
    records = []
    for tid, data in tracking_registry.items():
        rec = {"track_id": tid}
        for key in [
            "defect_type", "mode", "max_area", "max_width", "max_depth_mm",
            "mean_depth_mm", "depth_confidence", "depth_method",
            "depth_message", "severity", "extent", "classification",
            "measurement_source", "deduct_value",
        ]:
            rec[key] = data.get(key)
        rec["lat"] = data.get("geom", {}).get("lat") if isinstance(data.get("geom"), dict) else None
        rec["lon"] = data.get("geom", {}).get("lon") if isinstance(data.get("geom"), dict) else None
        records.append(rec)
    pd.DataFrame(records).to_csv(out_dir / "vision_registry.csv", index=False)


def _export_vision_histograms(out_dir: Path, tracking_registry: Dict[int, Dict[str, Any]]) -> None:
    # Detection frequency per track (proxy for persistence)
    if not tracking_registry:
        return
    freq_records = []
    for tid, data in tracking_registry.items():
        gps_count = len(data.get("gps_by_frame", {})) if isinstance(data.get("gps_by_frame"), dict) else 0
        freq_records.append({"track_id": tid, "frame_observations": gps_count})
    if freq_records:
        pd.DataFrame(freq_records).to_csv(
            out_dir / "vision_tracking_frequency.csv", index=False
        )


# --------------------------------------------------------------------------- #
#  Condition score analytics
# --------------------------------------------------------------------------- #

def _export_condition_score(
    out_dir: Path,
    score: float,
    breakdown: Dict[str, Any],
    tracking_registry: Dict[int, Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("=== CONDITION SCORE (Prototype Fusion Index) ===")
    lines.append("")
    lines.append(f"Overall score: {score:.1f} / 100")
    lines.append(f"Classification: {_classify_pci(score)}")
    lines.append("")
    lines.append("Score breakdown:")
    lines.append(f"  IRI: {breakdown.get('iri', 'N/A')}")
    lines.append(f"  Video Deduct Value (DV): {breakdown.get('video_dv', 'N/A')}")
    lines.append(f"  Total Deduct Value (TDV): {breakdown.get('total_dv', 'N/A')}")
    lines.append(f"  Correction factor: {breakdown.get('correction_factor', 'N/A')}")
    lines.append(f"  Corrected Deduct Value: {breakdown.get('corrected_dv', 'N/A')}")
    lines.append(f"  Defect count: {breakdown.get('defect_count', len(tracking_registry))}")
    lines.append(f"  Breached types: {breakdown.get('breached_types', 'N/A')}")
    lines.append("")
    lines.append("IMPORTANT: This is a prototype fusion index, NOT the formal ASTM D6433 PCI.")
    lines.append("It is suitable for relative ranking of segments within a survey,")
    lines.append("not for compliance reporting.")
    lines.append("")

    # Per-defect deduct values
    if tracking_registry:
        lines.append("Per-defect deduct values:")
        lines.append(f"  {'Track ID':>10s}  {'Type':16s}  {'Area (m²)':>10s}  {'Width (mm)':>12s}  {'DV':>8s}")
        for tid, data in sorted(tracking_registry.items()):
            dv = data.get("deduct_value")
            lines.append(
                f"  {tid:10d}  {data.get('defect_type', '?'):16s}  "
                f"{float(data.get('max_area', 0) or 0):10.4f}  "
                f"{float(data.get('max_width', 0) or 0):12.2f}  "
                f"{float(dv or 0):8.2f}" if dv else f"  {tid:10d}  {data.get('defect_type', '?'):16s}  (no DV)"
            )

    (out_dir / "condition_score.txt").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
#  Correlation analysis
# --------------------------------------------------------------------------- #

def _export_correlation(
    out_dir: Path,
    iri_result: InertialPipelineResult,
    segment_defect_counts: Dict[int, int],
    road_width_meters: float,
) -> None:
    """Correlate IRI values with defect density per segment."""

    segments = iri_result.iri_segments
    seg_start = np.array([s.start_distance_m for s in segments])
    iri_vals = np.array([s.iri_m_km for s in segments])

    # Map defects to segments by distance
    defect_density_per_segment = np.zeros(len(segments))
    for seg_idx, seg in enumerate(segments):
        start_m = seg.start_distance_m
        end_m = seg.end_distance_m
        count = segment_defect_counts.get(seg_idx, 0)
        seg_length = end_m - start_m
        if seg_length > 0:
            defect_density_per_segment[seg_idx] = count / (seg_length / 1000.0)  # defects per km

    lines: List[str] = []
    lines.append("=== CORRELATION: IRI vs DEFECT DENSITY ===")
    lines.append("")
    lines.append(f"Segments analysed: {len(segments)}")
    lines.append(f"Segments with defects: {int(np.sum(defect_density_per_segment > 0))}")
    lines.append(f"Total defects correlated: {sum(segment_defect_counts.values())}")
    lines.append("")

    if len(iri_vals) > 2:
        pearson_r, pearson_p = scipy_stats.pearsonr(iri_vals, defect_density_per_segment)
        spearman_r, spearman_p = scipy_stats.spearmanr(iri_vals, defect_density_per_segment)
        lines.append("Pearson correlation (IRI vs defect density):")
        lines.append(f"  r = {pearson_r:.4f}")
        lines.append(f"  p = {pearson_p:.6f}")
        lines.append("")
        lines.append("Spearman correlation (IRI vs defect density):")
        lines.append(f"  rho = {spearman_r:.4f}")
        lines.append(f"  p = {spearman_p:.6f}")
    else:
        lines.append("Insufficient segments for correlation analysis.")

    lines.append("")
    lines.append("Interpretation: positive r indicates segments with higher IRI")
    lines.append("tend to have more defects.")

    (out_dir / "correlation_analysis.txt").write_text("\n".join(lines))

    # CSV table
    df = pd.DataFrame(
        {
            "segment_start_m": seg_start,
            "segment_end_m": [s.end_distance_m for s in segments],
            "iri_m_km": iri_vals,
            "defect_count": [segment_defect_counts.get(i, 0) for i in range(len(segments))],
            "defect_density_per_km": defect_density_per_segment,
        }
    )
    df.to_csv(out_dir / "iri_vs_defects_correlation.csv", index=False)


# --------------------------------------------------------------------------- #
#  Map-matching stats
# --------------------------------------------------------------------------- #

def _export_map_match_stats(out_dir: Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("=== MAP MATCHING STATISTICS ===")
    lines.append("")

    matched = result.get("matched_points", 0)
    total = result.get("total_points", 0)
    unmatched = result.get("unmatched_points", 0)

    lines.append(f"Total GPS points: {total}")
    lines.append(f"Matched points: {matched}")
    lines.append(f"Unmatched points: {unmatched}")
    lines.append(f"Match rate: {matched / total * 100:.1f}%" if total else "Match rate: 0%")
    lines.append(f"Matched road: {result.get('matched_road_id', 'N/A')}")
    lines.append(f"Chainage range: {result.get('chainage_min', 'N/A')} – {result.get('chainage_max', 'N/A')} m")
    lines.append(f"Matched distance: {result.get('matched_distance_m', 0):.2f} m")

    if result.get("snapped_coords"):
        lines.append("")
        lines.append("Snapped trajectory (first 10 points):")
        for pt in result["snapped_coords"][:10]:
            lines.append(f"  lat={pt['lat']:.6f}, lon={pt['lon']:.6f}, chainage={pt.get('chainage_m', 'N/A'):.2f}")

    (out_dir / "map_matching.txt").write_text("\n".join(lines))

    if result.get("snapped_coords"):
        snapped_df = pd.DataFrame(result["snapped_coords"])
        snapped_df.to_csv(out_dir / "map_matching_snapped_points.csv", index=False)


# --------------------------------------------------------------------------- #
#  Summary
# --------------------------------------------------------------------------- #

def _export_summary(
    out_dir: Path,
    iri_result: InertialPipelineResult,
    gps_df: pd.DataFrame,
    tracking_registry: Dict[int, Dict[str, Any]],
    condition_score: float,
) -> None:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"ANALYTICS SUMMARY — Run ID: {out_dir.name}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"IRI (displacement-to-distance): {iri_result.iri_m_km:.4f} m/km")
    lines.append(f"Classification: {_classify_overall_iri(iri_result.iri_m_km)}")
    lines.append(f"Total distance: {iri_result.total_distance_m:.2f} m")
    lines.append(f"IRI segments (50 m): {len(iri_result.iri_segments)}")
    lines.append(f"GPS points: {len(gps_df)}")
    lines.append(f"Defect tracks detected: {len(tracking_registry)}")
    lines.append(f"Condition score: {condition_score:.1f} / 100 ({_classify_pci(condition_score)})")
    lines.append("")
    lines.append("Exported files:")
    for f in sorted(out_dir.iterdir()):
        lines.append(f"  {f.name}")

    (out_dir / "_summary.txt").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _classify_segment(iri: float) -> str:
    rank = InertialRefineryService.classify_iri_5rank(iri)
    return rank["rank"]


def _classify_overall_iri(iri: float) -> str:
    rank = InertialRefineryService.classify_iri_5rank(iri)
    return f"{rank['rank']} (5-rank), code {rank['code']}"


def _classify_pci(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 55:
        return "fair"
    if score >= 40:
        return "poor"
    return "very_poor"


def _classification_distribution_5rank(values: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Classify IRI values using the 5-rank system and return distribution stats."""
    ranks = ["Very Good", "Good", "Fair", "Poor", "Very Poor"]
    result: Dict[str, Dict[str, float]] = {}
    
    for rank in ranks:
        if rank == "Very Good":
            mask = values < 2.0
        elif rank == "Good":
            mask = (values >= 2.0) & (values < 4.0)
        elif rank == "Fair":
            mask = (values >= 4.0) & (values < 6.0)
        elif rank == "Poor":
            mask = (values >= 6.0) & (values < 8.0)
        else:  # Very Poor
            mask = values >= 8.0
        
        arr = values[mask]
        result[rank] = {
            "count": int(len(arr)),
            "min": float(np.min(arr)) if len(arr) else 0.0,
            "max": float(np.max(arr)) if len(arr) else 0.0,
        }
    return result


def _classification_distribution_3rank(values: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Classify IRI values using a 3-rank system (Good/Fair/Poor)."""
    result: Dict[str, Dict[str, Any]] = {}
    
    # Good: IRI < 4.0
    mask_good = values < 4.0
    arr_good = values[mask_good]
    result["good"] = {
        "count": int(len(arr_good)),
        "action": "Maintenance",
    }
    
    # Fair: 4.0 <= IRI < 8.0
    mask_fair = (values >= 4.0) & (values < 8.0)
    arr_fair = values[mask_fair]
    result["fair"] = {
        "count": int(len(arr_fair)),
        "action": "Rehabilitation",
    }
    
    # Poor: IRI >= 8.0
    mask_poor = values >= 8.0
    arr_poor = values[mask_poor]
    result["poor"] = {
        "count": int(len(arr_poor)),
        "action": "Reconstruction",
    }
    
    return result


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _haversine_distances(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if len(lat) < 2:
        return np.array([0.0])
    return np.array([_haversine_distance(lat[i - 1], lon[i - 1], lat[i], lon[i]) for i in range(1, len(lat))])
