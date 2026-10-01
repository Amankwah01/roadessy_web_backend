"""Verify quarter-car method on reference CSV with RecIntvl-based timing."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from app.services.inertial_service import InertialRefineryService

CSV = "2025-12-11_07-15-52_Mempeasem_Trial 1.csv"

# Reconstruct time from RecIntvl (as reference pipeline does)
df = pd.read_csv(CSV)
df.columns = [str(c).strip() for c in df.columns]
rec_col = df.columns[[c.lower() for c in df.columns].index("recintvl")]
dt = pd.to_numeric(df[rec_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
time = np.cumsum(dt)
print(f"RecIntvl-based time: len={len(time)}, last={time[-1]}, median_dt={np.median(np.diff(time))}, fs={1/np.median(np.diff(time))}")

acc_col = "LinAccelZ"
speed_col = "SpeedKmh"
acc = pd.to_numeric(df[acc_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
speed = pd.to_numeric(df[speed_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
if "kmh" in speed_col.lower():
    speed = speed * 1000.0 / 3600.0

# Use the new process_vibration_data (simple displacement method)
simple_result = InertialRefineryService.process_vibration_data(time, acc, speed)
print(f"\nSimple displacement IRI: {simple_result.iri_m_km:.4f} m/km")
print(f"  Total distance: {simple_result.total_distance_m:.2f} m")
print(f"  Segments: {len(simple_result.iri_segments)}")

# Use the new process_astm_qcm (ASTM quarter-car method)
astm_qcm = InertialRefineryService.process_astm_qcm(
    timestamps=time, accel_z=acc, speeds_kmh=speed * 3.6
)
print(f"\nASTM QCM IRI: {astm_qcm.iri_m_km:.4f} m/km")
print(f"  Total distance: {astm_qcm.total_distance_m:.2f} m")
print(f"  Quality status: {astm_qcm.quality_status}")
print(f"  Quality flags: {astm_qcm.quality_flags}")
print(f"  Segments: {len(astm_qcm.iri_segments)}")
for seg in astm_qcm.iri_segments:
    print(f"  {seg.start_distance_m:.0f}-{seg.end_distance_m:.0f}m: iri={seg.iri_m_km:.4f}")