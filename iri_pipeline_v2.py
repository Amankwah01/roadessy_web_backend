from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

from app.services.inertial_service import InertialRefineryService

FloatArray = NDArray[np.float64]


def compute_iri_simple(profile: FloatArray, distance_km: FloatArray) -> float:
    if len(profile) == 0 or len(distance_km) == 0:
        return 0.0
    total_distance = float(distance_km[-1])
    if total_distance <= 0:
        return 0.0
    accumulated_vert_disp = float(np.sum(np.abs(profile)))
    iri = (accumulated_vert_disp / total_distance) * 100.0
    return float(iri)


def compute_iri_segmented_simple(profile: FloatArray, distance_km: FloatArray, interval_m: float = 100.0):
    interval_km = interval_m / 1000.0
    segments = []
    overall_accum = 0.0
    overall_distance = 0.0
    accumulated_distance = 0.0
    accumulated_vert_disp = 0.0
    seg_start_m = 0.0
    seg_idx = 0

    for i in range(len(profile)):
        d_km = distance_km[i] if i == 0 else max(float(distance_km[i] - distance_km[i - 1]), 0.0)
        accumulated_distance += d_km
        accumulated_vert_disp += float(np.abs(profile[i]))

        if accumulated_distance >= interval_km and accumulated_distance > 0:
            iri = (accumulated_vert_disp / accumulated_distance) * 100.0
            seg_end_m = seg_start_m + interval_m
            segments.append({
                "start_distance_m": round(seg_start_m, 2),
                "end_distance_m": round(seg_end_m, 2),
                "iri_m_km": round(iri, 4),
                "sample_count": 1,
            })
            overall_accum += iri * (seg_end_m - seg_start_m) / 1000.0
            overall_distance += (seg_end_m - seg_start_m)
            seg_start_m = seg_end_m
            accumulated_distance = 0.0
            accumulated_vert_disp = 0.0
            seg_idx += 1

    if accumulated_distance > 0:
        seg_end_m = seg_start_m + accumulated_distance * 1000.0
        iri = (accumulated_vert_disp / accumulated_distance) * 100.0
        segments.append({
            "start_distance_m": round(seg_start_m, 2),
            "end_distance_m": round(seg_end_m, 2),
            "iri_m_km": round(iri, 4),
            "sample_count": 1,
        })
        overall_accum += iri * (seg_end_m - seg_start_m) / 1000.0
        overall_distance += (seg_end_m - seg_start_m)

    overall_iri = (overall_accum / overall_distance * 1000.0) if overall_distance > 0 else 0.0
    return float(overall_iri), segments


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("2025-12-11_07-15-52_Mempeasem_Trial 1.csv")

result = InertialRefineryService.process_imu_csv_simple("2025-12-11_07-15-52_Mempeasem_Trial 1.csv")
print(f"\nEstimated IRI (simple): {result.iri_m_km:.2f} mm/m")

overall_iri, segments = compute_iri_segmented_simple(result.profile, result.distance_axis)
print(f"Segmented IRI (100m): {overall_iri:.2f} mm/m across {len(segments)} segments")
for seg in segments:
    print(f"  {seg['start_distance_m']:.0f}m - {seg['end_distance_m']:.0f}m: {seg['iri_m_km']:.2f} mm/m")

# =========================
# PLOT
# =========================
plt.figure(figsize=(10,5))
plt.plot(result.distance_axis, result.profile)
plt.title("Road Profile (Simple IRI)")
plt.xlabel("Distance (km)")
plt.ylabel("Vertical Displacement (m)")
plt.grid()
plt.savefig("iri_profile_simple.png")
plt.show()

# Auto-detect columns
acc_col = None
for candidate in ["LinAccelZ", "AccelZ"]:
    for c in df.columns:
        if c.lower() == candidate.lower():
            acc_col = c
            break
    if acc_col:
        break

if not acc_col:
    for c in df.columns:
        if "acc" in c.lower() and "z" in c.lower():
            acc_col = c
            break

if not acc_col:
    for c in df.columns:
        if "acc" in c.lower():
            acc_col = c
            break

speed_col = None
for c in df.columns:
    if "speed" in c.lower():
        speed_col = c
        break

if acc_col is None or speed_col is None:
    raise ValueError("Acceleration or speed column not found")

print(f"Using acceleration: {acc_col}")
print(f"Using speed: {speed_col}")

# =========================
# TIME RECONSTRUCTION
# =========================
df['dt'] = df['RecIntvl']
df['time'] = df['dt'].cumsum()

time: FloatArray = df['time'].to_numpy(dtype=np.float64)
acc: FloatArray = df[acc_col].to_numpy(dtype=np.float64)
speed: FloatArray = df[speed_col].to_numpy(dtype=np.float64)

# =========================
# RESAMPLING (CRITICAL)
# =========================
t_uniform = np.arange(0, time[-1], 1 / TARGET_FS)

acc_u = np.interp(t_uniform, time, acc)
speed_u = np.interp(t_uniform, time, speed)

# =========================
# FILTERS
# =========================
def highpass(data: FloatArray, fs: float, cutoff: float) -> FloatArray:
    b, a = cast(
        tuple[FloatArray, FloatArray],
        butter(2, cutoff / (fs / 2), btype='highpass', output='ba'),
    )
    return np.asarray(filtfilt(b, a, data), dtype=np.float64)

def lowpass(data: FloatArray, fs: float, cutoff: float) -> FloatArray:
    b, a = cast(
        tuple[FloatArray, FloatArray],
        butter(4, cutoff / (fs / 2), btype='lowpass', output='ba'),
    )
    return np.asarray(filtfilt(b, a, data), dtype=np.float64)

acc_hp = highpass(acc_u, TARGET_FS, HP_CUTOFF)
acc_filt = lowpass(acc_hp, TARGET_FS, LP_CUTOFF)

# =========================
# INTEGRATION
# =========================
vel = np.cumsum(acc_filt) / TARGET_FS
disp = np.cumsum(vel) / TARGET_FS

# =========================
# DISTANCE DOMAIN
# =========================
distance = np.cumsum(speed_u) / TARGET_FS

# Remove bad values
mask = np.isfinite(distance) & np.isfinite(disp)
distance = distance[mask]
disp = disp[mask]

# Ensure monotonic
valid = np.diff(distance) > 0
distance = distance[:-1][valid]
disp = disp[:-1][valid]

# Uniform spacing
d_uniform = np.arange(0, distance[-1], DX)
profile = np.interp(d_uniform, distance, disp)

# =========================
# QUARTER CAR MODEL (IRI)
# =========================
def compute_iri(profile: FloatArray, dx: float, speed: float = 80 / 3.6) -> float:
    """
    profile: road elevation (m)
    dx: spacing (m)
    speed: vehicle speed (m/s), standard = 80 km/h
    """

    # Golden car parameters
    ms = 250.0
    mu = 30.0
    ks = 63300.0
    kt = 653000.0
    cs = 6000.0

    n = len(profile)

    # States:
    # x0 = x1 = sprung displacement
    # x1 = x2 = sprung velocity
    # x2 = x3 = unsprung displacement
    # x3 = x4 = unsprung velocity
    x = np.zeros(4)

    iri_accum = 0.0

    dt = dx / speed  # convert spatial step to time

    for i in range(n - 1):
        zr = profile[i]

        # State derivatives
        def f(x: FloatArray, zr: float) -> FloatArray:
            x1, x2, x3, x4 = x

            dx1 = x2
            dx2 = (-ks*(x1 - x3) - cs*(x2 - x4)) / ms
            dx3 = x4
            dx4 = (ks*(x1 - x3) + cs*(x2 - x4) - kt*(x3 - zr)) / mu

            return np.array([dx1, dx2, dx3, dx4])

        # RK4 integration
        k1 = f(x, zr)
        k2 = f(x + 0.5*dt*k1, zr)
        k3 = f(x + 0.5*dt*k2, zr)
        k4 = f(x + dt*k3, zr)

        x = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        # IRI accumulates absolute relative velocity over time
        iri_accum += abs(x[1] - x[3]) * dt

    # Total accumulated travel divided by total distance
    total_distance = (n - 1) * dx
    iri = iri_accum / total_distance  # m/m
    return iri * 1000  # mm/m


def _compute_iri_for_profile(profile: FloatArray, dx: float, speed: float = 80 / 3.6) -> float:
    ms = 250.0
    mu = 30.0
    ks = 63300.0
    kt = 653000.0
    cs = 6000.0

    n = len(profile)
    x = np.zeros(4)
    iri_accum = 0.0
    dt = dx / speed

    for i in range(n - 1):
        zr = profile[i]

        def f(x: FloatArray, zr: float) -> FloatArray:
            x1, x2, x3, x4 = x

            dx1 = x2
            dx2 = (-ks * (x1 - x3) - cs * (x2 - x4)) / ms
            dx3 = x4
            dx4 = (ks * (x1 - x3) + cs * (x2 - x4) - kt * (x3 - zr)) / mu

            return np.array([dx1, dx2, dx3, dx4])

        k1 = f(x, zr)
        k2 = f(x + 0.5 * dt * k1, zr)
        k3 = f(x + 0.5 * dt * k2, zr)
        k4 = f(x + dt * k3, zr)

        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        iri_accum += abs(x[1] - x[3]) * dt

    total_distance = (n - 1) * dx
    if total_distance <= 0:
        return 0.0
    iri = iri_accum / total_distance
    return iri * 1000


def compute_iri_segmented(profile: FloatArray, dx: float, speed: float = 80 / 3.6):
    SEGMENT_LENGTH_M = 100.0
    points_per_segment = int(SEGMENT_LENGTH_M / dx)
    if points_per_segment < 2:
        raise ValueError(f"Segment length {SEGMENT_LENGTH_M}m too short for dx={dx}m")

    segments = []
    overall_accum = 0.0
    overall_distance = 0.0
    n = len(profile)
    num_segments = (n - 1) // points_per_segment

    for seg_idx in range(num_segments):
        start_idx = seg_idx * points_per_segment
        end_idx = min(start_idx + points_per_segment + 1, n)
        segment_profile = profile[start_idx:end_idx]

        if len(segment_profile) < 2:
            continue

        seg_iri = _compute_iri_for_profile(segment_profile, dx, speed)
        seg_dist_start = start_idx * dx
        seg_dist_end = min((start_idx + points_per_segment) * dx, (n - 1) * dx)

        segments.append({
            "start_distance_m": round(seg_dist_start, 2),
            "end_distance_m": round(seg_dist_end, 2),
            "iri_m_km": round(seg_iri, 4),
            "sample_count": len(segment_profile) - 1,
        })

        seg_length = seg_dist_end - seg_dist_start
        overall_accum += seg_iri * seg_length / 1000.0
        overall_distance += seg_length

    remaining_start = num_segments * points_per_segment
    if remaining_start < n - 1:
        remaining_profile = profile[remaining_start:]
        remaining_iri = _compute_iri_for_profile(remaining_profile, dx, speed)
        remaining_dist_start = remaining_start * dx
        remaining_dist_end = (n - 1) * dx

        segments.append({
            "start_distance_m": round(remaining_dist_start, 2),
            "end_distance_m": round(remaining_dist_end, 2),
            "iri_m_km": round(remaining_iri, 4),
            "sample_count": len(remaining_profile) - 1,
        })

        seg_length = remaining_dist_end - remaining_dist_start
        overall_accum += remaining_iri * seg_length / 1000.0
        overall_distance += seg_length

    overall_iri = (overall_accum / overall_distance * 1000.0) if overall_distance > 0 else 0.0
    return float(overall_iri), segments


iri_est = compute_iri(profile, DX)
print(f"\nEstimated IRI: {iri_est:.2f} mm/m")

overall_iri, segments = compute_iri_segmented(profile, DX)
print(f"Segmented IRI (100m): {overall_iri:.2f} mm/m across {len(segments)} segments")
for seg in segments:
    print(f"  {seg['start_distance_m']:.0f}m - {seg['end_distance_m']:.0f}m: {seg['iri_m_km']:.2f} mm/m")

# =========================
# PLOT
# =========================
plt.figure(figsize=(10,5))
plt.plot(d_uniform, profile)
plt.title("Road Profile (Filtered & Corrected)")
plt.xlabel("Distance (m)")
plt.ylabel("Displacement (m)")
plt.grid()
plt.savefig("iri_profile.png")
plt.show()
