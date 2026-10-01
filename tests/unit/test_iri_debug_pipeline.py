#!/usr/bin/env python3
"""
Debug test to understand Simple IRI pipeline intermediate steps.
"""

import numpy as np
import pandas as pd

from app.services.inertial import InertialRefineryService
from app.services.pipeline.preprocessing import PreprocessingPipeline


def create_sine_wave_profile(
    amplitude: float = 0.005,      # 5mm in meters
    wavelength: float = 5.0,       # 5m wavelength
    total_length: float = 1000.0,  # 1000m total
    dx: float = 0.025,             # 2.5cm spatial sampling
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic sine wave profile z(x) = A * sin(2πx/L)."""
    x = np.arange(0, total_length + dx, dx)
    z = amplitude * np.sin(2 * np.pi * x / wavelength)
    
    speed_ms = 80.0 / 3.6  # 22.22 m/s
    t = x / speed_ms
    
    # Acceleration is second time derivative of z(t)
    omega = 2 * np.pi * speed_ms / wavelength
    a = -amplitude * omega**2 * np.sin(omega * t)
    
    return x, z, a, t


def debug_preprocessing_pipeline():
    """Debug the ST1-ST5b preprocessing pipeline step by step."""
    x, z, a, t = create_sine_wave_profile()
    speed_ms = 80.0 / 3.6
    speeds_kmh = np.full_like(t, 80.0)
    
    # Create synthetic lat/lon
    lat = np.zeros_like(x)
    lon = np.zeros_like(x)
    for i in range(1, len(x)):
        lon[i] = lon[i-1] + (x[i] - x[i-1]) / 111000.0
    
    print("=" * 70)
    print("DEBUGGING SIMPLE IRI PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Input signal stats:")
    print(f"  accel_z (a): min={a.min():.6f}, max={a.max():.6f}, mean={a.mean():.6f}")
    print(f"  timestamps: min={t.min():.3f}, max={t.max():.3f}, len={len(t)}")
    print(f"  speeds_kmh: min={speeds_kmh.min():.1f}, max={speeds_kmh.max():.1f}")
    print()
    
    # Build DataFrame for preprocessing
    n = len(a)
    df = pd.DataFrame({
        'time_ms': t * 1000,
        'accel_x': np.zeros_like(a),
        'accel_y': np.zeros_like(a),
        'accel_z': a,
        'gravi_x': np.zeros_like(a),
        'gravi_y': np.zeros_like(a),
        'gravi_z': np.full_like(a, 9.81),
        'lat': lat,
        'long': lon,
        'gps_speed': speeds_kmh,
        'time_ms': t * 1000,
    })
    
    print("Running preprocessing pipeline (ST1-ST5b)...")
    processed = PreprocessingPipeline.run_pipeline(df)
    
    print("\nPreprocessed output columns:", list(processed.columns))
    print(f"Processed length: {len(processed)}")
    
    for col in ['vertical_displacement', 'time_ms', 'gps_speed', 'lat', 'long']:
        if col in processed.columns:
            vals = processed[col].values
            if vals is not None and len(vals) > 0:
                print(f"  {col}: min={np.nanmin(vals):.6f}, max={np.nanmax(vals):.6f}, mean={np.nanmean(vals):.6f}, NaN%={np.isnan(vals).sum()/len(vals)*100:.1f}%")
    
    # Check displacement specifically
    disp = processed['vertical_displacement'].values
    if disp is not None:
        print(f"\nVertical displacement stats:")
        print(f"  min={np.nanmin(disp):.6f}, max={np.nanmax(disp):.6f}, mean={np.nanmean(disp):.6f}")
        print(f"  abs min={np.nanmin(np.abs(disp)):.6f}, abs max={np.nanmax(np.abs(disp)):.6f}")
        print(f"  sum(|disp|)={np.nansum(np.abs(disp)):.6f}")
    
    return processed


def debug_haversine_distance():
    """Debug Haversine distance calculation."""
    x, z, a, t = create_sine_wave_profile()
    speed_ms = 80.0 / 3.6
    
    lat = np.zeros_like(x)
    lon = np.zeros_like(x)
    for i in range(1, len(x)):
        lon[i] = lon[i-1] + (x[i] - x[i-1]) / 111000.0
    
    print("\n" + "=" * 70)
    print("DEBUGGING HAVERSINE DISTANCE")
    print("=" * 70)
    
    distances_m = InertialRefineryService._haversine_distances_m(lat, lon)
    print(f"Total distance: {distances_m[-1]:.3f} m")
    print(f"Expected ~1000m, got {distances_m[-1]:.3f} m")
    
    # Check per-step distances
    deltas = np.diff(distances_m)
    print(f"Mean step distance: {np.mean(deltas)*100:.3f} cm")
    print(f"dx expected: 2.5cm")


def debug_iri_segmentation():
    """Debug the IRI segment accumulation logic."""
    x, z, a, t = create_sine_wave_profile()
    speed_ms = 80.0 / 3.6
    
    lat = np.zeros_like(x)
    lon = np.zeros_like(x)
    for i in range(1, len(x)):
        lon[i] = lon[i-1] + (x[i] - x[i-1]) / 111000.0
    
    df = pd.DataFrame({
        'time_ms': t * 1000,
        'accel_x': np.zeros_like(a),
        'accel_y': np.zeros_like(a),
        'accel_z': a,
        'gravi_x': np.zeros_like(a),
        'gravi_y': np.zeros_like(a),
        'gravi_z': np.full_like(a, 9.81),
        'lat': lat,
        'long': lon,
        'gps_speed': np.full_like(a, 80.0),
        'time_ms': t * 1000,
    })
    
    processed = PreprocessingPipeline.run_pipeline(df)
    displacement = processed['vertical_displacement'].values
    vert_disp_m = displacement
    speeds_kmh = processed['gps_speed'].values
    
    lat_vals = processed['lat'].values
    lon_vals = processed['long'].values
    distances_m = InertialRefineryService._haversine_distances_m(lat_vals, lon_vals)
    total_distance = float(distances_m[-1])
    
    print("\n" + "=" * 70)
    print("DEBUGGING IRI SEGMENTATION (ACCUMULATION METHOD)")
    print("=" * 70)
    
    segment_length_m = 50.0  # IRI_SEGMENT_LENGTH_M
    segment_length_km = segment_length_m / 1000.0
    
    accumulated_dist = 0.0
    accumulated_disp = 0.0
    speed_sum = 0.0
    speed_count = 0
    iri_segments = []
    
    print(f"Segment length: {segment_length_m} m ({segment_length_km} km)")
    print(f"Total distance: {total_distance:.2f} m")
    print(f"Expected segments: {int(total_distance / segment_length_m)}")
    print()
    
    for i in range(len(distances_m)):
        dist_delta = distances_m[i] - (distances_m[i - 1] if i > 0 else 0.0)
        vert = float(np.abs(vert_disp_m[i]))
        sp = float(speeds_kmh[i]) if speeds_kmh is not None and i < len(speeds_kmh) else 0.0
        
        accumulated_dist += dist_delta
        accumulated_disp += vert
        
        if sp > 0:
            speed_sum += sp
            speed_count += 1
        
        if accumulated_dist >= segment_length_m:
            avg_speed = speed_sum / speed_count if speed_count > 0 else 0.0
            seg_iri = accumulated_disp / (accumulated_dist / 1000.0) if accumulated_dist > 0 else 0.0
            
            print(f"Segment {len(iri_segments)+1}:")
            print(f"  accumulated_dist = {accumulated_dist:.3f} m")
            print(f"  accumulated_disp = {accumulated_disp:.6f} m")
            print(f"  speed_count = {speed_count}")
            print(f"  seg_iri = {seg_iri:.6f} m/km")
            print(f"  avg_speed = {avg_speed:.2f} km/h")
            
            iri_segments.append(seg_iri)
            accumulated_dist = 0.0
            accumulated_disp = 0.0
            speed_sum = 0.0
            speed_count = 0
    
    overall_iri = float(np.mean(iri_segments)) if iri_segments else 0.0
    print(f"\nSegments generated: {len(iri_segments)}")
    print(f"Overall IRI: {overall_iri:.6f} m/km")
    
    # Also compute theoretical IRI for sine wave
    # For z(x) = A sin(2πx/L), slope dz/dx = A*(2π/L)*cos(2πx/L)
    # IRI from simple method ≈ (1/L_seg) ∫ |dz/dx| dx over segment
    # For sinusoidal: avg |cos| = 2/π
    # So IRI ≈ A * (2π/L) * (2/π) * 1000 = 4*A/L * 1000 (in m/km)
    # Wait, IRI = sum(|displacement|) / distance_km
    # For sine wave, displacement per sample = |dz| ≈ |dz/dx| * dx
    # sum(|displacement|) ≈ ∫ |dz/dx| dx
    # dz/dx = A*(2π/L)*cos(2πx/L)
    # avg |dz/dx| = A*(2π/L) * (2/π) = 4A/L
    # Over distance D, sum(|disp|) ≈ (4A/L) * D
    # IRI = sum(|disp|) / (D/1000) = 4*A/L * 1000 (m/km)
    # For A=0.005, L=5: IRI = 4*0.005/5*1000 = 4 m/km
    print(f"\nTheoretical Simple IRI for sine wave (A=5mm, L=5m):")
    print(f"  4*A/L * 1000 = 4*0.005/5*1000 = 4.0 m/km")
    print(f"  (This assumes no filtering in displacement calculation)")


def debug_qcm_vs_simple():
    """Compare QCM and Simple approaches conceptually."""
    print("\n" + "=" * 70)
    print("CONCEPTUAL COMPARISON: QCM vs SIMPLE")
    print("=" * 70)
    print("Simple IRI (Roadessy/Pipeline 2):")
    print("  - Uses ST1-ST5b preprocessing on raw acceleration")
    print("  - ST2: Speed filter (removes points with speed < 5 km/h)")
    print("  - ST3: Filtering (high-pass, low-pass)")
    print("  - ST4: Double integration (accel -> velocity -> displacement)")
    print("  - ST5: Location filter (keeps variation if std > 1e-6)")
    print("  - ST5b: Vertical displacement calculation (per-sample)")
    print("  - IRI = sum(|displacement|) / distance_km")
    print("  - Accumulation approach: emits segment when accumulated_dist >= 50m")
    print("  - Skips stopped periods naturally")
    print()
    print("ASTM QCM IRI (Pipeline 1):")
    print("  - Resamples to uniform spatial profile (dx=constant)")
    print("  - Fixed-range segmentation on uniform profile")
    print("  - Quarter-car model simulation (RK4)")
    print("  - No stopped-period issue (profile already resampled)")
    print("  - IRI = accumulated suspension motion / distance")
    print()
    print("Key difference: Simple IRI applies extensive preprocessing filters")
    print("  which may attenuate the signal before integration.")
    print("  QCM works on resampled uniform profile with quarter-car model.")


if __name__ == "__main__":
    debug_preprocessing_pipeline()
    debug_haversine_distance()
    debug_iri_segmentation()
    debug_qcm_vs_simple()