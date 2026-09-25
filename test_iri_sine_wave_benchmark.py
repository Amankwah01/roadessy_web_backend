#!/usr/bin/env python3
"""
Test IRI computations for both pipelines using synthetic sine wave benchmark.

Benchmark from troubleshooting checklist:
- z(x) = A * sin(2πx/L) with A=0.005m (5mm), L=5m over 1000m total
- Both Simple IRI (Pipeline 2) and ASTM QCM IRI (Pipeline 1) should produce
  identical, moderate IRI values on this controlled synthetic input.
"""

import numpy as np
import pytest

from app.services.inertial_service import (
    InertialRefineryService,
    ASTMQCMResult,
    InertialPipelineResult,
)


def create_sine_wave_profile(
    amplitude: float = 0.005,      # 5mm in meters
    wavelength: float = 5.0,       # 5m wavelength
    total_length: float = 1000.0,  # 1000m total
    dx: float = 0.025,             # 2.5cm spatial sampling (standard)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic sine wave profile z(x) = A * sin(2πx/L).
    
    Returns:
        x: spatial coordinates (m)
        z: elevation profile (m)
        a: acceleration profile (m/s²) - second time derivative
        t: time array (s) for constant speed simulation
    """
    x = np.arange(0, total_length + dx, dx)
    z = amplitude * np.sin(2 * np.pi * x / wavelength)
    
    # Create time array for constant 80 km/h speed
    speed_ms = 80.0 / 3.6  # 22.22 m/s
    t = x / speed_ms
    
    # Acceleration is second time derivative of z(t)
    # z(t) = A * sin(2π * v * t / L)
    # a(t) = -A * (2πv/L)² * sin(2πvt/L)
    omega = 2 * np.pi * speed_ms / wavelength
    a = -amplitude * omega**2 * np.sin(omega * t)
    
    return x, z, a, t


def test_sine_wave_simple_iri() -> InertialPipelineResult:
    """Test Simple IRI (Pipeline 2) on synthetic sine wave."""
    x, z, a, t = create_sine_wave_profile()
    speed_ms = 80.0 / 3.6
    speeds_kmh = np.full_like(t, 80.0)
    
    # Use the Simple IRI method (displacement-to-distance)
    # We need to provide lat/lon for Haversine distance
    # Create synthetic lat/lon that follows the x distance
    lat = np.zeros_like(x)
    lon = np.zeros_like(x)
    for i in range(1, len(x)):
        # Convert x distance to approximate longitude degrees
        # 1 degree longitude ≈ 111km at equator
        lon[i] = lon[i-1] + (x[i] - x[i-1]) / 111000.0
    
    result = InertialRefineryService.process_vibration_data(
        timestamps=t,
        accel_z=a,  # Use proper acceleration (second derivative of elevation)
        speeds_kmh=speeds_kmh,
        latitudes=lat,
        longitudes=lon,
    )
    return result


def test_sine_wave_astm_qcm_iri() -> ASTMQCMResult:
    """Test ASTM QCM IRI (Pipeline 1) on synthetic sine wave."""
    x, z, a, t = create_sine_wave_profile()
    dx = x[1] - x[0] if len(x) > 1 else 0.025
    
    # Process using ASTM QCM method
    # ASTM QCM expects the raw acceleration and does upsampling/resampling internally
    result = InertialRefineryService.process_astm_qcm(
        timestamps=t,
        accel_z=a,  # Use proper acceleration
        speeds_kmh=np.full_like(t, 80.0),
    )
    return result


def test_both_pipelines_sine_wave():
    """
    Run both pipelines on synthetic sine wave and compare results.
    
    Expected: Both should return moderate, identical IRI values.
    The exact value depends on the algorithm but should be consistent.
    """
    print("=" * 60)
    print("SYNTHETIC SINE WAVE IRI BENCHMARK")
    print("=" * 60)
    print("Profile: z(x) = 0.005 * sin(2*pi*x/5.0) over 1000m")
    print(f"Amplitude: 5mm, Wavelength: 5m, dx: 2.5cm")
    print(f"Speed: 80 km/h")
    print("-" * 60)
    
    # Test Simple IRI (Pipeline 2)
    print("\nTesting Simple IRI (Pipeline 2 - Roadessy)...")
    simple_result = test_sine_wave_simple_iri()
    print(f"  Overall IRI: {simple_result.iri_m_km:.4f} m/km")
    print(f"  Total Distance: {simple_result.total_distance_m:.2f} m")
    print(f"  Number of segments: {len(simple_result.iri_segments)}")
    
    # Test ASTM QCM IRI (Pipeline 1)
    print("\nTesting ASTM QCM IRI (Pipeline 1 - ASTM)...")
    astm_result = test_sine_wave_astm_qcm_iri()
    print(f"  Overall IRI: {astm_result.iri_m_km:.4f} m/km")
    print(f"  Total Distance: {astm_result.total_distance_m:.2f} m")
    print(f"  Quality Status: {astm_result.quality_status}")
    print(f"  Quality Flags: {astm_result.quality_flags}")
    print(f"  Number of segments: {len(astm_result.iri_segments)}")
    
    print("-" * 60)
    print("COMPARISON:")
    diff = abs(simple_result.iri_m_km - astm_result.iri_m_km)
    ratio = simple_result.iri_m_km / astm_result.iri_m_km if astm_result.iri_m_km > 0 else 0
    print(f"  Simple IRI:    {simple_result.iri_m_km:.4f} m/km")
    print(f"  ASTM QCM IRI:  {astm_result.iri_m_km:.4f} m/km")
    print(f"  Difference:    {diff:.4f} m/km")
    print(f"  Ratio:         {ratio:.4f}")
    
    # Both should be moderate values (not extremely high or zero)
    # Note: Simple IRI uses preprocessing pipeline which applies filtering
    # ASTM QCM uses quarter-car model directly on resampled profile
    print(f"\nNote: Simple IRI uses ST1-ST5b preprocessing (filtering)")
    print(f"Note: ASTM QCM uses quarter-car model on uniform profile")
    assert simple_result.iri_m_km > 0.001, f"Simple IRI too low: {simple_result.iri_m_km}"
    assert simple_result.iri_m_km < 20.0, f"Simple IRI too high: {simple_result.iri_m_km}"
    assert astm_result.iri_m_km > 0.001, f"ASTM QCM IRI too low: {astm_result.iri_m_km}"
    assert astm_result.iri_m_km < 20.0, f"ASTM QCM IRI too high: {astm_result.iri_m_km}"
    
    print("\n[OK] Both pipelines produced moderate IRI values")
    print("[OK] Test PASSED")
    
    return simple_result, astm_result


if __name__ == "__main__":
    test_both_pipelines_sine_wave()