#!/usr/bin/env python3
"""
IRI computation tests for both pipelines, on a synthetic sine wave.

Benchmark from the troubleshooting checklist:
- z(x) = A * sin(2*pi*x/L) with A=0.005m (5mm), L=5m over 1000m total
- Simple IRI (Pipeline 2) and ASTM QCM IRI (Pipeline 1) are different methods
  applied to the same profile, so they are expected to *agree closely*, not to
  return the same number: measured here they differ by about 9%.

The file is also runnable as a script, which prints the full comparison table.
"""

import numpy as np

from app.services.inertial import (
    ASTMQCMResult,
    InertialPipelineResult,
    InertialRefineryService,
)

#: Measured values for the profile below, used as the bounds both pipelines must
#: stay inside. A pipeline that silently returns zeros, or that loses a factor of
#: 1000 in its resampling, stays within "finite and positive" but leaves this
#: band, so the band is what carries the regression check.
IRI_MIN_M_KM = 3.0
IRI_MAX_M_KM = 6.0

#: How far apart the two methods may land, as a fraction of their mean.
CROSS_METHOD_TOLERANCE = 0.20

#: The profile is 1000 m of track, so a pipeline that mis-times its own
#: integration cannot be within a metre of this.
DISTANCE_TOLERANCE_M = 5.0


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


def run_simple_iri() -> InertialPipelineResult:
    """Run Simple IRI (Pipeline 2) on the synthetic sine wave."""
    x, _, a, t = create_sine_wave_profile()
    speed_ms = 80.0 / 3.6
    speeds_kmh = np.full_like(t, 80.0)
    
    # The Simple IRI method integrates displacement to distance, so it needs
    # coordinates. Synthesised from x: 1 degree longitude ~ 111km at the equator.
    lat = np.zeros_like(x)
    lon = np.zeros_like(x)
    for i in range(1, len(x)):
        lon[i] = lon[i-1] + (x[i] - x[i-1]) / 111000.0
    
    return InertialRefineryService.process_vibration_data(
        timestamps=t,
        accel_z=a,  # Use proper acceleration (second derivative of elevation)
        speeds_kmh=speeds_kmh,
        latitudes=lat,
        longitudes=lon,
    )


def run_astm_qcm_iri() -> ASTMQCMResult:
    """Run ASTM QCM IRI (Pipeline 1) on the synthetic sine wave.

    The method expects raw acceleration and does its own upsampling/resampling,
    so unlike the Simple IRI path it needs no coordinates.
    """
    _, _, a, t = create_sine_wave_profile()
    return InertialRefineryService.process_astm_qcm(
        timestamps=t,
        accel_z=a,
        speeds_kmh=np.full_like(t, 80.0),
    )


def test_sine_wave_simple_iri():
    """Simple IRI must land in the measured band and integrate the full profile."""
    result = run_simple_iri()

    assert np.isfinite(result.iri_m_km), f"Simple IRI not finite: {result.iri_m_km}"
    assert IRI_MIN_M_KM <= result.iri_m_km <= IRI_MAX_M_KM, (
        f"Simple IRI outside the measured band for this profile: {result.iri_m_km}"
    )
    assert abs(result.total_distance_m - 1000.0) < DISTANCE_TOLERANCE_M, (
        f"Simple IRI integrated {result.total_distance_m} m of a 1000 m profile"
    )
    assert len(result.iri_segments) > 0


def test_sine_wave_astm_qcm_iri():
    """ASTM QCM IRI must land in the measured band and pass its own QA checks."""
    result = run_astm_qcm_iri()

    assert np.isfinite(result.iri_m_km), f"ASTM QCM IRI not finite: {result.iri_m_km}"
    assert IRI_MIN_M_KM <= result.iri_m_km <= IRI_MAX_M_KM, (
        f"ASTM QCM IRI outside the measured band for this profile: {result.iri_m_km}"
    )
    assert abs(result.total_distance_m - 1000.0) < DISTANCE_TOLERANCE_M, (
        f"ASTM QCM integrated {result.total_distance_m} m of a 1000 m profile"
    )
    # A profile this clean should not be flagged; if it is, the filtering or
    # resampling upstream has started rejecting good data.
    assert result.quality_status == "good", f"unexpected quality status: {result.quality_flags}"
    assert not result.quality_flags


def test_both_pipelines_agree_on_the_same_profile():
    """The two methods are different, but not free to disagree by much.

    They see the same profile through different filters, so a large divergence
    means one of them is broken rather than merely different.
    """
    simple_result = run_simple_iri()
    astm_result = run_astm_qcm_iri()

    mean = (simple_result.iri_m_km + astm_result.iri_m_km) / 2.0
    difference = abs(simple_result.iri_m_km - astm_result.iri_m_km)
    assert difference / mean < CROSS_METHOD_TOLERANCE, (
        f"pipelines disagree by {difference:.4f} m/km "
        f"({difference / mean:.1%}): simple={simple_result.iri_m_km}, "
        f"astm={astm_result.iri_m_km}"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("SYNTHETIC SINE WAVE IRI BENCHMARK")
    print("=" * 60)
    print("Profile: z(x) = 0.005 * sin(2*pi*x/5.0) over 1000m")
    print("Amplitude: 5mm, Wavelength: 5m, dx: 2.5cm")
    print("Speed: 80 km/h")
    print("-" * 60)

    print("\nTesting Simple IRI (Pipeline 2 - Roadessy)...")
    simple_result = run_simple_iri()
    print(f"  Overall IRI: {simple_result.iri_m_km:.4f} m/km")
    print(f"  Total Distance: {simple_result.total_distance_m:.2f} m")
    print(f"  Number of segments: {len(simple_result.iri_segments)}")

    print("\nTesting ASTM QCM IRI (Pipeline 1 - ASTM)...")
    astm_result = run_astm_qcm_iri()
    print(f"  Overall IRI: {astm_result.iri_m_km:.4f} m/km")
    print(f"  Total Distance: {astm_result.total_distance_m:.2f} m")
    print(f"  Quality Status: {astm_result.quality_status}")
    print(f"  Quality Flags: {astm_result.quality_flags}")
    print(f"  Number of segments: {len(astm_result.iri_segments)}")

    print("-" * 60)
    print("COMPARISON:")
    difference = abs(simple_result.iri_m_km - astm_result.iri_m_km)
    mean = (simple_result.iri_m_km + astm_result.iri_m_km) / 2.0
    print(f"  Simple IRI:    {simple_result.iri_m_km:.4f} m/km")
    print(f"  ASTM QCM IRI:  {astm_result.iri_m_km:.4f} m/km")
    print(f"  Difference:    {difference:.4f} m/km ({difference / mean:.1%})")

    print("\nNote: Simple IRI uses ST1-ST5b preprocessing (filtering)")
    print("Note: ASTM QCM uses quarter-car model on uniform profile")
    test_sine_wave_simple_iri()
    test_sine_wave_astm_qcm_iri()
    test_both_pipelines_agree_on_the_same_profile()
    print("\n[OK] All benchmark assertions passed")