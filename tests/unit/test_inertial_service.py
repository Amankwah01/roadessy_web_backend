"""Tests for the inertial (IRI) service.

Scoped to what `InertialRefineryService` actually exposes. The previous version
of this file tested `_detect_column`, `parse_imu_csv`, `highpass`, `lowpass`,
`apply_butterworth_filter`, `build_road_profile`, `compute_iri`,
`classify_iri_3rank` and `classify_iri_both`; none of those exist on the class
and no production code called them, so the file failed on a phantom API. These
cover the real surface: the Haversine distance helper, uniform-distance
resampling, the quarter-car simulation, the simple displacement pipeline, the
ASTM QCM pipeline, CSV ingestion, and 5-rank IRI classification.

Expectations were read off the implementation and confirmed by running it
rather than inferred, since the earlier expectations described a different
service entirely.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.inertial import (
    ASTMQCMResult,
    InertialPipelineResult,
    InertialRefineryService,
)

TARGET_DX = 0.025
REFERENCE_SPEED_KMH = 80.0


def _track(n=300, interval_s=0.01, speed_kmh=60.0, freq=0.5):
    """A synthetic drive: known timestamps, speed, and a straight GPS path."""
    t = np.cumsum(np.full(n, interval_s))
    accel = np.sin(2 * np.pi * freq * t)
    speeds = np.full(n, speed_kmh)
    # ~5.5m of latitude per sample, giving a long enough run to emit segments.
    lat = 5.0 + np.arange(n) * 0.00005
    lon = np.zeros(n)
    return t, accel, speeds, lat, lon


class TestHaversineDistances:
    def test_single_point_has_no_distance(self):
        assert list(InertialRefineryService._haversine_distances_m(np.array([5.0]), np.array([0.0]))) == [0.0]

    def test_distance_starts_at_zero(self):
        """Cumulative distance is measured from the first fix, not from zero metres."""
        d = InertialRefineryService._haversine_distances_m(
            np.array([5.0, 5.001]), np.array([0.0, 0.0])
        )
        assert d[0] == 0.0
        assert d[1] > 0.0

    def test_is_cumulative_and_monotonic(self):
        d = InertialRefineryService._haversine_distances_m(
            np.full(5, 5.0), np.array([0.0, 0.001, 0.002, 0.003, 0.004])
        )
        assert np.all(np.diff(d) > 0)
        assert d[-1] == pytest.approx(d[1] * 4, rel=0.01)

    def test_one_degree_of_longitude_is_about_111km(self):
        """Sanity check on the earth radius: 0.001deg lon at the equator is ~111m."""
        d = InertialRefineryService._haversine_distances_m(
            np.array([0.0, 0.0]), np.array([0.0, 0.001])
        )
        assert d[1] == pytest.approx(111.2, rel=0.01)

    def test_stationary_track_accumulates_nothing(self):
        d = InertialRefineryService._haversine_distances_m(
            np.full(10, 5.6), np.zeros(10)
        )
        assert d[-1] == pytest.approx(0.0)

    def test_empty_input_returns_empty(self):
        assert len(InertialRefineryService._haversine_distances_m(np.array([]), np.array([]))) == 0

    def test_accepts_pandas_series(self):
        """The pipeline hands over pandas columns, not numpy arrays."""
        s = pd.Series([5.0, 5.001])
        d = InertialRefineryService._haversine_distances_m(s, pd.Series([0.0, 0.0]))
        assert d[1] > 0.0


class TestResampleToUniformDistance:
    def test_too_short_input_degrades_to_a_single_point(self):
        d, p = InertialRefineryService._resample_to_uniform_distance(
            np.array([0.0]), np.array([1.0])
        )
        assert len(d) == 1 and len(p) == 1

    def test_resampling_produces_uniform_spacing(self):
        distances = np.linspace(0, 10, 50)
        profile = np.sin(distances)
        d, p = InertialRefineryService._resample_to_uniform_distance(distances, profile)
        assert len(d) == len(p)
        assert np.allclose(np.diff(d), TARGET_DX)

    def test_duplicate_distances_are_dropped(self):
        """A GPS stall repeats a position; those zero-length steps cannot be interpolated."""
        distances = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0])
        profile = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0])
        d, p = InertialRefineryService._resample_to_uniform_distance(distances, profile)
        assert np.all(np.diff(d) > 0)
        assert len(d) == len(p)

    def test_non_finite_samples_are_dropped(self):
        distances = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        profile = np.array([0.0, np.nan, 2.0, np.inf, 4.0])
        d, p = InertialRefineryService._resample_to_uniform_distance(distances, profile)
        assert np.all(np.isfinite(p))

    def test_custom_target_spacing_is_honoured(self):
        distances = np.linspace(0, 10, 50)
        profile = np.sin(distances)
        d, _ = InertialRefineryService._resample_to_uniform_distance(
            distances, profile, target_dx=0.5
        )
        assert np.allclose(np.diff(d), 0.5)


class TestQuarterCarIri:
    def test_flat_profile_has_no_roughness(self):
        assert InertialRefineryService._compute_quarter_car_iri(np.zeros(200), TARGET_DX) == 0.0

    def test_rough_profile_scores_above_zero(self):
        profile = 0.02 * np.sin(np.linspace(0, 40 * np.pi, 200))
        assert InertialRefineryService._compute_quarter_car_iri(profile, TARGET_DX) > 0.0

    def test_amplitude_raises_the_score(self):
        """Rougher input means more suspension travel, so a higher IRI."""
        mild = 0.005 * np.sin(np.linspace(0, 40 * np.pi, 200))
        severe = 0.05 * np.sin(np.linspace(0, 40 * np.pi, 200))
        assert (
            InertialRefineryService._compute_quarter_car_iri(severe, TARGET_DX)
            > InertialRefineryService._compute_quarter_car_iri(mild, TARGET_DX)
        )

    def test_degenerate_profile_returns_zero(self):
        assert InertialRefineryService._compute_quarter_car_iri(np.array([1.0]), TARGET_DX) == 0.0


class TestProcessVibrationData:
    def test_insufficient_samples_return_an_empty_result(self):
        """Too few points to filter meaningfully yields zeros, not an exception."""
        result = InertialRefineryService.process_vibration_data(np.arange(5), np.zeros(5))
        assert isinstance(result, InertialPipelineResult)
        assert result.iri_m_km == 0.0
        assert result.total_distance_m == 0.0
        assert result.iri_segments == []

    def test_produces_segments_along_the_drive(self):
        t, accel, speeds, lat, lon = _track()
        result = InertialRefineryService.process_vibration_data(t, accel, speeds, latitudes=lat, longitudes=lon)
        assert result.total_distance_m > 0.0
        assert len(result.iri_segments) > 0
        assert result.iri_m_km >= 0.0

    def test_segments_are_contiguous_and_ordered(self):
        t, accel, speeds, lat, lon = _track()
        segments = InertialRefineryService.process_vibration_data(
            t, accel, speeds, latitudes=lat, longitudes=lon
        ).iri_segments
        for previous, current in zip(segments, segments[1:]):
            assert current.start_distance_m >= previous.start_distance_m
            assert current.end_distance_m > current.start_distance_m

    def test_no_gps_means_no_distance_and_no_segments(self):
        """Distance is Haversine from GPS, so a track without fixes measures nothing.

        Emitting an IRI here would mean dividing by a distance of zero.
        """
        t, accel, speeds, _, _ = _track()
        result = InertialRefineryService.process_vibration_data(t, accel, speeds)
        assert result.total_distance_m == 0.0
        assert result.iri_segments == []
        assert result.iri_m_km == 0.0

    def test_stationary_vehicle_produces_no_segments(self):
        t = np.cumsum(np.full(300, 0.01))
        result = InertialRefineryService.process_vibration_data(
            t, np.zeros(300), np.zeros(300), latitudes=np.full(300, 5.6), longitudes=np.zeros(300)
        )
        assert result.iri_segments == []


class TestProcessAstmQcm:
    def test_clean_run_is_reported_good_with_no_flags(self):
        t, accel, speeds, _, _ = _track(n=1000, interval_s=0.01, speed_kmh=REFERENCE_SPEED_KMH, freq=2.0)
        result = InertialRefineryService.process_astm_qcm(t, accel, speeds)
        assert isinstance(result, ASTMQCMResult)
        assert result.sample_rate_hz == 100.0
        assert result.quality_flags == []
        assert result.quality_status == "good"
        assert result.total_distance_m > 0.0
        assert result.iri_m_km >= 0.0
        assert len(result.iri_segments) > 0

    def test_insufficient_data_is_flagged_not_raised(self):
        t, accel, speeds, _, _ = _track(n=5)
        result = InertialRefineryService.process_astm_qcm(t[:5], accel[:5], speeds[:5])
        assert result.iri_m_km == 0.0
        assert "insufficient_data" in result.quality_flags
        assert result.quality_status == "warning"

    def test_low_sampling_rate_is_flagged(self):
        """Below 100Hz the run is not ASTM E950 Class 1, and says so."""
        t = np.cumsum(np.full(500, 0.02))  # 50Hz
        accel = np.sin(2 * np.pi * 2 * t)
        result = InertialRefineryService.process_astm_qcm(t, accel, np.full(500, REFERENCE_SPEED_KMH))
        assert "low_sampling_rate" in result.quality_flags
        assert result.astm_compliance["E950_class1_sampling"] is not True

    def test_short_distance_is_flagged(self):
        t, accel, speeds, _, _ = _track(n=100, speed_kmh=REFERENCE_SPEED_KMH)
        result = InertialRefineryService.process_astm_qcm(t, accel, speeds)
        assert "insufficient_distance" in result.quality_flags

    def test_speed_variation_is_flagged(self):
        """A coefficient of variation over 0.2 means the speed was not steady."""
        n = 2000
        t = np.cumsum(np.full(n, 0.01))
        accel = np.sin(2 * np.pi * 2 * t)
        speeds = np.where(np.arange(n) < n // 2, 20.0, 120.0)
        result = InertialRefineryService.process_astm_qcm(t, accel, speeds)
        assert "speed_variation" in result.quality_flags

    def test_distance_comes_from_speed_not_gps(self):
        """The QCM simulates at a reference speed, so it needs no GPS at all.

        Distance is speed x time by design; taking it from GPS would mix a real
        road's length into a simulated run.
        """
        t, accel, speeds, _, _ = _track(n=1000, speed_kmh=REFERENCE_SPEED_KMH, freq=2.0)
        result = InertialRefineryService.process_astm_qcm(t, accel, speeds)
        expected = (REFERENCE_SPEED_KMH / 3.6) * (len(t) / 100.0)
        assert result.total_distance_m == pytest.approx(expected, rel=0.05)

    def test_non_finite_samples_are_dropped(self):
        t, accel, speeds, _, _ = _track(n=1000, speed_kmh=REFERENCE_SPEED_KMH, freq=2.0)
        accel = accel.copy()
        accel[:50] = np.nan
        result = InertialRefineryService.process_astm_qcm(t, accel, speeds)
        assert np.isfinite(result.iri_m_km)
        assert result.total_distance_m > 0.0

    def test_upsampling_info_records_the_conversion(self):
        t, accel, speeds, _, _ = _track(n=1000, speed_kmh=REFERENCE_SPEED_KMH, freq=2.0)
        info = InertialRefineryService.process_astm_qcm(t, accel, speeds).upsampling_info
        assert info["target_fs_hz"] == 100.0
        assert "method" in info

    def test_quarter_car_parameters_are_reported(self):
        """The Golden Car constants are published so a result can be reproduced."""
        t, accel, speeds, _, _ = _track(n=1000, speed_kmh=REFERENCE_SPEED_KMH, freq=2.0)
        params = InertialRefineryService.process_astm_qcm(t, accel, speeds).quarter_car_params
        assert params["v_ref_kmh"] == REFERENCE_SPEED_KMH
        for key in ("ms", "mu", "ks", "kt", "cs"):
            assert key in params


class TestProcessImuCsv:
    def test_logger_export_with_datetime_timestamps(self, tmp_path):
        """A real logger export: datetime stamps, a padded speed header, and GPS."""
        path = tmp_path / "logger.csv"
        rows = [
            f"Tue Aug 04 07:04:{i % 60:02d} GMT 2026,0.475,29.0,5.0,{i * 0.0001},1.0"
            for i in range(200)
        ]
        path.write_text(
            "Timestamp,LinAccelZ,SpeedKmh ,Latitude,Longitude,RecIntvl\n" + "\n".join(rows) + "\n"
        )
        result = InertialRefineryService.process_imu_csv(str(path))
        assert isinstance(result, InertialPipelineResult)
        # Columns are matched after stripping, so "SpeedKmh " is still found.
        assert result.total_distance_m > 0.0
        assert len(result.iri_segments) > 0

    def test_without_gps_no_distance_is_measured(self, tmp_path):
        path = tmp_path / "plain.csv"
        n = 200
        pd.DataFrame(
            {
                "RecIntvl": np.full(n, 0.01),
                "acc_x": np.sin(np.arange(n) * 0.1),
                "speed": np.full(n, 60.0),
            }
        ).to_csv(path, index=False)
        result = InertialRefineryService.process_imu_csv(str(path))
        assert result.total_distance_m == 0.0
        assert result.iri_segments == []

    def test_recintvl_is_used_when_timestamps_are_unusable(self, tmp_path):
        """`RecIntvl` is authoritative, so a bad time column still yields a real span."""
        path = tmp_path / "recintvl.csv"
        n = 150
        pd.DataFrame(
            {
                "Timestamp": ["not-a-date"] * n,
                "acc_x": np.sin(np.arange(n) * 0.2),
                "speed": np.full(n, 60.0),
                "Latitude": 5.0 + np.arange(n) * 0.00005,
                "Longitude": np.zeros(n),
                "RecIntvl": np.full(n, 0.01),
            }
        ).to_csv(path, index=False)
        result = InertialRefineryService.process_imu_csv(str(path))
        assert result.total_distance_m > 0.0
        assert len(result.iri_segments) > 0

    def test_simple_alias_matches_the_main_entry_point(self, tmp_path):
        path = tmp_path / "imu.csv"
        n = 200
        pd.DataFrame(
            {
                "RecIntvl": np.full(n, 0.01),
                "acc_x": np.sin(np.arange(n) * 0.2),
                "speed": np.full(n, 60.0),
                "Latitude": 5.0 + np.arange(n) * 0.00005,
                "Longitude": np.zeros(n),
            }
        ).to_csv(path, index=False)
        main = InertialRefineryService.process_imu_csv(str(path))
        alias = InertialRefineryService.process_imu_csv_simple(str(path))
        assert alias.iri_m_km == main.iri_m_km
        assert alias.total_distance_m == main.total_distance_m

    def test_non_finite_rows_are_dropped(self, tmp_path):
        path = tmp_path / "gaps.csv"
        n = 200
        accel = np.sin(np.arange(n) * 0.2)
        accel[:20] = np.nan
        pd.DataFrame(
            {
                "RecIntvl": np.full(n, 0.01),
                "acc_x": accel,
                "speed": np.full(n, 60.0),
                "Latitude": 5.0 + np.arange(n) * 0.00005,
                "Longitude": np.zeros(n),
            }
        ).to_csv(path, index=False)
        result = InertialRefineryService.process_imu_csv(str(path))
        assert np.isfinite(result.iri_m_km)


class TestClassifyIri5Rank:
    @pytest.mark.parametrize(
        "iri,expected",
        [
            (0.0, "Very Good"),
            (1.9, "Very Good"),
            (2.0, "Good"),
            (3.9, "Good"),
            (4.0, "Fair"),
            (5.9, "Fair"),
            (6.0, "Poor"),
            (7.9, "Poor"),
            (8.0, "Very Poor"),
        ],
    )
    def test_each_band(self, iri, expected):
        assert InertialRefineryService.classify_iri_5rank(iri)["rank"] == expected

    def test_above_the_scale_stays_very_poor(self):
        """An IRI past the top of the scale is not clamped into a better band."""
        assert InertialRefineryService.classify_iri_5rank(50.0)["rank"] == "Very Poor"

    def test_returns_a_numeric_code(self):
        for iri in (1.0, 3.0, 5.0, 7.0, 9.0):
            result = InertialRefineryService.classify_iri_5rank(iri)
            assert result["code"] in {1, 2, 3, 4, 5}

    def test_code_increases_as_iri_worsens(self):
        codes = [InertialRefineryService.classify_iri_5rank(v)["code"] for v in (1.0, 3.0, 5.0, 7.0, 9.0)]
        assert codes == sorted(codes)

    def test_band_boundaries_are_half_open(self):
        """A value exactly on a bound belongs to the worse band.

        The comparison is `<`, so 2.0 is Good rather than Very Good; the
        boundary is included once, not twice.
        """
        assert InertialRefineryService.classify_iri_5rank(2.0)["rank"] == "Good"
        assert InertialRefineryService.classify_iri_5rank(1.999)["rank"] == "Very Good"
