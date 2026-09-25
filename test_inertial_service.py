import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.inertial_service import (
    HP_CUTOFF,
    LP_CUTOFF,
    TARGET_FS,
    DX,
    InertialPipelineResult,
    InertialRefineryService,
    STANDARD_SPEED,
)


class TestInertialRefineryServiceColumnDetection:
    def test_detect_column_found(self):
        df = pd.DataFrame({"acc_x": [1], "speed": [2]})
        assert InertialRefineryService._detect_column(df, "acc") == "acc_x"

    def test_detect_column_not_found(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError):
            InertialRefineryService._detect_column(df, "acc")


class TestInertialRefineryServiceParseImuCsv:
    def test_parse_logger_csv_strips_headers_and_converts_kmh(self, tmp_path):
        csv_path = tmp_path / "logger.csv"
        csv_path.write_text(
            "Timestamp,LinAccelZ,SpeedKmh ,Latitude,Longitude,RecIntvl\n"
            "Tue Aug 04 07:04:44 GMT 2026,0.475,29.0,6.68,-1.57,1.0\n"
            "Tue Aug 04 07:04:45 GMT 2026,-0.720,29.0,6.68,-1.57,1.0\n"
        )
        time, acc, speed, acc_col, speed_col = InertialRefineryService.parse_imu_csv(str(csv_path))
        assert time.tolist() == [1.0, 2.0]
        assert acc_col == "LinAccelZ"
        assert speed_col == "SpeedKmh"
        assert speed[0] == pytest.approx(29.0 * 1000.0 / 3600.0)

    def test_parse_imu_csv_returns_arrays(self, tmp_path):
        csv_path = tmp_path / "imu.csv"
        csv_path.write_text(
            "RecIntvl,acc_x,speed\n0.01,0.1,5.0\n0.01,0.2,5.1\n0.01,0.3,5.2\n"
        )
        time, acc, speed, acc_col, speed_col = InertialRefineryService.parse_imu_csv(str(csv_path))
        assert len(time) == 3
        assert len(acc) == 3
        assert len(speed) == 3
        assert acc_col == "acc_x"
        assert speed_col == "speed"

    def test_parse_imu_csv_missing_recintvl(self, tmp_path):
        csv_path = tmp_path / "imu.csv"
        csv_path.write_text("acc_x,speed\n0.1,5.0\n")
        with pytest.raises(ValueError):
            InertialRefineryService.parse_imu_csv(str(csv_path))


class TestInertialRefineryServiceFilters:
    def test_highpass_shape(self):
        data = np.sin(np.linspace(0, 10, 100))
        result = InertialRefineryService.highpass(data, TARGET_FS, HP_CUTOFF)
        assert result.shape == data.shape

    def test_lowpass_shape(self):
        data = np.random.randn(100)
        result = InertialRefineryService.lowpass(data, TARGET_FS, LP_CUTOFF)
        assert result.shape == data.shape

    def test_butterworth_filter_shape(self):
        data = np.random.randn(100)
        result = InertialRefineryService.apply_butterworth_filter(data, TARGET_FS)
        assert result.shape == data.shape


class TestInertialRefineryServiceBuildRoadProfile:
    def test_stationary_input_returns_empty_profile(self):
        time = np.linspace(0, 5, 500)
        acc = np.sin(2 * np.pi * 2 * time)
        speed = np.zeros_like(time)
        d_uniform, profile, acc_filt = InertialRefineryService.build_road_profile(time, acc, speed)
        assert len(d_uniform) == 0
        assert len(profile) == 0
        assert len(acc_filt) > 0

    def test_stationary_acceleration_does_not_create_road_profile(self):
        time = np.linspace(0, 10, 1000)
        acc = np.zeros_like(time)
        acc[:500] = 20 * np.sin(2 * np.pi * 2 * time[:500])
        speed = np.zeros_like(time)
        speed[500:] = STANDARD_SPEED
        d_uniform, profile, _ = InertialRefineryService.build_road_profile(time, acc, speed)
        assert len(d_uniform) > 0
        assert len(profile) == len(d_uniform)

    def test_build_road_profile_outputs(self):
        time = np.linspace(0, 5, 500)
        acc = np.sin(2 * np.pi * 0.5 * time)
        speed = np.full_like(time, STANDARD_SPEED)
        d_uniform, profile, acc_filt = InertialRefineryService.build_road_profile(time, acc, speed)
        assert len(d_uniform) > 0
        assert len(profile) == len(d_uniform)
        assert len(acc_filt) > 0

    def test_build_road_profile_monotonic_distance(self):
        time = np.linspace(0, 5, 500)
        acc = np.zeros_like(time)
        speed = np.full_like(time, STANDARD_SPEED)
        d_uniform, profile, _ = InertialRefineryService.build_road_profile(time, acc, speed)
        assert np.all(np.diff(d_uniform) > 0)


class TestInertialRefineryServiceComputeIri:
    def test_compute_iri_flat_profile(self):
        profile = np.zeros(100)
        iri = InertialRefineryService.compute_iri(profile, DX)
        assert iri == pytest.approx(0.0, abs=1e-3)

    def test_compute_iri_ramp_profile(self):
        x = np.linspace(0, 10, 100)
        profile = 0.01 * x
        iri = InertialRefineryService.compute_iri(profile, DX)
        assert iri >= 0.0

    def test_compute_iri_returns_float(self):
        profile = np.random.randn(50)
        iri = InertialRefineryService.compute_iri(profile, DX)
        assert isinstance(iri, float)

    def test_compute_iri_is_invariant_to_profile_vertical_offset(self):
        x = np.arange(401, dtype=np.float64) * DX
        profile = 0.001 * np.sin(2 * np.pi * x / 25.0)
        offset_profile = profile + 100.0

        iri = InertialRefineryService.compute_iri(profile, DX)
        offset_iri = InertialRefineryService.compute_iri(offset_profile, DX)

        assert offset_iri == pytest.approx(iri, abs=1e-9)

    def test_compute_iri_does_not_clamp_values_above_screening_threshold(self):
        x = np.arange(401, dtype=np.float64) * DX
        profile = 0.1 * np.sin(2 * np.pi * x / 5.0)

        assert InertialRefineryService.compute_iri(profile, DX) > 50.0


class TestInertialRefineryServiceClassification:
    def test_classify_iri_5rank_excellent(self):
        result = InertialRefineryService.classify_iri_5rank(1.0)
        assert result["rank"] == "excellent"
        assert result["scale"] == "5-Rank Serviceability"

    def test_classify_iri_5rank_good(self):
        result = InertialRefineryService.classify_iri_5rank(3.5)
        assert result["rank"] == "good"

    def test_classify_iri_5rank_fair(self):
        result = InertialRefineryService.classify_iri_5rank(5.5)
        assert result["rank"] == "fair"

    def test_classify_iri_5rank_poor(self):
        result = InertialRefineryService.classify_iri_5rank(7.0)
        assert result["rank"] == "poor"

    def test_classify_iri_5rank_very_poor(self):
        result = InertialRefineryService.classify_iri_5rank(10.0)
        assert result["rank"] == "very_poor"

    def test_classify_iri_5rank_above_max(self):
        result = InertialRefineryService.classify_iri_5rank(15.0)
        assert result["rank"] == "very_poor"

    def test_classify_iri_3rank_good(self):
        result = InertialRefineryService.classify_iri_3rank(2.0)
        assert result["status"] == "GOOD"
        assert result["scale"] == "3-Rank Generalized Status"

    def test_classify_iri_3rank_fair(self):
        result = InertialRefineryService.classify_iri_3rank(5.5)
        assert result["status"] == "FAIR"

    def test_classify_iri_3rank_poor(self):
        result = InertialRefineryService.classify_iri_3rank(7.0)
        assert result["status"] == "POOR"

    def test_classify_iri_both(self):
        result = InertialRefineryService.classify_iri_both(5.5)
        assert "serviceability_5rank" in result
        assert "generalized_status_3rank" in result
        assert result["iri_value"] == 5.5
        assert result["source"] == "Ghana Road Condition Survey Manual 2023"


class TestInertialRefineryServiceEndToEnd:
    def test_process_imu_csv_returns_result(self, tmp_path):
        n = 200
        time = np.cumsum(np.full(n, 0.01))
        acc = np.sin(2 * np.pi * 0.5 * time) + np.random.randn(n) * 0.01
        speed = np.full(n, STANDARD_SPEED)
        df = pd.DataFrame({"RecIntvl": np.full(n, 0.01), "acc_x": acc, "speed": speed})
        csv_path = tmp_path / "imu.csv"
        df.to_csv(csv_path, index=False)

        result = InertialRefineryService.process_imu_csv(str(csv_path))
        assert isinstance(result, InertialPipelineResult)
        assert result.iri_m_km >= 0.0
        assert result.sample_rate_hz == pytest.approx(TARGET_FS)
        assert result.raw_sample_count == n
        assert result.resampled_sample_count > 0
        assert result.distance_sample_count > 0
        assert result.total_distance_m > 0.0
        assert result.input_sample_rate_hz == pytest.approx(100.0)
        assert result.moving_distance_m > 0.0
        assert result.reference_speed_kmh == pytest.approx(80.0)
        assert result.quality_status in {"usable_with_validation_required", "insufficient_moving_distance"}
        assert result.acceleration_column == "acc_x"
        assert result.speed_column == "speed"
        assert len(result.profile) == result.distance_sample_count
        assert len(result.distance_axis) == result.distance_sample_count
        assert len(result.filtered_acceleration) == result.resampled_sample_count
