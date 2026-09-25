import pytest

from app.services.fusion_service import PavementConditionFusioneer


class TestPavementConditionFusioneerEvaluateSeverity:
    def test_linear_slight(self):
        assert PavementConditionFusioneer.evaluate_severity("linear", 5.0) == "slight"

    def test_linear_moderate_lower_boundary(self):
        assert PavementConditionFusioneer.evaluate_severity("linear", 6.0) == "moderate"

    def test_linear_moderate_upper_boundary(self):
        assert PavementConditionFusioneer.evaluate_severity("linear", 25.0) == "moderate"

    def test_linear_severe(self):
        assert PavementConditionFusioneer.evaluate_severity("linear", 30.0) == "severe"

    def test_pothole_slight(self):
        assert PavementConditionFusioneer.evaluate_severity("pothole", 100.0) == "slight"

    def test_pothole_moderate_upper_boundary(self):
        assert PavementConditionFusioneer.evaluate_severity("pothole", 500.0) == "moderate"

    def test_pothole_severe(self):
        assert PavementConditionFusioneer.evaluate_severity("pothole", 600.0) == "severe"

    def test_unknown_mode_passthrough(self):
        assert PavementConditionFusioneer.evaluate_severity("unknown", 100.0) == "slight"

    def test_mode_alias_resolution(self):
        assert PavementConditionFusioneer.evaluate_severity("crack", 30.0) == "severe"
        assert PavementConditionFusioneer.evaluate_severity("erosion_gully", 100.0, "erosion_gully") == "moderate"
        assert PavementConditionFusioneer.evaluate_severity("erosion_gully", 50.0, "erosion_gully") == "slight"
        assert PavementConditionFusioneer.evaluate_severity("erosion_gully", 250.0, "erosion_gully") == "severe"
        assert PavementConditionFusioneer.evaluate_severity("erosion_gully", 250.0) == "severe"


class TestPavementConditionFusioneerCalculateDeductValue:
    def test_linear_slight_localized(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "linear", 0.5, 5.0, 3.5
        )
        assert code == 1
        assert dv >= 0

    def test_linear_moderate_intermittent(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "linear", 2.0, 10.0, 3.5
        )
        assert code == 4

    def test_linear_severe_prevalent(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "linear", 5.0, 30.0, 3.5
        )
        assert code == 7

    def test_pothole_slight_localized(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "pothole", 0.1, 50.0, 3.5
        )
        assert code == 1

    def test_pothole_moderate_intermittent(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "pothole", 1.0, 300.0, 3.5
        )
        assert code == 4

    def test_pothole_severe_prevalent(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "pothole", 5.0, 600.0, 3.5
        )
        assert code == 7

    def test_severity_override(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "linear", 0.1, 1.0, 3.5, severity_override="severe"
        )
        assert code == 7

    def test_extent_override(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "linear", 0.1, 1.0, 3.5, extent_override="prevalent"
        )
        assert code == 3

    def test_extreme_severity_same_as_severe(self):
        code_s, _ = PavementConditionFusioneer.calculate_deduct_value(
            "linear", 0.1, 30.0, 3.5, severity_override="severe"
        )
        code_e, _ = PavementConditionFusioneer.calculate_deduct_value(
            "linear", 0.1, 30.0, 3.5, severity_override="extreme"
        )
        assert code_s == code_e

    def test_all_nine_defect_types(self):
        defect_types = [
            "bleeding",
            "corrugation",
            "crack",
            "erosion_gully",
            "patching",
            "pothole",
            "ravelling",
            "rutting",
            "shoving",
        ]
        for defect in defect_types:
            code, dv = PavementConditionFusioneer.calculate_deduct_value(
                defect, 1.0, 10.0, 3.5
            )
            assert 1 <= code <= 9
            assert dv >= 0

    def test_area_based_extent_localized(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "pothole", 1.0, 50.0, 3.5
        )
        density = (1.0 / (100.0 * 3.5)) * 100.0
        assert density < 30.0
        assert code == 1

    def test_area_based_extent_intermittent(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "pothole", 120.0, 50.0, 3.5
        )
        density = (120.0 / (100.0 * 3.5)) * 100.0
        assert 30.0 <= density <= 70.0
        assert code == 5

    def test_area_based_extent_prevalent(self):
        code, dv = PavementConditionFusioneer.calculate_deduct_value(
            "pothole", 300.0, 50.0, 3.5
        )
        density = (300.0 / (100.0 * 3.5)) * 100.0
        assert density > 70.0
        assert code == 7


class TestPavementConditionFusioneerComputePciCascade:
    def test_empty_registry(self):
        pci, breakdown = PavementConditionFusioneer.compute_pci_cascade({}, 2.0, 3.5)
        assert pci == 98.0
        assert breakdown == []

    def test_single_defect(self):
        registry = {
            1: {
                "defect_type": "crack",
                "mode": "linear",
                "max_area": 0.5,
                "max_width": 5.0,
                "severity": "slight",
                "extent": "localized",
            }
        }
        pci, breakdown = PavementConditionFusioneer.compute_pci_cascade(registry, 1.0, 3.5)
        assert 0.0 <= pci <= 100.0
        assert len(breakdown) == 1
        assert breakdown[0]["track_id"] == 1
        assert breakdown[0]["failure_mode"] == "crack"

    def test_multiple_defects_correction_factor(self):
        registry = {
            1: {
                 "defect_type": "pothole",
                "mode": "pothole",
                "max_area": 5.0,
                "max_width": 600.0,
                "severity": "severe",
                "extent": "prevalent",
            },
            2: {
                "defect_type": "crack",
                "mode": "linear",
                "max_area": 2.0,
                "max_width": 30.0,
                "severity": "severe",
                "extent": "prevalent",
            },
        }
        pci, breakdown = PavementConditionFusioneer.compute_pci_cascade(registry, 1.0, 3.5)
        assert 0.0 <= pci <= 100.0
        assert len(breakdown) == 2
        assert all("deduct_value" in r for r in breakdown)

    def test_iri_only_returns_high_pci(self):
        registry = {}
        pci, breakdown = PavementConditionFusioneer.compute_pci_cascade(registry, 0.5, 3.5)
        assert pci == 99.5
        assert breakdown == []

    def test_breakdown_report_fields(self):
        registry = {
            1: {
                "defect_type": "rutting",
                "mode": "pothole",
                "max_area": 1.0,
                "max_width": 300.0,
                "max_depth_mm": 25.0,
                "mean_depth_mm": 20.0,
                "depth_confidence": 0.8,
                "depth_method": "sfm",
                "depth_message": "ok",
                "severity": "moderate",
                "extent": "intermittent",
                "classification": {"score": 5},
                "measurement_source": "vision+depth",
                "geom_location": "SRID=4326;POINT(0 0)",
                "geom": {"lat": 0.0, "lon": 0.0},
            }
        }
        pci, breakdown = PavementConditionFusioneer.compute_pci_cascade(registry, 1.0, 3.5)
        report = breakdown[0]
        assert report["max_area_m2"] == round(1.0, 4)
        assert report["max_width_mm"] == round(300.0, 2)
        assert report["max_depth_mm"] == 25.0
        assert report["mean_depth_mm"] == 20.0
        assert report["depth_confidence"] == 0.8
        assert report["depth_method"] == "sfm"
        assert report["depth_message"] == "ok"
        assert report["severity"] == "moderate"
        assert report["extent"] == "intermittent"
        assert report["classification"] == {"score": 5}
        assert report["measurement_source"] == "vision+depth"
        assert report["geom_location"] == "SRID=4326;POINT(0 0)"

    def test_pci_clamped_to_zero(self):
        registry = {
            1: {
                 "defect_type": "pothole",
                "mode": "pothole",
                "max_area": 100.0,
                "max_width": 1000.0,
            }
        }
        pci, _ = PavementConditionFusioneer.compute_pci_cascade(registry, 50.0, 3.5)
        assert pci >= 0.0
        assert pci <= 100.0

    def test_pci_clamped_to_hundred(self):
        registry = {}
        pci, _ = PavementConditionFusioneer.compute_pci_cascade(registry, 0.0, 3.5)
        assert pci == 100.0
