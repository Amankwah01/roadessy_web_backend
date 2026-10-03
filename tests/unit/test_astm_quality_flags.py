"""The ASTM result's quality_flags column.

`ASTMQCMResult.quality_flags` is a list of flag names. The fusion layer folded it
into the JSON column with `dict(flags, pci_provenance=...)`, which raised
"dictionary update sequence element #0 has length 17; 2 is required" — `dict`
read the first flag, "low_sampling_rate", as a key/value pair. That killed runs
after ~14 minutes of vision and IRI work, once enough flags had accumulated for
the list to be non-empty.

The column is a mapping: the frontend types it `Record<string, boolean>`.
"""
import pytest

from app.services.fusion.service import PavementConditionFusioneer
from app.services.inertial.service import ASTMQCMResult, IRISegmentResult
from app.services.vision import ASTMVisionResult


def _qcm(flags):
    return ASTMQCMResult(
        iri_m_km=2.5,
        total_distance_m=1200.0,
        iri_segments=[
            IRISegmentResult(
                start_distance_m=0.0,
                end_distance_m=50.0,
                iri_m_km=2.4,
                sample_count=100,
            )
        ],
        sample_rate_hz=100.0,
        quality_status="warning" if flags else "good",
        quality_flags=list(flags),
    )


class TestFlagFolding:
    def test_flag_names_are_flag_names_in_the_source_result(self):
        """The precondition behind the crash: a list, not a mapping."""
        flags = ["low_sampling_rate", "speed_variation", "profile_gaps"]

        assert _qcm(flags).quality_flags == flags
        assert not isinstance(_qcm(flags).quality_flags, dict)

    def test_the_flag_list_is_not_something_dict_can_consume(self):
        """Pins the exact failure so the shape change cannot silently regress."""
        with pytest.raises(ValueError, match="length 17"):
            dict(["low_sampling_rate", "speed_variation"])

    @pytest.mark.parametrize(
        "flags",
        [
            [],
            ["low_sampling_rate"],
            ["low_sampling_rate", "speed_variation", "profile_gaps"],
            ["insufficient_distance"],
        ],
    )
    def test_every_flag_list_folds_into_a_mapping(self, flags):
        folded = {str(flag): True for flag in flags}

        assert folded == {flag: True for flag in flags}
        assert all(value is True for value in folded.values())

    def test_the_folded_shape_matches_the_frontend_type(self):
        flags = ["low_sampling_rate", "speed_variation"]
        column = {
            **{str(flag): True for flag in flags},
            "pci_provenance": "muted",
        }

        assert isinstance(column, dict)
        assert column["low_sampling_rate"] is True
        assert column["pci_provenance"] == "muted"

    def test_provenance_is_present_even_with_no_flags(self):
        """A clean survey still records why PCI is what it is."""
        column = {
            **{str(flag): True for flag in []},
            "pci_provenance": None,
        }

        assert column == {"pci_provenance": None}


class TestFusionProducesTheColumn:
    """Exercises the real entry point.

    The expression-level tests above pass whether or not `compute_astm_pipeline`
    folds the list correctly, so these call it. With the old `dict(flags, ...)`
    this raised ValueError only once a run had accumulated flags, which is why it
    surfaced as an intermittent failure after ~14 minutes of pipeline work.
    """

    @pytest.fixture(autouse=True)
    def astm_track_on(self, monkeypatch):
        """Opt in to the muted ASTM track, because a muted one folds no flags.

        While the track is off, `compute_astm_pipeline` returns a fixed payload
        whose `quality_flags` holds only `astm_track_disabled`, so these tests
        would pass on a mapping they never produced — the exact kind of vacuous
        pass this file exists to prevent.
        """
        from app.services.fusion import service as fusion_service

        monkeypatch.setattr(fusion_service, "ASTM_PCI_ENABLED", True)
        monkeypatch.setattr(fusion_service, "ASTM_TRACK_RUNNING", True)

    @pytest.mark.parametrize(
        "flags",
        [
            [],
            ["low_sampling_rate"],
            ["low_sampling_rate", "speed_variation"],
            ["low_sampling_rate", "speed_variation", "profile_gaps"],
            ["quarter_car_divergence", "insufficient_distance"],
        ],
    )
    def test_no_flag_list_raises(self, flags):
        result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=_qcm(flags),
            astm_vision=ASTMVisionResult(),
            run_id="test-flags",
        )

        assert isinstance(result["quality_flags"], dict)
        for flag in flags:
            assert result["quality_flags"][flag] is True

    def test_every_flag_becomes_a_true_entry(self):
        flags = ["low_sampling_rate", "speed_variation", "profile_gaps"]

        result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=_qcm(flags),
            astm_vision=ASTMVisionResult(),
            run_id="test-flags",
        )

        column = result["quality_flags"]
        for flag in flags:
            assert column[flag] is True
        assert column["pci_provenance"] is not None or "pci_provenance" in column

    def test_the_column_serialises_as_json(self):
        """It lands in a JSON column, so a non-serialisable value would fail at
        commit time instead of here."""
        import json

        result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=_qcm(["low_sampling_rate", "speed_variation"]),
            astm_vision=ASTMVisionResult(),
            run_id="test-flags",
        )

        assert json.loads(json.dumps(result["quality_flags"]))