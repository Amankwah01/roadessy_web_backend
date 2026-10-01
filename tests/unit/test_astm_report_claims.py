"""The run report is the artifact a reader sees, so its labels are a claim.

`astm_pci` cannot be renamed without a DB migration and an API break, and the
field name alone is not what someone takes away from a report. What they read is
the row label and the qualification beside it. These tests pin that surface, so
the word "ASTM" cannot creep back onto a number whose curves are provisional.
"""

import asyncio
import html.parser
import re
import uuid
from pathlib import Path

import pytest

from app.api.routes import runs
from app.services.astm import ASTMD6433PCI


# --------------------------------------------------------------------------
# the limitation record, which the report renders
# --------------------------------------------------------------------------

class TestLimitations:
    def test_it_states_what_the_number_is_not(self):
        assert ASTMD6433PCI.LIMITATIONS["not"] == \
            "an ASTM D6433 compliance value"

    def test_it_names_the_calculation_edition_and_the_curve_edition(self):
        """Two different editions, and the gap is the whole reason for the
        provisional label. Collapsing them into one string would hide it."""
        limits = ASTMD6433PCI.LIMITATIONS
        assert "D6433-24" in limits["method"]
        assert "D6433-99" in limits["curves"]
        assert "provisional" in limits["curves"]

    def test_every_reason_is_a_full_sentence_a_reader_could_understand(self):
        for reason in ASTMD6433PCI.LIMITATIONS["reasons"]:
            assert reason.strip().endswith(".")
            assert len(reason) > 40

    def test_the_sampling_difference_is_stated_first(self):
        """It is the limitation no amount of better curve data can clear, so
        it must not be buried after the digitisation caveats."""
        first = ASTMD6433PCI.LIMITATIONS["reasons"][0]
        assert "census" in first
        assert "random" in first

    def test_the_bands_are_flagged_as_not_the_ghana_vocabulary(self):
        assert "Good/Fair/Poor" in ASTMD6433PCI.LIMITATIONS["ratings_are"]


# --------------------------------------------------------------------------
# the generated report
# --------------------------------------------------------------------------

class FakeSegment:
    segment_end_m = 120.0
    astm_qcm_iri = 2.1
    astm_pci_segment = 91.4
    start_distance_m = 0.0
    end_distance_m = 50.0
    iri_m_km = 2.0


class FakeRun:
    run_id = uuid.UUID(int=1)
    name = "Test run"
    created_at = None
    astm_pci = 88.25
    roadessy_pci = 85.0
    simple_iri = 2.4
    astm_qcm_iri = 2.1


class FakeQuery:
    def filter(self, *a, **k):
        return self

    def count(self):
        return 4

    def all(self):
        return [FakeSegment()]

    def first(self):
        return FakeRun()


class FakeDB:
    def query(self, *a, **k):
        return FakeQuery()


class TagBalance(html.parser.HTMLParser):
    VOID = {"br", "hr", "img", "meta", "link", "input"}

    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append("unbalanced </%s>" % tag)
            return
        self.stack.pop()


@pytest.fixture
def document(tmp_path, monkeypatch):
    """Generate a real report through the real endpoint and return the HTML."""
    import app.api.routes.analysis as analysis

    monkeypatch.setattr(
        analysis, "_run_workspace_candidates", lambda run_id: [tmp_path])

    asyncio.run(runs.generate_run_report(str(uuid.UUID(int=1)), db=FakeDB()))
    report = tmp_path / ("run_report_%s.html" % uuid.UUID(int=1))
    return report.read_text(encoding="utf-8")


def _labels(document):
    return re.findall(r"<th[^>]*>([^<]*)</th>", document)


def _text(document):
    return " ".join(re.sub(r"<[^>]+>", " ", document).split())


class TestReportLabels:
    def test_the_pci_row_is_labelled_as_provisional(self, document):
        assert "PCI (D6433-derived, provisional)" in _labels(document)

    def test_no_row_presents_the_number_as_an_astm_value(self, document):
        labels = _labels(document)
        assert "ASTM PCI" not in labels
        assert "ASTM IRI" not in labels

    def test_the_value_itself_is_still_reported(self, document):
        """Renaming the label is not a reason to drop the measurement."""
        assert "88.25" in _text(document)

    def test_the_other_indexes_keep_their_own_names(self, document):
        """Only the D6433 number is qualified. Roadessy PCI and Simple IRI are
        separate measures and must not be swept up in the caveat."""
        labels = _labels(document)
        assert "Roadessy PCI" in labels
        assert "Simple IRI" in labels


class TestReportQualification:
    def test_it_travels_with_the_number(self, document):
        assert "About the D6433-derived PCI" in _text(document)

    def test_it_says_the_number_is_not_a_compliance_value(self, document):
        text = _text(document)
        assert "It is not an ASTM D6433 compliance value" in text

    def test_the_sentence_is_not_garbled_by_a_capitalize_call(self, document):
        """Regression guard.

        Interpolating the `not` phrase after "It is" produced "It is an ASTM
        D6433 compliance value. It is not a compliance value" -- directly
        self-contradicting, because `.capitalize()` also lowercased "ASTM" and
        "D6433" into "astm d6433".
        """
        text = _text(document)
        assert "It is an ASTM" not in text
        assert "An astm" not in text
        assert "astm d6433" not in text

    def test_it_names_the_method_and_the_curve_provenance(self, document):
        text = _text(document)
        assert "ASTM D6433-24" in text
        assert "D6433-99" in text
        assert "provisional first-pass digitization" in text

    def test_all_three_reasons_reach_the_reader(self, document):
        text = _text(document)
        assert "census of the whole section" in text
        assert "systematic rather than random" in text
        assert "rated on condition" in text

    def test_it_flags_the_band_vocabulary(self, document):
        assert "Good/Fair/Poor" in _text(document)


class TestReportIsWellFormed:
    def test_the_tags_balance(self, document):
        parser = TagBalance()
        parser.feed(document)
        assert parser.stack == []
        assert parser.errors == []

    def test_it_is_a_complete_html_document(self, document):
        assert document.startswith("<!doctype html>")
        assert document.rstrip().endswith("</html>")

    def test_interpolated_text_is_escaped(self, document):
        """The reasons contain apostrophes. Unescaped, they would be the only
        thing standing between this and a malformed document."""
        assert "&#x27;" in document
        # No raw quote from the data can close an attribute early.
        assert re.search(r"=\s*\"[^\"]*'[^\"]*\"", document) is None
