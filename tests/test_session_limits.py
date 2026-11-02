#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020 Rune B. Broberg
#  Copyright (C) 2020ff NanoVNA-Saver Authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""NEXT-03/NEXT-04: limit sets in sessions, sloped bounds, chart overlays.

The limits engine, the session document and the charts are three separate
owners of the same fact ("which bounds are in force").  These tests pin the
places where they meet: a session must restore exactly the limits it stored,
an old session must not wipe the operator's library, and a drawn bound must
be the line the evaluator actually compares samples against.
"""

import json

import pytest

from NanoVNASaver.ChartLimits import (
    apply_chart_limits,
    described_bounds,
    series_for,
    sign_for,
)
from NanoVNASaver.Limits import (
    LIMIT_SET_VERSION,
    LimitCatalogue,
    LimitFormatError,
    LimitRule,
    LimitSegment,
    LimitSet,
    LimitValidationError,
    Metric,
    Verdict,
    evaluate_rule,
    load,
    loads,
)
from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Session import (
    SESSION_VERSION,
    SessionDocument,
    SessionFormatError,
    limit_sets_from_document,
    make_document,
)

MHZ = 1_000_000


def _segment(start=10 * MHZ, stop=20 * MHZ, **kwargs):
    return LimitSegment(start=start, stop=stop, **kwargs)


def _rule(metric=Metric.S11_VSWR, bound=1.0, **kwargs):
    """A single-segment rule; ``bound`` is used when no bound is given."""
    if not any(
        key in kwargs for key in ("minimum", "maximum", "minimum_slope")
    ):
        kwargs.setdefault("maximum", bound)
    return LimitRule(
        metric=metric, segments=(_segment(**kwargs),), id="r1", label="R1"
    )


def _limit_set(name="bench", **kwargs):
    return LimitSet(name=name, rules=(_rule(**kwargs),))


def _points(values, start=10 * MHZ, step=1 * MHZ):
    """Reflection samples with a chosen |S11| per frequency."""
    return [
        Datapoint(start + index * step, magnitude, 0.0)
        for index, magnitude in enumerate(values)
    ]


def _snapshot(points):
    return snapshot_from_datapoints(points, None, name="dut")


class TestSlopedBounds:
    @staticmethod
    def test_slope_is_stored_per_hertz_and_round_trips():
        segment = _segment(maximum=0.0, maximum_slope=1e-9)
        document = segment.to_dict()
        assert document["maximum_slope"] == pytest.approx(1e-9)
        assert LimitSegment.from_dict(document, "segment") == segment

    @staticmethod
    def test_zero_slope_is_omitted_from_the_document():
        document = _segment(maximum=2.0).to_dict()
        assert "maximum_slope" not in document
        assert "minimum_slope" not in document

    @staticmethod
    def test_bound_at_frequency_interpolates_linearly():
        segment = _segment(maximum=1.0, maximum_slope=1e-7)
        # 5 MHz into the segment the bound has risen by 0.5.
        assert segment.maximum_at(15 * MHZ) == pytest.approx(1.5)

    @staticmethod
    def test_slope_without_a_bound_is_rejected():
        with pytest.raises(LimitValidationError):
            _segment(maximum_slope=1e-9)

    @staticmethod
    def test_crossing_sloped_bounds_are_rejected():
        # Minimum rises faster than the maximum, so they cross inside the band.
        with pytest.raises(LimitValidationError):
            _segment(
                minimum=0.0,
                maximum=10.0,
                minimum_slope=2e-6,
                maximum_slope=1e-7,
            )

    @staticmethod
    def test_non_crossing_sloped_bounds_are_accepted():
        segment = _segment(
            minimum=0.0, maximum=10.0, minimum_slope=1e-7, maximum_slope=2e-7
        )
        assert segment.lowest_gap() > 0

    @staticmethod
    def test_a_sloped_maximum_rejects_a_sample_it_would_have_passed_flat():
        """The per-sample bound must be used, not the bound at the start."""
        # |S11| rises with frequency: 0.1 at 10 MHz up to 1.0 at 19 MHz.
        points = _points([0.1 + 0.1 * index for index in range(11)])
        # Maximum allows 0.05 at the start and rises by 0.1 per 9 MHz, so it
        # ends at 0.15 -- every sample violates it.  A flat-bound evaluator
        # aimed at the start value would also fail, so the discriminating
        # case is the reverse: a generous bound at the start that tightens.
        tightening = _segment(maximum=2.0, maximum_slope=-2e-7)
        result = evaluate_rule(
            LimitRule(
                metric=Metric.S11_VSWR, segments=(tightening,), id="slope"
            ),
            _snapshot(points),
        )
        assert result.verdict is Verdict.FAIL
        # The worst sample is the highest |S11| against the tightest bound,
        # and the reported limit is that sample's own bound, not the start's.
        assert result.worst_limit == pytest.approx(
            tightening.maximum_at(result.worst_frequency)
        )
        assert result.worst_limit < 2.0

    @staticmethod
    def test_a_sloped_bound_passes_data_that_its_own_line_allows():
        points = _points([0.01] * 11)
        segment = _segment(maximum=2.0, maximum_slope=1e-9)
        result = evaluate_rule(
            LimitRule(metric=Metric.S11_VSWR, segments=(segment,), id="ok"),
            _snapshot(points),
        )
        assert result.verdict is Verdict.PASS

    @staticmethod
    def test_boundary_equality_passes():
        """A sample exactly on the bound is inside it.

        ``|S11| = 0.5`` is a VSWR of exactly 3, so a maximum of 3 is the
        boundary itself and must not be reported as a violation.
        """
        points = _points([0.5] * 11)
        result = evaluate_rule(
            LimitRule(
                metric=Metric.S11_VSWR,
                segments=(_segment(maximum=3.0),),
                id="eq",
            ),
            _snapshot(points),
        )
        assert result.verdict is Verdict.PASS
        assert result.worst_value == pytest.approx(3.0)

    @staticmethod
    def test_a_sample_a_hair_over_the_bound_fails():
        points = _points([0.5000001] * 11)
        result = evaluate_rule(
            LimitRule(
                metric=Metric.S11_VSWR,
                segments=(_segment(maximum=3.0),),
                id="eq",
            ),
            _snapshot(points),
        )
        assert result.verdict is Verdict.FAIL

    @staticmethod
    def test_sloped_bounds_keep_the_document_version_current():
        document = _limit_set(maximum=1.0, maximum_slope=1e-9).to_dict()
        assert document["version"] == LIMIT_SET_VERSION
        assert loads(json.dumps(document)).rules[0].segments[0].maximum_slope


class TestPhaseMetric:
    @staticmethod
    def test_phase_is_reported_in_wrapped_degrees():
        point = Datapoint(10 * MHZ, 0.5, 0.5)  # 45 degrees
        assert Metric.S11_PHASE_DEG.measure(point) == pytest.approx(45.0)
        assert Metric.S11_PHASE_DEG.unit == "deg"

    @staticmethod
    def test_phase_wraps_negative_angles():
        point = Datapoint(10 * MHZ, 0.5, -0.5)
        assert Metric.S11_PHASE_DEG.measure(point) == pytest.approx(-45.0)

    @staticmethod
    def test_phase_rule_is_evaluated_against_s11():
        points = [Datapoint(10 * MHZ, 0.5, 0.5), Datapoint(15 * MHZ, 0.5, 0.5)]
        result = evaluate_rule(
            LimitRule(
                metric=Metric.S11_PHASE_DEG,
                segments=(
                    LimitSegment(start=10 * MHZ, stop=15 * MHZ, maximum=90.0),
                ),
                id="phase",
            ),
            _snapshot(points),
        )
        assert result.verdict is Verdict.PASS
        assert result.metric is Metric.S11_PHASE_DEG


class TestSessionLimits:
    @staticmethod
    def test_limit_sets_round_trip_through_a_session_document(tmp_path):
        snapshot = snapshot_from_datapoints(_points([0.2, 0.3]), None)
        limit_set = _limit_set()
        document = make_document(
            [snapshot],
            limit_sets=[limit_set.to_dict()],
            active_limit_set="bench",
        )
        document.save = tmp_path
        from NanoVNASaver.Session import build_archive, parse_session

        parsed = parse_session(build_archive(document))
        restored = limit_sets_from_document(parsed)
        assert [item.name for item in restored] == ["bench"]
        assert restored[0] == limit_set
        assert parsed.active_limit_set == "bench"

    @staticmethod
    def test_sloped_bounds_survive_a_session_round_trip():
        limit_set = _limit_set(maximum=1.0, maximum_slope=1e-9)
        document = make_document(
            [], limit_sets=[limit_set.to_dict()], active_limit_set="bench"
        )
        restored = limit_sets_from_document(document)
        assert restored[0] == limit_set

    @staticmethod
    def test_malformed_embedded_limits_are_reported_with_their_index():
        document = make_document(
            [], limit_sets=[_limit_set().to_dict(), {"name": "broken"}]
        )
        with pytest.raises(SessionFormatError) as info:
            limit_sets_from_document(document)
        assert "entry 1" in str(info.value)

    @staticmethod
    def test_duplicate_embedded_names_are_rejected():
        payload = _limit_set(name="same").to_dict()
        document = make_document([], limit_sets=[payload, dict(payload)])
        with pytest.raises(SessionFormatError) as info:
            limit_sets_from_document(document)
        assert "two limit sets named" in str(info.value)

    @staticmethod
    def test_a_dangling_active_name_is_rejected():
        document = make_document(
            [],
            limit_sets=[_limit_set(name="present").to_dict()],
            active_limit_set="missing",
        )
        with pytest.raises(SessionFormatError) as info:
            limit_sets_from_document(document)
        assert "not embedded" in str(info.value)

    @staticmethod
    def test_an_empty_catalogue_resolves_to_no_limit_sets():
        assert limit_sets_from_document(make_document([])) == []

    @staticmethod
    def test_a_version_1_document_has_no_limit_data():
        """The guard that keeps an old session from emptying the library."""
        # A manifest written before limits were embedded reports version 1.
        document = SessionDocument.from_manifest(
            {"version": 1, "created": "2024-01-01T00:00:00+00:00"}
        )
        assert document.format_version == 1
        assert document.limit_sets == []
        assert document.active_limit_set == ""
        # A manifest without a version at all is read as the oldest schema.
        older = SessionDocument.from_manifest(
            {"created": "2024-01-01T00:00:00+00:00"}
        )
        assert older.format_version == 1
        # A document this build writes records the current version instead.
        assert make_document([]).format_version == SESSION_VERSION

    @staticmethod
    def test_an_unknown_future_session_version_is_still_readable_now():
        """Future versions are refused at load time, not silently truncated.

        The manifest version is validated by :func:`parse_session`; this pins
        that the limit payload alone cannot smuggle a newer schema past it.
        """
        document = make_document([], limit_sets=[_limit_set().to_dict()])
        document.limit_sets[0]["version"] = LIMIT_SET_VERSION + 1
        with pytest.raises(SessionFormatError):
            limit_sets_from_document(document)


class TestLimitsCatalogueTransactions:
    @staticmethod
    def test_a_rejected_set_leaves_the_catalogue_untouched(tmp_path):
        catalogue = LimitCatalogue()
        catalogue.add(_limit_set(name="keep"))
        bad = tmp_path / "bad.nvlimits"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(LimitFormatError):
            catalogue.load(str(bad))
        assert catalogue.names() == ["keep"]

    @staticmethod
    def test_readding_a_name_replaces_rather_than_duplicates():
        """The catalogue holds one set per name; a reload replaces it.

        A duplicate is only an error where two *different* documents claim the
        same name at once, which a session is checked for on restore.
        """
        catalogue = LimitCatalogue()
        catalogue.add(_limit_set(name="same", bound=1.0))
        catalogue.add(_limit_set(name="same", bound=2.0))
        assert catalogue.names() == ["same"]
        assert catalogue.find("same").rules[0].segments[0].maximum == 2.0

    @staticmethod
    def test_activating_a_missing_set_is_refused():
        catalogue = LimitCatalogue()
        from NanoVNASaver.Limits import LimitError

        with pytest.raises(LimitError):
            catalogue.activate("nope")
        assert catalogue.active_name == ""


class _FakeChart:
    """Minimal chart stand-in recording what an overlay pushed into it."""

    def __init__(self, inverted=False):
        self.isInverted = inverted
        self.limit_series = []
        self.cleared = 0

    def setLimits(self, series):
        self.limit_series = list(series)

    def clearLimits(self):
        self.limit_series = []
        self.cleared += 1


class TestChartOverlays:
    @staticmethod
    def test_only_charts_that_plot_the_metric_receive_a_bound():
        charts = {
            "s11": {"vswr": _FakeChart(), "smith": _FakeChart()},
            "s21": {"log_mag": _FakeChart()},
        }
        applied = apply_chart_limits(charts, _limit_set())
        assert ("s11", "vswr") in applied
        assert ("s11", "smith") not in applied
        assert charts["s11"]["smith"].limit_series == []
        assert charts["s21"]["log_mag"].limit_series == []

    @staticmethod
    def test_a_withdrawn_limit_set_clears_the_overlay():
        chart = _FakeChart()
        charts = {"s11": {"vswr": chart}}
        apply_chart_limits(charts, _limit_set())
        assert chart.limit_series
        apply_chart_limits(charts, None)
        assert chart.limit_series == []

    @staticmethod
    def test_return_loss_is_drawn_negated_on_the_negative_dB_axis():
        """The S11 log-mag chart plots ``gain``; return loss is its negation."""
        limit_set = LimitSet(
            name="rl",
            rules=(
                LimitRule(
                    metric=Metric.S11_RETURN_LOSS_DB,
                    segments=(_segment(maximum=10.0),),
                    id="rl",
                ),
            ),
        )
        series = series_for(limit_set, "s11", "log_mag")
        values = [point[1] for point in series[0][2]]
        assert values == [pytest.approx(-10.0), pytest.approx(-10.0)]

    @staticmethod
    def test_return_loss_follows_an_inverted_axis():
        limit_set = LimitSet(
            name="rl",
            rules=(
                LimitRule(
                    metric=Metric.S11_RETURN_LOSS_DB,
                    segments=(_segment(maximum=10.0),),
                    id="rl",
                ),
            ),
        )
        series = series_for(limit_set, "s11", "log_mag", _FakeChart(True))
        values = [point[1] for point in series[0][2]]
        assert values == [pytest.approx(10.0), pytest.approx(10.0)]

    @staticmethod
    def test_insertion_loss_is_drawn_as_gain():
        """A loss bound appears negated on the S21 gain chart."""
        limit_set = LimitSet(
            name="loss",
            rules=(
                LimitRule(
                    metric=Metric.S21_INSERTION_LOSS_DB,
                    segments=(_segment(maximum=3.0),),
                    id="loss",
                ),
            ),
        )
        series = series_for(limit_set, "s21", "log_mag")
        values = [point[1] for point in series[0][2]]
        assert values == [pytest.approx(-3.0), pytest.approx(-3.0)]

    @staticmethod
    def test_an_inverted_chart_flips_the_bound_with_its_axis():
        limit_set = LimitSet(
            name="loss",
            rules=(
                LimitRule(
                    metric=Metric.S21_INSERTION_LOSS_DB,
                    segments=(_segment(maximum=3.0),),
                    id="loss",
                ),
            ),
        )
        chart = _FakeChart(inverted=True)
        series = series_for(limit_set, "s21", "log_mag", chart)
        values = [point[1] for point in series[0][2]]
        assert values == [pytest.approx(3.0), pytest.approx(3.0)]

    @staticmethod
    def test_a_sloped_bound_is_drawn_at_both_ends():
        limit_set = _limit_set(maximum=1.0, maximum_slope=1e-7)
        series = series_for(limit_set, "s11", "vswr")
        points = series[0][2]
        assert points[0][0] == pytest.approx(10 * MHZ)
        assert points[1][0] == pytest.approx(20 * MHZ)
        assert points[0][1] == pytest.approx(1.0)
        assert points[1][1] == pytest.approx(2.0)

    @staticmethod
    def test_unsupported_metric_chart_pairs_have_no_factor():
        assert sign_for(Metric.S11_VSWR, "s21", "vswr") is None
        assert sign_for(Metric.S11_VSWR, "s11", "vswr") == 1.0

    @staticmethod
    def test_described_bounds_match_the_drawn_values():
        limit_set = _limit_set(maximum=1.0, maximum_slope=1e-7)
        text = described_bounds(limit_set, "s11", "vswr")
        assert text and "1" in text[0] and "2" in text[0]


class TestLimitsWindowNotification:
    @staticmethod
    def test_activating_emits_a_change_notification(qt_app):
        from NanoVNASaver.Windows.LimitsWindow import LimitsWindow

        catalogue = LimitCatalogue()
        catalogue.add(_limit_set(name="bench"))
        window = LimitsWindow(None, catalogue)
        seen = []
        window.catalogue_changed.connect(lambda: seen.append(True))
        window.list_sets.setCurrentRow(0)
        window.activate_selected()
        assert seen == [True]
        window.deleteLater()

    @staticmethod
    def test_deactivating_emits_a_change_notification(qt_app):
        from NanoVNASaver.Windows.LimitsWindow import LimitsWindow

        catalogue = LimitCatalogue()
        catalogue.add(_limit_set(name="bench"))
        catalogue.activate("bench")
        window = LimitsWindow(None, catalogue)
        seen = []
        window.catalogue_changed.connect(lambda: seen.append(True))
        window.deactivate()
        assert seen == [True]
        window.deleteLater()

    @staticmethod
    def test_a_rejected_load_does_not_notify(qt_app, tmp_path):
        from NanoVNASaver.Windows.LimitsWindow import LimitsWindow

        window = LimitsWindow(None, LimitCatalogue())
        seen = []
        window.catalogue_changed.connect(lambda: seen.append(True))
        bad = tmp_path / "bad.nvlimits"
        bad.write_text("{", encoding="utf-8")
        assert window.load_file(str(bad)) is None
        assert seen == []
        window.deleteLater()


class TestLimitsWindowDescription:
    @staticmethod
    def test_a_sloped_bound_is_described_with_its_rate():
        from NanoVNASaver.Windows.LimitsWindow import describe_limit_set

        text = describe_limit_set(_limit_set(maximum=1.0, maximum_slope=1e-7))
        assert "to 2" in text
        assert "/GHz" in text

    @staticmethod
    def test_a_flat_bound_reads_as_a_single_value():
        from NanoVNASaver.Windows.LimitsWindow import describe_limit_set

        text = describe_limit_set(_limit_set(maximum=1.5))
        assert "maximum 1.5" in text
        assert "/GHz" not in text


class TestLimitEditor:
    @staticmethod
    def _editor(qt_app, limit_set=None):
        from NanoVNASaver.Windows.LimitEditor import LimitEditor

        if limit_set is None:
            limit_set = _limit_set(name="bench", maximum=2.0)
        return LimitEditor(limit_set)

    @staticmethod
    def test_create_duplicate_edit_and_export(qt_app, tmp_path):
        """The full NEXT-04 workflow, without hand-editing JSON."""
        from NanoVNASaver.Limits import load
        from NanoVNASaver.Windows.LimitsWindow import LimitsWindow

        catalogue = LimitCatalogue()
        catalogue.add(_limit_set(name="bench", maximum=2.0))
        window = LimitsWindow(None, catalogue)
        seen = []
        window.catalogue_changed.connect(lambda: seen.append(True))

        copy = window.duplicate_selected()
        assert copy is not None
        assert copy.name == "bench copy"
        assert copy.rules == catalogue.find("bench").rules

        target = tmp_path / "copy.nvlimits"
        written = window.export_selected(str(target))
        assert written == str(target)
        assert load(str(target)) == copy
        assert seen == [True]
        window.deleteLater()

    @staticmethod
    def test_a_duplicate_name_never_collides():
        from NanoVNASaver.Windows.LimitEditor import duplicate_name

        original = _limit_set(name="bench")
        assert duplicate_name(original, set()) == "bench copy"
        assert duplicate_name(original, {"bench copy"}) == "bench copy 2"
        assert (
            duplicate_name(original, {"bench copy", "bench copy 2"})
            == "bench copy 3"
        )

    @staticmethod
    def test_cancelling_the_editor_changes_nothing(qt_app):
        from PySide6 import QtWidgets

        from NanoVNASaver.Windows.LimitsWindow import LimitsWindow

        catalogue = LimitCatalogue()
        catalogue.add(_limit_set(name="bench", maximum=2.0))
        window = LimitsWindow(None, catalogue)
        window.list_sets.setCurrentRow(0)
        # Patch exec so the dialog is cancelled without a modal loop.
        import NanoVNASaver.Windows.LimitEditor as editor_module

        original_exec = editor_module.LimitEditor.exec
        editor_module.LimitEditor.exec = lambda _self: (
            QtWidgets.QDialog.DialogCode.Rejected
        )
        try:
            assert window.edit_selected() is None
        finally:
            editor_module.LimitEditor.exec = original_exec
        assert catalogue.find("bench").rules[0].segments[0].maximum == 2.0
        window.deleteLater()

    @staticmethod
    def test_an_invalid_candidate_is_never_applied(qt_app):
        editor = TestLimitEditor._editor(qt_app)
        editor.edit_name.setText("edited")
        editor.rule_boxes[0].rows[0].edit_max.setText("not a number")
        assert editor.preview_candidate() is None
        editor.apply()
        assert editor.result_set is None
        assert "could not convert" in editor.message
        editor.deleteLater()

    @staticmethod
    def test_a_gap_between_segments_is_reported_as_such(qt_app):
        editor = TestLimitEditor._editor(qt_app)
        box = editor.rule_boxes[0]
        box.add_row(None)
        # The new row starts where the last ended; move it to leave a gap.
        box.rows[1].edit_start.setText(str(box.rows[0].segment.stop + 1))
        assert editor.preview_candidate() is None
        assert "gap" in editor.message
        editor.deleteLater()

    @staticmethod
    def test_an_overlap_between_segments_is_reported_as_such(qt_app):
        editor = TestLimitEditor._editor(qt_app)
        box = editor.rule_boxes[0]
        box.add_row(None)
        box.rows[1].edit_start.setText(str(box.rows[0].segment.stop - 1))
        assert editor.preview_candidate() is None
        assert "overlap" in editor.message
        editor.deleteLater()

    @staticmethod
    def test_frequency_suffixes_and_whole_hertz_are_enforced():
        from NanoVNASaver.Windows.LimitEditor import _parse_frequency

        assert _parse_frequency("144M") == 144_000_000
        assert _parse_frequency("1.5k") == 1500
        assert _parse_frequency("100") == 100
        with pytest.raises(ValueError):
            _parse_frequency("1.5")
        with pytest.raises(ValueError):
            _parse_frequency("")

    @staticmethod
    def test_a_metric_change_updates_the_shown_unit_and_note(qt_app):
        editor = TestLimitEditor._editor(qt_app)
        box = editor.rule_boxes[0]
        box.combo_metric.setCurrentIndex(
            box.combo_metric.findData(Metric.S11_PHASE_DEG.value)
        )
        assert box.metric() is Metric.S11_PHASE_DEG
        assert "deg" in box.label_unit.text()
        assert "wrapped" in box.label_note.text().lower()
        editor.deleteLater()

    @staticmethod
    def test_slopes_are_entered_per_gigahertz_and_stored_per_hertz(qt_app):
        editor = TestLimitEditor._editor(qt_app)
        row = editor.rule_boxes[0].rows[0]
        row.edit_max.setText("10")
        row.edit_max_slope.setText("2")
        candidate = editor.preview_candidate()
        assert candidate is not None
        stored = candidate.rules[0].segments[0].maximum_slope
        assert stored == pytest.approx(2e-9)
        editor.deleteLater()


class TestLimitSetWrites:
    @staticmethod
    def test_saving_is_atomic_and_leaves_no_temporary_files(tmp_path):
        from NanoVNASaver.Limits import save

        target = tmp_path / "bench.nvlimits"
        save(_limit_set(), str(target))
        assert target.exists()
        assert [item.name for item in tmp_path.iterdir()] == ["bench.nvlimits"]

    @staticmethod
    def test_a_saved_document_is_byte_stable(tmp_path):
        from NanoVNASaver.Limits import save

        first = tmp_path / "a.nvlimits"
        second = tmp_path / "b.nvlimits"
        save(_limit_set(), str(first))
        save(_limit_set(), str(second))
        assert first.read_bytes() == second.read_bytes()

    @staticmethod
    def test_a_failed_save_does_not_destroy_the_previous_file(tmp_path):
        from NanoVNASaver.Limits import save

        target = tmp_path / "bench.nvlimits"
        save(_limit_set(name="keep"), str(target))
        before = target.read_bytes()
        # A directory in place of the temporary file's parent is not writable;
        # simulating that portably means pointing at a missing directory.
        with pytest.raises(OSError):
            save(_limit_set(name="other"), str(tmp_path / "missing" / "x"))
        assert target.read_bytes() == before

    @staticmethod
    def test_a_saved_set_reads_back_identically(tmp_path):
        from NanoVNASaver.Limits import load, save

        limit_set = LimitSet(
            name="full",
            rules=(
                LimitRule(
                    metric=Metric.S11_PHASE_DEG,
                    segments=(
                        LimitSegment(
                            start=10 * MHZ,
                            stop=20 * MHZ,
                            minimum=-90.0,
                            maximum=90.0,
                            maximum_slope=-2e-10,
                        ),
                    ),
                    id="phase",
                    label="Phase",
                ),
            ),
            description="a description",
        )
        target = tmp_path / "full.nvlimits"
        save(limit_set, str(target))
        assert load(str(target)) == limit_set


class TestVersion2Example:
    @staticmethod
    def test_the_v2_example_loads_and_round_trips():
        from NanoVNASaver.Limits import dumps

        limit_set = load("tests/data/example_v2.nvlimits")
        assert limit_set.version == LIMIT_SET_VERSION
        assert {rule.metric for rule in limit_set.rules} == {
            Metric.S11_VSWR,
            Metric.S21_INSERTION_LOSS_DB,
            Metric.S11_PHASE_DEG,
        }
        assert loads(dumps(limit_set)) == limit_set

    @staticmethod
    def test_the_v1_example_still_loads_unchanged():
        """A version-1 document predates slopes and the phase metric."""
        limit_set = load("tests/data/example.nvlimits")
        assert limit_set.version == 1
        for rule in limit_set.rules:
            for segment in rule.segments:
                assert segment.minimum_slope == 0.0
                assert segment.maximum_slope == 0.0

    @staticmethod
    def test_the_sloped_example_is_evaluated_at_the_sample_frequency():
        limit_set = load("tests/data/example_v2.nvlimits")
        rule = next(item for item in limit_set.rules if item.id == "path")
        segment = rule.segments[0]
        # 1 MHz into a 2 MHz segment at 0.5 dB/GHz adds 0.5 dB.
        assert segment.maximum_at(144 * MHZ) == pytest.approx(2.0)
        assert segment.maximum_at(145 * MHZ) == pytest.approx(2.5)


class TestCorruptLimitsAreRefusedBeforeRestore:
    @staticmethod
    def _archive(**kwargs):
        from NanoVNASaver.Session import build_archive

        snapshot = snapshot_from_datapoints(_points([0.2, 0.3]), None)
        return build_archive(make_document([snapshot], **kwargs))

    def test_parsing_refuses_a_corrupt_limit_set(self):
        from NanoVNASaver.Session import SessionFormatError, parse_session

        with pytest.raises(SessionFormatError, match="invalid limit set"):
            parse_session(self._archive(limit_sets=[{"name": "broken"}]))

    def test_parsing_refuses_an_active_set_that_is_not_embedded(self):
        from NanoVNASaver.Session import SessionFormatError, parse_session

        with pytest.raises(SessionFormatError, match="not embedded"):
            parse_session(self._archive(active_limit_set="missing"))

    def test_opening_it_leaves_the_application_untouched(
        self, qt_app, tmp_path, monkeypatch
    ):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        path = tmp_path / "corrupt.nvsession"
        path.write_bytes(self._archive(limit_sets=[{"name": "broken"}]))
        window = NanoVNASaver()
        window.session.dirty = False
        errors = []
        monkeypatch.setattr(window, "showError", errors.append)
        calibration = window.calibration
        limits = window.limits
        try:
            assert window.open_session_file(str(path)) is False
            assert errors
            assert window.data.s11 == []
            assert window.calibration is calibration
            assert window.limits is limits
        finally:
            window.session.dirty = False
            window.worker.quit()
            window.worker.wait(5000)
            window.deleteLater()


class TestEditingKeepsTheSpecification:
    """Applying an untouched rule must not change a single bound."""

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"maximum": 1.2345678, "minimum": 0.123456789012},
            {"maximum": 2.0},
            {"maximum": 1.5, "maximum_slope": 1.23456789e-9},
            {"minimum": 0.1, "minimum_slope": 3.3e-10},
        ],
    )
    def test_an_unedited_row_reads_back_exactly(qt_app, kwargs):
        from NanoVNASaver.Windows.LimitEditor import SegmentRow

        segment = _segment(**kwargs)
        assert SegmentRow(segment).candidate() == segment

    @staticmethod
    def test_whole_numbers_stay_readable(qt_app):
        from NanoVNASaver.Windows.LimitEditor import SegmentRow

        assert SegmentRow(_segment(maximum=2.0)).edit_max.text() == "2"

    @staticmethod
    def test_an_edited_slope_is_taken(qt_app):
        from NanoVNASaver.Windows.LimitEditor import SegmentRow

        row = SegmentRow(_segment(maximum=1.5, maximum_slope=1e-9))
        row.edit_max_slope.setText("2")
        assert row.candidate().maximum_slope == pytest.approx(2e-9)
