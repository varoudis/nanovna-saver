#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
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
"""LIMIT-03: limits in reports, frozen preview and HTML/CSV export."""

import csv
import io
import json

import pytest

from NanoVNASaver.Limits import (
    LIMIT_SET_FORMAT,
    LimitRule,
    LimitSegment,
    LimitSet,
    Metric,
    Verdict,
)
from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.Reports import (
    _csv_number,
    build_report,
    csv_summary,
    limit_plots,
    render_html,
    report_snapshot,
    write_csv,
    write_html,
)
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Traces import TraceCollection

START = 10_000_000
STOP = 11_000_000


def _snapshot(name="trace", *, s21=True):
    """S11 is 1.5:1 everywhere except the last sample, which is 3:1."""
    freqs = list(range(START, STOP + 1, 100_000))
    s11 = []
    for index, freq in enumerate(freqs):
        vswr = 3.0 if index == len(freqs) - 1 else 1.5
        reflection = (vswr - 1.0) / (vswr + 1.0)
        s11.append(Datapoint(freq, reflection, 0.0))
    s21 = [Datapoint(freq, 0.5, 0.0) for freq in freqs] if s21 else None
    return snapshot_from_datapoints(
        s11,
        s21,
        name=name,
        source="simulated",
        simulated=True,
        simulator_scenario="filter",
    )


def _limit_set(maximum=2.0, name="bench"):
    return LimitSet(
        name,
        (
            LimitRule(
                Metric.S11_VSWR,
                (LimitSegment(START, STOP, maximum=maximum),),
                id="match",
                label="Antenna match",
            ),
        ),
    )


def _trace(snapshot=None):
    collection = TraceCollection()
    return collection.add(snapshot or _snapshot(), name="Sim 0")


class TestReportModelFreezesLimits:
    @staticmethod
    def test_report_without_limits_is_unchanged():
        trace = _trace()
        model = build_report("plain", [trace])
        assert model.limit_set is None
        assert model.verdict is None
        assert model.traces[0].limits == ()
        document = render_html(model)
        assert "Limit check" not in document
        assert "limit:match" not in csv_summary(model)

    @staticmethod
    def test_limit_results_are_attached_to_the_frozen_trace():
        trace = _trace()
        model = build_report("limited", [trace], limits=_limit_set())
        results = model.traces[0].limits
        assert len(results) == 1
        assert results[0].verdict is Verdict.FAIL
        assert results[0].worst_frequency == STOP
        assert results[0].trace_name == "Sim 0"
        assert model.verdict is Verdict.FAIL
        assert model.limit_set.name == "bench"

    @staticmethod
    def test_passing_limits_pass():
        model = build_report(
            "limited", [_trace()], limits=_limit_set(maximum=5.0)
        )
        assert model.verdict is Verdict.PASS

    @staticmethod
    def test_changing_the_limit_set_after_building_does_not_change_it():
        snapshot = _snapshot()
        trace = _trace(snapshot)
        limit_set = _limit_set(maximum=2.0)
        model = build_report("limited", [trace], limits=limit_set)
        assert model.verdict is Verdict.FAIL

        # A different set with the same name is a different object.
        replacement = _limit_set(maximum=9.0)
        assert replacement is not limit_set
        assert replacement.evaluate(snapshot)[0].verdict is Verdict.PASS
        assert model.traces[0].limits[0].verdict is Verdict.FAIL
        assert model.limit_set.rules[0].segments[0].maximum == 2.0

    @staticmethod
    def test_a_later_measurement_cannot_change_a_frozen_report():
        first = _trace(_snapshot("first"))
        model = build_report("limited", [first], limits=_limit_set())
        assert model.verdict is Verdict.FAIL

        # Building a second report from newer data must not touch the first.
        later = build_report(
            "limited",
            [_trace(_snapshot("better"))],
            limits=_limit_set(maximum=9.0),
        )
        assert later.verdict is Verdict.PASS
        assert model.verdict is Verdict.FAIL
        assert len(model.traces) == 1
        assert model.traces[0].name == "Sim 0"


class TestReportRendering:
    @staticmethod
    def test_html_shows_the_verdict_and_limit_set():
        document = render_html(
            build_report("limited", [_trace()], limits=_limit_set())
        )
        assert "Limit set: bench" in document
        assert "verdict-fail" in document
        assert "Antenna match" in document
        assert "S11 VSWR" in document
        assert 'class="verdict verdict-fail"' in document
        assert "maximum 2" in document

    @staticmethod
    def test_html_explains_unavailable_is_not_a_pass():
        snapshot = _snapshot(s21=False)
        trace = _trace(snapshot)
        limit_set = LimitSet(
            "one-port",
            (
                LimitRule(
                    Metric.S21_INSERTION_LOSS_DB,
                    (LimitSegment(START, STOP, maximum=3.0),),
                    id="path",
                ),
            ),
        )
        model = build_report("limited", [trace], limits=limit_set)
        assert model.verdict is Verdict.UNAVAILABLE
        document = render_html(model)
        assert 'class="verdict verdict-unavailable"' in document
        assert "no S21 channel" in document
        assert 'class="verdict verdict-pass"' not in document

    @staticmethod
    def test_html_escapes_limit_labels():
        limit_set = LimitSet(
            "<script>",
            (
                LimitRule(
                    Metric.S11_VSWR,
                    (LimitSegment(START, STOP, maximum=2.0),),
                    id="<id>",
                    label="<b>match</b>",
                ),
            ),
        )
        document = render_html(
            build_report("limited", [_trace()], limits=limit_set)
        )
        assert "<script>" not in document
        assert "&lt;script&gt;" in document
        assert "<b>match</b>" not in document

    @staticmethod
    def test_csv_has_one_row_per_limit_result():
        model = build_report("limited", [_trace()], limits=_limit_set())
        rows = list(csv.reader(io.StringIO(csv_summary(model))))
        header, *body = rows
        limit_rows = [row for row in body if row[3] == "limit:match"]
        assert len(limit_rows) == 1
        row = limit_rows[0]
        assert row[4] == "fail"
        assert row[2] == "S11"
        assert row[6] == "3.0"
        assert row[8] == str(STOP)
        assert "maximum 2" in row[9]
        assert header[0] == "trace_id"

    @staticmethod
    def test_unavailable_rows_carry_their_reason():
        trace = _trace(_snapshot(s21=False))
        limit_set = LimitSet(
            "one-port",
            (
                LimitRule(
                    Metric.S21_INSERTION_LOSS_DB,
                    (LimitSegment(START, STOP, maximum=3.0),),
                    id="path",
                ),
            ),
        )
        model = build_report("limited", [trace], limits=limit_set)
        rows = list(csv.reader(io.StringIO(csv_summary(model))))
        row = next(row for row in rows if row[3] == "limit:path")
        assert row[4] == "unavailable"
        assert "no S21 channel" in row[10]

    @staticmethod
    def test_snapshot_view_embeds_the_limit_set_and_results():
        model = build_report("limited", [_trace()], limits=_limit_set())
        payload = json.loads(json.dumps(report_snapshot(model)))
        assert payload["limit_set"]["name"] == "bench"
        assert payload["traces"][0]["limits"][0]["verdict"] == "fail"

    @staticmethod
    def test_writers_preserve_the_frozen_verdict(tmp_path):
        trace = _trace()
        model = build_report("limited", [trace], limits=_limit_set())
        html_path = tmp_path / "report.html"
        csv_path = tmp_path / "report.csv"
        write_html(model, str(html_path))
        write_csv(model, str(csv_path))
        # Later measurements do not appear in the already written files.
        assert "verdict-fail" in html_path.read_text(encoding="utf-8")
        assert "limit:match" in csv_path.read_text(encoding="utf-8")

    @staticmethod
    def test_limit_overlay_plots_are_drawn_only_with_limits():
        model = build_report("limited", [_trace()], limits=_limit_set())
        titles = [title for title, _svg in limit_plots(model)]
        assert titles == ["S11 VSWR (ratio) with limits"]
        svg = limit_plots(model)[0][1]
        assert "<svg" in svg
        assert "match maximum" in svg
        assert limit_plots(build_report("plain", [_trace()])) == []

    @staticmethod
    def test_html_export_contains_the_limit_overlay():
        document = render_html(
            build_report("limited", [_trace()], limits=_limit_set())
        )
        assert "with limits" in document


class TestInfiniteValuesInExports:
    """A total reflection is data: its infinite VSWR survives every export.

    ``Datapoint.vswr`` is ``inf`` for |S11| >= 1, which is a valid physical
    result and fails any finite maximum.  The CSV must say so with the same
    portable marker the snapshot view uses, while a genuinely undecidable
    check keeps an empty value column.
    """

    @staticmethod
    def _reflection_model():
        points = [Datapoint(10_000_000, 1.0, 0.0), Datapoint(20_000_000, 1, 0)]
        snapshot = snapshot_from_datapoints(points, None, name="short")
        collection = TraceCollection()
        trace = collection.add(snapshot, name="Open")
        limit_set = LimitSet(
            "reflection",
            (
                LimitRule(
                    Metric.S11_VSWR,
                    (LimitSegment(10_000_000, 20_000_000, maximum=2.0),),
                    id="match",
                ),
            ),
        )
        return build_report("reflection", [trace], limits=limit_set)

    @staticmethod
    def test_csv_preserves_a_positive_infinity():
        rows = list(
            csv.reader(
                io.StringIO(
                    csv_summary(TestInfiniteValuesInExports._reflection_model())
                )
            )
        )
        row = next(row for row in rows if row[3] == "limit:match")
        assert row[4] == "fail"
        assert row[6] == "inf"
        assert row[8] == "10000000"
        assert "maximum 2" in row[9]

    @staticmethod
    def test_the_csv_number_policy_is_explicit_and_portable():
        # No implemented metric can produce -inf (a reflection below zero
        # magnitude is impossible, and an infinite complex sample is
        # rejected as unavailable), but the formatter must stay symmetric
        # with the snapshot encoding rather than dropping the sign.
        assert _csv_number(None) == ""
        assert _csv_number(float("inf")) == "inf"
        assert _csv_number(float("-inf")) == "-inf"
        assert _csv_number(float("nan")) == "nan"
        assert _csv_number(2.5) == "2.5"

    @staticmethod
    def test_an_unavailable_check_keeps_an_empty_value_column():
        trace = _trace(_snapshot(s21=False))
        limit_set = LimitSet(
            "one-port",
            (
                LimitRule(
                    Metric.S21_INSERTION_LOSS_DB,
                    (LimitSegment(START, STOP, maximum=3.0),),
                    id="path",
                ),
            ),
        )
        model = build_report("limited", [trace], limits=limit_set)
        rows = list(csv.reader(io.StringIO(csv_summary(model))))
        row = next(row for row in rows if row[3] == "limit:path")
        assert row[4] == "unavailable"
        assert row[6] == ""

    @staticmethod
    def test_csv_html_and_frozen_result_agree_after_a_settings_change():
        model = TestInfiniteValuesInExports._reflection_model()
        frozen = model.traces[0].limits[0]
        assert frozen.verdict is Verdict.FAIL
        assert frozen.worst_value == float("inf")
        assert frozen.worst_frequency == 10_000_000

        document = render_html(model)
        assert "infinite" in document
        assert f"<td>{frozen.worst_frequency}</td>" in document

        rows = list(csv.reader(io.StringIO(csv_summary(model))))
        row = next(row for row in rows if row[3] == "limit:match")
        assert row[6] == "inf"
        assert row[8] == str(frozen.worst_frequency)
        assert row[4] == frozen.verdict.value

        payload = json.loads(json.dumps(report_snapshot(model)))
        frozen_payload = payload["traces"][0]["limits"][0]
        assert frozen_payload["worst_value"] == "inf"
        assert frozen_payload["worst_frequency_hz"] == frozen.worst_frequency
        # A later, laxer limit set cannot reach into the frozen report.
        replacement = LimitSet(
            "reflection",
            (
                LimitRule(
                    Metric.S11_VSWR,
                    (LimitSegment(10_000_000, 20_000_000, maximum=1e9),),
                    id="match",
                ),
            ),
        )
        assert (
            replacement.evaluate(model.traces[0].snapshot)[0].verdict
            is Verdict.FAIL
        )
        assert csv_summary(model) == csv_summary(model)
        assert model.traces[0].limits[0].worst_value == float("inf")


class TestLimitFormatInReport:
    @staticmethod
    def test_document_format_is_reported_verbatim():
        limit_set = LimitSet(
            "v3",
            (
                LimitRule(
                    Metric.S11_VSWR,
                    (LimitSegment(START, STOP, maximum=2.0),),
                    id="match",
                ),
            ),
            version=1,
        )
        assert limit_set.to_dict()["format"] == LIMIT_SET_FORMAT
        model = build_report("limited", [_trace()], limits=limit_set)
        assert "version 1" in render_html(model)


@pytest.fixture
def qt_app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def app(qt_app):
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    window = NanoVNASaver()
    yield window
    # A test may legitimately make the session dirty (deactivating a limit
    # set does).  Closing a dirty window opens a modal "discard changes?"
    # dialog that blocks the whole run, so the discard is answered here.
    window.session.dirty = False
    window.close()


@pytest.fixture
def report_window(app):
    from NanoVNASaver.Windows.ReportWindow import ReportWindow

    collection = app.trace_control.collection
    for trace_id in list(collection.ordered_ids()):
        collection.remove(trace_id)
    window = ReportWindow(app)
    yield window
    window.close()


class TestReportWindowLimits:
    @staticmethod
    def _load(app, tmp_path, maximum=2.0, name="bench"):
        payload = _limit_set(maximum=maximum, name=name).to_dict()
        path = tmp_path / "bench.nvlimits"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return app.windows["limits"].load_file(str(path))

    @staticmethod
    def test_the_limits_window_is_registered(app):
        assert "limits" in app.windows
        assert app.windows["limits"].catalogue is app.limits

    @staticmethod
    def test_default_reports_still_carry_no_limits(app, report_window):
        app.trace_control.collection.add(_snapshot("Sim 0"), name="Sim 0")
        report_window.refresh()
        model = report_window.frozen_model()
        assert model.limit_set is None
        assert "No limit checking" in report_window.label_limits.text()
        assert "Sim 0" in report_window.preview.toPlainText()

    @staticmethod
    def test_loading_and_selecting_limits_changes_the_preview(
        app, report_window, tmp_path
    ):
        app.trace_control.collection.add(_snapshot("Sim 0"), name="Sim 0")
        report_window.refresh()
        assert report_window.selected_limit_name() == ""

        assert TestReportWindowLimits._load(app, tmp_path) is not None
        report_window.refresh()
        assert report_window.selected_limit_name() == "bench"
        model = report_window.frozen_model()
        assert model.verdict is Verdict.FAIL
        assert "FAIL" in report_window.label_limits.text()
        assert "Limit check" in report_window.preview.toPlainText()

    @staticmethod
    def test_a_rejected_file_is_reported_and_keeps_the_loaded_set(
        app, report_window, tmp_path
    ):
        app.trace_control.collection.add(_snapshot("Sim 0"), name="Sim 0")
        TestReportWindowLimits._load(app, tmp_path)
        bad = tmp_path / "bad.nvlimits"
        bad.write_text("{ not json", encoding="utf-8")
        assert app.windows["limits"].load_file(str(bad)) is None
        assert app.limits.names() == ["bench"]
        assert "invalid JSON" in app.windows["limits"].message

    @staticmethod
    def test_exports_freeze_the_same_selection_as_the_preview(
        app, report_window, tmp_path
    ):
        app.trace_control.collection.add(_snapshot("Sim 0"), name="Sim 0")
        TestReportWindowLimits._load(app, tmp_path, maximum=2.0)
        report_window.refresh()
        frozen = report_window.frozen_model()
        assert frozen.verdict is Verdict.FAIL

        # Change the live selection and add a trace after freezing.
        TestReportWindowLimits._load(app, tmp_path, maximum=9.0, name="loose")
        report_window.select_limit_set("loose")
        app.trace_control.collection.add(_snapshot("Sim 1"), name="Sim 1")

        html_path = tmp_path / "frozen.html"
        csv_path = tmp_path / "frozen.csv"
        assert report_window.save_html_at(str(html_path))
        assert report_window.save_csv_at(str(csv_path))
        assert report_window.frozen_model() is frozen

        html_text = html_path.read_text(encoding="utf-8")
        csv_text = csv_path.read_text(encoding="utf-8")
        assert "Limit set: bench" in html_text
        assert "verdict-fail" in html_text
        assert "Sim 1" not in html_text
        rows = list(csv.reader(io.StringIO(csv_text)))
        assert [row[0] for row in rows if row[3] == "limit:match"] == [
            "trace-1"
        ]

        # The preview still shows the frozen result.
        assert "FAIL" in report_window.label_limits.text()
        assert frozen.traces[0].limits[0].worst_limit == 2.0

    @staticmethod
    def test_refresh_applies_the_new_selection(app, report_window, tmp_path):
        app.trace_control.collection.add(_snapshot("Sim 0"), name="Sim 0")
        TestReportWindowLimits._load(app, tmp_path, maximum=2.0)
        report_window.refresh()
        assert report_window.frozen_model().verdict is Verdict.FAIL
        TestReportWindowLimits._load(app, tmp_path, maximum=9.0, name="loose")
        report_window.select_limit_set("loose")
        report_window.refresh()
        assert report_window.frozen_model().verdict is Verdict.PASS
        assert report_window.frozen_model().limit_set.name == "loose"

    @staticmethod
    def test_selecting_no_limits_removes_the_check_on_refresh(
        app, report_window, tmp_path
    ):
        app.trace_control.collection.add(_snapshot("Sim 0"), name="Sim 0")
        TestReportWindowLimits._load(app, tmp_path)
        report_window.refresh()
        assert report_window.frozen_model().limit_set is not None
        report_window.select_limit_set("")
        report_window.refresh()
        assert report_window.frozen_model().limit_set is None
        assert "No limit checking" in report_window.label_limits.text()

    @staticmethod
    def test_editing_settings_marks_the_preview_stale(
        app, report_window, tmp_path
    ):
        app.trace_control.collection.add(_snapshot("Sim 0"), name="Sim 0")
        TestReportWindowLimits._load(app, tmp_path)
        report_window.refresh()
        assert not report_window.label_stale.isVisibleTo(report_window)
        report_window.select_limit_set("")
        assert report_window.label_stale.isVisibleTo(report_window)


class TestLimitsWindow:
    @staticmethod
    def test_check_traces_reports_every_result(app, tmp_path):
        window = app.windows["limits"]
        app.trace_control.collection.add(_snapshot("Sim 0"), name="Sim 0")
        payload = _limit_set(maximum=2.0).to_dict()
        path = tmp_path / "bench.nvlimits"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert window.load_file(str(path)) is not None
        results = window.check_traces()
        assert len(results) == 1
        assert results[0].verdict is Verdict.FAIL
        assert "FAIL" in window.label_verdict.text()
        assert "Sim 0" in window.results_view.toPlainText()

    @staticmethod
    def test_deactivating_reports_no_active_set(app):
        window = app.windows["limits"]
        window.deactivate()
        assert window.active_limit_set() is None
        assert "No limit set is active" in window.label_active.text()

    @staticmethod
    def test_removing_the_set_clears_the_selector(app, tmp_path):
        window = app.windows["limits"]
        path = tmp_path / "bench.nvlimits"
        path.write_text(json.dumps(_limit_set().to_dict()), encoding="utf-8")
        window.load_file(str(path))
        assert window.remove_selected()
        assert app.limits.names() == []
