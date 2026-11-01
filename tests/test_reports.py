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
"""REPORT-01/02/03: report model, HTML/CSV export and reproducibility."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from NanoVNASaver.Analysis.Results import (
    AnalysisKind,
    analyse_resonance,
    analyse_snapshot,
    analyse_vswr,
)
from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.CalibrationCoverage import coverage_from_calibration
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Measurement import (
    CalibrationIdentity,
    SweepInfo,
    snapshot_from_datapoints,
)
from NanoVNASaver.Reports import (
    build_report,
    csv_summary,
    provenance_labels,
    render_html,
    report_plots,
    report_snapshot,
    svg_plot,
    write_csv,
    write_html,
)
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Traces import TraceCollection


def _snapshot(scenario, start, stop, points=201, name="trace"):
    vna = make_simulated_vna(scenario)
    vna.datapoints = points
    vna.setSweep(start, stop)
    frequencies = vna.read_frequencies()
    convert = lambda values: [  # noqa: E731
        Datapoint(freq, value.real, value.imag)
        for freq, value in zip(frequencies, values, strict=True)
    ]
    s11 = convert(vna.readValues("data 0"))
    s21 = None if scenario == "antenna" else convert(vna.readValues("data 1"))
    snapshot = snapshot_from_datapoints(
        s11,
        s21,
        name=name,
        simulated=True,
        simulator_scenario=scenario,
        source="simulated",
    )
    sweep = SweepInfo(
        start=start, end=stop, points=points, segments=1, averages=3
    )
    return replace(
        snapshot, provenance=replace(snapshot.provenance, sweep=sweep)
    )


def _antenna_report():
    collection = TraceCollection()
    trace = collection.add(_snapshot("antenna", 13e6, 15e6), name="Antenna")
    analyses = {
        trace.id: [
            analyse_resonance(trace.snapshot, trace_id=trace.id),
            analyse_vswr(trace.snapshot, limit=2.0, trace_id=trace.id),
        ]
    }
    return build_report(
        "Antenna sweep",
        [trace],
        dut="2 m dipole",
        notes="Measured outdoors.",
        analyses=analyses,
        app_version="0.6.5",
    )


class TestReportModel:
    @staticmethod
    def test_report_freezes_traces_and_metadata():
        model = _antenna_report()
        assert model.title == "Antenna sweep"
        assert model.dut == "2 m dipole"
        assert model.notes == "Measured outdoors."
        assert model.app_version == "0.6.5"
        assert model.generated.tzinfo is not None
        assert len(model.traces) == 1
        assert model.traces[0].name == "Antenna"
        assert not model.multi_trace

    @staticmethod
    def test_simulation_is_detected_from_provenance():
        model = _antenna_report()
        assert model.simulated

    @staticmethod
    def test_generation_time_is_injectable_for_reproducibility():
        stamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        model = build_report("x", [], generated=stamp)
        assert model.generated == stamp

    @staticmethod
    def test_analysis_results_stay_attached_to_their_trace():
        model = _antenna_report()
        trace = model.traces[0]
        assert {result.kind for result in trace.analyses} == {
            AnalysisKind.RESONANCE,
            AnalysisKind.VSWR,
        }
        assert all(
            result.trace_id == trace.trace_id for result in trace.analyses
        )

    @staticmethod
    def test_unavailable_results_keep_their_reason():
        collection = TraceCollection()
        trace = collection.add(_snapshot("antenna", 13e6, 15e6))
        results = analyse_snapshot(
            trace.snapshot, [AnalysisKind.INSERTION_LOSS], trace_id=trace.id
        )
        model = build_report("x", [trace], analyses={trace.id: results})
        result = model.traces[0].analyses[0]
        assert not result.available
        assert "S21" in result.reason


class TestProvenanceLabels:
    @staticmethod
    def test_simulated_trace_is_labelled():
        model = _antenna_report()
        labels = provenance_labels(model.traces[0])
        assert any("SIMULATED" in label for label in labels)

    @staticmethod
    def test_uncalibrated_data_is_labelled():
        model = _antenna_report()
        assert "UNCALIBRATED" in provenance_labels(model.traces[0])

    @staticmethod
    def test_one_port_data_is_labelled():
        model = _antenna_report()
        assert any(
            "one-port" in label for label in provenance_labels(model.traces[0])
        )

    @staticmethod
    def test_two_port_data_is_not_labelled_one_port():
        snapshot = _snapshot("filter", 10e6, 200e6)
        collection = TraceCollection()
        trace = collection.add(snapshot)
        labels = provenance_labels(build_report("x", [trace]).traces[0])
        assert not any("one-port" in label for label in labels)
        # No calibration was supplied, so this must still be flagged.
        assert "UNCALIBRATED" in labels

    @staticmethod
    def test_calibrated_data_loses_the_uncalibrated_label():
        snapshot = _snapshot("filter", 10e6, 200e6)
        calibration = Calibration()
        calibration.load("tests/data/sol_27_30.cal")
        calibrated = replace(
            snapshot,
            provenance=replace(
                snapshot.provenance,
                calibration=CalibrationIdentity(
                    identity="abc123", source="test", one_port=True
                ),
            ),
        )
        collection = TraceCollection()
        trace = collection.add(calibrated)
        labels = provenance_labels(build_report("x", [trace]).traces[0])
        assert "UNCALIBRATED" not in labels


class TestPlots:
    @staticmethod
    def test_plots_include_reflection_and_transmission():
        collection = TraceCollection()
        trace = collection.add(_snapshot("filter", 10e6, 200e6))
        plots = report_plots(build_report("x", [trace]))
        titles = [title for title, _ in plots]
        assert "S11 magnitude" in titles
        assert "S21 magnitude" in titles
        assert "Group delay" in titles

    @staticmethod
    def test_one_port_report_has_no_transmission_plots():
        titles = [title for title, _ in report_plots(_antenna_report())]
        assert "S11 magnitude" in titles
        assert "S21 magnitude" not in titles

    @staticmethod
    def test_svg_is_self_contained_and_escaped():
        svg = svg_plot(
            [("<evil>", "#ff0000", [(0.0, 0.0), (1.0, 1.0)])],
            title="Test <plot>",
        )
        assert svg.startswith("<svg")
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg
        assert "<evil>" not in svg
        assert "&lt;evil&gt;" in svg
        assert "script" not in svg.lower()

    @staticmethod
    def test_plot_theme_is_independent_of_dark_mode():
        svg = svg_plot([("a", "#00ff00", [(0.0, 0.0), (1.0, 2.0)])], title="t")
        assert 'fill="#ffffff"' in svg
        assert 'fill="#111111"' in svg


class TestHtmlExport:
    @staticmethod
    def test_html_is_a_complete_document():
        model = _antenna_report()
        document = render_html(model)
        assert document.startswith("<!DOCTYPE html>")
        assert "<title>Antenna sweep</title>" in document
        assert "2 m dipole" in document
        assert "Measured outdoors." in document
        assert document.rstrip().endswith("</html>")

    @staticmethod
    def test_html_states_reference_impedance_and_sweep():
        document = render_html(_antenna_report())
        assert "Reference impedance: 50 ohm" in document
        assert "13" in document and "15" in document
        assert "Sweep:" in document

    @staticmethod
    def test_html_reports_unavailable_analysis_without_zero():
        collection = TraceCollection()
        trace = collection.add(_snapshot("antenna", 13e6, 15e6))
        results = analyse_snapshot(
            trace.snapshot, [AnalysisKind.CUTOFF], trace_id=trace.id
        )
        document = render_html(
            build_report("x", [trace], analyses={trace.id: results})
        )
        assert "unavailable" in document
        # No fabricated 0 Hz / 0.0 result row for the missing analysis.
        assert "cutoff: unavailable" not in document

    @staticmethod
    def test_notes_with_markup_are_escaped():
        collection = TraceCollection()
        trace = collection.add(_snapshot("antenna", 13e6, 15e6))
        document = render_html(
            build_report("x", [trace], notes="<script>alert(1)</script>")
        )
        assert "<script>alert(1)</script>" not in document
        assert "&lt;script&gt;" in document

    @staticmethod
    def test_long_notes_survive_render():
        notes = "\n".join(f"line {index}" for index in range(200))
        collection = TraceCollection()
        trace = collection.add(_snapshot("antenna", 13e6, 15e6))
        document = render_html(build_report("x", [trace], notes=notes))
        assert "line 199" in document

    @staticmethod
    def test_write_html_produces_a_readable_file(tmp_path):
        target = tmp_path / "report.html"
        write_html(_antenna_report(), str(target))
        text = target.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in text
        assert "Antenna sweep" in text


class TestCsvExport:
    @staticmethod
    def test_csv_has_one_row_per_value_with_units():
        text = csv_summary(_antenna_report())
        lines = text.strip().splitlines()
        assert lines[0].startswith("trace_id,trace,channel,analysis,status")
        assert any("resonance" in line for line in lines[1:])
        assert any(",Hz," in line or ",Hz" in line for line in lines[1:])

    @staticmethod
    def test_csv_records_unavailable_rows_with_reason():
        collection = TraceCollection()
        trace = collection.add(_snapshot("antenna", 13e6, 15e6))
        results = analyse_snapshot(
            trace.snapshot, [AnalysisKind.INSERTION_LOSS], trace_id=trace.id
        )
        text = csv_summary(
            build_report("x", [trace], analyses={trace.id: results})
        )
        row = [line for line in text.splitlines() if "insertion_loss" in line]
        assert row
        assert "unavailable" in row[0]
        assert "S21" in row[0]

    @staticmethod
    def test_write_csv(tmp_path):
        target = tmp_path / "summary.csv"
        write_csv(_antenna_report(), str(target))
        assert "resonance" in target.read_text(encoding="utf-8")


class TestReproducibility:
    @staticmethod
    def test_rendering_the_same_model_twice_is_identical():
        model = _antenna_report()
        assert render_html(model) == render_html(model)

    @staticmethod
    def test_snapshot_view_excludes_generation_time_only():
        model = _antenna_report()
        first = report_snapshot(model)
        later = build_report(
            model.title,
            [],
            dut=model.dut,
            notes=model.notes,
            generated=model.generated,
            app_version=model.app_version,
        )
        assert report_snapshot(later)["generated"] == first["generated"]

    @staticmethod
    def test_multi_trace_report_lists_each_trace():
        collection = TraceCollection()
        one = collection.add(_snapshot("antenna", 13e6, 14e6), name="A")
        two = collection.add(_snapshot("antenna", 13e6, 14e6), name="B")
        model = build_report("pair", [one, two])
        assert model.multi_trace
        document = render_html(model)
        assert document.count("<h3>") == 2
        assert ">A<" in document and ">B<" in document

    @staticmethod
    def test_report_never_mutates_the_source_traces():
        collection = TraceCollection()
        trace = collection.add(_snapshot("antenna", 13e6, 15e6))
        before = trace.snapshot
        model = build_report("x", [trace])
        assert trace.snapshot is before
        assert model.traces[0].snapshot is before

    @staticmethod
    @pytest.mark.skipif(
        not Path("tests/data/sol_27_30.cal").exists(),
        reason="calibration fixture missing",
    )
    def test_coverage_section_mentions_state():
        from NanoVNASaver.Calibration import Calibration

        calibration = Calibration()
        calibration.load("tests/data/sol_27_30.cal")
        snapshot = _snapshot("antenna", 13e6, 15e6)
        coverage = coverage_from_calibration(
            calibration, [dp.freq for dp in snapshot.channel("S11")]
        )
        collection = TraceCollection()
        trace = collection.add(snapshot)
        document = render_html(
            build_report("x", [trace], coverage={trace.id: coverage})
        )
        assert "Calibration coverage" in document
        assert "coverage%" not in document.lower()

    @staticmethod
    def test_analysis_values_match_structured_results():
        model = _antenna_report()
        document = render_html(model)
        for result in model.traces[0].analyses:
            for value in result.values:
                if not value.available:
                    continue
                if value.unit == "Hz":
                    token = f"{value.value / 1e6:.6g} MHz"
                elif value.unit:
                    token = f"{value.value:.6g} {value.unit}"
                else:
                    token = f"{value.value:.6g}"
                assert token in document


class TestCoverageLabels:
    """Regression: a dead `EDGE_HELD` state crashed every calibrated report."""

    @staticmethod
    def _trace_with(coverage):
        from NanoVNASaver.Reports import ReportTrace

        collection = TraceCollection()
        trace = collection.add(
            _snapshot("antenna", 27_000_000, 29_000_000, points=5)
        )
        return ReportTrace.from_trace(trace, analyses=(), coverage=coverage)

    @staticmethod
    def test_fully_covered_trace_labels_without_raising():
        from NanoVNASaver.Reports import provenance_labels

        calibration = Calibration()
        calibration.load("tests/data/sol_27_30.cal")
        calibration.calc_corrections()
        frequencies = [27_500_000, 28_000_000, 28_500_000]
        coverage = coverage_from_calibration(calibration, frequencies)

        labels = provenance_labels(TestCoverageLabels._trace_with(coverage))
        assert isinstance(labels, list)
        assert not any("held at calibration edge" in label for label in labels)

    @staticmethod
    def test_edge_held_trace_says_so():
        from NanoVNASaver.Reports import provenance_labels

        calibration = Calibration()
        calibration.load("tests/data/sol_27_30.cal")
        calibration.calc_corrections()
        coverage = coverage_from_calibration(
            calibration, [26_000_000, 28_000_000, 31_000_000]
        )
        assert coverage.edge_held

        labels = provenance_labels(TestCoverageLabels._trace_with(coverage))
        assert any("held at calibration edge" in label for label in labels)

    @staticmethod
    def test_build_report_accepts_coverage_for_every_trace():
        from NanoVNASaver.Reports import build_report, provenance_labels

        calibration = Calibration()
        calibration.load("tests/data/sol_27_30.cal")
        calibration.calc_corrections()
        collection = TraceCollection()
        trace = collection.add(
            _snapshot("antenna", 27_000_000, 29_000_000, points=5)
        )
        coverage = coverage_from_calibration(
            calibration, [dp.freq for dp in trace.channel("S11")]
        )

        model = build_report("Covered", [trace], coverage={trace.id: coverage})
        labels = provenance_labels(model.traces[0])
        assert isinstance(labels, list)


class TestReportReproducibility:
    """REPORT-03.2: the same saved session must produce the same report."""

    @staticmethod
    def _document(tmp_path):
        from NanoVNASaver.Session import (
            load_session,
            make_document,
            save_session,
        )

        collection = TraceCollection()
        trace = collection.add(
            _snapshot("filter", 13_000_000, 15_000_000, points=51)
        )
        document = make_document(
            [trace.snapshot],
            trace_ids=[trace.id],
            title="Reproducible",
            trace_names={trace.id: "Alpha"},
            app_version="test",
        )
        path = tmp_path / "repro.nvsession"
        save_session(document, str(path))
        return load_session(str(path))

    @staticmethod
    def _report(document, generated=None):
        from NanoVNASaver.Session import document_snapshots

        collection = TraceCollection()
        names = {
            entry["trace_id"]: entry.get("trace_name", "")
            for entry in document.traces
        }
        traces = [
            collection.add(
                snapshot,
                name=names.get(trace_id) or None,
                trace_id=trace_id,
            )
            for trace_id, snapshot in document_snapshots(document).items()
        ]
        analyses = {
            trace.id: analyse_snapshot(
                trace.snapshot, [AnalysisKind.RESONANCE, AnalysisKind.VSWR]
            )
            for trace in traces
        }
        return build_report(
            "Reproducible",
            traces,
            analyses=analyses,
            generated=generated or datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

    @staticmethod
    def test_two_exports_of_one_session_are_identical(tmp_path):
        from NanoVNASaver.Reports import csv_summary, render_html

        document = TestReportReproducibility._document(tmp_path)
        first = TestReportReproducibility._report(document)
        second = TestReportReproducibility._report(document)

        assert render_html(first) == render_html(second)
        assert csv_summary(first) == csv_summary(second)

    @staticmethod
    def test_only_the_generation_timestamp_varies(tmp_path):
        import re

        from NanoVNASaver.Reports import render_html

        document = TestReportReproducibility._document(tmp_path)
        fixed = render_html(TestReportReproducibility._report(document))
        live = render_html(
            TestReportReproducibility._report(
                document, datetime.now(timezone.utc)
            )
        )

        def strip_time(text):
            return re.sub(r"Generated: [^<]+", "Generated: <TS>", text)

        assert strip_time(live) == strip_time(fixed)

    @staticmethod
    def test_the_trace_name_survives_the_round_trip(tmp_path):
        document = TestReportReproducibility._document(tmp_path)
        report = TestReportReproducibility._report(document)
        assert report.traces[0].name == "Alpha"
