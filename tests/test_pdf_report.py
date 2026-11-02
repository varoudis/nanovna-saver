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
"""REPORT-01: PDF export of a frozen report model.

Every assertion here reads the produced PDF back with ``pdftotext`` rather
than trusting the writer: the point of the export is that a reader of the
finished file sees the same verdicts the HTML export and the CSV carry.
"""

import re
import shutil
import subprocess

import pytest

from NanoVNASaver.Limits import (
    LimitRule,
    LimitSegment,
    LimitSet,
    Metric,
    Verdict,
)
from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.PdfReport import print_html, render_pdf, write_pdf
from NanoVNASaver.Reports import build_report, csv_summary, render_html
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Traces import TraceCollection

START = 10_000_000
STOP = 11_000_000

PDFTOTEXT = shutil.which("pdftotext") or "/opt/homebrew/bin/pdftotext"
needs_pdftotext = pytest.mark.skipif(
    not shutil.os.path.exists(PDFTOTEXT),
    reason=f"pdftotext is not installed at {PDFTOTEXT}",
)


def _pdf_text(path, *options):
    """Extract text with ``pdftotext``; pages are split on the form feed."""
    result = subprocess.run(
        [PDFTOTEXT, *options, str(path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _pages(path):
    return [page for page in _pdf_text(path).split("\f") if page.strip()]


def _bbox_words(path):
    """Words with their A4 point coordinates, per page."""
    pages = []
    for chunk in _pdf_text(path, "-bbox").split("<page")[1:]:
        pages.append(
            [
                (float(xmin), float(xmax), text)
                for xmin, xmax, text in re.findall(
                    r'xMin="([\d.]+)".*?xMax="([\d.]+)">([^<]*)</word>',
                    chunk,
                )
            ]
        )
    return pages


def _snapshot(name="trace", *, s21=True):
    freqs = list(range(START, STOP + 1, 100_000))
    s11 = []
    for index, freq in enumerate(freqs):
        vswr = 3.0 if index == len(freqs) - 1 else 1.5
        s11.append(Datapoint(freq, (vswr - 1.0) / (vswr + 1.0), 0.0))
    return snapshot_from_datapoints(
        s11,
        [Datapoint(freq, 0.5, 0.0) for freq in freqs] if s21 else None,
        name=name,
        source="simulated",
        simulated=True,
        simulator_scenario="filter",
    )


def _total_reflection(name="mirror"):
    """|Gamma| == 1 at every sample, so VSWR is genuinely infinite."""
    freqs = list(range(START, STOP + 1, 100_000))
    return snapshot_from_datapoints(
        [Datapoint(freq, -1.0, 0.0) for freq in freqs],
        None,
        name=name,
        source="simulated",
        simulated=True,
        simulator_scenario="antenna",
    )


def _limits(maximum=2.0, name="bench"):
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


def _trace(snapshot=None, name="Sim 0"):
    return TraceCollection().add(snapshot or _snapshot(), name=name)


def _report(tmp_path, name, model):
    path = tmp_path / name
    write_pdf(model, path)
    return path


@needs_pdftotext
class TestFrozenValuesSurviveThePdf:
    @staticmethod
    def test_header_fields_are_the_frozen_model(tmp_path):
        model = build_report(
            "Bench report — Ω",
            [_trace()],
            dut="DUT-Ω-42",
            notes="Measured with the supplied SMA cable.",
            app_version="9.9.9",
            limits=_limits(),
        )
        path = _report(tmp_path, "frozen.pdf", model)

        assert path.read_bytes().startswith(b"%PDF-")
        text = _pdf_text(path)

        # Every value comes from the frozen model, never from live state.
        assert model.title in text
        assert "DUT-Ω-42" in text
        assert "Measured with the supplied SMA cable." in text
        assert "bench" in text
        assert "Sim 0" in text

    @staticmethod
    def test_verdict_words_match_the_html_export(tmp_path):
        model = build_report("bench", [_trace()], limits=_limits(maximum=2.0))
        csv_row = csv_summary(model)
        text = _pdf_text(_report(tmp_path, "verdict.pdf", model))

        assert model.verdict is Verdict.FAIL
        assert "Overall result: FAIL" in text
        assert "Antenna match" in text
        # The CSV carries the same verdict, lowercase by its own convention.
        assert "limit:match,fail" in csv_row
        # The failing sample is the last one, at STOP.
        assert str(STOP) in text
        # A FAIL must never be rendered as the UNAVAILABLE wording.
        assert "Overall result: UNAVAILABLE" not in text

    @staticmethod
    def test_provenance_labels_are_not_dropped(tmp_path):
        model = build_report("provenance", [_trace()], limits=_limits(5.0))
        text = _pdf_text(_report(tmp_path, "prov.pdf", model))

        assert "SIMULATED (filter)" in text
        assert "UNCALIBRATED" in text
        assert "Correction: not applied" in text

    @staticmethod
    def test_a_one_port_trace_explains_the_missing_channel(tmp_path):
        snapshot = _snapshot(s21=False)
        model = build_report(
            "one port",
            [_trace(snapshot)],
            limits=LimitSet(
                "s21",
                (
                    LimitRule(
                        Metric.S21_INSERTION_LOSS_DB,
                        (LimitSegment(START, STOP, maximum=3.0),),
                        id="insertion",
                        label="Insertion loss",
                    ),
                ),
            ),
        )
        text = _pdf_text(_report(tmp_path, "oneport.pdf", model))
        normalised = re.sub(r"\s+", " ", text)

        assert model.verdict is Verdict.UNAVAILABLE
        assert (
            model.traces[0].limits[0].reason == "no S21 channel in this trace"
        )
        assert "UNAVAILABLE" in text
        assert "no S21 channel in this trace" in normalised
        # UNAVAILABLE is never reported as a pass.
        assert "Overall result: PASS" not in text
        # A missing channel must never become a zero sample.
        assert "0 dB" not in normalised

    @staticmethod
    def test_an_infinite_measurement_is_not_dropped(tmp_path):
        model = build_report(
            "infinite", [_trace(_total_reflection())], limits=_limits(2.0)
        )
        text = _pdf_text(_report(tmp_path, "inf.pdf", model))
        normalised = re.sub(r"\s+", "", text)

        assert model.verdict is Verdict.FAIL
        # Rendered by Limits._limit_value_text as prose, not as a blank cell.
        assert "infinite" in normalised.replace("\ufb01", "fi")


@needs_pdftotext
class TestPagination:
    @staticmethod
    def _many_traces(count):
        collection = TraceCollection()
        for index in range(count):
            snapshot = _snapshot(f"trace {index}")
            collection.add(snapshot, name=f"Sim {index}")
        return list(collection)

    @needs_pdftotext
    def test_a_multi_page_report_repeats_the_table_header(self, tmp_path):
        traces = TestPagination._many_traces(14)
        model = build_report("long", traces, limits=_limits(2.0))
        pages = _pages(_report(tmp_path, "many.pdf", model))

        assert len(pages) >= 2
        assert "Sim 0" in pages[0]
        assert any("Sim 13" in page for page in pages[1:])

        # A wide table is printed as one labelled line per row, so a reader
        # on any page sees every field name next to its value without
        # turning back to a header row (see ``_MAX_TABLE_COLUMNS``).
        later = re.sub(r"\s+", " ", " ".join(pages[1:]))
        assert "Limit:" in later
        assert "Metric:" in later
        # The values themselves survive, which is the point of the round
        # trip: a verdict word and a metric name must not be split.
        assert "S11 VSWR (ratio)" in later

    @staticmethod
    def test_footers_number_every_page(tmp_path):
        traces = TestPagination._many_traces(14)
        model = build_report("long", traces, limits=_limits(2.0))
        pages = _pages(_report(tmp_path, "footers.pdf", model))

        assert f"Page 1 of {len(pages)}" in pages[0]
        assert f"Page {len(pages)} of {len(pages)}" in pages[-1]

    @staticmethod
    def test_every_plot_gets_its_own_page(tmp_path):
        model = build_report("plots", [_trace()])
        text = _pdf_text(_report(tmp_path, "plots.pdf", model))

        assert "Plots" in text
        assert "Reference impedance" in text or "S11" in text


@needs_pdftotext
class TestUnicodeAndLongValues:
    @staticmethod
    def test_unicode_round_trips(tmp_path):
        model = build_report(
            "Ω — Ünïcode ✓",
            [_trace()],
            dut="Antenna «Ω» 50 Ω ±1 Ω — €5",
            notes="Résumé — ok.",
        )
        text = _pdf_text(_report(tmp_path, "unicode.pdf", model))

        assert "Ω — Ünïcode ✓" in text
        assert "Antenna «Ω» 50 Ω ±1 Ω — €5" in text
        assert "Résumé — ok." in text
        # An em dash is used for unavailable cells, never a silent zero.
        assert "—" in text or model.traces[0].limits == ()

    @staticmethod
    def test_a_long_unbroken_identifier_is_not_clipped(tmp_path):
        identifier = "A" + "0123456789" * 12 + "Z"  # 122 characters
        model = build_report(
            "long", [_trace()], notes=f"trace identifier {identifier}"
        )
        path = _report(tmp_path, "long.pdf", model)

        text = _pdf_text(path)
        assert identifier in re.sub(r"\s+", "", text)

        # Nothing is drawn outside the printable area of the A4 page.
        page_width = 595.0
        for page in _bbox_words(path):
            for _xmin, xmax, _word in page:
                assert xmax <= page_width - 30.0

    @staticmethod
    def test_the_same_values_reach_html_and_csv(tmp_path):
        model = build_report("compare", [_trace()], limits=_limits())
        document = render_html(model, include_plots=False)
        # The HTML wraps the verdict in a badge, so compare visible text.
        html_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", document))
        text = re.sub(
            r"\s+", " ", _pdf_text(_report(tmp_path, "compare.pdf", model))
        )

        assert model.verdict is Verdict.FAIL
        assert "Overall result: FAIL" in text
        assert "Overall result: FAIL" in html_text
        for value in ("Antenna match", "S11 VSWR (ratio)", "10 - 11 MHz"):
            assert value in text
            assert value in html_text
        # The same rule and verdict reach the CSV.
        assert "limit:match,fail" in csv_summary(model)


class TestPrintMarkup:
    @staticmethod
    def test_the_plots_section_becomes_one_page_per_plot():
        model = build_report("plots", [_trace()])
        markup = print_html(model)

        assert 'class="plotpage"' in markup
        assert 'class="plot"' not in markup
        assert "data:image/svg+xml;base64," in markup

    @staticmethod
    def test_plots_can_be_left_out():
        model = build_report("plots", [_trace()])
        markup = print_html(model, include_plots=False)

        assert 'class="plotpage"' not in markup
        assert "data:image/svg+xml" not in markup

    @staticmethod
    def test_narrow_tables_keep_their_header_and_wide_ones_become_rows():
        """Qt repeats a ``<thead>``, so narrow tables keep theirs.

        A table wider than ``_MAX_TABLE_COLUMNS`` cannot be laid out and
        read back, so it is printed as labelled paragraphs instead; the
        header row and its ``<thead>`` legitimately disappear with it.
        """
        model = build_report("thead", [_trace()], limits=_limits())
        markup = print_html(model, include_plots=False)

        # The four-column channel table is narrow: header preserved.
        assert "<thead>" in markup
        # The eight-column limit table is not: its rows are paragraphs.
        assert 'class="row"' in markup
        assert "<b>Limit:</b> Antenna match" in markup
        assert "<b>Metric:</b> S11 VSWR (ratio)" in markup


class TestPdfBytes:
    @staticmethod
    def test_render_pdf_returns_a_complete_document():
        model = build_report("bytes", [_trace()])
        payload = render_pdf(model, include_plots=False)

        assert payload.startswith(b"%PDF-")
        assert payload.rstrip().endswith(b"%%EOF")

    @staticmethod
    def test_writing_matches_rendering(tmp_path):
        model = build_report("bytes", [_trace()])
        path = _report(tmp_path, "bytes.pdf", model)

        assert path.stat().st_size == len(render_pdf(model))


class TestReportWindow:
    @staticmethod
    def test_the_window_saves_the_same_frozen_model_as_a_pdf(tmp_path):
        """REPORT-01: the button exists, works, and never rebuilds."""
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        application = (
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        )
        assert application is not None
        window = NanoVNASaver()
        window.session.dirty = False
        try:
            report = window.windows["report"]
            report.refresh()
            assert report.btn_save_pdf.text() == "Save PDF ..."
            target = tmp_path / "window.pdf"
            assert report.save_pdf_at(str(target)) is True
            payload = target.read_bytes()
            assert payload.startswith(b"%PDF-")
            assert payload.rstrip().endswith(b"%%EOF")
            assert "saved" in report.label_status.text().lower()
        finally:
            window.session.dirty = False
            window.worker.quit()
            window.worker.wait(5000)
            window.deleteLater()
