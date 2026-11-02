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
"""RF-02: the reference impedance of a loaded file reaches every display.

These are the integration half of the renormalization work: the module tests
cover the maths, and these cover the wiring that made a 75 ohm file read as
50 ohm on every impedance chart, marker and analysis.
"""

import os

import pytest

from NanoVNASaver.RFTools import Datapoint

_APP = None


@pytest.fixture(scope="session")
def qt_app():
    global _APP  # noqa: PLW0603 - one application per interpreter
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    if _APP is None:
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


@pytest.fixture
def window(qt_app):
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    app = NanoVNASaver()
    app.session.dirty = False
    try:
        yield app
    finally:
        app.session.dirty = False
        app.worker.quit()
        app.worker.wait(5000)
        app.deleteLater()


def _write_s1p(path, resistance):
    lines = [f"# HZ S RI R {resistance}"]
    for index in range(5):
        lines.append(f"{14_000_000 + index * 1_000_000} 0.2 0.1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestImportedFile:
    @staticmethod
    def test_a_75_ohm_file_makes_the_application_75_ohm(window, tmp_path):
        from NanoVNASaver.Touchstone import Touchstone

        path = _write_s1p(tmp_path / "z75.s1p", 75)
        touchstone = Touchstone(str(path))
        touchstone.load()
        assert touchstone.opts.resistance == 75

        window.data.s11 = touchstone.s11
        window.data.s21 = touchstone.s21
        window.adopt_reference_impedance(touchstone.opts.resistance)
        window.dataUpdated()

        assert window.reference_impedance == 75
        # Every chart that derives an impedance was told, not just the app.
        for chart in window.s11charts:
            assert chart.reference_impedance == 75

    @staticmethod
    def test_a_50_ohm_file_leaves_the_application_at_50(window, tmp_path):
        from NanoVNASaver.Touchstone import Touchstone

        path = _write_s1p(tmp_path / "z50.s1p", 50)
        touchstone = Touchstone(str(path))
        touchstone.load()
        window.adopt_reference_impedance(touchstone.opts.resistance)
        assert window.reference_impedance == 50

    @staticmethod
    def test_an_unusable_value_keeps_the_previous_reference(window):
        """A failed import must not silently revert to 50 ohm."""
        window.adopt_reference_impedance(75)
        for value in (float("nan"), float("inf"), 0, -75, "seventy-five"):
            window.adopt_reference_impedance(value)
            assert window.reference_impedance == 75, value

    @staticmethod
    def test_the_impedance_readout_actually_changes(window, tmp_path):
        """The wiring is only correct if the numbers move."""
        from NanoVNASaver.Touchstone import Touchstone

        path = _write_s1p(tmp_path / "z75.s1p", 75)
        touchstone = Touchstone(str(path))
        touchstone.load()
        point = touchstone.s11[0]

        window.data.s11 = touchstone.s11
        window.data.s21 = touchstone.s21
        window.adopt_reference_impedance(75)
        window.dataUpdated()

        from NanoVNASaver.Charts.RIZ import RealImaginaryZChart

        riz = next(
            chart
            for chart in window.s11charts
            if isinstance(chart, RealImaginaryZChart)
        )
        assert riz.reference_impedance == 75
        # Independent of the chart: the same reflection coefficient at two
        # references cannot give the same impedance.
        assert point.impedance(75).real != point.impedance(50).real


class TestChartContract:
    @staticmethod
    def test_every_chart_accepts_the_reference_argument(window):
        """A subclass with a narrower ``setData`` would break the wiring."""
        for chart in window.selectable_charts:
            chart.setData([Datapoint(1_000_000, 0.3, 0.2)], 75)
            assert chart.reference_impedance == 75

    @staticmethod
    def test_omitting_the_reference_keeps_the_previous_one(window):
        chart = window.s11charts[0]
        chart.setData([Datapoint(1_000_000, 0.3, 0.2)], 75)
        chart.setData([Datapoint(2_000_000, 0.3, 0.2)])
        assert chart.reference_impedance == 75


class TestDisplayFollowsTheDataOnScreen:
    @staticmethod
    def _snapshot(reference):
        from NanoVNASaver.Measurement import snapshot_from_datapoints
        from NanoVNASaver.RFTools import Datapoint

        return snapshot_from_datapoints(
            [Datapoint(1_000_000, 0.2, 0.1), Datapoint(2_000_000, 0.3, 0.1)],
            None,
            reference_impedance=reference,
        )

    def test_a_live_sample_replaces_a_file_reference(self, window):
        from types import SimpleNamespace

        window.adopt_reference_impedance(75)
        snapshot = self._snapshot(50.0)
        window.applySample(
            SimpleNamespace(
                sweep_id=window.worker.sweep_id,
                s11=snapshot.corrected["S11"],
                s21=(),
                snapshot=snapshot,
            )
        )
        assert window.reference_impedance == 50.0

    def test_a_renormalized_live_sample_is_drawn_at_its_reference(self, window):
        from types import SimpleNamespace

        snapshot = self._snapshot(75.0)
        window.applySample(
            SimpleNamespace(
                sweep_id=window.worker.sweep_id,
                s11=snapshot.corrected["S11"],
                s21=(),
                snapshot=snapshot,
            )
        )
        assert window.reference_impedance == 75.0

    def test_restoring_a_session_adopts_its_reference(self, window):
        from NanoVNASaver.Session import make_document

        document = make_document([self._snapshot(75.0)])
        window._apply_session(document)
        assert window.reference_impedance == 75.0
        for chart in window.s11charts:
            assert chart.reference_impedance == 75.0

    def test_a_new_session_returns_to_the_default(self, window, monkeypatch):
        monkeypatch.setattr(window, "_confirm_discard", lambda: True)
        window.adopt_reference_impedance(75)
        window.action_new_session()
        assert window.reference_impedance == 50.0
