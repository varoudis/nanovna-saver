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
"""Regression: the persisted marker count must not grow without bound.

`Marker.count()` counts every marker instance ever created in the process.
Persisting that value meant the saved count grew on each run, and once it
reached thousands the application spent over a minute building marker
widgets at startup. Measured before the fix: 88 s to construct the window.
"""

import os

import pytest

from NanoVNASaver.Defaults import ChartConfig

_APP = None


@pytest.fixture(scope="session")
def qt_app():
    global _APP  # noqa: PLW0603 - one application per interpreter
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    if _APP is None:
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


class TestChartConfigClamp:
    @staticmethod
    def test_a_normal_count_is_kept():
        assert ChartConfig(marker_count=3).marker_count == 3
        assert ChartConfig(marker_count=7).marker_count == 7

    @staticmethod
    @pytest.mark.parametrize(
        "value", [9216, 0, -1, 10_000, ChartConfig.MAX_MARKER_COUNT + 1]
    )
    def test_an_out_of_range_count_falls_back_to_the_default(value):
        assert ChartConfig(marker_count=value).marker_count == 3

    @staticmethod
    def test_a_non_numeric_count_falls_back_to_the_default():
        assert ChartConfig(marker_count="not a number").marker_count == 3

    @staticmethod
    def test_the_ceiling_is_small_enough_to_be_instant():
        assert ChartConfig.MAX_MARKER_COUNT <= 20


class TestMarkerControlClamp:
    @staticmethod
    def test_a_restored_oversized_count_is_clamped_not_built(qt_app):
        """Settings restore uses `setattr`, so __post_init__ never runs."""
        from NanoVNASaver.Controls.MarkerControl import MarkerControl

        class FakeApp:
            settings = None
            markers = []  # noqa: RUF012 - instance keeps its own list

            def markerUpdated(self, _marker):
                return None

        from NanoVNASaver.Defaults import get_app_config

        config = get_app_config()
        original = config.chart.marker_count
        config.chart.marker_count = 5000
        try:
            app = FakeApp()
            app.markers = []
            control = MarkerControl(app)
            assert len(app.markers) == ChartConfig.MAX_MARKER_COUNT
            assert len(app.markers) > 0
            assert control is not None
        finally:
            config.chart.marker_count = original

    @staticmethod
    def test_the_window_persists_its_own_marker_count(qt_app):
        """`Marker.count()` is process-wide, so it must not be persisted."""
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        window = NanoVNASaver()
        try:
            # A second window would inflate Marker.count() without this
            # window owning any extra markers.
            second = NanoVNASaver()
            try:
                from NanoVNASaver.Marker.Widget import Marker

                assert Marker.count() > len(window.markers)
                assert len(window.markers) < 100
            finally:
                second.worker.quit()
                second.deleteLater()
        finally:
            window.worker.quit()
            window.deleteLater()
