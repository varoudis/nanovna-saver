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
"""PERF-03: coalesced chart redraws.

The measurement that motivated this (offscreen, this machine, 4001 points,
27 visible charts):

    one dataUpdated             ~10 ms   (model work only)
    one repaint of all charts  ~259 ms   (what setData() invalidates)

A 16 ms timer under that cost schedules paints faster than they can be
drawn, so a live sweep spends the whole GUI thread in ``paintEvent``.  The
coalescer merges repaint *requests*; it never merges or drops samples.

These tests assert the properties that make that safe: a burst costs one
paint, a terminal state draws immediately, and the data a coalesced paint
finally shows is the newest, not the oldest.
"""

import os
import time

import pytest

from NanoVNASaver.NanoVNASaver import LIVE_REDRAW_INTERVAL_MS, NanoVNASaver
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
    app = NanoVNASaver()
    # A dirty session makes close() raise a modal "discard?" dialog, which
    # would hang an offscreen run.
    app.session.dirty = False
    try:
        yield app
    finally:
        app.session.dirty = False
        app.worker.quit()
        app.worker.wait(5000)
        app.deleteLater()


def _points(count, scale=1.0):
    return [
        Datapoint(1_000_000 + index * 1000, 0.3 * scale, 0.2 * scale)
        for index in range(count)
    ]


class TestCoalescing:
    @staticmethod
    def test_a_burst_of_schedule_calls_repaints_once(window, qt_app):
        """Two hundred samples must not cost two hundred repaints."""
        window.data.s11 = _points(401)
        window.data.s21 = _points(401)

        repaints = []
        original = window.dataUpdated

        def counting():
            repaints.append(1)
            original()

        window.dataUpdated = counting
        # No event-loop spin inside the burst: a timer cannot fire without
        # one, so this measures the coalescer rather than how long the loop
        # happens to take on a loaded machine (an earlier version spun the
        # loop and asserted zero repaints, which failed whenever the loop
        # itself outlasted the 50 ms window).
        for _ in range(200):
            window._schedule_redraw()
        assert repaints == [], "a repaint ran before the coalescing window"

        # Let the timer fire.  Bounding on wall time, not on iteration count:
        # 400 spins of processEvents can finish in less than the interval.
        deadline = time.monotonic() + 5.0
        while not repaints and time.monotonic() < deadline:
            qt_app.processEvents()
        assert len(repaints) == 1, f"{len(repaints)} repaints for 200 requests"

    @staticmethod
    def test_a_burst_with_an_event_loop_still_costs_far_fewer_paints(
        window, qt_app
    ):
        """Spinning the loop may let the timer fire, but never 200 times."""
        window.data.s11 = _points(101)
        window.data.s21 = _points(101)
        repaints = []
        original = window.dataUpdated

        def counting():
            repaints.append(1)
            original()

        window.dataUpdated = counting
        for _ in range(200):
            window._schedule_redraw()
            qt_app.processEvents()

        # The property is coalescing, not "the timer never fires": however
        # long the burst takes, repaints must be a tiny fraction of requests.
        assert len(repaints) <= 3, f"{len(repaints)} repaints for 200 requests"

    @staticmethod
    def test_the_coalesced_paint_shows_the_newest_data(window, qt_app):
        """Merging repaints must never leave stale samples on screen."""
        window.data.s11 = _points(11, 0.1)
        window.data.s21 = _points(11, 0.1)

        window._schedule_redraw()
        # Newer data arrives inside the window.
        newest = _points(21, 0.9)
        window.data.s11 = list(newest)
        window.data.s21 = list(newest)
        window._schedule_redraw()

        deadline = time.monotonic() + 5.0
        while window._redraw_pending and time.monotonic() < deadline:
            qt_app.processEvents()

        chart = window.s11charts[0]
        assert [point.freq for point in chart.data] == [
            point.freq for point in newest
        ]

    @staticmethod
    def test_a_terminal_state_draws_inside_the_window(window):
        """Completion must not wait for the timer."""
        window.data.s11 = _points(11)
        window.data.s21 = _points(11)
        window._schedule_redraw()

        repaints = []
        original = window.dataUpdated
        window.dataUpdated = lambda: (repaints.append(1), original())

        assert window.flush_pending_redraw() is True
        assert len(repaints) == 1
        # Nothing is owed afterwards, so a second flush does nothing.
        assert window.flush_pending_redraw() is False
        assert len(repaints) == 1

    @staticmethod
    def test_a_cancelled_sweep_is_drawn_before_the_window_returns(
        window, qt_app
    ):
        """The restored previous sweep must be on screen, not queued."""
        from NanoVNASaver.SweepState import SweepState

        window.data.s11 = _points(11, 0.1)
        window.data.s21 = _points(11, 0.1)
        window.dataUpdated()
        window._remember_complete_sweep()

        # A partial sweep, then a cancel arriving inside the window.
        window.data.s11 = _points(5, 0.5)
        window.data.s21 = _points(5, 0.5)
        window._schedule_redraw()
        window.sweepRunFinished(SweepState.CANCELLED)
        qt_app.processEvents()

        assert window._redraw_pending is False
        assert [point.freq for point in window.s11charts[0].data] == [
            point.freq for point in window.data.s11
        ]
        assert len(window.data.s11) == 11

    @staticmethod
    def test_the_interval_is_longer_than_a_full_repaint_takes():
        """The constant must stay justified by the measured repaint cost.

        Painting all 27 charts of a 4001-point sweep measured ~259 ms here;
        the interval only has to be long enough that a paint can finish, and
        short enough that a sweep still looks continuous.  Asserting the
        range keeps a future edit from silently reverting to a frame-rate
        number that starves the GUI thread.
        """
        assert 20 <= LIVE_REDRAW_INTERVAL_MS <= 100
