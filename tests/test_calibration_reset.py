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
"""Resetting the calibration republishes what an uncalibrated sweep would.

The inline S21 attenuator is undone in ``SweepWorker.applyCalibration``, not
in ``saveData``.  ``CalibrationWindow.reset()`` used to hand the raw samples
straight to ``saveData``, so with a pad configured the redrawn S21 was low by
exactly the pad until the next sweep.
"""

import pytest

from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.RFTools import Datapoint

FREQUENCIES = (1_000_000, 2_000_000, 3_000_000)
RAW11 = [Datapoint(f, 0.1, 0.02) for f in FREQUENCIES]
RAW21 = [Datapoint(f, 0.05, -0.01) for f in FREQUENCIES]

_APP = None


@pytest.fixture
def app(qt_app):
    global _APP  # noqa: PLW0603 - building the window is the expensive part
    if _APP is None:
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        _APP = NanoVNASaver()
    _APP.vna = make_simulated_vna("antenna")
    yield _APP
    _APP.s21att = 0.0
    _APP.worker.offsetDelay = 0.0
    _APP.worker.rawData11 = []
    _APP.worker.rawData21 = []


def _reset(app, att: float, delay: float = 0.0):
    app.s21att = att
    app.worker.offsetDelay = delay
    app.worker.rawData11 = list(RAW11)
    app.worker.rawData21 = list(RAW21)
    app.calibrationWindow.reset()
    with app.dataLock:
        return list(app.data.s11), list(app.data.s21)


@pytest.mark.parametrize("att", [6.0, 20.0])
def test_reset_undoes_the_attenuator(app, att):
    _, s21 = _reset(app, att)
    pad = 10 ** (att / 20)
    assert [p.z for p in s21] == pytest.approx([p.z * pad for p in RAW21])


def test_reset_without_attenuator_publishes_the_raw_samples(app):
    s11, s21 = _reset(app, 0.0)
    assert s11 == RAW11
    assert s21 == RAW21


def test_reset_matches_an_uncalibrated_sweep(app):
    """Same samples, same attenuator and delay: same published data."""
    s11, s21 = _reset(app, 20.0, delay=50e-12)
    swept11, swept21 = app.worker.applyCalibration(RAW11, RAW21)
    assert s11 == swept11
    assert s21 == swept21
    assert app.worker.data11 == swept11
    assert app.worker.data21 == swept21
