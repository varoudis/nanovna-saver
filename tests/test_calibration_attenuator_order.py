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
"""The inline S21 attenuator is undone before the calibration, not after.

A physical pad in front of the S21 input attenuates what the DUT transmits,
but not the isolation leakage, which couples from port 1 to port 2 inside the
instrument.  The standard capture stores a transmission standard with the pad
applied (``_standard_samples``), so the stored ``e30`` carries the pad's
factor while the measured reading carries it only on the transmitted part.
Correcting the raw reading and scaling the *result* mixed the two footings::

    published - ideal = -isolation * (a - 1) / tracking

with ``a = 10**(att/20)`` -- 2.2 % of S21 at 20 dB.  Un-padding the raw
reading before the correction puts both on the stored footing, and the term
cancels exactly.

These tests are analytic: ``_dut`` is closed form, so a pass says the
correction inverted the error box exactly.  No device is involved.
"""

import cmath
import math
import threading

import numpy as np
import pytest

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.RFTools import Datapoint, corr_att_data
from NanoVNASaver.Settings.Sweep import Sweep
from NanoVNASaver.SweepWorker import SweepCapture, SweepWorker

FREQUENCIES = tuple(20_000_000 + i * 5_000_000 for i in range(9))

#: An error box with every forward term enabled and a leakage that is small
#: but far above double precision, so the residue it causes can be both
#: measured and told apart from rounding noise.
DIRECTIVITY = complex(0.03, -0.02)
SOURCE_MATCH = complex(0.05, 0.03)
REFLECTION_TRACKING = complex(0.95, 0.10)
TRANSMISSION_TRACKING = complex(0.90, -0.05)
ISOLATION = complex(1.0e-4, 2.0e-4)

ATTENUATIONS = (3.0, 6.0, 12.0, 20.0)


def _pad(att: float) -> float:
    """The linear factor a pad of ``att`` dB applies to transmission."""
    return 10 ** (att / 20)


def _dut(freq: int) -> tuple[complex, complex]:
    """A lossy delay line: reflection and transmission both vary with f."""
    rotation = cmath.exp(complex(0.0, -2 * math.pi * freq * 1.0e-9))
    return 0.08 * rotation, 0.7 * rotation


class _Constant:
    """A stand-in for a SciPy interpolator that answers one value."""

    def __init__(self, value: complex):
        self.value = value

    def __call__(self, freq):
        array = np.asarray(freq, dtype=np.float64)
        return np.full(array.shape, self.value, dtype=np.complex128)


def _calibration(att: float) -> Calibration:
    """The error box above, with ``e30`` stored on the padded footing.

    ``_standard_samples`` applies the pad to a captured transmission
    standard, so the isolation a `.cal` records was measured through the same
    pad and carries its factor.  ``e22`` stays disabled: port 2 then has no
    standing wave and the forward model is two scalar equations.

    The interpolators are supplied directly because the standard *capture* is
    covered by ``test_calibration_standard_capture.py``; here the subject is
    the ordering of the pad and the correction.
    """
    calibration = Calibration()
    calibration.interp = {
        "e00": _Constant(DIRECTIVITY),
        "e11": _Constant(SOURCE_MATCH),
        "e10e01": _Constant(REFLECTION_TRACKING),
        "e10e32": _Constant(TRANSMISSION_TRACKING),
        # Stored as measured through the pad, exactly as the capture does.
        "e30": _Constant(ISOLATION * _pad(att)),
        "delta_e": _Constant(DIRECTIVITY * SOURCE_MATCH - REFLECTION_TRACKING),
    }
    calibration.isCalculated = True
    calibration.isValid1Port = lambda: True
    calibration.isValid2Port = lambda: True
    return calibration


def _raw(att: float) -> tuple[list[Datapoint], list[Datapoint]]:
    """The samples the instrument reads for the DUT behind a pad.

    ``S11m = e00 + e10e01*S11 / (1 - e11*S11)`` is untouched by the pad.  For
    ``S21m = e30 + e10e32*S21 / (1 - e11*S11)`` the pad drops the transmitted
    term and leaves ``e30`` alone, because the leakage never reaches the pad.
    """
    raw11: list[Datapoint] = []
    raw21: list[Datapoint] = []
    pad = _pad(att)
    for freq in FREQUENCIES:
        dut_s11, dut_s21 = _dut(freq)
        denominator = 1 - SOURCE_MATCH * dut_s11
        s11 = DIRECTIVITY + REFLECTION_TRACKING * dut_s11 / denominator
        s21 = ISOLATION + TRANSMISSION_TRACKING * dut_s21 / denominator / pad
        raw11.append(Datapoint(freq, s11.real, s11.imag))
        raw21.append(Datapoint(freq, s21.real, s21.imag))
    return raw11, raw21


def _ideal() -> tuple[list[complex], list[complex]]:
    """The DUT the correction should recover, independent of the pad."""
    reflections = []
    transmissions = []
    for freq in FREQUENCIES:
        dut_s11, dut_s21 = _dut(freq)
        reflections.append(dut_s11)
        transmissions.append(dut_s21)
    return reflections, transmissions


def _worker(calibration: Calibration, att: float):
    """A worker whose live configuration is the one under test."""
    worker = SweepWorker.__new__(SweepWorker)
    worker.app = type(
        "App",
        (),
        {"calibration": calibration, "s21att": att},
    )()
    worker.offsetDelay = 0.0
    return worker


def _capture(calibration: Calibration, att: float) -> SweepCapture:
    return SweepCapture(
        sweep_id=1,
        sweep=Sweep(
            start=FREQUENCIES[0], end=FREQUENCIES[-1], points=len(FREQUENCIES)
        ),
        calibration=calibration,
        offset_delay=0.0,
        averages=1,
        truncates=0,
        continuous=False,
        s21_att=att,
    )


@pytest.mark.parametrize("att", ATTENUATIONS)
def test_a_padded_correction_recovers_the_dut_exactly(att):
    calibration = _calibration(att)
    worker = _worker(calibration, att)
    raw11, raw21 = _raw(att)

    data11, data21 = worker.applyCalibration(
        raw11, raw21, capture=_capture(calibration, att)
    )
    ideal11, ideal21 = _ideal()

    for now, ideal in zip(data11, ideal11, strict=True):
        assert now.z == pytest.approx(ideal, rel=1e-12, abs=1e-15), now.freq
    for now, ideal in zip(data21, ideal21, strict=True):
        assert now.z == pytest.approx(ideal, rel=1e-12, abs=1e-15), now.freq


@pytest.mark.parametrize("att", ATTENUATIONS)
def test_no_isolation_residue_survives_the_pad(att):
    """The residue this guards is gone, not merely small.

    The old ordering left a term proportional to ``isolation * (a - 1)`` --
    about ``7.5e-4`` at 20 dB here.  Anything above the double-precision
    floor means the pad was applied after the correction instead of before.
    """
    calibration = _calibration(att)
    worker = _worker(calibration, att)
    raw11, raw21 = _raw(att)

    _, data21 = worker.applyCalibration(
        raw11, raw21, capture=_capture(calibration, att)
    )
    _, ideal21 = _ideal()

    worst = max(
        abs(now.z - ideal) for now, ideal in zip(data21, ideal21, strict=True)
    )
    old_residue = abs(ISOLATION) * (_pad(att) - 1) / abs(TRANSMISSION_TRACKING)
    assert worst < 1e-12, (
        f"residue {worst:.3e} at {att:g} dB;"
        f" the old ordering left about {old_residue:.3e}"
    )


def test_the_removed_residue_is_large_enough_to_matter():
    """The tolerance above is not vacuous: the old seam is visible.

    Re-running the correction on the padded reading and scaling the result
    afterwards -- the previous arrangement -- reproduces the law, so the
    fixed path differs from it by a measurable amount and not by rounding.
    """
    att = 20.0
    pad = _pad(att)
    calibration = _calibration(att)
    worker = _worker(calibration, att)
    raw11, raw21 = _raw(att)
    _, ideal21 = _ideal()

    # The previous behaviour: correct the reading as it came from the device,
    # then divide the pad out of the published channel.
    _, old_data21 = worker.applyCalibration(
        raw11, raw21, capture=_capture(calibration, 0.0)
    )
    published = corr_att_data(old_data21, att)

    # The stored e30 carries the pad, so the leakage term the old ordering
    # subtracted was ``a*isolation`` while the measurement supplied
    # ``isolation``: the residue is ``a*isolation*(1 - a) / tracking``,
    # scaled by ``1 - e11*S11``.
    # The correction applies that same factor to both terms, so it scales
    # the residue too.
    worst = 0.0
    for index, (now, ideal) in enumerate(zip(published, ideal21, strict=True)):
        dut_s11, _ = _dut(FREQUENCIES[index])
        residue = (
            pad
            * ISOLATION
            * (1 - pad)
            / TRANSMISSION_TRACKING
            * (1 - SOURCE_MATCH * dut_s11)
        )
        assert now.z - ideal == pytest.approx(residue, rel=1e-9, abs=1e-15)
        worst = max(worst, abs(now.z - ideal))
    assert worst > 1e-3, worst


def test_without_an_attenuator_nothing_changes():
    """A zero pad is a no-op: the published S21 is the corrected raw one."""
    calibration = _calibration(0.0)
    worker = _worker(calibration, 0.0)
    raw11, raw21 = _raw(0.0)

    data11, data21 = worker.applyCalibration(
        raw11, raw21, capture=_capture(calibration, 0.0)
    )
    _, ideal21 = _ideal()

    assert [point.freq for point in data11] == list(FREQUENCIES)
    for now, ideal in zip(data21, ideal21, strict=True):
        assert now.z == pytest.approx(ideal, rel=1e-12, abs=1e-15)


def test_published_s21_is_not_scaled_twice():
    """The attenuator reaches the correction once, and only once.

    ``saveData`` used to scale the published channel; leaving that in place
    while also un-padding would divide the pad out twice.
    """
    att = 20.0
    calibration = _calibration(att)
    worker = _worker(calibration, att)
    raw11, raw21 = _raw(att)

    _, data21 = worker.applyCalibration(
        raw11, raw21, capture=_capture(calibration, att)
    )
    _, ideal21 = _ideal()

    by_itself = _pad(att)
    for now, ideal in zip(data21, ideal21, strict=True):
        assert now.z != pytest.approx(ideal / by_itself, rel=1e-6, abs=1e-9)


def test_a_captureless_correction_uses_the_live_attenuator():
    """Re-correcting already acquired raw data must un-pad it too.

    The calibration window re-runs the correction without a capture when the
    offset delay or the calibration changes; the live attenuator is the one
    in force for those samples.
    """
    att = 12.0
    calibration = _calibration(att)
    worker = _worker(calibration, att)
    raw11, raw21 = _raw(att)

    _, data21 = worker.applyCalibration(raw11, raw21)
    _, ideal21 = _ideal()

    for now, ideal in zip(data21, ideal21, strict=True):
        assert now.z == pytest.approx(ideal, rel=1e-12, abs=1e-15)


def test_a_capture_freezes_the_attenuator():
    """An edit mid-sweep cannot change what published samples mean."""
    calibration = _calibration(20.0)
    worker = _worker(calibration, 3.0)
    capture = _capture(calibration, 20.0)
    raw11, raw21 = _raw(20.0)

    _, data21 = worker.applyCalibration(raw11, raw21, capture=capture)
    _, ideal21 = _ideal()

    for now, ideal in zip(data21, ideal21, strict=True):
        assert now.z == pytest.approx(ideal, rel=1e-12, abs=1e-15)


def test_corr_att_data_is_the_inverse_of_the_pad():
    """The un-padding helper itself: only a positive pad, exact factor."""
    points = [Datapoint(1_000_000, 0.5, -0.25)]
    assert corr_att_data(points, 0.0) is points
    scaled = corr_att_data(points, 20.0)
    assert scaled[0].z == pytest.approx(points[0].z * _pad(20.0))


def test_savedata_publishes_the_channel_unscaled():
    """``saveData`` no longer applies the attenuator to the live data."""
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    app = NanoVNASaver.__new__(NanoVNASaver)
    app.dataLock = threading.RLock()
    app.data = type("Data", (), {"s11": [], "s21": []})()
    app.s21att = 20.0
    app.sweepSource = ""
    points = [Datapoint(1_000_000, 0.5, -0.25)]

    app.saveData(points, points, "synthetic")

    assert app.data.s21 == points
    assert app.data.s11 == points
