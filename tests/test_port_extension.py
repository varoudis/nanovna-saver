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
"""RF-01: port extension and its automatic fit.

The synthetic fixture is built from physics, not from the code under test: a
standard at the device plane is transported through a length of line by
multiplying by ``exp(-j*2*pi*f*tau)`` and scaling by the line's loss.  The
fit must recover the parameters the fixture was built with, and the
processing stage must then remove it.
"""

import math
import os

import pytest

from NanoVNASaver.PortExtension import (
    ExtensionFit,
    FitStatus,
    StandardKind,
    fit_port_extension,
    fit_reflection_extension,
    fit_through_extension,
)
from NanoVNASaver.Processing import (
    ProcessingConfig,
    apply_processing,
    port_extension_factor,
    port_extension_stage,
)
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


START_HZ = 50_000_000
STOP_HZ = 450_000_000
POINTS = 201
GRID = [
    START_HZ + round(index * (STOP_HZ - START_HZ) / (POINTS - 1))
    for index in range(POINTS)
]


def through_fixture(gamma: complex, delay_s: float, loss_db_per_ghz: float):
    """Measure ``gamma`` through a line of ``delay_s`` and ``loss``.

    Written from the physics: the wave travels to the standard and back, so
    the reflection sees twice the delay and twice the loss.
    """
    out = []
    for freq in GRID:
        loss_db = 2.0 * loss_db_per_ghz * (freq / 1e9)
        rotation = complex(
            math.cos(-math.tau * freq * 2.0 * delay_s),
            math.sin(-math.tau * freq * 2.0 * delay_s),
        )
        value = gamma * rotation * 10 ** (-loss_db / 20.0)
        out.append(Datapoint(freq, value.real, value.imag))
    return out


class TestReflectionFit:
    @staticmethod
    def test_an_open_recovers_the_fixture_delay_and_loss():
        delay = 1_500e-12
        loss = 0.35
        points = through_fixture(complex(1.0, 0.0), delay, loss)
        fit = fit_port_extension(points, StandardKind.OPEN)
        assert fit.status is FitStatus.OK, fit.reason
        assert fit.delay_ps == pytest.approx(delay * 1e12, rel=1e-6)
        assert fit.loss_db_per_ghz == pytest.approx(loss, rel=1e-6)
        assert fit.residual == pytest.approx(0.0, abs=1e-9)

    @staticmethod
    def test_a_short_recovers_the_same_fixture():
        delay = 1_450e-12
        loss = 1.1
        points = through_fixture(complex(-1.0, 0.0), delay, loss)
        fit = fit_port_extension(points, StandardKind.SHORT)
        assert fit.ok, fit.reason
        assert fit.delay_ps == pytest.approx(delay * 1e12, rel=1e-6)
        assert fit.loss_db_per_ghz == pytest.approx(loss, rel=1e-6)

    @staticmethod
    def test_the_fit_stage_removes_the_fixture_from_the_measurement():
        delay = 1_500e-12
        loss = 0.35
        points = through_fixture(complex(1.0, 0.0), delay, loss)
        fit = fit_port_extension(points, StandardKind.OPEN)
        stage = fit.stage(port=1)
        assert stage is not None
        config = ProcessingConfig.build(stage)
        result = apply_processing({"S11": points}, config)
        corrected = result.channels["S11"]
        # An ideal open must come back as gamma = 1 + 0j at every sample.
        for point in corrected:
            assert point.z.real == pytest.approx(1.0, abs=1e-9)
            assert point.z.imag == pytest.approx(0.0, abs=1e-9)

    @staticmethod
    def test_the_fit_reports_the_samples_it_used():
        points = through_fixture(complex(1.0, 0.0), 1_500e-12, 0.0)
        fit = fit_port_extension(points, StandardKind.OPEN)
        assert fit.samples == POINTS
        assert fit.used == POINTS

    @staticmethod
    def test_a_band_restricts_the_fit_and_is_reported():
        points = through_fixture(complex(1.0, 0.0), 1_500e-12, 0.5)
        full = fit_port_extension(points, StandardKind.OPEN)
        narrow = fit_port_extension(
            points, StandardKind.OPEN, start=100_000_000, stop=300_000_000
        )
        assert narrow.ok
        assert narrow.samples < full.samples
        assert narrow.delay_ps == pytest.approx(full.delay_ps, rel=1e-9)


class TestRefusals:
    @staticmethod
    def test_a_wrong_standard_is_refused_by_its_magnitude():
        # A 50-ohm load is neither an open nor a short: |Gamma| near zero.
        points = through_fixture(complex(0.0, 0.0), 1_500e-12, 0.5)
        fit = fit_port_extension(points, StandardKind.OPEN)
        assert fit.status is FitStatus.UNAVAILABLE
        assert "does not look like a open" in fit.reason

    @staticmethod
    def test_a_band_with_too_little_phase_turn_cannot_identify_a_delay():
        points = through_fixture(complex(1.0, 0.0), 20e-12, 0.0)
        fit = fit_port_extension(points, StandardKind.OPEN)
        assert fit.status is FitStatus.UNAVAILABLE
        assert "cycles" in fit.reason

    @staticmethod
    def test_too_few_samples_in_the_band_is_refused():
        points = through_fixture(complex(1.0, 0.0), 1_500e-12, 0.0)[:2]
        fit = fit_port_extension(points, StandardKind.OPEN)
        assert fit.status is FitStatus.UNAVAILABLE
        assert "three samples" in fit.reason

    @staticmethod
    def test_empty_data_is_refused():
        fit = fit_port_extension([], StandardKind.OPEN)
        assert not fit.ok
        assert "no samples" in fit.reason

    @staticmethod
    def test_a_non_finite_sample_is_refused_rather_than_fitted():
        points = through_fixture(complex(1.0, 0.0), 1_500e-12, 0.5)
        points[10] = Datapoint(points[10].freq, math.nan, 0.0)
        fit = fit_port_extension(points, StandardKind.OPEN)
        assert not fit.ok
        assert "non-finite" in fit.reason

    @staticmethod
    def test_a_negative_delay_is_refused_politely():
        # Transporting *backwards* in time looks like a negative delay.
        delay = 1_500e-12
        points = through_fixture(complex(1.0, 0.0), -delay, 0.0)
        fit = fit_port_extension(points, StandardKind.OPEN)
        assert fit.status is FitStatus.UNAVAILABLE
        assert "negative" in fit.reason
        # And it must not be usable as a stage.
        assert fit.stage() is None

    @staticmethod
    def test_an_unavailable_fit_produces_no_stage():
        fit = ExtensionFit(FitStatus.UNAVAILABLE, reason="nothing fitted")
        assert fit.stage() is None
        assert "unavailable" in fit.describe()


class TestThroughFit:
    @staticmethod
    def test_a_through_recovers_the_one_way_delay():
        delay = 2_000e-12
        loss = 0.75
        out = []
        for freq in GRID:
            loss_db = loss * (freq / 1e9)
            rotation = complex(
                math.cos(-math.tau * freq * delay),
                math.sin(-math.tau * freq * delay),
            )
            value = rotation * 10 ** (-loss_db / 20.0)
            out.append(Datapoint(freq, value.real, value.imag))
        fit = fit_through_extension(out)
        assert fit.ok, fit.reason
        assert fit.delay_ps == pytest.approx(delay * 1e12, rel=1e-6)
        assert fit.loss_db_per_ghz == pytest.approx(loss, rel=1e-6)

    @staticmethod
    def test_the_through_stage_restores_an_ideal_through():
        delay = 2_000e-12
        out = []
        for freq in GRID:
            rotation = complex(
                math.cos(-math.tau * freq * delay),
                math.sin(-math.tau * freq * delay),
            )
            out.append(Datapoint(freq, rotation.real, rotation.imag))
        fit = fit_through_extension(out)
        stage = fit.stage(port=1)
        # A through needs port 2 as well; the fit is one port at a time, so
        # the caller assembles both.  Here port 1 carries all of it.
        config = ProcessingConfig.build(stage)
        result = apply_processing({"S21": out}, config)
        # S21 goes through FORWARD_CHANNELS, which applies both ports once
        # each; port 2 is zero, so only port 1's fit acts.
        for point, origin in zip(result.channels["S21"], out, strict=True):
            assert abs(point.z - 1.0) < 1e-9, (point, origin)


class TestCombinedFit:
    @staticmethod
    def test_open_and_short_agree_so_the_average_is_the_answer():
        delay = 1_500e-12
        loss = 0.6
        open_points = through_fixture(complex(1.0, 0.0), delay, loss)
        short_points = through_fixture(complex(-1.0, 0.0), delay, loss)
        combined = fit_reflection_extension(open_points, short_points)
        alone = fit_port_extension(open_points, StandardKind.OPEN)
        assert combined.ok
        assert combined.delay_ps == pytest.approx(alone.delay_ps, rel=1e-9)
        assert combined.loss_db_per_ghz == pytest.approx(
            alone.loss_db_per_ghz, rel=1e-9
        )

    @staticmethod
    def test_only_one_standard_is_still_enough():
        delay = 1_500e-12
        points = through_fixture(complex(-1.0, 0.0), delay, 0.2)
        combined = fit_reflection_extension(short_points=points)
        assert combined.ok
        assert combined.delay_ps == pytest.approx(delay * 1e12, rel=1e-6)

    @staticmethod
    def test_neither_standard_is_refused():
        combined = fit_reflection_extension()
        assert combined.status is FitStatus.UNAVAILABLE
        assert "neither" in combined.reason


class TestRoundTripLoss:
    """The fixture is traversed twice in reflection, once in transmission."""

    @staticmethod
    def test_a_reflection_doubles_both_delay_and_loss():
        factor = port_extension_factor(
            1_000_000_000, delay_seconds=100e-12, loss_db_per_ghz=1.0
        )
        # One-way 100 ps and 1 dB/GHz => 2 dB of round-trip gain at 1 GHz.
        assert 20 * math.log10(abs(factor)) == pytest.approx(2.0, abs=1e-12)
        # 100 ps one way at 1 GHz is 0.2 cycles round trip, i.e. 72 degrees.
        assert math.degrees(math.atan2(factor.imag, factor.real)) == (
            pytest.approx(72.0, abs=1e-9)
        )

    @staticmethod
    def test_a_transmission_applies_the_loss_once():
        factor = port_extension_factor(
            1_000_000_000,
            delay_seconds=100e-12,
            loss_db_per_ghz=1.0,
            doubled=False,
        )
        assert 20 * math.log10(abs(factor)) == pytest.approx(1.0, abs=1e-12)

    @staticmethod
    def test_a_negative_delay_or_loss_is_refused():
        from NanoVNASaver.Processing import ProcessingError

        with pytest.raises(ProcessingError):
            port_extension_factor(1_000_000, delay_seconds=-1e-12)
        with pytest.raises(ProcessingError):
            port_extension_factor(1_000_000, loss_db_per_ghz=-1.0)

    @staticmethod
    def test_an_all_zero_stage_is_not_created():
        assert port_extension_stage() is None


class TestSweepSettingsControls:
    """The panel must fit from the live sweep and store a real stage."""

    @staticmethod
    def make_window(qt_app):
        """A window that can be closed without the unsaved-session prompt.

        Every control here marks the session dirty by design, so the test
        must settle the session before closing or ``closeEvent`` blocks on a
        modal dialog that no headless run can answer.
        """
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        window = NanoVNASaver()
        window.show()
        return window

    @staticmethod
    def close_window(window):
        window.session.dirty = False
        window.close()

    @staticmethod
    def test_the_fit_button_recovers_a_fixture_from_the_live_sweep(qt_app):
        window = TestSweepSettingsControls.make_window(qt_app)
        try:
            delay, loss = 1_500e-12, 0.4
            window.data.s11 = through_fixture(complex(1.0, 0.0), delay, loss)
            settings = window.windows["sweep_settings"]
            assert settings.fit_port_extension_from_standard() is True
            assert settings.spin_port1_delay.value() == pytest.approx(
                1500.0, rel=1e-6
            )
            stage = window.processing_config.find("port_extension")
            assert stage is not None
            assert stage.as_dict["port1_delay_ps"] == pytest.approx(1500.0)
        finally:
            TestSweepSettingsControls.close_window(window)

    @staticmethod
    def test_a_wrong_standard_shows_the_reason_and_stores_nothing(qt_app):
        window = TestSweepSettingsControls.make_window(qt_app)
        try:
            window.data.s11 = [Datapoint(freq, 0.01, 0.0) for freq in GRID]
            settings = window.windows["sweep_settings"]
            assert settings.fit_port_extension_from_standard() is False
            assert "does not look like" in settings.label_extension.text()
            assert window.processing_config.find("port_extension") is None
        finally:
            TestSweepSettingsControls.close_window(window)

    @staticmethod
    def test_zeroing_the_delay_removes_the_stage(qt_app):
        window = TestSweepSettingsControls.make_window(qt_app)
        try:
            settings = window.windows["sweep_settings"]
            settings.spin_port1_delay.setValue(500.0)
            assert window.processing_config.find("port_extension") is not None
            settings.spin_port1_delay.setValue(0.0)
            settings.spin_port1_loss.setValue(0.0)
            assert window.processing_config.find("port_extension") is None
        finally:
            TestSweepSettingsControls.close_window(window)
