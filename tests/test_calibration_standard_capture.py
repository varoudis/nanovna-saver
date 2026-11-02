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
"""DSP-01: a calibration standard is stored as the device measured it.

``cal_save`` used to insert ``app.data``, the *published* view.  That view is
already corrected and processed, so a standard swept while smoothing, a
running average, an offset delay, a fixture or a reference impedance was in
force was stored transformed.  A calibration built from such standards makes
every later measurement inherit that transformation a second time.

Each test sweeps the virtual standards through the real window and the real
worker, presses the window's own ``cal_save``, and compares what was stored
with the standard as the error box measured it -- a value computed by
``VirtualStandards``, not read back from the application.  The calibrated
result is then checked against the DUT's closed-form response, so a pass
means the correction undid the error box and nothing else.

TOLERANCE
---------
``1e-12`` relative, ``1e-15`` absolute per component.  The stored samples
are the doubles the device returned, so anything above rounding noise means
something else was stored; the defects this guards against are ``1e-2`` to
a factor of two.
"""

import cmath
import math

import pytest

from NanoVNASaver.Calibration import CalElement
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Hardware.SimulatedScenarios import (
    CableLine,
    ErrorModel,
    VirtualStandards,
)
from NanoVNASaver.Processing import (
    ProcessingConfig,
    port_extension_stage,
    smoothing_stage,
)
from NanoVNASaver.RFTools import Datapoint

POINTS = 11
FREQUENCIES = tuple(10_000_000 + i * 1_000_000 for i in range(POINTS))
MIDDLE = POINTS // 2

#: Directivity, source match, tracking and isolation of an imperfect set.
#: No load match, so the correction returns the DUT rather than the DUT as
#: seen through a port-2 termination; that keeps the expected values the
#: DUT's own S-parameters.
BOX_TERMS = {
    "directivity": complex(0.03, -0.02),
    "source_match": complex(0.05, 0.03),
    "reflection_tracking": complex(0.95, 0.10),
    "transmission_tracking": complex(0.90, -0.05),
    "isolation": complex(1e-4, 2e-4),
}

#: One-way offset delay in front of the ports, in seconds.
OFFSET_DELAY = 1e-9
#: Inline attenuator in front of the S21 input, in dB.
ATTENUATOR = 6.0

STANDARDS = ("short", "open", "load", "through", "isolation", "thrurefl")
TRANSMISSION_STANDARDS = ("through", "isolation")
#: ``thrurefl`` is the reflection of a through, not a component of its own.
SCENARIO_OF = {"thrurefl": "through"}


def _close(actual: complex, expected: complex) -> bool:
    return math.isclose(
        actual.real, expected.real, rel_tol=1e-12, abs_tol=1e-15
    ) and math.isclose(actual.imag, expected.imag, rel_tol=1e-12, abs_tol=1e-15)


def _box(*, isolation: complex = BOX_TERMS["isolation"]) -> ErrorModel:
    return ErrorModel(**{**BOX_TERMS, "isolation": isolation})


#: The attenuator correction is applied to the whole S21 reading, leakage
#: included, so a padded measurement leaves ``|isolation| * (factor - 1)``
#: behind -- an existing property of that seam, not of the standard capture.
#: The padded tests use a negligible isolation so they can assert exactly.
NEGLIGIBLE_ISOLATION = complex(1e-9, 2e-9)


def _padded(device):
    """A device behind a physical pad on its S21 input.

    The pad reduces what the instrument reads for transmission while the
    window's attenuator field tells the application to scale it back up.
    """
    factor = 10 ** (ATTENUATOR / 20)
    inner = device.readValues

    def read_values(name):
        values = inner(name)
        if name != "data 1":
            return values
        return [value / factor for value in values]

    device.readValues = read_values
    return device


def _ideal_elements() -> CalElement:
    element = CalElement()
    element.short_state = "IDEAL"
    element.open_state = "IDEAL"
    element.load_state = "IDEAL"
    element.through_is_ideal = True
    return element


@pytest.fixture
def window(qt_app, monkeypatch):
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    app = NanoVNASaver()
    app.session.dirty = False
    app.sweep.update(FREQUENCIES[0], FREQUENCIES[-1], 1, POINTS)
    app.calibration.cal_element = _ideal_elements()
    # A failed sweep opens a modal dialog, which would hang the run instead
    # of failing it; record the message and assert on it instead.
    errors: list[str] = []
    monkeypatch.setattr(app, "showError", errors.append)
    app.sweep_errors = errors
    try:
        yield app
    finally:
        app.session.dirty = False
        app.worker.quit()
        app.worker.wait(5000)
        app.deleteLater()


def _sweep(window, device, qt_app) -> None:
    """One sweep through the worker, exactly as a run would do it."""
    window.vna = device
    window.worker._cancel.clear()
    window.worker.run_once()
    qt_app.processEvents()
    assert not window.sweep_errors, window.sweep_errors


def _device(scenario, *, padded=False, error_model=None):
    device = make_simulated_vna(scenario, seed=1, error_model=error_model)
    device.datapoints = POINTS
    return _padded(device) if padded else device


def _standard_device(standards, name, *, padded=False):
    return _device(
        standards.standard(SCENARIO_OF.get(name, name)), padded=padded
    )


def _capture(window, standards, qt_app, *, padded=False) -> dict[str, list]:
    """Sweep and store every standard through the calibration window."""
    panel = window.windows["calibration"]
    for name in STANDARDS:
        _sweep(window, _standard_device(standards, name, padded=padded), qt_app)
        panel.cal_save(name)
    return {
        name: [
            getattr(window.calibration.dataset.get(freq), name)
            for freq in FREQUENCIES
        ]
        for name in STANDARDS
    }


def _measured(standards, name: str) -> complex:
    """What the error box reports for that standard, i.e. the raw reading."""
    measured = standards.measured(SCENARIO_OF.get(name, name))
    if name in TRANSMISSION_STANDARDS:
        return measured.s21
    return measured.s11


PROCESSING = {
    "identity": lambda: ProcessingConfig(),
    "smoothing": lambda: ProcessingConfig.build(smoothing_stage(POINTS)),
    "port_extension": lambda: ProcessingConfig.build(
        port_extension_stage(port1_delay_ps=1000.0, port2_delay_ps=1000.0)
    ),
}
#: Renormalization is deliberately absent: the engine refuses to renormalize
#: a transmission channel, so a two-port standard sweep fails outright.  That
#: refusal is covered by ``tests/test_processing_integration.py``; here it
#: would only mask what the standard capture does.


@pytest.mark.parametrize("config_name", sorted(PROCESSING))
@pytest.mark.parametrize("delay", (0.0, OFFSET_DELAY))
def test_no_correction_reaches_a_stored_standard(
    window, qt_app, config_name, delay
):
    """Neither processing nor the offset delay may alter a standard."""
    standards = VirtualStandards(_box())
    window.processing_config = PROCESSING[config_name]()
    window.worker.offsetDelay = delay
    stored = _capture(window, standards, qt_app)

    for name in STANDARDS:
        expected = _measured(standards, name)
        for freq, value in zip(FREQUENCIES, stored[name], strict=True):
            assert _close(value, expected), (
                f"{name} at {freq} Hz stored {value}, expected {expected}"
            )


def test_a_stored_standard_is_not_the_published_trace(window, qt_app):
    """The old behaviour is distinguishable, not just numerically equal.

    Frequency smoothing cannot tell the two apart here -- a virtual standard
    is flat across the span, so its smoothed value equals its raw one -- but
    the offset delay can: it rotates every sample, and it must not reach the
    stored standard.
    """
    standards = VirtualStandards(_box())
    window.worker.offsetDelay = OFFSET_DELAY
    _sweep(window, _standard_device(standards, "open"), qt_app)
    published = [point.z for point in window.data.s11]
    window.windows["calibration"].cal_save("open")
    stored = [window.calibration.dataset.get(freq).open for freq in FREQUENCIES]

    assert any(
        not _close(now, shown)
        for now, shown in zip(stored, published, strict=True)
    ), "the offset delay did not reach the published trace"
    for freq, value in zip(FREQUENCIES, stored, strict=True):
        assert _close(value, standards.measured("open").s11), freq


def test_transmission_standards_keep_the_inline_attenuator(window, qt_app):
    """The pad is measurement configuration, so it is undone exactly once.

    ``saveData`` scales it into every published sweep, so the stored
    standard is the physical value the pad hid -- not the padded reading,
    and not the padded reading scaled a second time.
    """
    standards = VirtualStandards(_box())
    window.s21att = ATTENUATOR
    stored = _capture(window, standards, qt_app, padded=True)
    for name in TRANSMISSION_STANDARDS:
        expected = _measured(standards, name)
        for freq, value in zip(FREQUENCIES, stored[name], strict=True):
            assert _close(value, expected), f"{name} at {freq} Hz"
    # A reflection standard never carries the S21 pad.
    for freq, value in zip(FREQUENCIES, stored["short"], strict=True):
        assert _close(value, standards.measured("short").s11), freq


def test_a_padded_calibration_still_recovers_the_dut(window, qt_app):
    """End to end: the pad does not survive the correction."""
    box = _box(isolation=NEGLIGIBLE_ISOLATION)
    standards = VirtualStandards(box)
    window.s21att = ATTENUATOR
    _capture(window, standards, qt_app, padded=True)
    window.calibration.calc_corrections()
    assert window.calibration.isCalculated

    window.processing_config = ProcessingConfig()
    window.worker.offsetDelay = 0.0
    dut = CableLine()
    _sweep(window, _device(dut, padded=True, error_model=box), qt_app)

    # ``saveData`` divides the pad out again, so the published trace is the
    # corrected one, and the correction returns the DUT's own S-parameters.
    # ``S11`` is exact; ``S21`` keeps the leakage-scaling residue named
    # above (``|isolation| * (factor - 1)``, about 2.5e-9 here).
    for index, freq in enumerate(FREQUENCIES):
        ideal = dut(freq)
        assert _close(window.data.s11[index].z, ideal.s11), freq
        assert window.data.s21[index].z == pytest.approx(
            ideal.s21, rel=1e-6, abs=1e-8
        ), freq


def test_a_processed_standard_does_not_poison_the_next_measurement(
    window, qt_app
):
    """The regression: standards captured under processing still calibrate.

    With ideal elements and this box the correction returns the DUT itself,
    so the published trace must be the cable's own response with the offset
    delay applied once per pass -- doubled for the reflection, which travels
    both ways -- whatever was configured while the standards were swept.
    """
    standards = VirtualStandards(_box())
    window.processing_config = ProcessingConfig.build(smoothing_stage(POINTS))
    window.worker.offsetDelay = OFFSET_DELAY
    _capture(window, standards, qt_app)
    window.calibration.calc_corrections()
    assert window.calibration.isCalculated

    window.processing_config = ProcessingConfig()
    dut = CableLine()
    _sweep(window, _device(dut, error_model=_box()), qt_app)

    for index, freq in enumerate(FREQUENCIES):
        factor = cmath.exp(complex(0.0, -2 * math.pi * freq * OFFSET_DELAY))
        ideal = dut(freq)
        assert _close(window.data.s11[index].z, ideal.s11 * factor**2), freq
        assert _close(window.data.s21[index].z, ideal.s21 * factor), freq


def test_a_trace_without_an_acquisition_is_stored_as_displayed(window):
    """A loaded file has no raw channel, so it is stored unchanged."""
    displayed = [Datapoint(freq, 0.5, 0.25) for freq in FREQUENCIES]
    window.saveData(displayed, [])
    window._live_snapshot = None

    window.windows["calibration"].cal_save("load")
    stored = [window.calibration.dataset.get(freq).load for freq in FREQUENCIES]
    assert stored == [point.z for point in displayed]
