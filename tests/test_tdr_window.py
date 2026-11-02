"""TDR-01 GUI: gating controls and the gated overlay.

The engine (`TDRGating.py`) has its own numerical tests.  These cover what
only the window can get wrong: that the controls exist and are off by
default, that a gate is never applied to data it cannot support, that the
overlay is indexed on the *same* axis as the measured curve, and that a new
sweep re-evaluates the gate instead of leaving a stale overlay.

Every sweep here starts at DC, because that is the precondition the engine
enforces for a lowpass time-domain transform.
"""

from __future__ import annotations

import pytest

from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.RFTools import Datapoint

START = 0
STOP = 900_000_000

_APP = None


@pytest.fixture(scope="session")
def tdr_app(qt_app):
    global _APP  # noqa: PLW0603 - one main window for the whole module
    if _APP is None:
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        _APP = NanoVNASaver()
    return _APP


@pytest.fixture
def app(tdr_app):
    window = tdr_app
    window.session.dirty = False
    _load(window, START, STOP)
    return window


@pytest.fixture
def tdr(app):
    return app.windows["tdr"]


def _load(window, start: int, stop: int, scenario: str = "cable") -> None:
    """Put one real sweep into the window, as the worker would."""
    vna = make_simulated_vna(scenario)
    vna.setSweep(start, stop)
    frequencies = list(vna.read_frequencies())
    s11 = list(vna.readValues("data 0"))
    s21 = list(vna.readValues("data 1"))
    window.vna = vna
    window.saveData(
        [
            Datapoint(int(f), c.real, c.imag)
            for f, c in zip(frequencies, s11, strict=True)
        ],
        [
            Datapoint(int(f), c.real, c.imag)
            for f, c in zip(frequencies, s21, strict=True)
        ],
    )
    window.windows["tdr"].updateTDR()


class TestGatingControls:
    @staticmethod
    def test_gating_is_off_by_default(tdr):
        assert tdr.gate_enabled.isChecked() is False
        assert tdr.gated_trace() == ()
        assert tdr.app.tdr_chart.overlay == []
        assert tdr.gate_status.text() == "Gate off"

    @staticmethod
    def test_the_gate_kinds_are_the_engine_ones(tdr):
        from NanoVNASaver.TDRGating import GateKind

        kinds = {
            tdr.gate_kind.itemData(index)
            for index in range(tdr.gate_kind.count())
        }
        assert kinds == set(GateKind)

    @staticmethod
    def test_the_gated_trace_keeps_the_measured_grid(tdr):
        tdr.gate_start.setValue(0.0)
        tdr.gate_stop.setValue(3.0)
        tdr.gate_edge.setValue(0.5)
        tdr.gate_enabled.setChecked(True)

        trace = tdr.gated_trace()
        measured = tdr.app.data.s11
        # A gate filters; it never resamples.  The frequencies must match
        # exactly, or the trace would no longer line up with the markers.
        assert [point.freq for point in trace] == [
            point.freq for point in measured
        ]
        assert len(trace) == len(measured)

    @staticmethod
    def test_a_narrow_gate_changes_the_samples_but_a_full_one_does_not(tdr):
        tdr.gate_start.setValue(0.0)
        tdr.gate_stop.setValue(1.0)
        tdr.gate_edge.setValue(0.0)
        tdr.gate_enabled.setChecked(True)
        narrow = tdr.gated_trace()
        measured = tdr.app.data.s11
        changed = max(
            abs(complex(a.re, a.im) - complex(b.re, b.im))
            for a, b in zip(narrow, measured, strict=True)
        )
        assert changed > 1e-3, "a gate over 0-1 m must remove something"

        # A gate over the whole record is an exact identity, which is how
        # the operator can tell the gate is not quietly distorting data
        # outside its band.
        tdr.gate_stop.setValue(1e9)
        tdr.gate_edge.setValue(0.0)
        full = tdr.gated_trace()
        residual = max(
            abs(complex(a.re, a.im) - complex(b.re, b.im))
            for a, b in zip(full, measured, strict=True)
        )
        assert residual < 1e-9, f"full-record gate residual {residual:g}"


class TestOverlay:
    @staticmethod
    def test_the_overlay_is_the_gate_profile_not_a_second_amplitude(tdr):
        """The overlay must be dimensionless.

        The engine normalises an on-grid echo to its true coefficient while
        the window's legacy transform does not (measured: 0.497 against
        0.006 on one sweep), so an amplitude overlay would be plotted on a
        scale it does not share.  A weight in 0..1 is true in either
        normalisation.
        """
        tdr.gate_start.setValue(0.0)
        tdr.gate_stop.setValue(2.0)
        tdr.gate_edge.setValue(0.2)
        tdr.gate_enabled.setChecked(True)

        weights = [weight for _, weight in tdr.app.tdr_chart.overlay]
        assert weights, "a gate must produce a profile"
        assert min(weights) >= 0.0 and max(weights) <= 1.0
        # A raised-cosine transition and a kept band both appear, so this
        # is a profile and not a constant.
        assert 0.0 in weights and 1.0 in weights
        assert any(0.0 < weight < 1.0 for weight in weights)

    @staticmethod
    def test_the_overlay_is_placed_by_distance_on_the_measured_axis(tdr):
        """A kept band must be drawn at its own distance on the chart.

        The gate and the chart have different grids, so plotting a gate bin
        at the same index would squeeze the whole profile into the start of
        the axis while still looking plausible.  So the drawn band is read
        back through the chart's own axis and compared in metres.
        """
        tdr.gate_start.setValue(1.5)
        tdr.gate_stop.setValue(3.0)
        tdr.gate_edge.setValue(0.0)
        tdr.gate_enabled.setChecked(True)
        assert tdr.gate_result.source.max_distance > 3.0

        overlay = tdr.app.tdr_chart.overlay
        assert overlay, "enabling the gate must produce an overlay"
        axis = tdr.distance_axis
        assert all(0 <= index < len(axis) for index, _ in overlay)
        kept = [
            float(axis[0] + index * (axis[1] - axis[0])) / 2
            for index, weight in overlay
            if weight > 0.5
        ]
        step = tdr.gate_result.source.distance_step
        assert min(kept) == pytest.approx(1.5, abs=step)
        assert max(kept) == pytest.approx(3.0, abs=step)

    @staticmethod
    def test_turning_the_gate_off_clears_the_overlay(tdr):
        tdr.gate_start.setValue(0.0)
        tdr.gate_stop.setValue(2.0)
        tdr.gate_enabled.setChecked(True)
        assert tdr.app.tdr_chart.overlay

        tdr.gate_enabled.setChecked(False)
        assert tdr.app.tdr_chart.overlay == []
        assert tdr.gate_status.text() == "Gate off"

    @staticmethod
    def test_a_new_sweep_re_evaluates_the_gate(tdr):
        """A stale overlay would describe the previous measurement."""
        tdr.gate_start.setValue(0.0)
        tdr.gate_stop.setValue(2.0)
        tdr.gate_enabled.setChecked(True)
        before = list(tdr.app.tdr_chart.overlay)

        _load(tdr.app, START, 400_000_000)
        after = list(tdr.app.tdr_chart.overlay)
        assert after, "the gate must still apply after a new sweep"
        assert after != before, "the overlay must follow the new data"


class TestUnsupportedData:
    @staticmethod
    def test_a_gate_that_cannot_be_applied_is_reported(tdr):
        """A gate start below zero is refused by the engine, not drawn."""
        tdr.gate_start.setValue(0.0)
        tdr.gate_stop.setValue(2.0)
        tdr.gate_enabled.setChecked(True)
        # `stop <= start` is rejected by the engine's validation.
        tdr.gate_stop.setValue(0.0)

        assert tdr.gated_trace() == ()
        assert tdr.app.tdr_chart.overlay == []
        assert "Gate not applied" in tdr.gate_status.text()

    @staticmethod
    def test_the_error_names_the_reason(tdr):
        tdr.gate_start.setValue(0.0)
        tdr.gate_stop.setValue(2.0)
        tdr.gate_enabled.setChecked(True)
        tdr.gate_stop.setValue(0.0)
        tdr.gated_trace()
        assert tdr.gate_error
        assert "below" in tdr.gate_error or "not below" in tdr.gate_error

    @staticmethod
    def test_too_few_samples_is_not_a_crash(tdr):
        tdr.app.saveData([], [])
        tdr.gate_enabled.setChecked(True)
        assert tdr.gated_trace() == ()
        assert tdr.app.tdr_chart.overlay == []


class TestReferenceImpedance:
    @staticmethod
    def test_the_step_response_uses_the_configured_reference(tdr):
        """RF-02: the impedance scale follows the reference, not a literal."""
        tdr.updateTDR()
        default_peak = max(float(value) for value in tdr.step_response_Z)

        tdr.app.reference_impedance = 75.0
        try:
            tdr.updateTDR()
            changed_peak = max(float(value) for value in tdr.step_response_Z)
        finally:
            tdr.app.reference_impedance = 50.0
        tdr.updateTDR()

        assert changed_peak != pytest.approx(default_peak)
