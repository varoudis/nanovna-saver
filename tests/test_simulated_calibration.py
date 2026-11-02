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
"""SIM-01: calibration-aware virtual DUTs, as regression evidence only.

Nothing here claims physical accuracy.  What is claimed, and asserted, is
that the *correction* path is exercised end to end by data whose answer is
known in closed form:

* the ideal scenarios are unchanged, bit for bit, by the new optional
  layers (the golden table below was captured from the code before the
  layers existed);
* an injected error box distorts the ideal response in the way the
  application's own ``Calibration`` expects, and a calibration built from
  the virtual standards recovers the ideal DUT within a stated tolerance;
* the uncalibrated path shows the error, so a test cannot pass by accident
  on an error model that does nothing;
* virtual standards are computed from their definitions, never read back
  from data the application has already corrected;
* a one-port DUT keeps declaring ``data 0`` only, so no S21 is invented.

TOLERANCES
----------
``RECOVERY_TOLERANCE`` is ``1e-9``.  The observed recovery error is at the
double-precision floor (a few times ``1e-16``: the calibration solves a
3x3 system and interpolates between the calibration frequencies), so the
tolerance sits about seven orders of magnitude above the arithmetic noise
while still being nine orders below the injected error (``~1e-1``).  A test
that ignored the error box, or applied it with the wrong sign, cannot land
inside it.

``UNCALIBRATED_MIN_ERROR`` is ``5e-2``.  The smallest error this particular
box injects into the DUTs below is about ``1.1e-1``, so a bound of ``5e-2``
states "the error is visible" without pinning the exact value.
"""

import math

import pytest

from NanoVNASaver.Acquisition import (
    CancelledError,
    CancelToken,
    SweepRequest,
    open_device,
)
from NanoVNASaver.Calibration import CalElement, Calibration
from NanoVNASaver.Hardware.ReplayScenario import ReplayChannelError
from NanoVNASaver.Hardware.Simulated import SimulatedDevice, make_simulated_vna
from NanoVNASaver.Hardware.SimulatedScenarios import (
    DEFAULT_SCENARIOS,
    REFERENCE_IMPEDANCE,
    SCENARIO_TYPES,
    SIMULATION_DISCLAIMER,
    AntennaResonance,
    Attenuator,
    BandPassFilter,
    ButterworthLowPass,
    DualBandAntenna,
    ErrorModel,
    ErrorModelError,
    ReversalError,
    ReversedTwoPort,
    ScenarioResult,
    VirtualStandards,
)
from NanoVNASaver.Reports import ReportTrace, provenance_labels
from NanoVNASaver.RFTools import Datapoint

#: Recovery tolerance; see the module docstring.
RECOVERY_TOLERANCE = 1e-9

#: Smallest error the uncalibrated path must show.
UNCALIBRATED_MIN_ERROR = 5e-2

#: One error box used throughout, in the units of the forward 12-term
#: model: (e00, e11, e22, e10e01, e10e32, e30).
BOX_TERMS = {
    "directivity": complex(0.03, -0.02),
    "source_match": complex(0.05, 0.03),
    "load_match": complex(-0.04, 0.06),
    "reflection_tracking": complex(0.95, 0.10),
    "transmission_tracking": complex(0.90, -0.05),
    "isolation": complex(1e-4, 2e-4),
}

#: The single-port box: the same terms minus the port-2 ones.
ONE_PORT_TERMS = {
    name: value
    for name, value in BOX_TERMS.items()
    if name not in ("load_match", "transmission_tracking")
}

CAL_FREQUENCIES = [1_000_000 + 500_000 * index for index in range(20)]

#: Ideal-mode output captured from the code *before* the SIM-01 layers were
#: added: (frequency, s11.real, s11.imag, s21.real, s21.imag).  Exact
#: equality is the point: an optional layer must cost the ideal path nothing.
GOLDEN_IDEAL = {
    "antenna": (
        (1_000_000, 0.999949555456055, -0.00710225311383161, 0.0, 0.0),
        (
            14_150_000,
            1.2924697071141058e-30,
            -1.1368683772161603e-15,
            0.0,
            0.0,
        ),
        (
            100_000_000,
            0.9997915557528925,
            0.014436093588760999,
            0.0,
            0.0,
        ),
        (
            1_000_000_000,
            0.9999979969769902,
            0.0014152805367297066,
            0.0,
            0.0,
        ),
    ),
    "filter": (
        (1_000_000, 0.0, 0.0, 0.9994764255629298, -0.03235544380702039),
        (
            14_150_000,
            5.672600613843772e-05,
            0.0,
            0.8964594439463979,
            -0.4431257859135207,
        ),
        (
            100_000_000,
            0.7071067811865469,
            0.0,
            -0.5000000000000003,
            0.5000000000000006,
        ),
        (
            1_000_000_000,
            0.99999999995,
            0.0,
            3.1838072974064064e-06,
            -9.47962926927956e-06,
        ),
    ),
    "cable": (
        (
            1_000_000,
            0.008050096741147029,
            -0.005932616830554118,
            0.9391279565986218,
            -0.30866794794845126,
        ),
        (
            14_150_000,
            -0.009055811401964483,
            -0.004241730761381498,
            -0.20806840361478704,
            0.934739256458575,
        ),
        (
            100_000_000,
            0.007784509241027807,
            -0.006277054713506381,
            0.8404385114588482,
            -0.2966333480597167,
        ),
        (
            1_000_000_000,
            0.008762193444921419,
            -0.004819125027821557,
            -0.6729966831996953,
            0.17286119393073612,
        ),
    ),
    "attenuator": tuple(
        (freq, 0.03162277660168379, 0.0, 0.7079457843841379, 0.0)
        for freq in (1_000_000, 14_150_000, 100_000_000, 1_000_000_000)
    ),
}

#: Seeded-noise output captured before the layers existed.  The draw order
#: and count of the random stream must not change.
GOLDEN_NOISE = {
    "filter": {
        "seed": 3,
        "noise_floor": 0.005,
        "span": (50_000_000, 150_000_000),
        "points": 3,
        "data 0": (
            0.03170829256915775 + 0.006250121905417752j,
            0.7058110084980798 - 0.0013075549199058698j,
            0.9912257816525004 + 0.0036474924321780455j,
        ),
        "data 1": (
            -0.1084925393547412 - 0.9959956040597401j,
            -0.5081345636497855 + 0.49880673572278156j,
            0.09582739833542915 + 0.08236790867145494j,
        ),
    },
    "antenna": {
        "seed": 7,
        "noise_floor": 0.01,
        "span": (12_000_000, 16_000_000),
        "points": 5,
        "data 0": (
            0.9138528621722729 - 0.27165512900598526j,
            0.7330309673989284 - 0.43948369579459473j,
            0.053827238141989 - 0.20139773224134927j,
            0.5601056387386384 + 0.502624306811049j,
            0.8416370826267024 + 0.33104551639269664j,
        ),
    },
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _box(**overrides) -> ErrorModel:
    """The standard error box, with individual terms replaced or dropped."""
    terms = dict(BOX_TERMS)
    for name, value in overrides.items():
        if value is None:
            terms.pop(name, None)
        else:
            terms[name] = value
    return ErrorModel(**terms)


def _one_port_box() -> ErrorModel:
    return ErrorModel(**ONE_PORT_TERMS)


def _samples(device: SimulatedDevice, frequencies) -> list:
    return [device.sample(freq) for freq in frequencies]


def _calibration(standards: VirtualStandards, frequencies) -> Calibration:
    """Build a calibration from the virtual standards, as a user would.

    The measured values are the standards' *raw* S-parameters (what the
    device reads), which is exactly what a calibration captures.  No
    corrected data is read back anywhere.
    """
    calibration = Calibration()
    element = CalElement()
    element.short_state = "IDEAL"
    element.open_state = "IDEAL"
    element.load_state = "IDEAL"
    element.through_is_ideal = True
    calibration.cal_element = element

    measured = standards.measured_all()
    for name in ("short", "open", "load"):
        value = measured[name].s11
        calibration.insert(
            name,
            [Datapoint(f, value.real, value.imag) for f in frequencies],
        )
    through = measured["through"]
    calibration.insert(
        "through",
        [Datapoint(f, through.s21.real, through.s21.imag) for f in frequencies],
    )
    calibration.insert(
        "thrurefl",
        [Datapoint(f, through.s11.real, through.s11.imag) for f in frequencies],
    )
    isolation = measured["isolation"].s21
    calibration.insert(
        "isolation",
        [Datapoint(f, isolation.real, isolation.imag) for f in frequencies],
    )
    calibration.calc_corrections()
    return calibration


def _corrected(
    calibration: Calibration, freq: int, measured: ScenarioResult
) -> tuple[complex, complex]:
    """Correct one measured point with the application's own solver."""
    s11 = calibration.correct11(
        Datapoint(freq, measured.s11.real, measured.s11.imag)
    ).z
    s21 = calibration.correct21(
        Datapoint(freq, measured.s21.real, measured.s21.imag),
        Datapoint(freq, measured.s11.real, measured.s11.imag),
    ).z
    return s11, s21


def _load_match_closed_form(
    ideal: ScenarioResult, error_model: ErrorModel
) -> tuple[complex, complex]:
    """What the application's correction returns for a two-port DUT.

    With load match enabled the correction does not return raw S11/S21: it
    returns the DUT as seen through a port-2 termination of ``e22``.  For a
    symmetric reciprocal DUT that is ``S11c = (S11 - e22*delta)/(1 - e22*S22)``
    and ``S21c = S21/(1 - e22*S22)`` with ``delta = S11*S22 - S12*S21``.
    Written out here from the DUT's own S-parameters and ``e22``, so the
    recovery test does not compare the implementation with itself.
    """
    e22 = error_model.terms()[2]
    delta = ideal.s11 * ideal.s22 - ideal.s12 * ideal.s21
    denominator = 1.0 - e22 * ideal.s22
    return (
        (ideal.s11 - e22 * delta) / denominator,
        ideal.s21 / denominator,
    )


# ----------------------------------------------------------------------
# The ideal path must stay exactly what it was
# ----------------------------------------------------------------------
#: The golden values above were recorded on CPython 3.10-3.13.  CPython 3.14
#: returns a value one ulp different from ``cmath``/``math`` (measured on the
#: ``filter`` scenario: S21 ``0.9994764255629298-0.03235544380702039j`` became
#: ``0.9994764255629297-0.032355443807020386j``), so the pin is "within a few
#: ulps of the recorded value", not "bit-identical to it".  Pinning exact bits
#: across interpreters would assert a property of ``libm``, not of this
#: program.
#:
#: The comparison is per component.  ``pytest.approx`` on a ``complex`` falls
#: back to *angle* comparison for its relative term, which both rejects a
#: correct value near the origin and accepts a wrong argument.
ULP_TOLERANCE = 4 * 2.220446049250313e-16


def _assert_close(actual: complex, expected: complex) -> None:
    """Compare per component, within a few ulps scaled by the magnitude."""
    for got, wanted in (
        (actual.real, expected.real),
        (actual.imag, expected.imag),
    ):
        # 1e-12 relative, not one ulp: the measured difference is ~4e-14,
        # because ``cmath.sqrt`` itself moved by an ulp between 3.13 and 3.14
        # and that error is carried through the rest of the model. Widening
        # further would stop this pin detecting a real change, which is what
        # the absolute term is for.
        assert math.isclose(
            got, wanted, rel_tol=1e-12, abs_tol=ULP_TOLERANCE
        ), f"{actual!r} is not close to {expected!r}"


class TestIdealPathUnchanged:
    @staticmethod
    @pytest.mark.parametrize("scenario", sorted(GOLDEN_IDEAL))
    def test_ideal_scenarios_match_the_recorded_values(scenario):
        device = SimulatedDevice(scenario, seed=1)
        for freq, real11, imag11, real21, imag21 in GOLDEN_IDEAL[scenario]:
            result = device.sample(freq)
            _assert_close(result.s11, complex(real11, imag11))
            _assert_close(result.s21, complex(real21, imag21))

    @staticmethod
    @pytest.mark.parametrize("scenario", sorted(GOLDEN_IDEAL))
    def test_ideal_scenarios_are_deterministic_within_an_interpreter(
        scenario,
    ):
        """Same input, same output, bit for bit, on this interpreter.

        This is the property the simulator actually promises.  Whether the
        last bit matches a value recorded on an *older* interpreter is a
        property of the platform's maths library, and is covered by
        ``test_ideal_scenarios_match_the_recorded_values`` instead.
        """
        first = SimulatedDevice(scenario, seed=1)
        second = SimulatedDevice(scenario, seed=1)
        for freq, *_ in GOLDEN_IDEAL[scenario]:
            assert first.sample(freq) == second.sample(freq)

    @staticmethod
    @pytest.mark.parametrize("scenario", sorted(GOLDEN_NOISE))
    def test_seeded_noise_draws_are_unchanged(scenario):
        golden = GOLDEN_NOISE[scenario]
        start, stop = golden["span"]
        device = SimulatedDevice(
            scenario,
            seed=golden["seed"],
            noise_floor=golden["noise_floor"],
            sweep_start=start,
            sweep_stop=stop,
        )
        device.set_sweep(start, stop, golden["points"])
        for channel, expected in golden.items():
            if not channel.startswith("data"):
                continue
            # As above: the *recorded* values are an inter-interpreter
            # reference, so they are compared to within an ulp.  Exact
            # equality is asserted by
            # ``test_seeded_noise_is_repeatable_within_an_interpreter``.
            got = tuple(device.read_channel(channel))
            assert len(got) == len(expected)
            for actual, wanted in zip(got, expected, strict=True):
                _assert_close(actual, wanted)

    @staticmethod
    @pytest.mark.parametrize("scenario", sorted(GOLDEN_NOISE))
    def test_seeded_noise_is_repeatable_within_an_interpreter(scenario):
        """The same seed must give the same draws, bit for bit, every run."""
        golden = GOLDEN_NOISE[scenario]
        start, stop = golden["span"]
        readings = []
        for _ in range(2):
            device = SimulatedDevice(
                scenario,
                seed=golden["seed"],
                noise_floor=golden["noise_floor"],
                sweep_start=start,
                sweep_stop=stop,
            )
            device.set_sweep(start, stop, golden["points"])
            readings.append(tuple(device.read_channel("data 0")))
        assert readings[0] == readings[1]

    @staticmethod
    def test_an_identity_box_is_never_applied():
        """The identity path returns the same object, not an equal copy."""
        scenario_result = ScenarioResult(complex(0.1, 0.2), complex(0.3, 0.4))
        assert ErrorModel().apply(scenario_result) is scenario_result
        assert (
            _box(**{name: None for name in BOX_TERMS}).apply(scenario_result)
            is scenario_result
        )

    @staticmethod
    def test_identity_box_leaves_every_sample_untouched():
        plain = SimulatedDevice("filter", seed=5, noise_floor=0.01)
        boxed = SimulatedDevice(
            "filter", seed=5, noise_floor=0.01, error_model=ErrorModel()
        )
        assert _samples(plain, CAL_FREQUENCIES) == _samples(
            boxed, CAL_FREQUENCIES
        )


# ----------------------------------------------------------------------
# The error box itself, term by term
# ----------------------------------------------------------------------
class TestErrorModel:
    @staticmethod
    def test_terms_fill_disabled_entries_with_the_identity():
        assert ErrorModel().terms() == (0, 0, 0, 1, 1, 0)
        only_isolation = ErrorModel(isolation=BOX_TERMS["isolation"])
        assert only_isolation.terms() == (0, 0, 0, 1, 1, BOX_TERMS["isolation"])
        assert only_isolation.enabled_terms == ("isolation",)

    @staticmethod
    def test_is_identity_is_exactly_the_ideal_instrument():
        assert ErrorModel().is_identity
        assert ErrorModel(directivity=0).is_identity
        assert not ErrorModel(directivity=1e-12).is_identity
        assert not ErrorModel(reflection_tracking=1.0 + 1e-12).is_identity

    @staticmethod
    def test_a_zero_tracking_term_is_rejected():
        with pytest.raises(ValueError, match="reflection_tracking"):
            ErrorModel(reflection_tracking=0)
        with pytest.raises(ValueError, match="transmission_tracking"):
            ErrorModel(transmission_tracking=0)

    @staticmethod
    def test_non_finite_terms_are_rejected():
        with pytest.raises(ValueError, match="directivity"):
            ErrorModel(directivity=complex(math.nan, 0.0))

    @staticmethod
    def test_directivity_adds_an_offset():
        """e00 only: the measured reflection is S11 + e00, nothing else."""
        ideal = ScenarioResult(complex(0.2, -0.1), complex(0.0, 0.0))
        measured = ErrorModel(directivity=BOX_TERMS["directivity"]).apply(ideal)
        assert measured.s11 == ideal.s11 + BOX_TERMS["directivity"]
        assert measured.s21 == ideal.s21

    @staticmethod
    def test_reflection_tracking_scales_the_reflection():
        ideal = ScenarioResult(complex(0.2, -0.1), complex(0.0, 0.0))
        tracking = ErrorModel(
            reflection_tracking=BOX_TERMS["reflection_tracking"]
        ).apply(ideal)
        assert tracking.s11 == ideal.s11 * BOX_TERMS["reflection_tracking"]

    @staticmethod
    def test_transmission_tracking_and_isolation_scale_s21():
        ideal = ScenarioResult(complex(0.0, 0.0), complex(0.6, 0.0))
        model = ErrorModel(
            transmission_tracking=BOX_TERMS["transmission_tracking"],
            isolation=BOX_TERMS["isolation"],
        )
        expected = (
            ideal.s21 * BOX_TERMS["transmission_tracking"]
            + BOX_TERMS["isolation"]
        )
        assert model.apply(ideal).s21 == expected

    @staticmethod
    def test_source_match_alone_follows_the_one_port_closed_form():
        """e11 only: S11m = S11 / (1 - e11*S11), derived by hand."""
        s11 = complex(0.4, 0.3)
        e11 = BOX_TERMS["source_match"]
        measured = ErrorModel(source_match=e11).apply(
            ScenarioResult(s11, complex(0.0, 0.0))
        )
        assert measured.s11 == pytest.approx(s11 / (1 - e11 * s11), abs=1e-15)

    @staticmethod
    def test_load_match_alone_uses_the_two_port_closed_form():
        s11, s21, s12, s22 = 0.1 + 0.2j, 0.5 - 0.1j, 0.5 - 0.1j, -0.3 + 0.1j
        e22 = BOX_TERMS["load_match"]
        ideal = ScenarioResult(s11, s21, s12, s22)
        delta = s11 * s22 - s12 * s21
        denominator = (1 - 0.0 * s11) * (1 - e22 * s22) - 0.0
        measured = ErrorModel(load_match=e22).apply(ideal)
        assert measured.s11 == pytest.approx(
            (s11 - e22 * delta) / denominator, abs=1e-15
        )
        assert measured.s21 == pytest.approx(s21 / denominator, abs=1e-15)

    @staticmethod
    def test_each_term_can_be_switched_off_again():
        """Switching a term off restores the ideal value for that channel."""
        ideal = ScenarioResult(
            complex(0.2, -0.1), complex(0.6, 0.2), complex(0.6, 0.2), 0.0j
        )
        full = _box().apply(ideal)
        without_directivity = _box(directivity=None).apply(ideal)
        assert without_directivity.s11 != full.s11
        assert full.s11 - without_directivity.s11 == pytest.approx(
            BOX_TERMS["directivity"], abs=1e-15
        )

    @staticmethod
    def test_a_one_port_dut_keeps_its_missing_channels():
        """No reverse sample is invented for a DUT that states none."""
        measured = _one_port_box().apply(
            ScenarioResult(complex(0.4, -0.2), complex(0.0, 0.0))
        )
        assert measured.s12 is None
        assert measured.s22 is None

    @staticmethod
    def test_a_transmitting_dut_must_state_its_reverse_channels():
        with pytest.raises(ErrorModelError, match="no S12/S22"):
            _box().apply(ScenarioResult(complex(0.1, 0.0), complex(0.7, 0.0)))

    @staticmethod
    def test_the_source_match_still_applies_when_no_load_match_is_set():
        """e11 alone needs no S22, so a transmitting DUT is not refused."""
        e11 = BOX_TERMS["source_match"]
        s11 = complex(0.4, 0.3)
        measured = ErrorModel(source_match=e11).apply(
            ScenarioResult(s11, complex(0.7, 0.0))
        )
        assert measured.s11 == pytest.approx(s11 / (1 - e11 * s11), abs=1e-15)
        assert measured.s12 is None
        assert measured.s22 is None

    @staticmethod
    def test_a_singular_box_is_reported_not_silently_ignored():
        # e11 = 1 and S11 = 1 makes 1 - e11*S11 exactly zero.
        with pytest.raises(ErrorModelError, match="singular"):
            ErrorModel(source_match=1 + 0j).apply(
                ScenarioResult(complex(1.0, 0.0), complex(0.0, 0.0))
            )

    @staticmethod
    def test_provenance_names_only_the_enabled_terms():
        text = ErrorModel(isolation=BOX_TERMS["isolation"]).describe()
        assert "isolation" in text
        assert "directivity" not in text
        assert SIMULATION_DISCLAIMER in text
        assert ErrorModel().describe() == (
            "simulated error box: none (ideal instrument)"
        )
        assert ErrorModel(isolation=1e-4).to_dict() == {
            "isolation": [1e-4, 0.0]
        }


# ----------------------------------------------------------------------
# Virtual standards: defined, then measured
# ----------------------------------------------------------------------
class TestVirtualStandards:
    @staticmethod
    def test_ideal_standards_are_exactly_their_definitions():
        standards = VirtualStandards()
        assert standards.ideal("short").s11 == -1 + 0j
        assert standards.ideal("open").s11 == 1 + 0j
        assert standards.ideal("load").s11 == 0j
        assert standards.ideal("through").s11 == 0j
        assert standards.ideal("through").s21 == 1 + 0j
        assert standards.ideal("isolation").s21 == 0j

    @staticmethod
    def test_an_identity_box_measures_the_ideal_standards():
        standards = VirtualStandards()
        for name in standards.names:
            assert standards.measured(name) == standards.ideal(name)

    @staticmethod
    def test_standards_are_measured_through_the_same_box_as_the_dut():
        box = _one_port_box()
        standards = VirtualStandards(box)
        raw = standards.measured("short")
        assert raw == box.apply(standards.ideal("short"))
        # The load now reads the directivity, not a perfect zero.
        assert standards.measured("load").s11 == pytest.approx(
            BOX_TERMS["directivity"], abs=1e-15
        )

    @staticmethod
    def test_a_standard_individually_switches_the_terms_it_needs():
        """A box with only directivity gives short = e00 - 1, by hand."""
        standards = VirtualStandards(
            ErrorModel(directivity=BOX_TERMS["directivity"])
        )
        assert standards.measured("short").s11 == pytest.approx(
            -1 + BOX_TERMS["directivity"], abs=1e-15
        )
        assert standards.measured("open").s11 == pytest.approx(
            1 + BOX_TERMS["directivity"], abs=1e-15
        )

    @staticmethod
    def test_standards_never_read_corrected_display_data():
        """The measured value is a pure function of definition and box."""
        standards = VirtualStandards(_box())
        again = VirtualStandards(_box())
        for name in standards.names:
            assert standards.measured(name) == again.measured(name)
        # Nothing in the model has any channel to read back from.
        assert not hasattr(standards, "snapshot")
        assert not hasattr(standards, "corrected")

    @staticmethod
    def test_a_standard_can_be_swept_as_a_device_scenario():
        standards = VirtualStandards(_one_port_box())
        short = standards.standard("short")
        device = SimulatedDevice(short)
        assert device.scenario_key == "standard_short"
        assert device.supported_channels == ("data 0",)
        assert device.read_channel("data 0")[0] == pytest.approx(
            standards.measured("short").s11, abs=1e-15
        )
        through = SimulatedDevice(standards.standard("through"))
        assert through.supported_channels == ("data 0", "data 1")

    @staticmethod
    def test_the_two_port_calibration_is_complete():
        standards = VirtualStandards(_box())
        device = SimulatedDevice(standards.standard("through"))
        device.set_sweep(1_000_000, 2_000_000, 3)
        assert all(
            math.isfinite(abs(s.s21))
            for s in _samples(device, device.frequency_grid())
        )

    @staticmethod
    def test_provenance_records_the_standards_and_the_box():
        standards = VirtualStandards(_box())
        state = standards.acquisition_state()
        assert set(state["standards"]) == set(standards.names)
        assert state["error_model"] == _box().to_dict()
        assert state["disclaimer"] == SIMULATION_DISCLAIMER
        assert "short" in standards.describe()

    @staticmethod
    def test_an_unknown_standard_is_named_not_guessed():
        with pytest.raises(ValueError, match="unknown standard"):
            VirtualStandards().ideal("open_short")


# ----------------------------------------------------------------------
# Injected error, and its recovery by a calibration
# ----------------------------------------------------------------------
class TestInjectedErrorRecovery:
    @staticmethod
    def test_the_error_box_reaches_the_device_samples():
        ideal = SimulatedDevice("filter")
        boxed = SimulatedDevice("filter", error_model=_box())
        freq = 50_000_000
        assert boxed.sample(freq) != ideal.sample(freq)
        assert boxed.sample(freq) == _box().apply(ideal.sample(freq))

    @staticmethod
    def test_one_port_recovery_recovers_the_ideal_dut():
        """A three-term calibration undoes the one-port box completely."""
        box = _one_port_box()
        calibration = _calibration(VirtualStandards(box), CAL_FREQUENCIES)
        assert calibration.isValid1Port()

        antenna = AntennaResonance()
        worst = 0.0
        worst_uncalibrated = 0.0
        for freq in CAL_FREQUENCIES:
            reference = antenna(freq)
            measured = box.apply(reference)
            corrected = calibration.correct11(
                Datapoint(freq, measured.s11.real, measured.s11.imag)
            ).z
            worst = max(worst, abs(corrected - reference.s11))
            worst_uncalibrated = max(
                worst_uncalibrated, abs(measured.s11 - reference.s11)
            )
        assert REFERENCE_IMPEDANCE == 50.0
        assert worst <= RECOVERY_TOLERANCE
        assert worst_uncalibrated >= UNCALIBRATED_MIN_ERROR

    @staticmethod
    def test_two_port_recovery_recovers_the_ideal_dut():
        """Through/isolation/thrurefl plus load match recover the DUT."""
        box = _box()
        calibration = _calibration(VirtualStandards(box), CAL_FREQUENCIES)
        assert calibration.isValid2Port()

        duts = (ButterworthLowPass(), Attenuator())
        worst = 0.0
        worst_uncalibrated = 0.0
        for dut in duts:
            for freq in CAL_FREQUENCIES:
                reference = dut(freq)
                measured = box.apply(reference)
                s11, s21 = _corrected(calibration, freq, measured)
                expected11, expected21 = _load_match_closed_form(reference, box)
                worst = max(worst, abs(s11 - expected11), abs(s21 - expected21))
                worst_uncalibrated = max(
                    worst_uncalibrated,
                    abs(measured.s11 - reference.s11),
                    abs(measured.s21 - reference.s21),
                )
        assert worst <= RECOVERY_TOLERANCE
        assert worst_uncalibrated >= UNCALIBRATED_MIN_ERROR

    @staticmethod
    def test_the_recovered_dut_is_the_ideal_one_too_where_e22_is_zero():
        """With no load match the correction must return the DUT exactly."""
        box = _box(load_match=None)
        calibration = _calibration(VirtualStandards(box), CAL_FREQUENCIES)
        dut = ButterworthLowPass()
        for freq in CAL_FREQUENCIES:
            reference = dut(freq)
            measured = box.apply(reference)
            s11, s21 = _corrected(calibration, freq, measured)
            assert abs(s11 - reference.s11) <= RECOVERY_TOLERANCE
            assert abs(s21 - reference.s21) <= RECOVERY_TOLERANCE

    @staticmethod
    def test_a_calibration_built_without_the_box_does_not_recover():
        """Positive control: the recovery comes from the standards, not luck."""
        box = _box()
        # A calibration built from a *different* box: every standard is
        # measurable (none is exactly zero), so the completeness rule is
        # satisfied and the calibration is simply wrong for this instrument.
        wrong = _calibration(VirtualStandards(_one_port_box()), CAL_FREQUENCIES)
        dut = ButterworthLowPass()
        worst = 0.0
        for freq in CAL_FREQUENCIES:
            reference = dut(freq)
            measured = box.apply(reference)
            s11, s21 = _corrected(wrong, freq, measured)
            worst = max(
                worst, abs(s11 - reference.s11), abs(s21 - reference.s21)
            )
        assert worst >= UNCALIBRATED_MIN_ERROR

    @staticmethod
    def test_an_ideal_load_is_rejected_by_the_completeness_rule():
        """Documented limitation: a perfect zero load is not "present"."""
        calibration = Calibration()
        element = CalElement()
        element.short_state = "IDEAL"
        element.open_state = "IDEAL"
        element.load_state = "IDEAL"
        calibration.cal_element = element
        for name, value in (("short", -1 + 0j), ("open", 1 + 0j), ("load", 0j)):
            calibration.insert(
                name,
                [Datapoint(f, value.real, value.imag) for f in CAL_FREQUENCIES],
            )
        assert not calibration.isValid1Port()
        with pytest.raises(ValueError):
            calibration.calc_corrections()


# ----------------------------------------------------------------------
# Reversal
# ----------------------------------------------------------------------
class _AsymmetricTwoPort:
    """A deliberately asymmetric ideal two-port, for reversal semantics.

    Only used as a test double so that swapping the ports is observable:
    the shipped analytic models are symmetric, where a reversal would be a
    numerical no-op and could hide a wrong swap.
    """

    supported_channels = ("data 0", "data 1")

    def __call__(self, _freq: float) -> ScenarioResult:
        return ScenarioResult(
            s11=complex(0.10, 0.00),
            s21=complex(0.80, 0.10),
            s12=complex(0.30, -0.20),
            s22=complex(-0.40, 0.05),
        )


class TestReversal:
    @staticmethod
    def test_reversal_swaps_the_ports_of_an_asymmetric_dut():
        forward = _AsymmetricTwoPort()(1.0)
        backward = ReversedTwoPort(_AsymmetricTwoPort())(1.0)
        assert backward.s11 == forward.s22
        assert backward.s22 == forward.s11
        assert backward.s21 == forward.s12
        assert backward.s12 == forward.s21

    @staticmethod
    def test_a_reciprocal_symmetric_dut_reads_the_same_either_way():
        forward = ButterworthLowPass()(50_000_000)
        backward = ReversedTwoPort(ButterworthLowPass())(50_000_000)
        assert backward.s11 == forward.s11
        assert backward.s21 == forward.s21

    @staticmethod
    def test_a_one_port_cannot_be_reversed_and_says_so():
        with pytest.raises(ReversalError, match="S12/S22"):
            SimulatedDevice("antenna", reverse=True).sample(14_000_000)

    @staticmethod
    def test_reversal_keeps_the_channel_set_and_names_itself():
        device = SimulatedDevice("filter", reverse=True)
        assert device.supported_channels == ("data 0", "data 1")
        assert device.scenario_key == "filter_reversed"
        assert "reversed" in device.scenario_name
        assert "ports exchanged" in device.provenance_note
        one_port = SimulatedDevice(DualBandAntenna(), reverse=True)
        assert one_port.supported_channels == ("data 0",)

    @staticmethod
    def test_a_reversed_one_port_device_is_still_refused_at_read_time():
        device = SimulatedDevice(DualBandAntenna(), reverse=True)
        with pytest.raises(ReversalError):
            device.read_channel("data 0")


# ----------------------------------------------------------------------
# Channels, new scenarios, markers
# ----------------------------------------------------------------------
class TestChannelsAndScenarios:
    @staticmethod
    @pytest.mark.parametrize(
        "scenario", ["antenna", "dual_band", "bandpass", "filter"]
    )
    def test_only_real_channels_are_declared(scenario):
        device = SimulatedDevice(scenario)
        expected = (
            ("data 0",)
            if scenario in ("antenna", "dual_band")
            else ("data 0", "data 1")
        )
        assert device.supported_channels == expected
        if "data 1" not in expected:
            with pytest.raises(ReplayChannelError):
                device.read_channel("data 1")

    @staticmethod
    def test_an_error_box_never_adds_a_channel():
        device = SimulatedDevice("antenna", error_model=_one_port_box())
        assert device.supported_channels == ("data 0",)
        with pytest.raises(ReplayChannelError):
            device.read_channel("data 1")

    @staticmethod
    def test_the_dual_band_antenna_has_two_distinct_minima():
        antenna = DualBandAntenna()
        low, high = 7.1e6, 14.2e6
        assert antenna.vswr(low) < 1.1
        assert antenna.vswr(high) < 1.1
        # Between the bands it is badly mismatched: the model is not a
        # broadband match with two cosmetic notches.
        assert antenna.vswr(10e6) > 10.0
        assert antenna.supported_channels == ("data 0",)

    @staticmethod
    def test_the_band_pass_is_3db_at_its_stated_edges():
        filt = BandPassFilter(center_frequency=100e6, bandwidth=10e6, order=3)
        ratio = filt.bandwidth / (2 * filt.center_frequency)
        root = math.sqrt(1 + ratio * ratio)
        lower = filt.center_frequency * (root - ratio)
        upper = filt.center_frequency * (root + ratio)
        assert math.isclose(upper - lower, filt.bandwidth, rel_tol=1e-12)
        assert filt.magnitude_db(lower) == pytest.approx(-3.0103, abs=1e-4)
        assert filt.magnitude_db(upper) == pytest.approx(-3.0103, abs=1e-4)
        assert filt.magnitude_db(filt.center_frequency) == pytest.approx(
            0.0, abs=1e-12
        )
        # Two decades up the response is deep in the stop band.
        assert filt.magnitude_db(1_000e6) < -100.0

    @staticmethod
    def test_new_scenarios_are_offered_by_name():
        for name in ("dual_band", "bandpass"):
            assert name in SCENARIO_TYPES
            assert name in DEFAULT_SCENARIOS
            device = SimulatedDevice(name)
            device.set_sweep(1_000_000, 200_000_000, 5)
            assert len(device.read_channel("data 0")) == 5


# ----------------------------------------------------------------------
# Provenance and labelling
# ----------------------------------------------------------------------
class TestProvenance:
    @staticmethod
    def test_a_plain_analytic_scenario_adds_no_note():
        assert SimulatedDevice("filter").provenance_note == ""

    @staticmethod
    def test_an_active_box_is_named_in_the_note_and_the_banner():
        device = SimulatedDevice("filter", error_model=_box())
        assert SIMULATION_DISCLAIMER in device.provenance_note
        assert "directivity" in device.provenance_note
        assert "simulation layers" in device.read_info()
        assert SIMULATION_DISCLAIMER in device.read_info()

    @staticmethod
    def test_the_state_record_is_machine_readable_and_labelled():
        device = SimulatedDevice("filter", error_model=_box(), reverse=True)
        state = device.simulation_state()
        assert state["simulated"] is True
        assert state["scenario"] == "filter_reversed"
        assert state["reversed"] is True
        assert state["error_model"] == _box().to_dict()
        assert state["disclaimer"] == SIMULATION_DISCLAIMER
        assert state["reference_impedance"] == 50.0

    @staticmethod
    def test_the_state_record_carries_the_standards_when_they_are_active():
        standards = VirtualStandards(_box())
        device = SimulatedDevice(standards.standard("through"))
        state = device.simulation_state()
        assert set(state["standards"]["standards"]) == set(standards.names)

    @staticmethod
    def test_a_sweep_reports_the_note_and_the_simulated_label():
        vna = make_simulated_vna("filter", error_model=_box())
        request = SweepRequest(start_hz=1_000_000, stop_hz=10_000_000, points=5)
        with open_device(vna, name="simulated") as session:
            result = session.sweep(request)
        # The adapter, not ``Acquisition``, is what carries the layer note;
        # the snapshot itself is labelled simulated.
        assert vna.provenance_note != ""
        assert SIMULATION_DISCLAIMER in vna.provenance_note
        assert vna.scenario_key == "filter"
        assert vna.simulated is True
        assert result.snapshot.simulated is True

    @staticmethod
    def test_a_report_labels_the_data_simulated():
        vna = make_simulated_vna("filter", error_model=_box())
        request = SweepRequest(start_hz=1_000_000, stop_hz=10_000_000, points=5)
        with open_device(vna, name="simulated") as session:
            snapshot = session.sweep(request).snapshot
        trace = ReportTrace(
            trace_id="t1", name="S11", color="#fff", snapshot=snapshot
        )
        labels = provenance_labels(trace)
        assert any(label.startswith("SIMULATED") for label in labels)


# ----------------------------------------------------------------------
# Repeatability, cancellation, ownership
# ----------------------------------------------------------------------
class TestRepeatability:
    @staticmethod
    def test_the_same_seed_gives_the_same_samples_with_a_box():
        first = SimulatedDevice(
            "filter", seed=11, noise_floor=0.01, error_model=_box()
        )
        second = SimulatedDevice(
            "filter", seed=11, noise_floor=0.01, error_model=_box()
        )
        assert _samples(first, CAL_FREQUENCIES) == _samples(
            second, CAL_FREQUENCIES
        )

    @staticmethod
    def test_a_different_seed_gives_different_samples():
        first = SimulatedDevice(
            "filter", seed=11, noise_floor=0.01, error_model=_box()
        )
        second = SimulatedDevice(
            "filter", seed=12, noise_floor=0.01, error_model=_box()
        )
        assert _samples(first, CAL_FREQUENCIES) != _samples(
            second, CAL_FREQUENCIES
        )

    @staticmethod
    def test_an_identity_box_does_not_consume_the_random_stream():
        """Turning the layer on must not shift the seeded draws."""
        plain = SimulatedDevice("filter", seed=11, noise_floor=0.01)
        identity = SimulatedDevice(
            "filter", seed=11, noise_floor=0.01, error_model=ErrorModel()
        )
        assert _samples(plain, CAL_FREQUENCIES) == _samples(
            identity, CAL_FREQUENCIES
        )


class _CancelOnSecondChannel:
    """Forwards to the simulator and cancels before the second read."""

    def __init__(self, device, token: CancelToken) -> None:
        self.device = device
        self.token = token

    def __getattr__(self, name):
        return getattr(self.device, name)

    @property
    def datapoints(self) -> int:
        return self.device.datapoints

    @datapoints.setter
    def datapoints(self, points: int) -> None:
        # ``Acquisition`` sets the point count as an attribute before
        # calling ``setSweep``; it has to reach the device, not this wrapper.
        self.device.datapoints = points

    def readValues(self, value: str) -> list[complex]:
        if value == "data 1":
            self.token.cancel()
        return self.device.readValues(value)


class TestCancellation:
    @staticmethod
    def test_a_cancelled_sweep_publishes_nothing():
        device = make_simulated_vna("filter", error_model=_box())
        token = CancelToken()
        request = SweepRequest(start_hz=1_000_000, stop_hz=10_000_000, points=5)
        session = open_device(_CancelOnSecondChannel(device, token))
        published = []
        try:
            with pytest.raises(CancelledError):
                published.append(session.sweep(request, token))
        finally:
            session.close()
        assert published == []
        assert token.cancelled

    @staticmethod
    def test_a_cancelled_token_publishes_nothing_before_the_first_read():
        device = SimulatedDevice("filter")
        token = CancelToken()
        token.cancel()
        request = SweepRequest(start_hz=1_000_000, stop_hz=10_000_000, points=5)
        with open_device(device) as session:
            with pytest.raises(CancelledError):
                session.sweep(request, token)

    @staticmethod
    def test_the_same_device_still_sweeps_after_a_cancellation():
        device = make_simulated_vna("filter")
        token = CancelToken()
        request = SweepRequest(start_hz=1_000_000, stop_hz=10_000_000, points=5)
        session = open_device(_CancelOnSecondChannel(device, token))
        try:
            with pytest.raises(CancelledError):
                session.sweep(request, token)
            # A fresh token: the cancelled run left no queued completion
            # behind that could be published under these settings.
            fresh = session.sweep(request, CancelToken())
        finally:
            session.close()
        assert len(fresh.snapshot.available) == 2
        assert fresh.request == request
