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
"""RF-03: independent verification of fixture de-embedding.

Every expected value here is derived in this file from textbook two-port
algebra: the fixture and the device under test are described by ABCD
matrices, cascaded by plain matrix multiplication, and converted to
S-parameters with the standard formulas.  ``Fixture.s_to_t``/``t_to_s`` are
never used to build an expectation, so the module's own parameter transform
is under test rather than assumed.  The measurements handed to
``remove_fixture`` are therefore exactly what a perfect instrument would
record for the cascade, and de-embedding must return the DUT to the last bit.
"""

import cmath
import json
import math

import pytest

from NanoVNASaver.Fixture import (
    Fixture,
    FixtureError,
    FixturePoint,
    deembed_point,
    fixture_from_stage,
    fixture_from_touchstone,
    fixture_stage,
    remove_fixture,
)
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Touchstone import Touchstone

Z0 = 50.0

#: Frequencies shared by the fixture and the measurement.
FREQUENCIES = (1_000_000, 2_500_000, 5_000_000, 7_500_000, 10_000_000)

#: The lab fixture's line: a stated delay and a stated attenuation.
DELAY = 2.5e-9
LOSS_DB = 1.5

#: Exactness demanded of a de-embedding that inverts an analytic cascade.
EXACT = 1e-9

ABCD = tuple[tuple[complex, complex], tuple[complex, complex]]
IDENTITY: ABCD = ((1 + 0j, 0j), (0j, 1 + 0j))


# -- analytic two-ports -------------------------------------------------
def _cascade(*blocks: ABCD) -> ABCD:
    """Cascade two-ports left to right, signal entering the first block."""
    result = IDENTITY
    for block in blocks:
        (a, b), (c, d) = result
        (e, f), (g, h) = block
        result = (
            (a * e + b * g, a * f + b * h),
            (c * e + d * g, c * f + d * h),
        )
    return result


def _to_s(abcd: ABCD) -> dict[str, complex]:
    """ABCD to S for equal reference impedances at both ports."""
    (a, b), (c, d) = abcd
    denominator = a + b / Z0 + c * Z0 + d
    determinant = a * d - b * c
    return {
        "s11": (a + b / Z0 - c * Z0 - d) / denominator,
        "s21": 2 / denominator,
        "s12": 2 * determinant / denominator,
        "s22": (-a + b / Z0 - c * Z0 + d) / denominator,
    }


def _from_s(params: dict[str, complex]) -> ABCD:
    """S to ABCD, the textbook rearrangement written out independently."""
    s11 = params["s11"]
    s21 = params["s21"]
    s12 = params["s12"]
    s22 = params["s22"]
    return (
        (
            ((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21),
            Z0 * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21),
        ),
        (
            ((1 - s11) * (1 - s22) - s12 * s21) / (2 * Z0 * s21),
            ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21),
        ),
    )


def _series(impedance: complex) -> ABCD:
    return ((1 + 0j, complex(impedance)), (0j, 1 + 0j))


def _shunt(admittance: complex) -> ABCD:
    return ((1 + 0j, 0j), (complex(admittance), 1 + 0j))


def _line(freq: int, delay: float, loss_db: float) -> ABCD:
    """Lossy line with propagation ``gamma*l = alpha + j*2*pi*f*tau``."""
    gamma_length = complex(
        loss_db * math.log(10.0) / 20.0, 2 * math.pi * freq * delay
    )
    cosh = cmath.cosh(gamma_length)
    sinh = cmath.sinh(gamma_length)
    return ((cosh, Z0 * sinh), (sinh / Z0, cosh))


def _attenuator(loss_db: float) -> ABCD:
    """Matched resistive T pad with ``loss_db`` of attenuation."""
    ratio = 10 ** (loss_db / 20)
    series = Z0 * (ratio - 1) / (ratio + 1)
    to_ground = 2 * ratio * Z0 / (ratio**2 - 1)
    return _cascade(_series(series), _shunt(1 / to_ground), _series(series))


def _isolator(insertion_db: float, isolation_db: float) -> ABCD:
    """Non-reciprocal two-port: forward loss, much deeper reverse loss."""
    return _from_s(
        {
            "s11": 0j,
            "s21": 10 ** (-insertion_db / 20),
            "s12": 10 ** (-isolation_db / 20),
            "s22": 0j,
        }
    )


# -- the two fixtures ---------------------------------------------------
def _lab_fixture(freq: int) -> ABCD:
    """Line plus a mismatch that looks different from each port."""
    return _cascade(
        _line(freq, DELAY, LOSS_DB),
        _series(complex(15.0, 20.0)),
        _shunt(complex(0.0, 2 * math.pi * freq * 1.5e-12)),
    )


def _rig_fixture(freq: int) -> ABCD:
    """Non-reciprocal fixture: S11 != S22 and S21 != S12."""
    return _cascade(
        _line(freq, 1.8e-9, 0.7),
        _isolator(1.0, 20.0),
        _series(complex(8.0, -14.0)),
        _shunt(complex(0.0, 2 * math.pi * freq * 0.8e-12)),
    )


def _through(_freq: int) -> ABCD:
    return IDENTITY


# -- devices under test -------------------------------------------------
DUT_ATTENUATOR: ABCD = _attenuator(6.0)
DUT_SERIES: ABCD = _series(complex(25.0, 30.0))


# -- plumbing -----------------------------------------------------------
def _fixture(
    freqs, factory, name: str, reference_impedance: float = Z0
) -> Fixture:
    return Fixture(
        name=name,
        reference_impedance=reference_impedance,
        points=tuple(
            FixturePoint(freq, **_to_s(factory(freq))) for freq in freqs
        ),
    )


def _measured(freqs, dut: ABCD, factory=_lab_fixture) -> dict:
    """The S-parameters a perfect two-port measurement would report."""
    channels: dict[str, list[Datapoint]] = {
        name: [] for name in ("S11", "S21", "S12", "S22")
    }
    for freq in freqs:
        sample = _to_s(_cascade(factory(freq), dut))
        for name, points in channels.items():
            value = sample[name.lower()]
            points.append(Datapoint(freq, value.real, value.imag))
    return {name: tuple(points) for name, points in channels.items()}


def _assert_matches(result, dut: ABCD, tolerance: float) -> None:
    expected = _to_s(dut)
    for name in ("S11", "S21", "S12", "S22"):
        want = expected[name.lower()]
        for freq, got in zip(FREQUENCIES, result[name], strict=True):
            assert got.freq == freq
            assert abs(got.z - want) <= tolerance, (
                f"{name} at {freq} Hz: de-embedded {got.z!r}, analytic {want!r}"
            )


class TestAnalyticModel:
    """The expectations are only meaningful if this model is right."""

    @staticmethod
    def test_the_pad_attenuates_by_its_stated_amount():
        sample = _to_s(DUT_ATTENUATOR)
        assert abs(sample["s21"]) == pytest.approx(10 ** (-6.0 / 20), abs=1e-15)
        assert abs(sample["s11"]) < 1e-15
        assert abs(sample["s22"]) < 1e-15

    @staticmethod
    def test_a_series_arm_matches_the_closed_form_reflection():
        impedance = complex(25.0, 30.0)
        expected = impedance / (impedance + 2 * Z0)
        sample = _to_s(DUT_SERIES)
        assert abs(sample["s11"] - expected) < 1e-15
        assert abs(sample["s22"] - expected) < 1e-15
        assert abs(sample["s21"] - 2 * Z0 / (impedance + 2 * Z0)) < 1e-15


class TestExactCascade:
    @staticmethod
    def test_the_measurement_really_carries_the_fixture():
        """A fixture that did nothing would make the next tests vacuous."""
        channels = _measured(FREQUENCIES, DUT_ATTENUATOR)
        expected = _to_s(DUT_ATTENUATOR)
        assert abs(channels["S11"][0].z - expected["s11"]) > 1e-3
        assert abs(channels["S21"][0].z - expected["s21"]) > 1e-3
        assert abs(channels["S22"][0].z - expected["s22"]) > 1e-3

    @staticmethod
    def test_a_resistive_attenuator_is_recovered_exactly():
        channels = _measured(FREQUENCIES, DUT_ATTENUATOR)
        fixture = _fixture(FREQUENCIES, _lab_fixture, "lab rig")
        result = remove_fixture(channels, fixture, Z0)
        _assert_matches(result, DUT_ATTENUATOR, EXACT)

    @staticmethod
    def test_a_series_impedance_is_recovered_exactly():
        channels = _measured(FREQUENCIES, DUT_SERIES)
        fixture = _fixture(FREQUENCIES, _lab_fixture, "lab rig")
        result = remove_fixture(channels, fixture, Z0)
        _assert_matches(result, DUT_SERIES, EXACT)

    @staticmethod
    def test_the_lab_fixture_is_asymmetric_but_reciprocal():
        sample = _to_s(_lab_fixture(FREQUENCIES[0]))
        assert abs(sample["s11"] - sample["s22"]) > 1e-3
        assert abs(sample["s21"] - sample["s12"]) < 1e-15


class TestAsymmetricFixture:
    @staticmethod
    def test_the_rig_fixture_is_genuinely_non_reciprocal():
        sample = _to_s(_rig_fixture(FREQUENCIES[0]))
        assert abs(sample["s11"] - sample["s22"]) > 1e-3
        assert abs(sample["s21"] - sample["s12"]) > 0.1

    @staticmethod
    def test_a_non_reciprocal_fixture_is_removed_exactly():
        channels = _measured(FREQUENCIES, DUT_SERIES, _rig_fixture)
        fixture = _fixture(FREQUENCIES, _rig_fixture, "isolator rig")
        result = remove_fixture(channels, fixture, Z0)
        _assert_matches(result, DUT_SERIES, EXACT)

    @staticmethod
    def test_an_attenuator_behind_a_non_reciprocal_fixture():
        channels = _measured(FREQUENCIES, DUT_ATTENUATOR, _rig_fixture)
        fixture = _fixture(FREQUENCIES, _rig_fixture, "isolator rig")
        result = remove_fixture(channels, fixture, Z0)
        _assert_matches(result, DUT_ATTENUATOR, EXACT)


class TestRefusals:
    @staticmethod
    def test_a_fixture_on_another_grid_is_refused_not_interpolated():
        shifted = tuple(freq + 1 for freq in FREQUENCIES)
        fixture = _fixture(shifted, _lab_fixture, "shifted rig")
        channels = _measured(FREQUENCIES, DUT_SERIES)
        missing = FREQUENCIES[0]
        with pytest.raises(FixtureError) as info:
            remove_fixture(channels, fixture, Z0)
        message = str(info.value)
        assert str(missing) in message
        assert "no sample" in message
        with pytest.raises(FixtureError):
            fixture.value_at(missing)

    @staticmethod
    def test_a_fixture_without_transmission_is_refused():
        freq = FREQUENCIES[0]
        transmission = 10 ** (-80.0 / 20)
        fixture = Fixture(
            name="80 dB pad",
            points=(FixturePoint(freq, 0j, transmission, transmission, 0j),),
        )
        channels = {
            name: (Datapoint(freq, 1.0, 0.0),)
            for name in ("S11", "S21", "S12", "S22")
        }
        with pytest.raises(FixtureError) as info:
            remove_fixture(channels, fixture, Z0)
        message = str(info.value)
        assert "ill-conditioned" in message
        assert str(freq) in message

    @staticmethod
    def test_the_single_point_path_refuses_it_too():
        freq = FREQUENCIES[0]
        transmission = 10 ** (-80.0 / 20)
        fixture = FixturePoint(freq, 0j, transmission, transmission, 0j)
        measured = FixturePoint(freq, 1.0, 1.0, 1.0, 1.0)
        with pytest.raises(FixtureError) as info:
            deembed_point(freq, measured, fixture)
        assert "ill-conditioned" in str(info.value)

    @staticmethod
    def test_a_measurement_without_s12_is_refused():
        fixture = _fixture(FREQUENCIES, _lab_fixture, "lab rig")
        channels = _measured(FREQUENCIES, DUT_SERIES)
        without_reverse = {
            name: points for name, points in channels.items() if name != "S12"
        }
        with pytest.raises(FixtureError) as info:
            remove_fixture(without_reverse, fixture, Z0)
        assert "S12" in str(info.value)

    @staticmethod
    def test_an_empty_reverse_channel_is_refused():
        fixture = _fixture(FREQUENCIES, _lab_fixture, "lab rig")
        channels = dict(_measured(FREQUENCIES, DUT_SERIES))
        channels["S12"] = ()
        with pytest.raises(FixtureError) as info:
            remove_fixture(channels, fixture, Z0)
        assert "S12" in str(info.value)

    @staticmethod
    def test_a_75_ohm_fixture_against_a_50_ohm_measurement_is_refused():
        fixture = _fixture(
            FREQUENCIES, _lab_fixture, "75 ohm rig", reference_impedance=75.0
        )
        channels = _measured(FREQUENCIES, DUT_SERIES)
        with pytest.raises(FixtureError) as info:
            remove_fixture(channels, fixture, Z0)
        message = str(info.value)
        assert "75" in message
        assert "50" in message
        assert "impedance" in message


class TestThroughFixture:
    @staticmethod
    def test_a_perfect_through_leaves_the_measurement_unchanged():
        channels = _measured(FREQUENCIES, DUT_SERIES)
        fixture = _fixture(FREQUENCIES, _through, "through")
        result = remove_fixture(channels, fixture, Z0)
        for name, points in channels.items():
            for original, got in zip(points, result[name], strict=True):
                assert got.freq == original.freq
                assert abs(got.z - original.z) <= 1e-12


class TestStorageRoundTrip:
    @staticmethod
    def test_a_touchstone_file_round_trips_identity_and_samples(tmp_path):
        fixture = _fixture(FREQUENCIES, _rig_fixture, "isolator rig")
        path = tmp_path / "fixture.s2p"
        store = Touchstone(str(path))
        store.s11 = _channel(fixture.points, "s11")
        store.s21 = _channel(fixture.points, "s21")
        store.s12 = _channel(fixture.points, "s12")
        store.s22 = _channel(fixture.points, "s22")
        store.save(4)

        loaded = Touchstone(str(path))
        loaded.load()
        assert loaded.present_parameters() == ("S11", "S21", "S12", "S22")
        reopened = fixture_from_touchstone(loaded, name="isolator rig")
        assert reopened.identity() == fixture.identity()
        assert reopened.reference_impedance == fixture.reference_impedance
        for index in (0, 2, len(FREQUENCIES) - 1):
            assert reopened.points[index] == fixture.points[index]

    @staticmethod
    def test_mismatched_channel_lengths_are_named_not_indexed():
        """A short column must not surface as a bare IndexError."""
        fixture = _fixture(FREQUENCIES, _rig_fixture, "isolator rig")
        store = Touchstone("")
        store.s11 = _channel(fixture.points, "s11")
        store.s21 = _channel(fixture.points, "s21")[:-1]
        store.s12 = _channel(fixture.points, "s12")
        store.s22 = _channel(fixture.points, "s22")
        with pytest.raises(FixtureError, match="different numbers of points"):
            fixture_from_touchstone(store)

    @staticmethod
    def test_a_stage_round_trips_the_fixture():
        fixture = _fixture(FREQUENCIES, _rig_fixture, "isolator rig")
        stage = fixture_stage(fixture)
        assert stage.as_dict["identity"] == fixture.identity()
        rebuilt = fixture_from_stage(stage)
        assert rebuilt.identity() == fixture.identity()
        assert rebuilt.points == fixture.points

    @staticmethod
    def test_a_persisted_stage_round_trips_the_fixture():
        """A session stores the stage as JSON: identity must survive it."""
        from NanoVNASaver.Processing import ProcessingStage

        fixture = _fixture(FREQUENCIES, _rig_fixture, "isolator rig")
        stage = fixture_stage(fixture)
        stored = json.loads(json.dumps(stage.to_dict()))
        rebuilt = fixture_from_stage(ProcessingStage.from_dict(stored))
        assert rebuilt.identity() == fixture.identity()
        assert rebuilt.points == fixture.points
        assert rebuilt.name == fixture.name

    @staticmethod
    def test_a_fixture_restored_from_a_stage_still_de_embeds():
        fixture = _fixture(FREQUENCIES, _rig_fixture, "isolator rig")
        channels = _measured(FREQUENCIES, DUT_SERIES, _rig_fixture)
        restored = fixture_from_stage(fixture_stage(fixture))
        result = remove_fixture(channels, restored, Z0)
        _assert_matches(result, DUT_SERIES, EXACT)


def _channel(points, attribute: str) -> list[Datapoint]:
    return [
        Datapoint(
            point.freq,
            getattr(point, attribute).real,
            getattr(point, attribute).imag,
        )
        for point in points
    ]
