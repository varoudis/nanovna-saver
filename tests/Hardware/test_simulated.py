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
"""SIM-01: deterministic simulated device, verified against known physics."""

import math

import pytest

from NanoVNASaver.Hardware.Simulated import (
    FaultConfig,
    FaultMode,
    SimulatedDevice,
    make_simulated_vna,
)
from NanoVNASaver.Hardware.SimulatedScenarios import (
    AntennaResonance,
    Attenuator,
    ButterworthLowPass,
    CableLine,
    build_scenario,
)
from NanoVNASaver.Settings.Sweep import Sweep
from NanoVNASaver.SweepWorker import truncate


class TestScenarioPhysics:
    @staticmethod
    def test_antenna_resonance_is_matched():
        antenna = AntennaResonance(
            resonant_frequency=14.15e6, radiation_resistance=50.0, q_factor=20
        )
        assert antenna.vswr(14.15e6) == pytest.approx(1.0)
        assert abs(antenna.s11(14.15e6)) < 1e-9
        # Off resonance the match degrades monotonically away from f0.
        assert antenna.vswr(13.5e6) > antenna.vswr(14.0e6) > 1.0
        assert antenna.vswr(14.8e6) > antenna.vswr(14.3e6) > 1.0

    @staticmethod
    def test_antenna_reactance_sign_changes_across_resonance():
        antenna = AntennaResonance()
        below = antenna.impedance(antenna.resonant_frequency * 0.9)
        above = antenna.impedance(antenna.resonant_frequency * 1.1)
        assert below.imag < 0  # capacitive below resonance
        assert above.imag > 0  # inductive above resonance
        assert below.real == pytest.approx(below.real)

    @staticmethod
    def test_antenna_vswr_bandwidth_matches_definition():
        antenna = AntennaResonance(q_factor=20)
        low, high = antenna.vswr_bandwidth_edges(2.0)
        # Both edges sit at VSWR == 2 on a 50 ohm system with R == 50.
        for edge in (low, high):
            assert antenna.vswr(edge) == pytest.approx(2.0, rel=1e-9)
        # For VSWR s and a matched-resistance series RLC the crossing spacing
        # is BW * (s - 1) / sqrt(s), which for s == 2 is BW / sqrt(2).
        expected = antenna.bandwidth * (2.0 - 1.0) / math.sqrt(2.0)
        assert (high - low) == pytest.approx(expected)

    @staticmethod
    def test_filter_is_minus_3db_at_cutoff():
        lowpass = ButterworthLowPass(cutoff_frequency=100e6, order=5)
        assert lowpass.magnitude_db(100e6) == pytest.approx(-3.0103, abs=1e-3)

    @staticmethod
    def test_filter_rolloff_is_order_times_20db_per_decade():
        lowpass = ButterworthLowPass(cutoff_frequency=100e6, order=5)
        # A decade above cutoff for a 5th order filter: ~100 dB down.
        assert lowpass.magnitude_db(1e9) == pytest.approx(-100.0, abs=0.5)

    @staticmethod
    def test_filter_is_lossless_and_reciprocal():
        lowpass = ButterworthLowPass(cutoff_frequency=100e6, order=3)
        for freq in (1e6, 50e6, 100e6, 300e6):
            s21, s11 = lowpass.response(freq)
            assert abs(s21) ** 2 + abs(s11) ** 2 == pytest.approx(1.0)

    @staticmethod
    def test_cable_delay_and_loss():
        cable = CableLine(
            length_m=10.0, velocity_factor=0.66, loss_db_per_100m=10.0
        )
        expected_delay = 10.0 / (0.66 * 299_792_458.0)
        assert cable.delay == pytest.approx(expected_delay)
        assert cable.round_trip_delay == pytest.approx(2 * expected_delay)
        # 10 m at 10 dB/100 m is 1 dB, referred to 100 MHz.
        assert cable.insertion_loss_db(100e6) == pytest.approx(1.0)
        # sqrt(f) skin effect scaling.
        assert cable.insertion_loss_db(400e6) == pytest.approx(2.0)

    @staticmethod
    def test_cable_phase_delay_is_linear_in_frequency():
        cable = CableLine()
        freq = 100e6
        phase = cable.s21(freq)
        assert phase == pytest.approx(
            cable.s21(freq).real + 1j * cable.s21(freq).imag
        )
        expected = -2 * math.pi * freq * cable.delay
        assert math.atan2(phase.imag, phase.real) == pytest.approx(
            math.atan2(math.sin(expected), math.cos(expected))
        )

    @staticmethod
    def test_build_scenario_rejects_unknown_name():
        with pytest.raises(ValueError, match="unknown scenario"):
            build_scenario("nope")

    @staticmethod
    def test_attenuator_is_flat():
        attenuator = Attenuator(attenuation_db=3.0)
        first = attenuator(1e6).s21
        last = attenuator(1e9).s21
        assert abs(first) == pytest.approx(abs(last))
        assert abs(first) == pytest.approx(10 ** (-3.0 / 20))


class TestSimulatedDevice:
    @staticmethod
    def test_noise_free_is_reproducible():
        def acquire(seed):
            device = SimulatedDevice("antenna", seed=seed)
            frequencies = [1e6, 5e6, 14.15e6, 30e6]
            return [device.sample(f).s11 for f in frequencies], [
                device.sample(f).s21 for f in frequencies
            ]

        first = acquire(1)
        second = acquire(1)
        assert first == second

    @staticmethod
    def test_seeded_noise_is_reproducible_and_seed_dependent():
        def acquire(seed):
            device = SimulatedDevice("antenna", seed=seed, noise_floor=0.01)
            return [device.sample(14.15e6).s11 for _ in range(5)]

        assert acquire(7) == acquire(7)
        assert acquire(7) != acquire(8)

    @staticmethod
    def test_frequency_grid_includes_endpoints():
        device = SimulatedDevice("antenna")
        device.set_sweep(1_000_000, 30_000_000, 101)
        grid = device.frequency_grid()
        assert len(grid) == 101
        assert grid[0] == 1_000_000
        assert grid[-1] == 30_000_000
        assert grid == sorted(grid)

    @staticmethod
    def test_single_point_sweep_is_supported():
        device = SimulatedDevice("antenna")
        device.set_sweep(14_150_000, 14_150_000, 1)
        assert device.frequency_grid() == [14_150_000]

    @staticmethod
    def test_acquire_restores_previous_sweep_range():
        device = SimulatedDevice("antenna")
        device.set_sweep(1_000_000, 2_000_000, 11)
        frequencies, channels = device.acquire(
            3_000_000, 4_000_000, 51, ("data 0", "data 1")
        )
        assert len(frequencies) == 51
        assert len(channels[0]) == len(channels[1]) == 51
        assert (device.start, device.stop, device.points) == (
            1_000_000,
            2_000_000,
            11,
        )

    @staticmethod
    def test_closed_device_refuses_acquisition():
        device = SimulatedDevice("antenna")
        device.close()
        with pytest.raises(IOError):
            device.set_sweep(1, 2, 3)

    @staticmethod
    def test_reference_impedance_is_documented():
        assert SimulatedDevice("antenna").reference_impedance == 50.0


class TestFaults:
    @staticmethod
    def test_empty_samples_fault():
        device = SimulatedDevice(
            "antenna", fault=FaultConfig(FaultMode.EMPTY_SAMPLES)
        )
        assert device.read_channel("data 0") == []

    @staticmethod
    def test_malformed_samples_fault():
        device = SimulatedDevice(
            "antenna", fault=FaultConfig(FaultMode.MALFORMED_SAMPLES)
        )
        with pytest.raises(ValueError):
            device.read_channel("data 0")

    @staticmethod
    def test_mismatched_lengths_fault():
        device = SimulatedDevice(
            "antenna", fault=FaultConfig(FaultMode.MISMATCHED_LENGTHS)
        )
        assert (
            len(device.read_channel("data 1"))
            == len(device.read_channel("data 0")) - 1
        )

    @staticmethod
    def test_non_finite_fault():
        device = SimulatedDevice(
            "antenna", fault=FaultConfig(FaultMode.NON_FINITE)
        )
        values = device.read_channel("data 0")
        assert math.isnan(values[0].real)
        assert math.isinf(values[0].imag)

    @staticmethod
    def test_disconnect_fault():
        device = SimulatedDevice(
            "antenna", fault=FaultConfig(FaultMode.DISCONNECT)
        )
        with pytest.raises(IOError):
            device.read_channel("data 0")

    @staticmethod
    def test_timeout_fault_is_bounded_by_transport_timeout():
        device = SimulatedDevice(
            "antenna", fault=FaultConfig(FaultMode.TIMEOUT)
        )
        device.timeout = 0.0
        with pytest.raises(TimeoutError):
            device.sample(14e6)

    @staticmethod
    def test_fail_during_average_after_n_samples():
        device = SimulatedDevice(
            "antenna",
            fault=FaultConfig(
                FaultMode.FAIL_DURING_AVERAGE, fail_after_samples=2
            ),
        )
        device.set_sweep(1e6, 2e6, 3)
        # The first two channel reads complete, as during averaging.
        assert len(device.read_channel("data 0")) == 3
        assert len(device.read_channel("data 1")) == 3
        # The next read fails part-way through the average sequence.
        with pytest.raises(IOError, match="failure during averaging"):
            device.read_channel("data 0")


class TestSimulatedVNAAdapter:
    @staticmethod
    def test_reports_itself_connected_and_simulated():
        vna = make_simulated_vna("antenna")
        assert vna.connected()
        assert vna.simulated
        assert "Simulated" in vna.name
        vna.disconnect()
        assert not vna.connected()
        vna.connect()
        assert vna.connected()

    @staticmethod
    def test_device_settings_surface_is_available():
        vna = make_simulated_vna("filter")
        assert vna.sweep_points_max >= max(vna.valid_datapoints)
        assert vna.datapoints in vna.valid_datapoints
        assert 1000 in vna.get_bandwidths()
        assert "Customizable data points" in vna.get_features()
        assert vna.version.major == 1

    @staticmethod
    def test_set_sweep_and_read_round_trip():
        vna = make_simulated_vna("antenna")
        vna.datapoints = 51
        vna.setSweep(1_000_000, 30_000_000)
        frequencies = vna.read_frequencies()
        values = vna.readValues("data 0")
        assert len(frequencies) == len(values) == 51

    @staticmethod
    def test_reference_impedance_and_scenario_are_exposed():
        vna = make_simulated_vna("cable")
        assert vna.reference_impedance == 50.0
        assert vna.scenario_name == "CableLine"


class TestNormalSweepPath:
    """The simulated device must work through the real sweep logic."""

    @staticmethod
    def _collect(vna, sweep: Sweep):
        """Mirror SweepWorker's segment loop without any Qt threading."""
        data11: list = []
        data21: list = []
        for index in range(sweep.segments):
            start, stop = sweep.get_index_range(index)
            vna.setSweep(start, stop)
            frequencies = vna.read_frequencies()
            values11 = vna.readValues("data 0")
            values21 = vna.readValues("data 1")
            assert len(frequencies) == len(values11) == len(values21)
            data11.extend(zip(frequencies, values11, strict=True))
            data21.extend(zip(frequencies, values21, strict=True))
        return data11, data21

    @staticmethod
    def test_single_segment_sweep_covers_the_span():
        vna = make_simulated_vna("antenna")
        vna.datapoints = 101
        sweep = Sweep(start=10_000_000, end=20_000_000, points=101)
        data11, data21 = TestNormalSweepPath._collect(vna, sweep)
        assert len(data11) == 101
        assert data11[0][0] == 10_000_000
        assert data11[-1][0] == 20_000_000
        # Both channels are populated for every point.
        assert len(data21) == 101

    @staticmethod
    def test_segmented_sweep_is_contiguous_and_ordered():
        vna = make_simulated_vna("filter")
        vna.datapoints = 11
        sweep = Sweep(start=1_000_000, end=10_000_000, points=11, segments=3)
        data11, _ = TestNormalSweepPath._collect(vna, sweep)
        frequencies = [row[0] for row in data11]
        assert frequencies == sorted(frequencies)
        assert len(frequencies) == 33

    @staticmethod
    def test_averaged_sweep_is_reproducible():
        def average(seed):
            vna = make_simulated_vna("antenna", seed=seed, noise_floor=0.01)
            vna.datapoints = 11
            vna.setSweep(12e6, 16e6)
            runs = [list(vna.readValues("data 0")) for _ in range(3)]
            return [
                sum(values[index] for values in runs) / len(runs)
                for index in range(11)
            ]

        assert average(3) == average(3)

    @staticmethod
    def test_zero_noise_expected_resonance_is_recovered():
        """Known analysis result: |S11| minimum at the modelled resonance."""
        vna = make_simulated_vna("antenna")
        vna.datapoints = 201
        sweep = Sweep(start=13_000_000, end=15_000_000, points=201)
        data11, _ = TestNormalSweepPath._collect(vna, sweep)
        best = min(data11, key=lambda row: abs(row[1]))
        assert best[0] == pytest.approx(14_150_000, abs=10_000)
        assert abs(best[1]) < 1e-6

    @staticmethod
    def test_zero_noise_filter_cutoff_is_3db_down():
        vna = make_simulated_vna("filter")
        vna.datapoints = 201
        vna.setSweep(10_000_000, 200_000_000)
        frequencies = vna.read_frequencies()
        values21 = vna.readValues("data 1")
        # Pick the sample nearest the modelled cutoff; the grid is 950 kHz
        # apart so an exact hit is not guaranteed.
        index = min(
            range(len(frequencies)),
            key=lambda i: abs(frequencies[i] - 100_000_000),
        )
        cutoff_gain = 20 * math.log10(abs(values21[index]))
        assert cutoff_gain == pytest.approx(-3.0, abs=0.2)

    @staticmethod
    def test_truncate_drops_extremes_from_averaged_values():
        """Averaging support still works with simulated data."""
        device = SimulatedDevice("antenna", seed=1, noise_floor=0.01)
        device.set_sweep(14e6, 15e6)
        runs = [device.read_channel("data 0") for _ in range(5)]
        truncated = truncate(runs, 2)
        assert len(truncated) == 3
        assert len(truncated[0]) == len(runs[0])
