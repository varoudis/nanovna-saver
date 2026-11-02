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
"""NEXT-02: the simulator's nominal latency is a real, documented cost.

``latency`` is charged once per *data read* (one ``read_channel`` call),
never per sample and never per sweep.  Fault modes that already delay a
read (``SLOW``, ``TIMEOUT``) replace it instead of stacking on top of it,
and the wait never draws from the seeded random stream.

Timing assertions only use lower bounds and deliberately generous upper
bounds, so a loaded machine cannot turn them into flakes.
"""

import math
import threading
import time

import pytest

from NanoVNASaver.Hardware.Simulated import (
    DELAY_SLICE,
    TIMED_FAULTS,
    FaultConfig,
    FaultMode,
    SimulatedDevice,
    SimulatedTransport,
    SimulatedVNA,
    make_simulated_vna,
)


def _timed(call, repeats=1):
    """Return the wall-clock seconds spent in ``repeats`` calls of ``call``."""
    start = time.monotonic()
    for _ in range(repeats):
        call()
    return time.monotonic() - start


def _device(latency=0.0, points=5, scenario="filter", **kwargs):
    # ``filter`` is two-port, so a test may read either channel.  The
    # antenna scenario is deliberately one-port.
    device = SimulatedDevice(scenario, latency=latency, **kwargs)
    device.set_sweep(1_000_000, 2_000_000, points)
    return device


def _raise_timeout(device):
    with pytest.raises(TimeoutError):
        device.sample(14_000_000)


class TestLatencyValidation:
    """A bad demo setting fails loudly instead of hanging or spinning."""

    @staticmethod
    @pytest.mark.parametrize(
        "bad",
        [-1.0, -1e-9, math.nan, math.inf, -math.inf],
        ids=["negative", "tiny_negative", "nan", "inf", "negative_inf"],
    )
    def test_invalid_latency_is_rejected(bad):
        with pytest.raises(ValueError, match="latency"):
            SimulatedDevice("antenna", latency=bad)

    @staticmethod
    def test_invalid_latency_is_rejected_by_every_entry_point():
        with pytest.raises(ValueError, match="latency"):
            SimulatedVNA(scenario="antenna", latency=-0.5)
        with pytest.raises(ValueError, match="latency"):
            make_simulated_vna("antenna", latency=math.nan)
        with pytest.raises(ValueError, match="latency"):
            SimulatedTransport(latency=-1.0)

    @staticmethod
    def test_latency_assignment_is_validated_too():
        device = _device()
        with pytest.raises(ValueError, match="latency"):
            device.latency = -1.0
        # The rejected assignment leaves the working value untouched.
        assert device.latency == 0.0
        device.latency = 0.25
        assert device.latency == pytest.approx(0.25)

    @staticmethod
    def test_non_numeric_latency_is_rejected():
        with pytest.raises(ValueError, match="latency"):
            SimulatedDevice("antenna", latency="fast")


class TestNominalLatency:
    """One data read costs one latency, whatever the sweep size."""

    @staticmethod
    def test_two_reads_pay_two_latencies():
        latency = 0.02
        device = _device(latency=latency)
        elapsed = _timed(lambda: device.read_channel("data 0"), repeats=2)
        assert elapsed >= 2 * latency * 0.9
        assert elapsed < 2 * latency + 1.0

    @staticmethod
    def test_adapter_read_values_pays_the_same_latency():
        latency = 0.02
        vna = make_simulated_vna("filter", latency=latency)
        vna.setSweep(1_000_000, 2_000_000)
        elapsed = _timed(lambda: vna.readValues("data 1"), repeats=2)
        assert elapsed >= 2 * latency * 0.9
        assert elapsed < 2 * latency + 1.0

    @staticmethod
    def test_latency_is_per_read_and_not_per_sample():
        """A 5-point and a 101-point read cost about the same."""

        latency = 0.02
        few = _timed(
            lambda: _device(latency=latency, points=5).read_channel("data 0")
        )
        many = _timed(
            lambda: _device(latency=latency, points=101).read_channel("data 0")
        )
        for elapsed in (few, many):
            assert elapsed >= latency * 0.9
            assert elapsed < latency + 1.0
        # Per-sample charging would be ~20x; allow a wide margin for noise.
        assert many < few + 0.5

    @staticmethod
    def test_zero_latency_keeps_reads_immediate():
        device = _device(latency=0.0, points=101)
        elapsed = _timed(lambda: device.read_channel("data 0"), repeats=3)
        assert elapsed < 0.5


class TestLatencyDoesNotStack:
    """Faults that own the timing of a read replace the nominal latency."""

    @staticmethod
    def test_slow_fault_is_not_extended_by_latency():
        latency = 0.25
        slow_delay = 0.01
        fault = FaultConfig(mode=FaultMode.SLOW, slow_delay=slow_delay)

        slow_only = _timed(
            lambda: _device(points=5, fault=fault).read_channel("data 0")
        )
        slow_and_latency = _timed(
            lambda: _device(
                latency=latency, points=5, fault=fault
            ).read_channel("data 0")
        )

        # SLOW alone already waits once per sample.
        assert slow_only >= 5 * slow_delay * 0.9
        # Stacking would cost slow_only + latency; it does not.
        assert slow_and_latency < slow_only + latency / 2

    @staticmethod
    def test_timeout_fault_does_not_wait_for_latency_first():
        device = _device(latency=0.5, fault=FaultConfig(mode=FaultMode.TIMEOUT))
        device.timeout = 0.0
        elapsed = _timed(lambda: _raise_timeout(device))
        assert elapsed < 0.25

    @staticmethod
    def test_timed_faults_are_declared():
        assert TIMED_FAULTS == {FaultMode.SLOW, FaultMode.TIMEOUT}


class TestLatencyKeepsSeededData:
    """Timing must never consume from the seeded random stream."""

    @staticmethod
    def test_same_seed_different_latency_gives_identical_samples():
        fast = _device(latency=0.0, points=11, seed=4242, noise_floor=0.01)
        slow = _device(latency=0.001, points=11, seed=4242, noise_floor=0.01)

        for channel in ("data 0", "data 1"):
            assert fast.read_channel(channel) == slow.read_channel(channel)

    @staticmethod
    def test_noise_floor_values_are_reproducible_across_latencies():
        fast = _device(latency=0.0, points=7, seed=99, noise_floor=0.02)
        slow = _device(latency=0.002, points=7, seed=99, noise_floor=0.02)
        assert fast.read_channel("data 0") == slow.read_channel("data 0")
        # A different seed must still change the noise.
        other = _device(latency=0.0, points=7, seed=100, noise_floor=0.02)
        assert other.read_channel("data 0") != fast.read_channel("data 0")


class TestLatencyIsInterruptible:
    """A long nominal delay never pins the device closed for its duration."""

    @staticmethod
    def test_close_interrupts_a_pending_latency():
        device = _device(latency=5.0, points=5)
        outcome = {}

        def read():
            try:
                device.read_channel("data 0")
            except OSError as err:  # IOError is an alias of OSError
                outcome["error"] = str(err)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        time.sleep(0.05)
        device.close()
        reader.join(timeout=1.0)

        assert not reader.is_alive()
        assert outcome.get("error") == "simulated transport disconnected"

    @staticmethod
    def test_delay_is_delivered_in_slices():
        """The slice constant keeps cancellation latency small."""
        assert 0.0 < DELAY_SLICE <= 0.05
        device = _device(latency=0.05)
        elapsed = _timed(lambda: device.read_channel("data 0"))
        assert elapsed >= 0.04
