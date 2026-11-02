"""PERF-02: an optimised read must reproduce the trusted read exactly.

The simulator provides the physics; the fake serial interface only speaks
the ASCII protocol the driver expects, decoding ``scan ... 0b110`` from the
device's own samples.  Parity is therefore a property of the read path, not
of a hand-written fixture: both paths see the same deterministic device and
must agree bit for bit.
"""

from __future__ import annotations

import pytest

from NanoVNASaver.Hardware.CombinedScan import PATH_TRUSTED
from NanoVNASaver.Hardware.NanoVNA import NanoVNA
from NanoVNASaver.Hardware.ScanCapability import ScanPreferences
from NanoVNASaver.Hardware.Simulated import SimulatedDevice

from .scan_fakes import FakeInterface, help_lines

START = 10_000_000
STOP = 20_000_000
POINTS = 11
TOLERANCE = 1e-12
MASKED = "0b110"


def _format(value: complex) -> str:
    return f"{value.real!r} {value.imag!r}"


class SimulatedProtocol:
    """Answers the ASCII commands by reading a real ``SimulatedDevice``."""

    def __init__(self, device: SimulatedDevice):
        self.device = device
        self.masked_reads = 0

    def __call__(self, command: str):
        words = command.split()
        if not words:
            return None
        head = words[0]
        if head == "version":
            return ["0.6.0"]
        if head == "help":
            return [help_lines()]
        if head == "frequencies":
            return [str(freq) for freq in self.device.read_frequencies()]
        if head == "data" and len(words) == 2:
            channel = f"data {words[1]}"
            return [
                _format(value) for value in self.device.read_channel(channel)
            ]
        if head == "scan" and len(words) >= 4:
            self.device.set_sweep(int(words[1]), int(words[2]), int(words[3]))
            if len(words) == 5 and words[4] == MASKED:
                self.masked_reads += 1
                return [
                    f"{_format(s11)} {_format(s21)}"
                    for s11, s21 in zip(
                        self.device.read_channel("data 0"),
                        self.device.read_channel("data 1"),
                        strict=True,
                    )
                ]
            return []
        return None


def _vna(scenario: str) -> tuple[NanoVNA, SimulatedProtocol]:
    device = SimulatedDevice(scenario, seed=11, noise_floor=0.0)
    protocol = SimulatedProtocol(device)
    vna = NanoVNA(FakeInterface(responder=protocol))
    return vna, protocol


def _sweep(vna: NanoVNA, preferences=None) -> list[list[complex]]:
    vna.datapoints = POINTS
    vna.setSweep(START, STOP)
    assert len(vna.read_frequencies()) == POINTS
    return [
        list(vna.read_values_optimised(command, preferences))
        for command in ("data 0", "data 1")
    ]


@pytest.fixture(params=["filter", "cable"])
def scenarios(request):
    return request.param


class TestParity:
    @staticmethod
    def test_the_combined_read_equals_the_trusted_read(scenarios):
        trusted_vna, _ = _vna(scenarios)
        combined_vna, protocol = _vna(scenarios)

        trusted = _sweep(trusted_vna, ScanPreferences())
        combined = _sweep(combined_vna, ScanPreferences(combined_text=True))

        assert protocol.masked_reads == 1, "one masked scan per sweep"
        assert len(combined[0]) == POINTS
        assert len(combined[1]) == POINTS
        for channel in range(2):
            for index, (left, right) in enumerate(
                zip(trusted[channel], combined[channel], strict=True)
            ):
                assert abs(left - right) <= TOLERANCE, (channel, index)

    @staticmethod
    def test_both_paths_match_the_devices_own_samples(scenarios):
        vna, _ = _vna(scenarios)
        combined = _sweep(vna, ScanPreferences(combined_text=True))

        device = SimulatedDevice(scenarios, seed=11, noise_floor=0.0)
        _, (s11, s21) = device.acquire(
            START, STOP, POINTS, ("data 0", "data 1")
        )

        assert combined[0] == pytest.approx(s11, abs=TOLERANCE)
        assert combined[1] == pytest.approx(s21, abs=TOLERANCE)

    @staticmethod
    def test_the_read_path_is_reported(scenarios):
        vna, _ = _vna(scenarios)
        _sweep(vna, ScanPreferences(combined_text=True))
        assert vna.last_read_path == "combined"

        trusted_vna, _ = _vna(scenarios)
        _sweep(trusted_vna, ScanPreferences())
        assert trusted_vna.last_read_path == PATH_TRUSTED


class TestTheFlagReallyGates:
    @staticmethod
    def test_the_trusted_path_runs_when_the_flag_is_off():
        vna, protocol = _vna("filter")

        _sweep(vna, ScanPreferences())

        assert protocol.masked_reads == 0
        assert not any(MASKED in command for command in vna.serial.writes)
        assert vna.serial.commands("data 0")
        assert vna.serial.commands("data 1")

    @staticmethod
    def test_a_one_port_style_failure_falls_back_to_the_trusted_read():
        vna, protocol = _vna("filter")

        def broken(command: str):
            if command.endswith(MASKED):
                return ["0.5 0"]  # two fields: not a two-channel reply
            return protocol(command)

        vna.serial.responder = broken
        values = _sweep(vna, ScanPreferences(combined_text=True))

        assert vna.last_read_path == PATH_TRUSTED
        assert all(len(channel) == POINTS for channel in values)
        assert vna.serial.commands("data 0")

    @staticmethod
    def test_a_short_masked_reply_does_not_become_a_measurement():
        vna, protocol = _vna("filter")

        def truncated(command: str):
            if command.endswith(MASKED):
                return protocol(command)[:-1]
            return protocol(command)

        vna.serial.responder = truncated
        values = _sweep(vna, ScanPreferences(combined_text=True))

        assert vna.last_read_path == PATH_TRUSTED
        assert all(len(channel) == POINTS for channel in values)
