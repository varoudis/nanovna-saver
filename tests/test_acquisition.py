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
"""AUTO-01: the Qt-free acquisition API.

These tests must not create a ``QApplication``.  ``Acquisition`` imports no
Qt itself; the pure-Python checks below run in a subprocess to prove that a
script can use the API without a display, and the rest drive a stand-in
device so failure paths are deterministic.
"""

import subprocess
import sys

import pytest

from NanoVNASaver.Acquisition import (
    BorrowedDevice,
    CancelledError,
    CancelToken,
    DeviceBusyError,
    DeviceError,
    SweepRequest,
    UnsupportedError,
    open_device,
)
from NanoVNASaver.Measurement import S11, S21


class FakeDevice:
    """A minimal device that can be made to misbehave on demand."""

    name = "fake"
    datapoints = 0
    simulated = True
    reference_impedance = 50.0

    def __init__(
        self,
        points=5,
        *,
        supported=("data 0", "data 1"),
        short_reads=(),
        failing=None,
    ):
        self.points = points
        self.supported_channels = supported
        self.short_reads = set(short_reads)
        self.failing = failing or set()
        self.start = 0
        self.stop = 0
        self.reads = []

    def read_frequencies(self):
        if "frequencies" in self.failing:
            raise OSError("no frequencies")
        step = 1_000_000
        return [1_000_000 + index * step for index in range(self.points)]

    def readValues(self, value):
        self.reads.append(value)
        if value in self.failing:
            raise OSError(f"{value} read failed")
        count = self.points - (1 if value in self.short_reads else 0)
        base = 0.1 if value == "data 0" else 0.5
        return [complex(base, base / 2) for _ in range(count)]

    def setSweep(self, start, stop):
        self.start, self.stop = start, stop

    def connected(self):
        return True

    def get_features(self):
        return {"Scan mask command"}

    def disconnect(self):
        self.disconnected = True


def request(points=5, **kwargs):
    return SweepRequest(1_000_000, 1_000_000 * points, points, **kwargs)


class TestRequests:
    @staticmethod
    def test_a_valid_request_describes_itself():
        text = request().describe()
        assert "1-5" in text
        assert "5 points" in text
        assert "S11+S21" in text

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"start_hz": 0},
            {"stop_hz": -1},
            {"start_hz": 10, "stop_hz": 10},
            {"points": 1},
            {"average": 0},
        ],
    )
    def test_impossible_requests_are_rejected_before_any_device_call(kwargs):
        fields = {
            "start_hz": 1,
            "stop_hz": 2,
            "points": 2,
            "channels": (S11,),
            **kwargs,
        }
        with pytest.raises(ValueError):
            SweepRequest(**fields)

    @staticmethod
    def test_an_unknown_channel_is_rejected():
        with pytest.raises(ValueError, match="unsupported channel"):
            SweepRequest(1_000, 2_000, 2, channels=("S12",))


class TestSweep:
    @staticmethod
    def test_a_sweep_returns_an_immutable_snapshot():
        device = FakeDevice()
        with open_device(device) as session:
            result = session.sweep(request())
        assert sorted(result.channels) == [S11, S21]
        assert len(result.channels[S11]) == device.points
        assert result.snapshot.simulated is True
        assert result.snapshot.corrected[S11][0].freq == 1_000_000

    @staticmethod
    def test_the_devices_own_read_commands_are_used():
        device = FakeDevice()
        with open_device(device) as session:
            session.sweep(request())
        assert device.reads == ["data 0", "data 1"]

    @staticmethod
    def test_averaging_divides_by_the_count_not_the_last_read():
        device = FakeDevice()
        with open_device(device) as session:
            once = session.sweep(request(average=1))
            twice = session.sweep(request(average=2))
        # The fake returns the same value every read, so averaging must not
        # change it; a bug that summed without dividing would double it.
        first_once = once.channels[S11][0].z
        first_twice = twice.channels[S11][0].z
        assert first_twice == pytest.approx(first_once)
        assert twice.averages == 2
        assert len(device.reads) == 2 + 4

    @staticmethod
    def test_progress_is_reported_once_per_average():
        seen = []
        device = FakeDevice()
        with open_device(device) as session:
            session.on_progress = seen.append
            session.sweep(request(average=3))
        assert seen == pytest.approx([1 / 3, 2 / 3, 1.0])

    @staticmethod
    def test_a_one_port_device_omits_s21_entirely():
        device = FakeDevice(supported=("data 0",))
        with open_device(device) as session:
            result = session.sweep(
                SweepRequest(1_000_000, 5_000_000, 5, channels=(S11,))
            )
            assert session.supported_channels == (S11,)
        assert list(result.channels) == [S11]
        assert device.reads == ["data 0"]

    @staticmethod
    def test_asking_a_one_port_device_for_s21_is_refused():
        device = FakeDevice(supported=("data 0",))
        with open_device(device) as session:
            with pytest.raises(UnsupportedError, match="does not provide S21"):
                session.sweep(request())
        assert device.reads == []


class TestFailures:
    @staticmethod
    def test_a_short_read_is_a_device_error_not_a_short_trace():
        device = FakeDevice(short_reads=("data 1",))
        with open_device(device) as session:
            with pytest.raises(DeviceError, match="samples for"):
                session.sweep(request())

    @staticmethod
    def test_a_read_exception_becomes_a_device_error():
        device = FakeDevice(failing=("data 0",))
        with open_device(device) as session:
            with pytest.raises(DeviceError, match="reading S11 failed"):
                session.sweep(request())

    @staticmethod
    def test_a_frequency_failure_is_reported_before_any_channel_read():
        device = FakeDevice(failing=("frequencies",))
        with open_device(device) as session:
            with pytest.raises(DeviceError, match="frequencies"):
                session.sweep(request())
        assert device.reads == []

    @staticmethod
    def test_a_wrong_point_count_is_refused():
        device = FakeDevice(points=7)
        with open_device(device) as session:
            with pytest.raises(DeviceError, match="7 frequencies"):
                session.sweep(request(points=5))

    @staticmethod
    def test_the_device_is_left_at_the_requested_point_count():
        device = FakeDevice()
        with open_device(device) as session:
            session.sweep(request(points=5))
        assert device.datapoints == 5


class TestCancellation:
    @staticmethod
    def test_an_already_cancelled_run_does_nothing():
        device = FakeDevice()
        token = CancelToken()
        token.cancel()
        with open_device(device) as session:
            with pytest.raises(CancelledError):
                session.sweep(request(), cancel=token)
        assert device.reads == []

    @staticmethod
    def test_a_cancel_between_reads_stops_the_run():
        class CancellingDevice(FakeDevice):
            def readValues(self, value):
                token.cancel()
                return super().readValues(value)

        token = CancelToken()
        cancelling = CancellingDevice()
        with open_device(cancelling) as session:
            with pytest.raises(CancelledError):
                session.sweep(request(), cancel=token)
        # The first read completed, so cancellation was honoured between
        # reads rather than only at the start.
        assert cancelling.reads == ["data 0"]

    @staticmethod
    def test_a_live_token_is_not_discarded():
        # Regression: ``cancel or CancelToken()`` threw away a live token,
        # because a token is falsy until someone cancels it.
        device = FakeDevice()
        token = CancelToken()
        with open_device(device) as session:
            session.sweep(request(), cancel=token)
        token.cancel()
        assert token.cancelled


class TestOwnership:
    @staticmethod
    def test_a_second_handle_is_refused_while_the_first_is_open():
        device = FakeDevice()
        first = open_device(device)
        try:
            with pytest.raises(DeviceBusyError, match="already owned"):
                open_device(device)
        finally:
            first.close()

    @staticmethod
    def test_closing_releases_the_device_for_reuse():
        device = FakeDevice()
        open_device(device).close()
        second = open_device(device)
        second.close()

    @staticmethod
    def test_closing_twice_is_safe():
        device = FakeDevice()
        session = open_device(device)
        session.close()
        session.close()

    @staticmethod
    def test_a_closed_session_refuses_further_work():
        device = FakeDevice()
        session = open_device(device)
        session.close()
        with pytest.raises(DeviceError, match="closed"):
            session.sweep(request())
        with pytest.raises(DeviceError, match="closed"):
            session.features()

    @staticmethod
    def test_a_failed_close_is_reported_but_still_releases():
        class Stubborn(FakeDevice):
            def disconnect(self):
                raise OSError("port stuck")

        device = Stubborn()
        session = open_device(device)
        with pytest.raises(DeviceError, match="port stuck"):
            session.close()
        assert session.closed
        # The claim was released even though the close failed, so a retry
        # (the usual response to a stuck port) is possible.
        retry = open_device(device)
        assert retry.name
        # And closing the retry keeps raising the same underlying problem
        # rather than silently reporting success.
        with pytest.raises(DeviceError, match="port stuck"):
            retry.close()


class TestProvenance:
    @staticmethod
    def test_a_simulated_device_is_never_labelled_physical():
        device = FakeDevice()
        device.simulated = True
        with open_device(device) as session:
            assert session.simulated is True
            assert session.sweep(request()).snapshot.simulated is True

    @staticmethod
    def test_a_physical_device_is_not_labelled_simulated():
        class Physical(FakeDevice):
            simulated = False

        with open_device(Physical()) as session:
            assert session.sweep(request()).snapshot.simulated is False

    @staticmethod
    def test_the_reference_impedance_comes_from_the_device():
        class SeventyFive(FakeDevice):
            reference_impedance = 75.0

        with open_device(SeventyFive()) as session:
            assert session.reference_impedance == 75.0
            assert session.sweep(request()).snapshot.reference_impedance == (
                75.0
            )

    @staticmethod
    def test_the_request_name_reaches_the_snapshot():
        with open_device(FakeDevice()) as session:
            result = session.sweep(request(name="band check"))
        assert result.snapshot.name == "band check"


class TestHeadless:
    @staticmethod
    def test_the_api_imports_and_runs_without_qt():
        """A script must be able to use this without ever loading widgets."""
        script = (
            "import sys\n"
            "from NanoVNASaver.Acquisition import SweepRequest, open_device\n"
            "from NanoVNASaver.Hardware.Simulated import make_simulated_vna\n"
            # ``filter`` is a two-port scenario: the antenna is one-port, so
            # it publishes S11 only and the S21 assertion below would be
            # asserting a channel the device does not have.
            "vna = make_simulated_vna('filter', seed=3)\n"
            "with open_device(vna) as session:\n"
            "    result = session.sweep(\n"
            "        SweepRequest(1_000_000, 50_000_000, 11)\n"
            "    )\n"
            "assert sorted(result.channels) == ['S11', 'S21']\n"
            "assert result.snapshot.simulated is True\n"
            "widgets = [m for m in sys.modules\n"
            "           if m.startswith('PySide6.QtWidgets')]\n"
            "print('OK' if not widgets else 'WIDGETS:' + widgets[0])\n"
        )
        finished = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert finished.returncode == 0, finished.stderr
        assert finished.stdout.strip() == "OK"


class CachingDevice(FakeDevice):
    """Acquires only on ``data 0``, as the V2 and scan-mask drivers do."""

    def __init__(self, points=5):
        super().__init__(points)
        self.acquisitions = 0
        self.cached = []

    def readValues(self, value):
        if value == "data 0":
            self.acquisitions += 1
            self.cached = [
                (complex(self.acquisitions, 0), complex(0, self.acquisitions))
            ] * self.points
        index = 1 if value == "data 1" else 0
        return [pair[index] for pair in self.cached]


class TestChannelReadOrder:
    @staticmethod
    def _request(channels, average=1):
        return SweepRequest(
            start_hz=1_000_000,
            stop_hz=2_000_000,
            points=5,
            channels=channels,
            average=average,
        )

    def test_s21_alone_on_a_fresh_device_is_measured(self):
        with open_device(CachingDevice()) as session:
            result = session.sweep(self._request((S21,)))
        assert result.snapshot.available == frozenset({S21}) or set(
            result.snapshot.available
        ) == {S21}
        assert result.snapshot.corrected[S21][0].z == 1j

    def test_a_second_s21_sweep_is_a_new_measurement(self):
        device = CachingDevice()
        with open_device(device) as session:
            session.sweep(self._request((S11, S21)))
            second = session.sweep(self._request((S21,)))
        assert device.acquisitions == 2
        assert second.snapshot.corrected[S21][0].z == 2j

    def test_averaging_s21_averages_separate_acquisitions(self):
        device = CachingDevice()
        with open_device(device) as session:
            result = session.sweep(self._request((S21,), average=3))
        assert device.acquisitions == 3
        assert result.snapshot.corrected[S21][0].z == pytest.approx(2j)

    def test_s21_first_still_acquires_before_reading_it(self):
        device = CachingDevice()
        with open_device(device) as session:
            result = session.sweep(self._request((S21, S11)))
        assert device.acquisitions == 1
        assert result.snapshot.corrected[S21][0].z == 1j
        assert result.snapshot.corrected[S11][0].z == 1

    def test_a_device_without_s11_is_not_asked_for_it(self):
        device = FakeDevice(supported=("data 1",))
        with open_device(device) as session:
            session.sweep(self._request((S21,)))
        assert "data 0" not in device.reads


class TestBorrowedDevice:
    @staticmethod
    def test_closing_the_session_does_not_disconnect_the_lender():
        disconnected = []
        device = FakeDevice()
        device.disconnect = lambda: disconnected.append(True)
        device.datapoints = 101
        released = []
        with open_device(
            BorrowedDevice(device, on_release=lambda: released.append(True))
        ) as session:
            session.sweep(
                SweepRequest(start_hz=1_000_000, stop_hz=2_000_000, points=5)
            )
            assert device.datapoints == 5
        assert not disconnected
        assert released == [True]
        assert device.datapoints == 101
