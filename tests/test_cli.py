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
"""AUTO-02: the command line, run as a real subprocess.

Exit codes are the contract, so they are asserted through ``subprocess``
rather than by calling ``main()`` — that is the only way to prove what an
unattended script actually observes.
"""

import json
import os
import subprocess
import sys

import pytest

from NanoVNASaver.cli import (
    EXIT_FAILED,
    EXIT_INVALID,
    EXIT_OK,
    EXIT_UNAVAILABLE,
    parse_frequency,
)

LIMIT_VSWR = {
    "format": "nanovna-saver-limits",
    "version": 2,
    "name": "resonance window",
    "rules": [
        {
            "id": "match",
            "label": "Antenna match",
            "metric": "s11_vswr",
            "segments": [
                {
                    "start_hz": 14_130_000,
                    "stop_hz": 14_170_000,
                    "maximum": 2.0,
                }
            ],
        }
    ],
}

LIMIT_IMPOSSIBLE = {
    "format": "nanovna-saver-limits",
    "version": 2,
    "name": "impossible",
    "rules": [
        {
            "id": "match",
            "label": "Antenna match",
            "metric": "s11_vswr",
            "segments": [
                {
                    "start_hz": 14_130_000,
                    "stop_hz": 14_170_000,
                    "maximum": 1.001,
                }
            ],
        }
    ],
}

LIMIT_S21 = {
    "format": "nanovna-saver-limits",
    "version": 2,
    "name": "s21",
    "rules": [
        {
            "id": "loss",
            "label": "Line loss",
            "metric": "s21_insertion_loss_db",
            "segments": [
                {
                    "start_hz": 14_130_000,
                    "stop_hz": 14_170_000,
                    "maximum": 2.0,
                }
            ],
        }
    ],
}


def run(*args, timeout=180):
    return subprocess.run(
        [sys.executable, "-m", "NanoVNASaver.cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture
def limits_file(tmp_path):
    def write(document, name="limits.nvlimits"):
        path = tmp_path / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return str(path)

    return write


class TestFrequencyParsing:
    @staticmethod
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1000", 1000),
            ("1k", 1000),
            ("1.5M", 1_500_000),
            ("2450MHz", 2_450_000_000),
            ("2g", 2_000_000_000),
            ("144.39 MHz", 144_390_000),
            ("7hz", 7),
        ],
    )
    def test_units_are_scaled(text, expected):
        assert parse_frequency(text) == expected

    @staticmethod
    @pytest.mark.parametrize("text", ["", "M", "abc", "1.2.3M"])
    def test_nonsense_is_rejected(text):
        with pytest.raises(ValueError):
            parse_frequency(text)


class TestExitCodes:
    @staticmethod
    def test_a_passing_check_exits_zero(limits_file):
        result = run(
            "--scenario",
            "antenna",
            "--start",
            "14.0M",
            "--stop",
            "14.3M",
            "--points",
            "41",
            "--limits",
            limits_file(LIMIT_VSWR),
        )
        assert result.returncode == EXIT_OK, result.stderr
        assert "PASS" in result.stdout
        assert "1 pass, 0 fail, 0 unavailable" in result.stdout

    @staticmethod
    def test_a_failing_check_exits_one(limits_file):
        result = run(
            "--scenario",
            "antenna",
            "--start",
            "14.0M",
            "--stop",
            "14.3M",
            "--points",
            "41",
            "--limits",
            limits_file(LIMIT_IMPOSSIBLE),
        )
        assert result.returncode == EXIT_FAILED, result.stderr
        assert "FAIL" in result.stdout

    @staticmethod
    def test_a_missing_channel_is_unavailable_and_never_a_pass(limits_file):
        """The whole point of exit 3: no data must not read as success."""
        result = run(
            "--scenario",
            "antenna",
            "--points",
            "41",
            "--limits",
            limits_file(LIMIT_S21),
        )
        assert result.returncode == EXIT_UNAVAILABLE, result.stderr
        assert result.returncode != EXIT_OK
        assert "does not provide S21" in result.stderr

    @staticmethod
    def test_a_missing_file_exits_two(tmp_path):
        result = run("--input", str(tmp_path / "absent.s1p"))
        assert result.returncode == EXIT_INVALID
        assert "cannot read" in result.stderr

    @staticmethod
    def test_an_unreadable_limit_set_exits_two(tmp_path):
        limit = tmp_path / "broken.nvlimits"
        limit.write_text("{not json", encoding="utf-8")
        result = run(
            "--scenario", "cable", "--points", "11", "--limits", str(limit)
        )
        assert result.returncode == EXIT_INVALID
        assert "cannot read limits" in result.stderr

    @staticmethod
    def test_an_unknown_scenario_exits_two():
        result = run("--scenario", "nonexistent")
        assert result.returncode == EXIT_INVALID

    @staticmethod
    def test_an_inverted_sweep_exits_two():
        result = run("--start", "20M", "--stop", "10M")
        assert result.returncode == EXIT_INVALID

    @staticmethod
    def test_a_help_request_exits_zero():
        assert run("--help").returncode == EXIT_OK


class TestOutputs:
    @staticmethod
    def test_json_carries_provenance_and_the_summary(limits_file, tmp_path):
        out = tmp_path / "result.json"
        result = run(
            "--scenario",
            "antenna",
            "--start",
            "14.0M",
            "--stop",
            "14.3M",
            "--points",
            "41",
            "--limits",
            limits_file(LIMIT_VSWR),
            "--json",
            str(out),
        )
        assert result.returncode == EXIT_OK, result.stderr
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["metadata"]["simulated"] is True
        assert payload["metadata"]["source"] == "scenario:antenna"
        assert payload["checked"] is True
        assert payload["summary"] == {
            "passed": 1,
            "failed": 0,
            "unavailable": 0,
        }
        assert payload["limits"][0]["verdict"] == "pass"

    @staticmethod
    def test_stdout_json_matches_the_file(limits_file, tmp_path):
        path = tmp_path / "result.json"
        args = (
            "--scenario",
            "cable",
            "--points",
            "11",
            "--stdout-format",
            "json",
            "--json",
            str(path),
        )
        result = run(*args)
        assert result.returncode == EXIT_OK, result.stderr
        assert json.loads(result.stdout) == json.loads(
            path.read_text(encoding="utf-8")
        )

    @staticmethod
    def test_a_touchstone_file_is_written(tmp_path):
        out = tmp_path / "sweep.s2p"
        result = run(
            "--scenario", "cable", "--points", "11", "--touchstone", str(out)
        )
        assert result.returncode == EXIT_OK, result.stderr
        text = out.read_text(encoding="utf-8")
        assert text.startswith("!")
        # The option line records the units, reference and format used.
        assert "# HZ S RI R 50" in text

    @staticmethod
    def test_an_existing_output_is_refused_without_overwrite(tmp_path):
        out = tmp_path / "sweep.s2p"
        out.write_text("existing", encoding="utf-8")
        result = run(
            "--scenario", "cable", "--points", "11", "--touchstone", str(out)
        )
        assert result.returncode == EXIT_INVALID
        assert "already exists" in result.stderr
        # Untouched, not truncated.
        assert out.read_text(encoding="utf-8") == "existing"

    @staticmethod
    def test_overwrite_replaces_the_file(tmp_path):
        out = tmp_path / "sweep.s2p"
        out.write_text("existing", encoding="utf-8")
        result = run(
            "--scenario",
            "cable",
            "--points",
            "11",
            "--touchstone",
            str(out),
            "--overwrite",
        )
        assert result.returncode == EXIT_OK, result.stderr
        assert out.read_text(encoding="utf-8").startswith("!")

    @staticmethod
    def test_a_path_with_spaces_works(tmp_path):
        directory = tmp_path / "a directory with spaces"
        directory.mkdir()
        out = directory / "sweep out.s2p"
        result = run(
            "--scenario", "cable", "--points", "11", "--touchstone", str(out)
        )
        assert result.returncode == EXIT_OK, result.stderr
        assert out.exists()

    @staticmethod
    def test_a_failed_run_leaves_no_partial_file(tmp_path):
        """A refused run must not leave something that looks like a result."""
        out = tmp_path / "never.s2p"
        result = run(
            "--scenario",
            "nonexistent",
            "--points",
            "11",
            "--touchstone",
            str(out),
        )
        assert result.returncode == EXIT_INVALID
        assert not out.exists()
        leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".part")]
        assert leftovers == []


class TestChannels:
    @staticmethod
    def test_a_one_port_device_is_asked_only_for_s11(limits_file):
        result = run(
            "--scenario",
            "antenna",
            "--start",
            "14.0M",
            "--stop",
            "14.3M",
            "--points",
            "41",
            "--limits",
            limits_file(LIMIT_VSWR),
            "--stdout-format",
            "json",
        )
        assert result.returncode == EXIT_OK, result.stderr
        payload = json.loads(result.stdout)
        assert payload["metadata"]["channels"] == ["S11"]

    @staticmethod
    def test_no_limit_set_acquires_both_channels():
        result = run(
            "--scenario", "cable", "--points", "11", "--stdout-format", "json"
        )
        assert result.returncode == EXIT_OK, result.stderr
        payload = json.loads(result.stdout)
        assert payload["metadata"]["channels"] == ["S11", "S21"]
        assert payload["checked"] is False


class TestOffline:
    @staticmethod
    def test_a_touchstone_file_can_be_checked_without_a_device(
        tmp_path, limits_file
    ):
        written = tmp_path / "cable.s2p"
        assert (
            run(
                "--scenario",
                "cable",
                "--points",
                "21",
                "--touchstone",
                str(written),
            ).returncode
            == EXIT_OK
        )
        result = run(
            "--input",
            str(written),
            "--limits",
            limits_file(
                {
                    **LIMIT_S21,
                    "rules": [
                        {
                            **LIMIT_S21["rules"][0],
                            "segments": [
                                {
                                    "start_hz": 1_000_000,
                                    "stop_hz": 100_000_000,
                                    "maximum": 50.0,
                                }
                            ],
                        }
                    ],
                }
            ),
            "--stdout-format",
            "json",
        )
        assert result.returncode == EXIT_OK, result.stderr
        payload = json.loads(result.stdout)
        # An imported file is not simulated, and must not say it is.
        assert payload["metadata"]["simulated"] is False
        assert payload["metadata"]["source"].endswith("cable.s2p")


class TestReadPath:
    """PERF-02: the optimised read paths are opt-in, and reported."""

    @staticmethod
    def test_the_default_run_uses_the_trusted_path():
        result = run(
            "--scenario", "filter", "--points", "11", "--stdout-format", "json"
        )
        assert result.returncode == EXIT_OK, result.stderr
        payload = json.loads(result.stdout)
        assert payload["metadata"]["read_path"] == "trusted"

    @staticmethod
    def test_the_opt_in_flags_are_accepted_and_still_succeed():
        result = run(
            "--scenario",
            "filter",
            "--points",
            "11",
            "--combined-text",
            "--binary-scan",
            "--stdout-format",
            "json",
        )
        assert result.returncode == EXIT_OK, result.stderr
        payload = json.loads(result.stdout)
        # The simulator offers neither path, so it honestly reports the
        # trusted read rather than claiming an optimisation happened.
        assert payload["metadata"]["read_path"] == "trusted"
        assert payload["metadata"]["channels"] == ["S11", "S21"]


class TestExitCodesStayHonest:
    """Exit 1 only ever means a limit check failed."""

    @staticmethod
    def _main(*args):
        from NanoVNASaver.cli import main

        return main(list(args))

    def test_an_unwritable_output_is_not_a_failed_unit(self, tmp_path, capsys):
        missing = tmp_path / "file-not-a-directory"
        missing.write_text("")
        code = self._main(
            "--scenario",
            "filter",
            "--start",
            "13M",
            "--stop",
            "15M",
            "--points",
            "11",
            "--json",
            str(missing / "x.json"),
        )
        assert code == 2
        assert "error" in capsys.readouterr().err

    @staticmethod
    def test_a_non_finite_frequency_is_invalid_input():
        from NanoVNASaver.cli import parse_frequency

        for text in ("inf", "nan", "infM"):
            with pytest.raises(ValueError, match="finite"):
                parse_frequency(text)

    def _sequence(self, tmp_path, steps):
        path = tmp_path / "unit.nvseq"
        path.write_text(
            json.dumps(
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "name": "unit",
                    "steps": steps,
                }
            )
        )
        return str(path)

    def test_a_sequence_defect_without_log_is_invalid_not_a_traceback(
        self, tmp_path
    ):
        path = self._sequence(
            tmp_path, [{"action": "log", "message": "{typo}"}]
        )
        assert self._main("--sequence", path) == 2

    def test_an_unknown_scenario_is_invalid_input(self, tmp_path):
        path = self._sequence(
            tmp_path,
            [{"action": "sweep", "start": "13M", "stop": "15M", "points": 11}],
        )
        assert self._main("--sequence", path, "--scenario", "bogus") == 2

    def test_a_stop_on_an_unavailable_step_is_unavailable(self, tmp_path):
        path = self._sequence(
            tmp_path, [{"action": "prompt", "message": "connect the DUT"}]
        )
        assert self._main("--sequence", path) == 3
