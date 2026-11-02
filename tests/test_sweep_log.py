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
"""NEXT-05: per-sweep verdict logging.

The contract is that a row in the log corresponds to one identifiable run
and says the same thing a frozen report would.  These tests read the file
back rather than inspecting the objects, because the file is what an auditor
sees.
"""

import csv
import json
import subprocess
import sys

import pytest

from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Limits import (
    LimitEvaluation,
    Metric,
    Verdict,
)
from NanoVNASaver.Limits import (
    load as load_limits,
)
from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.SweepLog import (
    CSV_FIELDS,
    LogFormat,
    LogWriteError,
    Outcome,
    SweepLog,
    cancelled_record,
    failed_record,
    record_for,
    summarise,
)


def snapshot(points=11, name="test"):
    data = [
        Datapoint(14_000_000 + index * 1_000_000, 0.2, 0.1)
        for index in range(points)
    ]
    return snapshot_from_datapoints(data, None, name=name)


def evaluation(verdict, label="rule", rule_id="r1"):
    return LimitEvaluation(
        rule_id=rule_id,
        rule_label=label,
        metric=Metric.S11_VSWR,
        trace_id="t1",
        trace_name="trace",
        verdict=verdict,
        band_start=14_000_000,
        band_stop=15_000_000,
        reason="" if verdict is Verdict.PASS else f"{label} {verdict.value}",
    )


class TestRecords:
    @staticmethod
    def test_a_passing_measurement_is_a_pass():
        record = record_for(
            snapshot(),
            run_id="run-1",
            evaluations=[evaluation(Verdict.PASS)],
        )
        assert record.outcome is Outcome.MEASURED
        assert record.is_pass
        assert record.rules_passed == 1

    @staticmethod
    def test_a_failed_rule_makes_the_run_fail():
        record = record_for(
            snapshot(),
            run_id="run-1",
            evaluations=[
                evaluation(Verdict.PASS, "ok", "r1"),
                evaluation(Verdict.FAIL, "bad", "r2"),
            ],
        )
        assert record.verdict == Verdict.FAIL.value
        assert not record.is_pass
        assert record.rules_failed == 1
        assert "bad" in record.reason

    @staticmethod
    def test_an_unavailable_rule_is_not_a_pass():
        record = record_for(
            snapshot(),
            run_id="run-1",
            evaluations=[evaluation(Verdict.UNAVAILABLE, "gap")],
        )
        assert record.verdict == Verdict.UNAVAILABLE.value
        assert not record.is_pass
        assert record.rules_unavailable == 1

    @staticmethod
    def test_no_rules_evaluated_is_not_a_pass():
        """A run nobody checked must not read as a successful one."""
        record = record_for(snapshot(), run_id="run-1")
        assert record.verdict == Verdict.UNAVAILABLE.value
        assert not record.is_pass
        assert "no limit rules" in record.reason

    @staticmethod
    def test_a_cancelled_run_is_never_a_measurement():
        record = cancelled_record(run_id="run-1", serial="SN1")
        assert record.outcome is Outcome.CANCELLED
        assert record.verdict == ""
        assert not record.is_pass

    @staticmethod
    def test_a_failed_run_is_never_a_measurement():
        record = failed_record(run_id="run-1", reason="device vanished")
        assert record.outcome is Outcome.FAILED
        assert record.verdict == ""
        assert not record.is_pass
        assert record.reason == "device vanished"

    @staticmethod
    def test_the_calibration_identity_is_carried_from_the_snapshot():
        record = record_for(snapshot(), run_id="run-1")
        assert record.calibration_identity == ""
        assert record.simulated is False
        assert record.reference_impedance == 50.0


class TestWriting:
    @staticmethod
    def test_a_csv_log_has_the_frozen_header(tmp_path):
        log = SweepLog(str(tmp_path / "v.csv"))
        log.append(record_for(snapshot(), run_id="run-1"))
        rows = log.read()
        assert list(rows[0]) == list(CSV_FIELDS)
        assert rows[0]["run_id"] == "run-1"
        assert rows[0]["outcome"] == "measured"

    @staticmethod
    def test_a_jsonl_log_round_trips(tmp_path):
        path = tmp_path / "v.jsonl"
        log = SweepLog(str(path), log_format=LogFormat.JSONL)
        log.append(record_for(snapshot(), run_id="run-1"))

        rows = log.read()
        assert rows[0]["run_id"] == "run-1"
        assert rows[0]["simulated"] is False
        raw = path.read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(raw[0])["run_id"] == "run-1"

    @staticmethod
    def test_two_runs_append_two_rows(tmp_path):
        log = SweepLog(str(tmp_path / "v.csv"))
        log.append(record_for(snapshot(), run_id="run-1"))
        log.append(record_for(snapshot(), run_id="run-2"))
        assert [row["run_id"] for row in log.read()] == ["run-1", "run-2"]

    @staticmethod
    def test_the_same_run_is_refused_twice(tmp_path):
        """A duplicated completion signal must not inflate the log."""
        path = tmp_path / "v.csv"
        log = SweepLog(str(path))
        log.append(record_for(snapshot(), run_id="run-1"))
        with pytest.raises(LogWriteError, match="already in this log"):
            log.append(record_for(snapshot(), run_id="run-1"))
        assert len(log.read()) == 1

    @staticmethod
    def test_a_mismatched_header_is_refused_not_misread(tmp_path):
        path = tmp_path / "v.csv"
        path.write_text("wrong,columns\n1,2\n", encoding="utf-8")
        log = SweepLog(str(path))
        with pytest.raises(LogWriteError, match="different column layout"):
            log.append(record_for(snapshot(), run_id="run-1"))
        # The existing file is untouched.
        assert path.read_text(encoding="utf-8") == "wrong,columns\n1,2\n"

    @staticmethod
    def test_no_file_is_created_until_something_is_logged(tmp_path):
        path = tmp_path / "v.csv"
        SweepLog(str(path))
        assert not path.exists()

    @staticmethod
    def test_an_unwritable_path_is_reported_as_a_log_error(tmp_path):
        directory = tmp_path / "missing"
        log = SweepLog(str(directory / "v.csv"))
        with pytest.raises(LogWriteError, match="cannot write"):
            log.append(record_for(snapshot(), run_id="run-1"))

    @staticmethod
    def test_a_later_failure_does_not_block_a_retry_of_a_different_run(
        tmp_path,
    ):
        path = tmp_path / "v.csv"
        path.write_text("bad\nheader\n", encoding="utf-8")
        log = SweepLog(str(path))
        with pytest.raises(LogWriteError):
            log.append(record_for(snapshot(), run_id="run-1"))


class TestSummary:
    @staticmethod
    def test_outcomes_are_counted_separately():
        records = [
            record_for(
                snapshot(), run_id="a", evaluations=[evaluation(Verdict.PASS)]
            ),
            record_for(
                snapshot(), run_id="b", evaluations=[evaluation(Verdict.FAIL)]
            ),
            record_for(
                snapshot(),
                run_id="c",
                evaluations=[evaluation(Verdict.UNAVAILABLE)],
            ),
            record_for(snapshot(), run_id="d"),
            cancelled_record(run_id="e"),
            failed_record(run_id="f", reason="x"),
        ]
        counts = summarise(records)
        assert counts["records"] == 6
        assert counts["passed"] == 1
        assert counts["failed"] == 1
        # Both "unavailable rule" and "no rules at all" land here: neither is
        # a pass, and both need a human.
        assert counts["unavailable"] == 2
        assert counts["cancelled"] == 1
        assert counts["errors"] == 1


class TestAgainstRealLimits:
    @staticmethod
    def test_a_logged_verdict_matches_the_limit_evaluation(tmp_path):
        limit_path = tmp_path / "l.nvlimits"
        limit_path.write_text(
            json.dumps(
                {
                    "format": "nanovna-saver-limits",
                    "version": 2,
                    "name": "t",
                    "rules": [
                        {
                            "id": "match",
                            "label": "Match",
                            "metric": "s11_vswr",
                            "segments": [
                                {
                                    "start_hz": 14_000_000,
                                    "stop_hz": 15_000_000,
                                    "maximum": 2.0,
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        data = snapshot()
        evaluations = load_limits(str(limit_path)).evaluate(data)
        record = record_for(
            data,
            run_id="run-1",
            evaluations=evaluations,
            limit_identity=str(limit_path),
        )
        assert record.verdict == evaluations[0].verdict.value
        assert record.is_pass == (evaluations[0].verdict is Verdict.PASS)


class TestSequenceRuns:
    """A sequence run is logged once, and only a complete run can pass."""

    @staticmethod
    def _provider():
        from NanoVNASaver.Hardware.Simulated import make_simulated_vna
        from NanoVNASaver.SequenceActions import DeviceProvider

        return DeviceProvider(
            factory=lambda: make_simulated_vna("antenna"), simulated=True
        )

    @staticmethod
    def _limits(path):
        path.write_text(
            json.dumps(
                {
                    "format": "nanovna-saver-limits",
                    "version": 2,
                    "name": "t",
                    "rules": [
                        {
                            "id": "m",
                            "label": "Match",
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
            ),
            encoding="utf-8",
        )
        return str(path)

    @staticmethod
    def _run(tmp_path, steps, *, log):
        from NanoVNASaver.Sequence import load_sequence
        from NanoVNASaver.SequenceActions import run_device_sequence

        sequence_path = tmp_path / "unit.nvseq"
        sequence_path.write_text(
            json.dumps(
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "name": "t",
                    "steps": steps,
                }
            ),
            encoding="utf-8",
        )
        sequence = load_sequence(str(sequence_path))
        return run_device_sequence(
            sequence,
            TestSequenceRuns._provider(),
            base_directory=str(tmp_path),
            output_directory=str(tmp_path),
            log=log,
        )

    def test_a_complete_passing_run_logs_one_pass(self, tmp_path):
        limit_path = self._limits(tmp_path / "l.nvlimits")
        log = SweepLog(str(tmp_path / "v.csv"))
        record = self._run(
            tmp_path,
            [
                {"action": "sweep", "start": "14.0M", "stop": "14.3M"},
                {"action": "check", "limits": "l.nvlimits"},
            ],
            log=log,
        )
        assert record.passed
        rows = log.read()
        assert len(rows) == 1
        assert rows[0]["verdict"] == "pass"
        assert rows[0]["limit_identity"] == limit_path
        assert rows[0]["outcome"] == "measured"

    def test_a_run_that_stops_after_a_passing_check_is_not_logged_as_a_pass(
        self, tmp_path
    ):
        """The unit met its limits; the program did not finish. Not a pass."""
        self._limits(tmp_path / "l.nvlimits")
        log = SweepLog(str(tmp_path / "v.csv"))
        record = self._run(
            tmp_path,
            [
                {"action": "sweep", "start": "14.0M", "stop": "14.3M"},
                {"action": "check", "limits": "l.nvlimits"},
                {"action": "save", "path": "missing/dir/{serial}.s1p"},
            ],
            log=log,
        )
        assert not record.passed
        rows = log.read()
        assert len(rows) == 1
        assert rows[0]["outcome"] == "failed"
        assert rows[0]["verdict"] == ""
        assert "step 3" in rows[0]["reason"]

    def test_a_failing_check_logs_a_fail(self, tmp_path):
        (tmp_path / "l.nvlimits").write_text(
            json.dumps(
                {
                    "format": "nanovna-saver-limits",
                    "version": 2,
                    "name": "t",
                    "rules": [
                        {
                            "id": "m",
                            "label": "Tight",
                            "metric": "s11_vswr",
                            "segments": [
                                {
                                    "start_hz": 14_130_000,
                                    "stop_hz": 14_170_000,
                                    "maximum": 1.0001,
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        log = SweepLog(str(tmp_path / "v.csv"))
        record = self._run(
            tmp_path,
            [
                {"action": "sweep", "start": "14.0M", "stop": "14.3M"},
                {"action": "check", "limits": "l.nvlimits"},
            ],
            log=log,
        )
        assert not record.passed
        rows = log.read()
        assert len(rows) == 1
        assert rows[0]["verdict"] == "fail"
        assert rows[0]["rules_failed"] == "1"

    def test_a_run_that_never_acquired_logs_a_failure(self, tmp_path):
        log = SweepLog(str(tmp_path / "v.csv"))
        self._run(tmp_path, [{"action": "log", "message": "hello"}], log=log)
        rows = log.read()
        assert len(rows) == 1
        assert rows[0]["outcome"] == "failed"
        assert "did not acquire" in rows[0]["reason"]

    def test_a_log_failure_is_raised_after_the_device_is_released(
        self, tmp_path
    ):
        from NanoVNASaver.Sequence import load_sequence
        from NanoVNASaver.SequenceActions import (
            DeviceProvider,
            run_device_sequence,
        )

        sequence_path = tmp_path / "unit.nvseq"
        sequence_path.write_text(
            json.dumps(
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "name": "t",
                    "steps": [
                        {
                            "action": "sweep",
                            "start": "14.0M",
                            "stop": "14.3M",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        holder = {}

        def factory():
            vna = make_simulated_vna("antenna")
            holder["vna"] = vna
            return vna

        provider = DeviceProvider(factory=factory, simulated=True)
        log = SweepLog(str(tmp_path / "missing" / "v.csv"))
        with pytest.raises(LogWriteError):
            run_device_sequence(
                load_sequence(str(sequence_path)),
                provider,
                base_directory=str(tmp_path),
                output_directory=str(tmp_path),
                log=log,
            )
        assert provider.session is None


class TestCli:
    @staticmethod
    def test_the_cli_appends_an_audit_row(tmp_path):
        limits = tmp_path / "l.nvlimits"
        limits.write_text(
            json.dumps(
                {
                    "format": "nanovna-saver-limits",
                    "version": 2,
                    "name": "t",
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
            ),
            encoding="utf-8",
        )
        log_path = tmp_path / "verdicts.csv"
        finished = subprocess.run(
            [
                sys.executable,
                "-m",
                "NanoVNASaver.cli",
                "--scenario",
                "antenna",
                "--start",
                "14.0M",
                "--stop",
                "14.3M",
                "--points",
                "41",
                "--limits",
                str(limits),
                "--log",
                str(log_path),
                "--serial-number",
                "SN9",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert finished.returncode == 0, finished.stderr
        with open(log_path, encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0]["serial"] == "SN9"
        assert rows[0]["verdict"] == "pass"
        assert rows[0]["simulated"] == "True"
        assert rows[0]["limit_identity"] == str(limits)
