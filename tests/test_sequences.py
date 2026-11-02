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
"""AUTO-03: versioned test sequences.

The promise a sequence makes is an *auditable* record, so these tests check
the record: what ran, what was skipped, why the run stopped, and that a run
which did not complete is never reported as passed.
"""

import json
import os
import subprocess
import sys

import pytest

from NanoVNASaver.Sequence import (
    ActionKind,
    SequenceError,
    SequenceFormatError,
    StepStatus,
    StopPolicy,
    Substitutions,
    load_sequence,
    parse_sequence,
    run_sequence,
    sequence_identity,
)
from NanoVNASaver.SequenceActions import DeviceProvider, run_device_sequence

DOCUMENT = {
    "format": "nanovna-saver-sequence",
    "version": 1,
    "name": "unit test",
    "steps": [
        {"action": "log", "message": "starting {serial}"},
        {
            "action": "sweep",
            "start": "14.0M",
            "stop": "14.3M",
            "points": 21,
            "name": "unit {serial}",
        },
    ],
}


def document(**overrides):
    merged = dict(DOCUMENT)
    merged.update(overrides)
    return merged


def write_sequence(tmp_path, payload, name="unit.nvseq"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def write_limits(tmp_path, maximum=2.0, name="limits.nvlimits"):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "format": "nanovna-saver-limits",
                "version": 2,
                "name": "test",
                "rules": [
                    {
                        "id": "match",
                        "label": "Antenna match",
                        "metric": "s11_vswr",
                        "segments": [
                            {
                                "start_hz": 14_130_000,
                                "stop_hz": 14_170_000,
                                "maximum": maximum,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path.name


class TestSchema:
    @staticmethod
    def test_a_valid_document_parses():
        sequence = parse_sequence(json.dumps(DOCUMENT))
        assert sequence.name == "unit test"
        assert [step.kind for step in sequence.steps] == [
            ActionKind.LOG,
            ActionKind.SWEEP,
        ]

    @staticmethod
    @pytest.mark.parametrize(
        "payload,match",
        [
            ({"format": "wrong", "version": 1, "steps": []}, "format must be"),
            (
                {"format": "nanovna-saver-sequence", "version": 9, "steps": []},
                "unsupported sequence version",
            ),
            (
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "steps": [],
                },
                "non-empty list",
            ),
            (
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "steps": [{"action": "nonsense"}],
                },
                "action must be one of",
            ),
            (
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "steps": [{"action": "sweep", "start": "1M"}],
                },
                "needs a stop",
            ),
            (
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "steps": [{"action": "check"}],
                },
                "needs a limits path",
            ),
            (
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "steps": [{"action": "save"}],
                },
                "needs a path",
            ),
            (document(extra=1), "unknown top-level key"),
        ],
    )
    def test_invalid_documents_are_refused_with_a_reason(payload, match):
        with pytest.raises(SequenceFormatError, match=match):
            parse_sequence(json.dumps(payload))

    @staticmethod
    def test_an_unknown_key_on_a_step_is_refused():
        """A typo must not be silently ignored: it could change the run."""
        with pytest.raises(SequenceFormatError, match="unknown key 'stpo'"):
            parse_sequence(
                json.dumps(
                    {
                        "format": "nanovna-saver-sequence",
                        "version": 1,
                        "steps": [
                            {"action": "sweep", "start": "1M", "stpo": "2M"}
                        ],
                    }
                )
            )

    @staticmethod
    def test_invalid_json_names_the_line():
        with pytest.raises(SequenceFormatError, match="invalid JSON"):
            parse_sequence("{")

    @staticmethod
    def test_a_bad_policy_is_refused():
        with pytest.raises(SequenceFormatError, match="must be 'continue'"):
            parse_sequence(
                json.dumps(
                    {
                        "format": "nanovna-saver-sequence",
                        "version": 1,
                        "steps": [
                            {
                                "action": "sweep",
                                "start": "1M",
                                "stop": "2M",
                                "fail": "explode",
                            }
                        ],
                    }
                )
            )

    @staticmethod
    def test_policies_are_accepted_on_any_action():
        payload = {
            "format": "nanovna-saver-sequence",
            "version": 1,
            "steps": [
                {"action": "log", "message": "x", "fail": "continue"},
                {"action": "save", "path": "a.s1p", "unavailable": "continue"},
            ],
        }
        sequence = parse_sequence(json.dumps(payload))
        assert sequence.steps[0].on_failure is StopPolicy.CONTINUE
        assert sequence.steps[1].on_unavailable is StopPolicy.CONTINUE


class TestSubstitutions:
    @staticmethod
    @pytest.mark.parametrize(
        "template,expected",
        [
            ("{serial}.s1p", "SN7.s1p"),
            ("{name}-{serial}", "dipole-SN7"),
            ("plain.s1p", "plain.s1p"),
            ("{date}/x", "2026-01-01/x"),
        ],
    )
    def test_known_values_expand(template, expected):
        values = Substitutions(serial="SN7", name="dipole", date="2026-01-01")
        assert values.apply(template, "test") == expected

    @staticmethod
    def test_unknown_substitution_is_refused_not_left_literal():
        values = Substitutions(serial="SN7")
        with pytest.raises(SequenceFormatError, match="unknown substitution"):
            values.apply("{operator}.s1p", "test")

    @staticmethod
    def test_document_values_become_defaults():
        sequence = parse_sequence(
            json.dumps(
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "substitutions": {"name": "from-file"},
                    "steps": [{"action": "log", "message": "x"}],
                }
            )
        )
        assert sequence.substitutions.name == "from-file"

    @staticmethod
    def test_a_caller_value_overrides_the_document():
        sequence = parse_sequence(
            json.dumps(
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "steps": [{"action": "log", "message": "x"}],
                }
            ),
            substitutions=Substitutions(serial="SN9"),
        )
        assert sequence.substitutions.serial == "SN9"


class TestPaths:
    @staticmethod
    def test_a_relative_output_lands_in_the_output_directory(tmp_path):
        output = tmp_path / "out"
        output.mkdir()
        sequence = parse_sequence(
            json.dumps(
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "steps": [{"action": "save", "path": "a.s1p"}],
                }
            )
        )
        context_path = run_sequence(
            sequence,
            {
                ActionKind.SAVE: lambda step, ctx: (
                    StepStatus.OK,
                    "",
                    ctx.path_for(step),
                )
            },
            base_directory=str(tmp_path),
            output_directory=str(output),
        )
        assert context_path.steps[0].artifact == str(output / "a.s1p")

    @staticmethod
    @pytest.mark.parametrize(
        "path",
        ["../escape.s1p", "../../../etc/passwd", "/etc/passwd"],
    )
    def test_an_escaping_path_is_refused(tmp_path, path):
        output = tmp_path / "out"
        output.mkdir()
        sequence = parse_sequence(
            json.dumps(
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "steps": [{"action": "save", "path": path}],
                }
            )
        )
        result = run_sequence(
            sequence,
            {ActionKind.SAVE: _save_via_context},
            base_directory=str(tmp_path),
            output_directory=str(output),
        )
        assert result.steps[0].status is StepStatus.FAILED
        assert "resolves outside" in result.steps[0].detail
        assert not os.path.exists(str(tmp_path / "escape.s1p"))


def _save_via_context(step, context):
    try:
        return StepStatus.OK, "", context.path_for(step)
    except Exception as error:
        return StepStatus.FAILED, str(error), ""


class TestRunning:
    @staticmethod
    def test_a_complete_successful_run_passes():
        sequence = parse_sequence(json.dumps(DOCUMENT))
        calls = []

        def handler(step, context):
            calls.append(step.kind)
            return StepStatus.OK, "", ""

        result = run_sequence(
            sequence,
            {ActionKind.LOG: handler, ActionKind.SWEEP: handler},
            base_directory=".",
            output_directory=".",
        )
        assert result.passed
        assert result.completed
        assert calls == [ActionKind.LOG, ActionKind.SWEEP]

    @staticmethod
    def test_a_failed_step_stops_the_run_by_default():
        sequence = parse_sequence(json.dumps(DOCUMENT))
        seen = []

        def ok(step, context):
            seen.append(step.index)
            return StepStatus.OK, "", ""

        def bad(step, context):
            seen.append(step.index)
            return StepStatus.FAILED, "nope", ""

        result = run_sequence(
            sequence,
            {ActionKind.LOG: bad, ActionKind.SWEEP: ok},
            base_directory=".",
            output_directory=".",
        )
        assert seen == [0]
        assert result.stopped_at == 0
        assert not result.completed
        assert not result.passed

    @staticmethod
    def test_a_continue_policy_runs_the_rest_but_still_does_not_pass():
        payload = document()
        payload["steps"][0]["fail"] = "continue"
        sequence = parse_sequence(json.dumps(payload))
        result = run_sequence(
            sequence,
            {
                ActionKind.LOG: lambda step, ctx: (
                    StepStatus.FAILED,
                    "nope",
                    "",
                ),
                ActionKind.SWEEP: lambda step, ctx: (StepStatus.OK, "done", ""),
            },
            base_directory=".",
            output_directory=".",
        )
        assert len(result.steps) == 2
        assert result.completed
        # A run with a failed step is never "passed", however it continued.
        assert not result.passed
        assert result.summary()["failed"] == 1

    @staticmethod
    def test_an_unavailable_step_is_not_success():
        sequence = parse_sequence(json.dumps(DOCUMENT))
        result = run_sequence(
            sequence,
            {
                ActionKind.LOG: lambda step, ctx: (
                    StepStatus.UNAVAILABLE,
                    "no data",
                    "",
                ),
                ActionKind.SWEEP: lambda step, ctx: (StepStatus.OK, "", ""),
            },
            base_directory=".",
            output_directory=".",
        )
        assert result.stopped_at == 0
        assert not result.passed
        assert result.summary()["unavailable"] == 1

    @staticmethod
    def test_unavailable_can_be_configured_to_continue():
        payload = document()
        payload["steps"][0]["unavailable"] = "continue"
        sequence = parse_sequence(json.dumps(payload))
        result = run_sequence(
            sequence,
            {
                ActionKind.LOG: lambda step, ctx: (
                    StepStatus.UNAVAILABLE,
                    "no data",
                    "",
                ),
                ActionKind.SWEEP: lambda step, ctx: (StepStatus.OK, "", ""),
            },
            base_directory=".",
            output_directory=".",
        )
        assert result.completed
        assert not result.passed

    @staticmethod
    def test_a_missing_handler_is_an_error_not_a_skip():
        payload = document()
        payload["steps"].append({"action": "check", "limits": "x.nvlimits"})
        sequence = parse_sequence(json.dumps(payload))
        with pytest.raises(SequenceError, match="no handler for action"):
            run_sequence(
                sequence,
                {ActionKind.LOG: lambda s, c: (StepStatus.OK, "", "")},
                base_directory=".",
                output_directory=".",
            )

    @staticmethod
    def test_an_unanswered_prompt_is_unavailable_by_default():
        sequence = parse_sequence(
            json.dumps(
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "steps": [{"action": "prompt", "message": "press enter"}],
                }
            )
        )
        result = run_sequence(
            sequence, {}, base_directory=".", output_directory="."
        )
        assert result.steps[0].status is StepStatus.UNAVAILABLE
        assert "headless" in result.steps[0].detail

    @staticmethod
    def test_a_prompt_can_be_configured_to_skip():
        sequence = parse_sequence(
            json.dumps(
                {
                    "format": "nanovna-saver-sequence",
                    "version": 1,
                    "steps": [
                        {
                            "action": "prompt",
                            "message": "press enter",
                            "headless": "skip",
                        }
                    ],
                }
            )
        )
        result = run_sequence(
            sequence, {}, base_directory=".", output_directory="."
        )
        assert result.steps[0].status is StepStatus.SKIPPED


class TestIdentity:
    @staticmethod
    def test_identity_is_stable_for_the_same_document():
        one = parse_sequence(json.dumps(DOCUMENT))
        two = parse_sequence(json.dumps(DOCUMENT))
        assert one.identity() == two.identity()

    @staticmethod
    def test_identity_changes_when_a_step_changes():
        changed = document()
        changed["steps"] = [
            dict(changed["steps"][0]),
            {**changed["steps"][1], "points": 51},
        ]
        assert (
            parse_sequence(json.dumps(DOCUMENT)).identity()
            != parse_sequence(json.dumps(changed)).identity()
        )

    @staticmethod
    def test_identity_ignores_the_file_location():
        sequence = parse_sequence(json.dumps(DOCUMENT), source="/a/b.nvseq")
        same = parse_sequence(json.dumps(DOCUMENT), source="/c/d.nvseq")
        assert sequence_identity(sequence) == sequence_identity(same)


class TestEndToEnd:
    @staticmethod
    def test_a_sequence_acquires_checks_saves_and_reports(tmp_path):
        limits = write_limits(tmp_path)
        output = tmp_path / "out"
        payload = document()
        payload["steps"] = [
            {"action": "log", "message": "unit {serial}"},
            {
                "action": "sweep",
                "start": "14.0M",
                "stop": "14.3M",
                "points": 41,
                "name": "unit {serial}",
            },
            {"action": "check", "limits": limits},
            {"action": "save", "path": "{serial}-{name}.s1p"},
            {"action": "report", "path": "{serial}-{name}.json"},
        ]
        path = write_sequence(tmp_path, payload)
        sequence = load_sequence(
            path, substitutions=Substitutions(serial="SN1", name="dipole")
        )
        provider = DeviceProvider(factory=lambda: _antenna(), simulated=True)
        record = run_device_sequence(
            sequence,
            provider,
            base_directory=str(tmp_path),
            output_directory=str(output),
        )
        assert record.passed, [step.detail for step in record.steps]
        assert (output / "SN1-dipole.s1p").exists()
        report = json.loads((output / "SN1-dipole.json").read_text())
        assert report["measurement"]["simulated"] is True
        assert report["measurement"]["name"] == "unit SN1"
        assert report["serial"] == "SN1"
        assert report["sequence"]["identity"]

    @staticmethod
    def test_a_failing_check_leaves_no_artifact(tmp_path):
        limits = write_limits(tmp_path, maximum=1.001)
        output = tmp_path / "out"
        payload = document()
        payload["steps"] = [
            {
                "action": "sweep",
                "start": "14.0M",
                "stop": "14.3M",
                "points": 41,
            },
            {"action": "check", "limits": limits},
            {"action": "save", "path": "should-not-exist.s1p"},
        ]
        path = write_sequence(tmp_path, payload)
        provider = DeviceProvider(factory=lambda: _antenna(), simulated=True)
        record = run_device_sequence(
            load_sequence(path),
            provider,
            base_directory=str(tmp_path),
            output_directory=str(output),
        )
        assert not record.passed
        assert record.summary()["failed"] == 1
        assert not (output / "should-not-exist.s1p").exists()
        # The device was released even though the run stopped.
        assert provider.session is None

    @staticmethod
    def test_the_cli_runs_a_sequence_and_reports_its_record(tmp_path):
        limits = write_limits(tmp_path)
        payload = document()
        payload["steps"] = [
            {
                "action": "sweep",
                "start": "14.0M",
                "stop": "14.3M",
                "points": 41,
            },
            {"action": "check", "limits": limits},
        ]
        path = write_sequence(tmp_path, payload)
        json_path = tmp_path / "record.json"
        finished = subprocess.run(
            [
                sys.executable,
                "-m",
                "NanoVNASaver.cli",
                "--sequence",
                path,
                "--serial",
                "SN3",
                "--output-dir",
                str(tmp_path / "out"),
                "--json",
                str(json_path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert finished.returncode == 0, finished.stderr
        record = json.loads(json_path.read_text())
        assert record["passed"] is True
        assert record["serial"] == "SN3"
        assert record["summary"]["ok"] == 2

    @staticmethod
    def test_the_cli_exits_one_when_a_sequence_fails(tmp_path):
        limits = write_limits(tmp_path, maximum=1.001)
        payload = document()
        payload["steps"] = [
            {
                "action": "sweep",
                "start": "14.0M",
                "stop": "14.3M",
                "points": 41,
                "fail": "stop",
            },
            {"action": "check", "limits": limits},
        ]
        path = write_sequence(tmp_path, payload)
        finished = subprocess.run(
            [
                sys.executable,
                "-m",
                "NanoVNASaver.cli",
                "--sequence",
                path,
                "--output-dir",
                str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert finished.returncode == 1, finished.stderr
        assert "NOT PASSED" in finished.stdout

    @staticmethod
    def test_the_cli_refuses_an_invalid_sequence(tmp_path):
        path = write_sequence(tmp_path, {"format": "nope", "version": 1})
        finished = subprocess.run(
            [sys.executable, "-m", "NanoVNASaver.cli", "--sequence", path],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert finished.returncode == 2
        assert "cannot load the sequence" in finished.stderr


def _antenna():
    from NanoVNASaver.Hardware.Simulated import make_simulated_vna

    return make_simulated_vna("antenna", seed=5)


class TestCallerSubstitutionsWin:
    @staticmethod
    def _parsed(supplied, **declared):
        payload = json.dumps(document(substitutions=declared))
        return parse_sequence(payload, substitutions=supplied).substitutions

    def test_the_caller_overrides_a_declared_serial(self):
        assert (
            self._parsed(
                Substitutions(serial="CLI123"), serial="DEFAULT"
            ).serial
            == "CLI123"
        )

    def test_a_declared_value_is_the_default_when_the_caller_has_none(self):
        assert (
            self._parsed(Substitutions(), serial="DEFAULT").serial == "DEFAULT"
        )

    def test_a_declared_date_is_used_unless_the_caller_sets_one(self):
        assert (
            self._parsed(Substitutions(), date="2020-01-01").date
            == "2020-01-01"
        )
        assert (
            self._parsed(
                Substitutions(date="2021-02-02"), date="2020-01-01"
            ).date
            == "2021-02-02"
        )


class TestAuditLogNeverPassesAnUnfinishedRun:
    @staticmethod
    def test_an_exception_after_a_passing_check_is_logged_as_failed(
        monkeypatch,
    ):
        from NanoVNASaver.Measurement import snapshot_from_datapoints
        from NanoVNASaver.RFTools import Datapoint
        from NanoVNASaver.SequenceActions import sequence_log_record
        from NanoVNASaver.SweepLog import LogRecord, Outcome

        passing = LogRecord(
            timestamp="2026-11-02T00:00:00+00:00",
            run_id="unit",
            outcome=Outcome.MEASURED,
            verdict="pass",
            rules_passed=1,
        )
        assert passing.is_pass
        monkeypatch.setattr(
            "NanoVNASaver.SweepLog.record_for",
            lambda *_args, **_kwargs: passing,
        )
        snapshot = snapshot_from_datapoints(
            [Datapoint(1_000_000, 0.1, 0.0), Datapoint(2_000_000, 0.1, 0.0)],
            None,
        )
        record = sequence_log_record(
            parse_sequence(json.dumps(document())),
            None,
            {"snapshot": snapshot},
        )
        assert not record.is_pass
        assert "did not complete" in record.reason
