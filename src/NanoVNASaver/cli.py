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
"""Unattended sweeps and limit checks from the command line (AUTO-02).

Run with ``python -m NanoVNASaver.cli --help``.

Exit codes are the contract an unattended script relies on, so they are
fixed and documented here:

===== ===========================================================
code  meaning
===== ===========================================================
0     success; every requested check passed
1     at least one limit check FAILED
2     invalid input: bad arguments, unreadable file, bad limit set
3     a result was UNAVAILABLE (missing channel, incomplete run, a
      check that could not be evaluated) and nothing failed
4     a transport or device error prevented the work
130   interrupted (Ctrl-C); no artifact is left behind
===== ===========================================================

UNAVAILABLE is deliberately *not* success.  A script that treats "we could
not evaluate this" as a pass is the failure mode this distinction exists to
prevent, so code 3 is separate from both 0 and 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Sequence

from .Acquisition import (
    AcquisitionError,
    CancelledError,
    CancelToken,
    DeviceError,
    SweepRequest,
    SweepResult,
    UnsupportedError,
    open_device,
)
from .Hardware.CombinedScan import PATH_TRUSTED
from .Hardware.ScanCapability import ScanPreferences
from .Limits import (
    LimitError,
    Verdict,
)
from .Limits import (
    load as load_limits,
)
from .Measurement import MeasurementSnapshot
from .Touchstone import Touchstone, TouchstoneExportError

#: Exit codes, named so the tests and the documentation cannot drift.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
EXIT_DEVICE = 4
EXIT_INTERRUPTED = 130


class CliError(Exception):
    """A problem that maps to one documented exit code."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def parse_frequency(text: str) -> int:
    """Parse ``1.5M``, ``30k``, ``2450MHz`` or a plain hertz count.

    Written here rather than reused from the GUI so the CLI has no Qt
    import and so a mistyped unit is an error instead of a silent factor of
    a thousand.
    """
    cleaned = text.strip().replace(" ", "")
    if not cleaned:
        raise ValueError("empty frequency")
    suffixes = {
        "ghz": 1_000_000_000,
        "g": 1_000_000_000,
        "mhz": 1_000_000,
        "m": 1_000_000,
        "khz": 1_000,
        "k": 1_000,
        "hz": 1,
        "h": 1,
    }
    lowered = cleaned.lower()
    for suffix, factor in suffixes.items():
        if lowered.endswith(suffix):
            number = lowered[: -len(suffix)]
            if not number:
                raise ValueError(f"no number in {text!r}")
            return _whole_hertz(float(number) * factor, text)
    return _whole_hertz(float(lowered), text)


def _whole_hertz(value: float, text: str) -> int:
    if not math.isfinite(value):
        raise ValueError(f"{text!r} is not a finite frequency")
    return int(round(value))


# ----------------------------------------------------------------------
# Results and output
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CheckOutcome:
    """The verdict summary for one run."""

    failed: int
    unavailable: int
    passed: int

    @property
    def exit_code(self) -> int:
        if self.failed:
            return EXIT_FAILED
        if self.unavailable:
            return EXIT_UNAVAILABLE
        return EXIT_OK

    def describe(self) -> str:
        return (
            f"{self.passed} pass, {self.failed} fail,"
            f" {self.unavailable} unavailable"
        )


def evaluate_limits(
    snapshot: MeasurementSnapshot, limit_path: str
) -> tuple[list[dict], CheckOutcome, list]:
    """Evaluate a limit set against a snapshot and summarise the verdicts."""
    try:
        limit_set = load_limits(limit_path)
    except LimitError as error:
        raise CliError(f"cannot read limits: {error}", EXIT_INVALID) from error
    evaluations = limit_set.evaluate(snapshot, trace_id="cli")
    rows = [item.to_dict() for item in evaluations]
    failed = sum(1 for item in evaluations if item.verdict is Verdict.FAIL)
    unavailable = sum(
        1 for item in evaluations if item.verdict is Verdict.UNAVAILABLE
    )
    passed = sum(1 for item in evaluations if item.verdict is Verdict.PASS)
    return (
        rows,
        CheckOutcome(failed=failed, unavailable=unavailable, passed=passed),
        list(evaluations),
    )


def _metadata(
    snapshot: MeasurementSnapshot,
    source: str,
    read_path: str = PATH_TRUSTED,
) -> dict:
    """Provenance every artifact must carry."""
    provenance = snapshot.provenance
    return {
        "source": source,
        "name": snapshot.name,
        "simulated": snapshot.simulated,
        "simulator_scenario": provenance.simulator_scenario,
        "completion": provenance.completion.value,
        "device": provenance.device,
        "app_version": provenance.app_version,
        "reference_impedance": snapshot.reference_impedance,
        "channels": sorted(snapshot.available),
        "notes": provenance.notes,
        "read_path": read_path,
    }


def write_atomically(path: str, writer) -> None:
    """Write via a temporary file in the same directory, then rename.

    An interrupted or failed write therefore never leaves a partial file
    that looks like a completed measurement.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".part")
    os.close(handle)
    try:
        writer(temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _snapshot_to_touchstone(snapshot: MeasurementSnapshot, path: str) -> None:
    """Write the measured channels, and only the measured channels.

    A one-port capture becomes a one-port file.  Writing it as an ``.s2p``
    would fill S21 with a zero placeholder, which is a fabricated
    transmission measurement — so the file type follows the data.
    """
    touchstone = Touchstone("cli")
    touchstone.s11 = list(snapshot.corrected.get("S11", ()))
    touchstone.s21 = list(snapshot.corrected.get("S21", ()))
    parameter_count = 4 if touchstone.s21 else 1
    expected = ".s2p" if parameter_count == 4 else ".s1p"
    if not path.lower().endswith(expected):
        raise CliError(
            f"this capture has"
            f" {'both channels' if parameter_count == 4 else 'S11 only'};"
            f" write it to a {expected} file, not {path}",
            EXIT_INVALID,
        )
    touchstone.filename = path
    touchstone.save(parameter_count)


def _channels_for(limit_path: str | None) -> tuple[str, ...]:
    """The channels a run must acquire, given what will be checked.

    Without a limit set the caller gets both channels, because that is what
    an interactive sweep provides.  With one, only the channels its metrics
    derive from are requested.
    """
    from .Measurement import S11, S21

    if not limit_path:
        return (S11, S21)
    try:
        limit_set = load_limits(limit_path)
    except LimitError:
        # The evaluation step reports this properly; do not mask it here.
        return (S11, S21)
    needed = {rule.metric.parameter for rule in limit_set.rules}
    ordered = tuple(name for name in (S11, S21) if name in needed)
    return ordered or (S11, S21)


def _acquire(
    args, cancel: CancelToken | None
) -> SweepResult | MeasurementSnapshot:
    """Take a sweep from the simulator, or load one from a file.

    A swept result keeps the transport it was read over (``read_path``); a
    loaded file has none, so it returns the snapshot directly.
    """
    if args.input:
        # ``Touchstone.load`` logs an unreadable file and returns with an
        # empty dataset, so an existence check is the only way to tell "the
        # file is missing" from "the file holds nothing".  Without it a
        # mistyped path would exit successfully with an empty measurement.
        if not os.path.isfile(args.input):
            raise CliError(
                f"cannot read {args.input}: no such file", EXIT_INVALID
            )
        touchstone = Touchstone(args.input)
        try:
            touchstone.load()
        except Exception as error:
            raise CliError(
                f"cannot read {args.input}: {error}", EXIT_INVALID
            ) from error
        if not touchstone.s11:
            raise CliError(f"{args.input} contains no S11 data", EXIT_INVALID)
        from .Measurement import snapshot_from_datapoints

        return snapshot_from_datapoints(
            touchstone.s11,
            touchstone.s21,
            name=os.path.basename(args.input),
            source=args.input,
        )

    from .Hardware.Simulated import make_simulated_vna

    try:
        device = make_simulated_vna(
            args.scenario, seed=args.seed, noise_floor=0.0
        )
    except (ValueError, KeyError) as error:
        raise CliError(f"unknown scenario: {error}", EXIT_INVALID) from error
    # Ask only for the channels the checks actually read.  Requesting S21
    # from a one-port device would be refused outright, and a check on S11
    # would then never run — the operator would see a failure to acquire
    # instead of the answer to the question they asked.
    channels = _channels_for(args.limits)
    try:
        request = SweepRequest(
            start_hz=parse_frequency(args.start),
            stop_hz=parse_frequency(args.stop),
            points=args.points,
            channels=channels,
            average=args.average,
            name=args.name or "cli sweep",
        )
    except ValueError as error:
        raise CliError(str(error), EXIT_INVALID) from error
    try:
        preferences = ScanPreferences(
            combined_text=bool(getattr(args, "combined_text", False)),
            binary_scan=bool(getattr(args, "binary_scan", False)),
        )
        with open_device(device, preferences=preferences) as session:
            return session.sweep(request, cancel=cancel)
    except UnsupportedError as error:
        raise CliError(str(error), EXIT_UNAVAILABLE) from error
    except CancelledError as error:
        raise CliError(str(error), EXIT_INTERRUPTED) from error
    except AcquisitionError as error:
        raise CliError(str(error), EXIT_DEVICE) from error


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m NanoVNASaver.cli",
        description=(
            "Sweep a simulated or replayed device and check limits without"
            " a display."
        ),
        epilog=(
            "exit codes: 0 pass, 1 failed limit, 2 invalid input,"
            " 3 unavailable, 4 device error, 130 interrupted"
        ),
    )
    source = parser.add_argument_group("source")
    source.add_argument(
        "--input",
        help="Touchstone file to check instead of sweeping",
    )
    source.add_argument(
        "--scenario",
        default="antenna",
        help="simulated scenario to sweep (default: antenna)",
    )
    source.add_argument("--seed", type=int, default=None)
    sweep = parser.add_argument_group("sweep")
    sweep.add_argument("--start", default="1M", help="start (default 1M)")
    sweep.add_argument("--stop", default="100M", help="stop (default 100M)")
    sweep.add_argument("--points", type=int, default=101)
    sweep.add_argument("--average", type=int, default=1)
    sweep.add_argument("--name", default="", help="name for this run")
    scan = parser.add_argument_group("scan read path")
    scan.add_argument(
        "--combined-text",
        action="store_true",
        help=(
            "try one masked text scan for all channels (unverified on"
            " hardware; falls back to per-channel reads)"
        ),
    )
    scan.add_argument(
        "--binary-scan",
        action="store_true",
        help=(
            "validate the binary scan frame instead of trusting it"
            " (unverified on hardware; falls back to per-channel reads)"
        ),
    )
    sequence = parser.add_argument_group("sequence")
    sequence.add_argument(
        "--sequence",
        help="run a versioned test sequence (.nvseq) instead of one sweep",
    )
    sequence.add_argument(
        "--serial", default="", help="serial number for {serial} substitutions"
    )
    sequence.add_argument(
        "--unit-name", default="", help="unit name for {name} substitutions"
    )
    sequence.add_argument(
        "--output-dir",
        default=".",
        help="where a sequence may write (default: the sequence directory)",
    )
    out = parser.add_argument_group("output")
    out.add_argument("--limits", help="limit set (.nvlimits) to evaluate")
    out.add_argument("--touchstone", help="write the sweep as Touchstone")
    out.add_argument(
        "--json",
        dest="json_path",
        help="write a machine-readable result to this path",
    )
    out.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output files (refused otherwise)",
    )
    out.add_argument(
        "--log",
        help="append a verdict record for this run to a CSV or .jsonl file",
    )
    out.add_argument(
        "--serial-number",
        default="",
        help="unit serial recorded in a --log or --sequence record",
    )
    out.add_argument(
        "--stdout-format",
        choices=("text", "json"),
        default="text",
        help="human summary or machine-readable result on stdout",
    )
    return parser


def _refuse_existing(path: str, overwrite: bool) -> None:
    if path and os.path.exists(path) and not overwrite:
        raise CliError(
            f"{path} already exists; pass --overwrite to replace it",
            EXIT_INVALID,
        )


def _append_verdict_log(args, snapshot, evaluations) -> None:
    """Append one audit record; a write failure is reported, not hidden."""
    from .SweepLog import LogFormat, LogWriteError, SweepLog, record_for

    log = SweepLog(
        args.log,
        log_format=(
            LogFormat.JSONL
            if str(args.log).lower().endswith(".jsonl")
            else LogFormat.CSV
        ),
    )
    record = record_for(
        snapshot,
        run_id=f"cli-{os.getpid()}-{snapshot.id}",
        evaluations=evaluations,
        limit_identity=os.path.abspath(args.limits) if args.limits else "",
        serial=args.serial_number,
        timestamp=None,
    )
    try:
        log.append(record)
    except LogWriteError as error:
        raise CliError(f"cannot log the verdict: {error}", EXIT_INVALID) from (
            error
        )


def _run_sequence(args) -> int:
    """Run a sequence document and report its record."""
    from .Sequence import (
        SequenceError,
        Substitutions,
        load_sequence,
    )
    from .SequenceActions import DeviceProvider, run_device_sequence

    try:
        sequence = load_sequence(
            args.sequence,
            substitutions=Substitutions(
                serial=args.serial, name=args.unit_name
            ),
        )
    except SequenceError as error:
        raise CliError(
            f"cannot load the sequence: {error}", EXIT_INVALID
        ) from (error)
    base = os.path.dirname(os.path.abspath(args.sequence))
    output = os.path.abspath(args.output_dir) if args.output_dir else base
    if args.output_dir and args.output_dir != ".":
        os.makedirs(output, exist_ok=True)
    else:
        output = base

    def factory():
        from .Hardware.Simulated import make_simulated_vna

        return make_simulated_vna(args.scenario, seed=args.seed)

    from .SweepLog import LogFormat, LogWriteError, SweepLog

    provider = DeviceProvider(factory=factory)
    log = None
    if args.log:
        log = SweepLog(
            args.log,
            log_format=(
                LogFormat.JSONL
                if str(args.log).lower().endswith(".jsonl")
                else LogFormat.CSV
            ),
        )
    try:
        record = run_device_sequence(
            sequence,
            provider,
            base_directory=base,
            output_directory=output,
            metadata={
                "app": "NanoVNASaver.cli",
                "scenario": args.scenario,
                "output_directory": output,
            },
            log=log,
        )
    except LogWriteError as error:
        raise CliError(
            f"the sequence ran but cannot log the verdict: {error}",
            EXIT_INVALID,
        ) from (error)
    except (SequenceError, ValueError) as error:
        raise CliError(
            f"the sequence could not run: {error}", EXIT_INVALID
        ) from (error)
    payload = record.to_dict()
    if args.json_path:

        def write(temporary: str) -> None:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")

        write_atomically(args.json_path, write)
    if args.stdout_format == "json":
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(record.sequence.name or "sequence")
        for item in record.steps:
            print(
                f"  {item.status.value.upper():<11} {item.kind.value}"
                f"{(': ' + item.detail) if item.detail else ''}"
            )
        print(f"  {record.describe()}")
    if not record.passed:
        # A stop caused by an unavailable step says nothing about the unit,
        # so only a step that actually failed may report a failed unit.
        return EXIT_FAILED if record.summary()["failed"] else EXIT_UNAVAILABLE
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cancel = CancelToken()
    try:
        if args.sequence:
            return _run_sequence(args)
        _refuse_existing(args.touchstone, args.overwrite)
        _refuse_existing(args.json_path, args.overwrite)
        acquired = _acquire(args, cancel)
        snapshot = getattr(acquired, "snapshot", acquired)
        read_path = getattr(acquired, "read_path", PATH_TRUSTED)
        rows: list[dict] = []
        evaluations: list = []
        outcome = CheckOutcome(failed=0, unavailable=0, passed=0)
        if args.limits:
            rows, outcome, evaluations = evaluate_limits(snapshot, args.limits)
        if args.touchstone:
            try:
                _snapshot_to_touchstone(snapshot, args.touchstone)
            except TouchstoneExportError as error:
                raise CliError(
                    f"cannot write {args.touchstone}: {error}", EXIT_INVALID
                ) from error
        payload = {
            "metadata": _metadata(
                snapshot,
                args.input or f"scenario:{args.scenario}",
                read_path,
            ),
            "checked": bool(args.limits),
            "summary": {
                "passed": outcome.passed,
                "failed": outcome.failed,
                "unavailable": outcome.unavailable,
            },
            "limits": rows,
        }
        if args.log:
            _append_verdict_log(args, snapshot, evaluations)
        if args.json_path:

            def write(temporary: str) -> None:
                with open(temporary, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True)
                    stream.write("\n")

            write_atomically(args.json_path, write)
        if args.stdout_format == "json":
            json.dump(payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            print(snapshot.name or "sweep")
            print(
                f"  {len(snapshot.available)} channel(s),"
                f" reference {snapshot.reference_impedance:g} ohm,"
                f" {'SIMULATED' if snapshot.simulated else 'measured'}"
            )
            for row in rows:
                print(f"  {row['verdict'].upper():<11} {row['rule_label']}")
            if args.limits:
                print(f"  {outcome.describe()}")
        return outcome.exit_code
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.code
    except KeyboardInterrupt:
        cancel.cancel()
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except DeviceError as error:
        print(f"device error: {error}", file=sys.stderr)
        return EXIT_DEVICE
    except OSError as error:
        # Exit status 1 is reserved for a failed limit check, and an unhandled
        # exception would look exactly like one to a calling script.
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
