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
"""Concrete handlers for test sequences (AUTO-03/AUTO-02).

``Sequence`` defines the schema and the run loop but knows nothing about
acquisition or file formats.  This module supplies the actions, so a
sequence can actually be executed, and the CLI can run one.

Every handler returns ``(status, detail, artifact)`` and never raises for an
expected condition: a missing sweep is ``UNAVAILABLE`` with a reason, not an
exception that aborts the whole record.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Mapping

from .Acquisition import (
    AcquisitionError,
    CancelledError,
    SweepRequest,
    UnsupportedError,
    open_device,
)
from .Limits import LimitError, Verdict
from .Limits import load as load_limits
from .Sequence import ActionKind, Handler, RunContext, StepStatus, TestSequence
from .Touchstone import Touchstone, TouchstoneExportError

__all__ = [
    "DeviceProvider",
    "handler_map",
    "run_device_sequence",
    "sequence_log_record",
]


@dataclass
class DeviceProvider:
    """Opens a device on demand and keeps it open across the steps.

    A sequence takes several sweeps; reopening (and reconnecting) the
    instrument for each one would be slow and, on real hardware, can change
    the reported calibration state between steps.
    """

    factory: Callable[[], object]
    session: object | None = None
    simulated: bool = False

    def get(self):
        if self.session is None:
            device = self.factory()
            self.session = open_device(device)
            self.simulated = bool(getattr(device, "simulated", False))
        return self.session

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None


def _parse_frequency(text: str) -> int:
    from .cli import parse_frequency

    return parse_frequency(text)


def _channels_for(path: str) -> tuple[str, ...]:
    from .cli import _channels_for as channels_for

    return channels_for(path)


def _sweep_handler(provider: DeviceProvider):
    def handle(step, context: RunContext):
        try:
            session = provider.get()
        except AcquisitionError as error:
            return StepStatus.FAILED, f"cannot open the device: {error}", ""
        # Ask only for channels the later checks read *and* the device can
        # actually provide.  A one-port antenna therefore runs fine unless a
        # check genuinely needs S21, in which case that check reports
        # UNAVAILABLE with the reason rather than the sweep failing for a
        # channel nobody asked about.
        needed = _channels_for_source(context)
        offered = tuple(
            name for name in needed if name in session.supported_channels
        )
        if not offered:
            return (
                StepStatus.UNAVAILABLE,
                f"the device provides none of {', '.join(needed)}",
                "",
            )
        try:
            request = SweepRequest(
                start_hz=_parse_frequency(str(step.fields["start"])),
                stop_hz=_parse_frequency(str(step.fields["stop"])),
                points=int(step.fields.get("points") or 101),
                channels=offered,
                average=int(step.fields.get("average") or 1),
                name=context.substitutions.apply(
                    str(step.fields.get("name") or ""), step.describe()
                ),
            )
        except ValueError as error:
            return StepStatus.FAILED, f"invalid sweep: {error}", ""
        try:
            context.snapshot = session.sweep(request).snapshot
        except UnsupportedError as error:
            return StepStatus.UNAVAILABLE, str(error), ""
        except CancelledError as error:
            # A cancellation is not a failure to diagnose; the audit record
            # says "cancelled" so nobody reads it as a bad unit.
            context.extras["cancelled"] = True
            return StepStatus.FAILED, str(error), ""
        except AcquisitionError as error:
            return StepStatus.FAILED, f"device error: {error}", ""
        # Freeze the snapshot at acquisition so a later failing step still
        # logs what was actually measured.
        context.extras["snapshot"] = context.snapshot
        return (
            StepStatus.OK,
            f"acquired {request.describe()}",
            "",
        )

    return handle


def _channels_for_source(context: RunContext) -> tuple[str, ...]:
    """Channels the remaining steps will read."""
    from .Measurement import S11, S21

    needed: set[str] = set()
    for step in context.sequence.steps:
        if step.kind is not ActionKind.CHECK:
            continue
        try:
            limit_set = load_limits(
                os.path.join(
                    context.base_directory, str(step.fields.get("limits", ""))
                )
            )
        except LimitError:
            continue
        needed.update(rule.metric.parameter for rule in limit_set.rules)
    ordered = tuple(name for name in (S11, S21) if name in needed)
    return ordered or (S11, S21)


def _check_handler():
    def handle(step, context: RunContext):
        if context.snapshot is None:
            return StepStatus.UNAVAILABLE, "no sweep to check", ""
        path = os.path.join(context.base_directory, str(step.fields["limits"]))
        try:
            limit_set = load_limits(path)
        except LimitError as error:
            return StepStatus.FAILED, f"cannot read limits: {error}", ""
        evaluations = limit_set.evaluate(context.snapshot, trace_id="sequence")
        # Freeze the verdict at this boundary: a later edit to the limit
        # catalogue must not relabel a record written from this run.
        context.extras["evaluations"] = list(evaluations)
        context.extras["limit_identity"] = path
        failed = [item for item in evaluations if item.verdict is Verdict.FAIL]
        unavailable = [
            item for item in evaluations if item.verdict is Verdict.UNAVAILABLE
        ]
        passed = [item for item in evaluations if item.verdict is Verdict.PASS]
        detail = (
            f"{len(passed)} pass, {len(failed)} fail,"
            f" {len(unavailable)} unavailable"
        )
        if failed:
            return (
                StepStatus.FAILED,
                f"{detail}: {failed[0].describe()}",
                "",
            )
        if unavailable:
            return (
                StepStatus.UNAVAILABLE,
                f"{detail}: {unavailable[0].describe()}",
                "",
            )
        return StepStatus.OK, detail, ""

    return handle


def _save_handler():
    def handle(step, context: RunContext):
        if context.snapshot is None:
            return StepStatus.UNAVAILABLE, "no sweep to save", ""
        try:
            path = context.path_for(step)
        except Exception as error:  # SequenceFormatError
            return StepStatus.FAILED, str(error), ""
        touchstone = Touchstone(path)
        touchstone.s11 = list(context.snapshot.corrected.get("S11", ()))
        touchstone.s21 = list(context.snapshot.corrected.get("S21", ()))
        parameter_count = 4 if touchstone.s21 else 1
        expected = ".s2p" if parameter_count == 4 else ".s1p"
        if not path.lower().endswith(expected):
            return (
                StepStatus.FAILED,
                f"this capture needs a {expected} file, not {path}",
                "",
            )
        try:
            touchstone.save(parameter_count)
        except (TouchstoneExportError, OSError) as error:
            return StepStatus.FAILED, f"cannot write {path}: {error}", ""
        return StepStatus.OK, f"wrote {os.path.basename(path)}", path

    return handle


def _report_handler():
    """Write a small JSON record beside the measurement.

    Deliberately JSON rather than a rendered page: a sequence runs
    unattended, so its output must be inspectable by another program, and
    the pretty report belongs to the GUI where someone is watching.
    """

    def handle(step, context: RunContext):
        if context.snapshot is None:
            return StepStatus.UNAVAILABLE, "no sweep to report", ""
        try:
            path = context.path_for(step)
        except Exception as error:
            return StepStatus.FAILED, str(error), ""
        snapshot = context.snapshot
        payload = {
            "title": str(step.fields.get("title") or context.sequence.name),
            "sequence": {
                "name": context.sequence.name,
                "identity": context.sequence.identity(),
            },
            "serial": context.substitutions.serial,
            "metadata": context.metadata,
            "measurement": {
                "name": snapshot.name,
                "simulated": snapshot.simulated,
                "completion": snapshot.provenance.completion.value,
                "reference_impedance": snapshot.reference_impedance,
                "channels": sorted(snapshot.available),
            },
            "steps": [
                item.to_dict()
                for item in (context.result.steps if context.result else [])
            ],
        }
        try:
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except OSError as error:
            return StepStatus.FAILED, f"cannot write {path}: {error}", ""
        return StepStatus.OK, f"wrote {os.path.basename(path)}", path

    return handle


def _log_handler():
    def handle(step, context: RunContext):
        message = context.text(step, "message")
        return StepStatus.OK, message, ""

    return handle


def handler_map(provider: DeviceProvider) -> Mapping[ActionKind, Handler]:
    """Every action a sequence may use, bound to ``provider``."""
    return {
        ActionKind.SWEEP: _sweep_handler(provider),
        ActionKind.CHECK: _check_handler(),
        ActionKind.SAVE: _save_handler(),
        ActionKind.REPORT: _report_handler(),
        ActionKind.LOG: _log_handler(),
    }


def sequence_log_record(sequence: TestSequence, result, extras):
    """One audit record for a finished sequence run.

    A run that stopped early, or that never acquired anything, is recorded
    as a failure: the file must not be able to show a pass for a unit whose
    own program did not finish.
    """
    from .SweepLog import failed_record, record_for

    snapshot = extras.get("snapshot")
    run_id = (
        f"{sequence.substitutions.serial or 'unit'}-{sequence.identity()[:12]}"
    )
    if snapshot is None:
        return failed_record(
            run_id=run_id,
            reason="the sequence did not acquire a measurement",
            serial=sequence.substitutions.serial,
            dut=sequence.substitutions.name,
            processing_identity=sequence.identity(),
        )
    record = record_for(
        snapshot,
        run_id=run_id,
        evaluations=extras.get("evaluations", ()),
        limit_identity=extras.get("limit_identity", ""),
        serial=sequence.substitutions.serial,
        dut=sequence.substitutions.name,
        processing_identity=sequence.identity(),
    )
    if (result is None or not result.passed) and record.is_pass:
        # The measurement passed its checks, but the rest of the program did
        # not run or did not succeed; calling that a pass would hide the
        # difference between a good unit and a good script.
        from .SweepLog import failed_record

        steps = result.steps if result is not None else ()
        first = next((step for step in steps if not step.success), None)
        if first is None:
            reason = "the run did not complete"
        else:
            reason = (
                f"step {first.index + 1} ({first.kind.value})"
                f" {first.status.value}: {first.detail}"
            )
        return failed_record(
            run_id=run_id,
            reason=reason,
            serial=sequence.substitutions.serial,
            dut=sequence.substitutions.name,
            processing_identity=sequence.identity(),
        )
    return record


def run_device_sequence(  # noqa: PLR0913
    sequence: TestSequence,
    provider: DeviceProvider,
    *,
    base_directory: str,
    output_directory: str,
    metadata: Mapping | None = None,
    log=None,
):
    """Run ``sequence`` against a device and close it afterwards.

    The device is closed in a ``finally`` so a failed step still releases
    the instrument; leaving a serial port held would block the next unit.

    ``log``, when given a :class:`SweepLog`, receives exactly one record for
    the run -- whatever happened, including a run that stopped early. A log
    failure is raised after the device is closed, so the measurement is
    complete and released before the operator learns the audit trail has a
    hole.
    """
    from .Sequence import run_sequence

    extras: dict = {}
    record = None
    try:
        record = run_sequence(
            sequence,
            handler_map(provider),
            base_directory=base_directory,
            output_directory=output_directory,
            metadata=metadata,
            extras=extras,
        )
        return record
    finally:
        provider.close()
        if log is not None:
            log.append(sequence_log_record(sequence, record, extras))
