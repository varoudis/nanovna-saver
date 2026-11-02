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
"""Touchstone replay scenario (REPLAY-01).

A replay scenario turns a Touchstone file into the response of the
simulated device, so a recorded measurement can be swept through the normal
simulator/worker pipeline (segments, averaging, cancellation, calibration
state, provenance) instead of through a separate offline trace import.

Interpolation contract
----------------------
* A frequency present in the file yields that sample unchanged: the real
  and imaginary parts are returned bit-exactly.
* Between two file samples the real part and the imaginary part are
  interpolated linearly and independently - ordinary complex linear
  interpolation on the real/imaginary pair.
* Outside the file's span nothing is produced.  ``ReplayOutOfSpanError`` is
  raised; held endpoint values, zeros or any other invented sample are
  never returned.

Honesty constraints
-------------------
* A one-port file replays S11 only.  S21 stays absent from every published
  snapshot; it is never replaced by zero-valued samples.
* Only S11 and S21 are *published*, because that is the whole channel set
  the sweep pipeline carries.  A two-port file's S12/S22 are retained but
  never presented as forward measurements: they are available only through
  :meth:`ReplayData.reversed`, which re-labels them the way a physical
  reverse sweep is re-labelled (RF-04), and the resulting view says its
  ports were exchanged.
* A non-50 ohm one-port file is accepted through an explicit, recorded
  renormalization: the file's own impedance is kept in
  ``source_impedance``, the samples are converted to the device's
  reference with ``Processing.renormalize_gamma``, and ``converted`` plus
  the summary/provenance text say so.  A *two-port* non-50 ohm file is
  still refused, because renormalizing S21 needs all four S-parameters.
* Touchstone 2.x is refused with an explicit error rather than parsed
  leniently by the 1.x reader.
"""

from __future__ import annotations

import logging
import math
from bisect import bisect_left
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

from ..Processing import renormalize_gamma
from ..RFTools import Datapoint
from ..Touchstone import PARAMETERS, Touchstone
from .Capability import SourceSpan, SweepCapability
from .SimulatedScenarios import REFERENCE_IMPEDANCE, ScenarioResult

logger = logging.getLogger(__name__)

#: Largest file accepted, before anything is parsed.
MAX_REPLAY_BYTES = 32 * 1024 * 1024

#: Largest number of samples per channel accepted after parsing.
MAX_REPLAY_POINTS = 250_000

#: Supported Touchstone file types and their number of ports.
SUPPORTED_SUFFIXES: dict[str, int] = {".s1p": 1, ".s2p": 2}

#: Channels the sweep pipeline can carry, in worker order.
CHANNEL_S11 = "data 0"
CHANNEL_S21 = "data 1"

#: S-parameters a file may contribute to a replay.
REPLAYABLE_PARAMETERS: tuple[str, ...] = ("S11", "S21")


class ReplayError(Exception):
    """A file cannot be used for replay; the message is meant for the user."""


class ReplayOutOfSpanError(ReplayError, RuntimeError):
    """A frequency outside the file's own span was requested.

    Deliberately not a ``ValueError``: the worker retries value errors as
    transient device trouble, while this is a deterministic request error.
    """


class ReplayReferenceImpedanceError(ReplayError, ValueError):
    """The file's reference impedance differs from the device's."""


class ReplayChannelError(ReplayError, RuntimeError):
    """A channel that the file does not provide was requested."""


def _check_finite(value: float, what: str, filename: str) -> float:
    if not math.isfinite(value):
        raise ReplayError(f"{filename}: {what} is not a finite number")
    return value


def _validate_channel(
    parameter: str,
    samples: list[Datapoint],
    expected_freqs: tuple[int, ...],
    filename: str,
) -> tuple[Datapoint, ...]:
    """Validate one channel of a loaded file and freeze it."""
    if len(samples) != len(expected_freqs):
        raise ReplayError(
            f"{filename}: {parameter} holds {len(samples)} samples but S11"
            f" holds {len(expected_freqs)}"
        )
    frozen = []
    for index, dp in enumerate(samples):
        if dp.freq != expected_freqs[index]:
            raise ReplayError(
                f"{filename}: {parameter} and S11 disagree on the frequency"
                f" of sample {index} ({dp.freq} Hz vs"
                f" {expected_freqs[index]} Hz)"
            )
        _check_finite(dp.re, f"{parameter} sample {index} real part", filename)
        _check_finite(dp.im, f"{parameter} sample {index} imag part", filename)
        frozen.append(Datapoint(dp.freq, dp.re, dp.im))
    return tuple(frozen)


@dataclass(frozen=True)
class ReplayData:
    """A validated Touchstone file, ready to be replayed.

    ``s11``/``s21`` are exact copies of the file's samples (real/imaginary
    pairs in hertz).  ``s21`` is empty for a one-port file, which is what
    keeps such a replay S11-only all the way to export.
    """

    path: str
    kind: str
    reference_impedance: float
    s11: tuple[Datapoint, ...]
    s21: tuple[Datapoint, ...] = ()
    ignored_parameters: tuple[str, ...] = ()
    #: The file's own reverse channels, when it describes a complete
    #: two-port.  Empty for a one-port file, which is what keeps "the file
    #: has no S12" distinguishable from "S12 is zero".
    s12: tuple[Datapoint, ...] = ()
    s22: tuple[Datapoint, ...] = ()
    #: The impedance the file *stated*.  Equal to ``reference_impedance``
    #: unless a conversion was applied, in which case the samples above are
    #: stated at the device's impedance and this records what they came
    #: from, so the conversion is never invisible.
    source_impedance: float = 0.0
    converted: bool = False
    #: Set on a reverse view produced by :meth:`reversed`, so provenance can
    #: say the ports were exchanged rather than silently reorder samples.
    reversed_from: str = ""

    @cached_property
    def frequencies(self) -> tuple[int, ...]:
        return tuple(dp.freq for dp in self.s11)

    @property
    def has_s21(self) -> bool:
        return bool(self.s21)

    @property
    def start(self) -> int:
        return self.frequencies[0]

    @property
    def stop(self) -> int:
        return self.frequencies[-1]

    @property
    def name(self) -> str:
        return Path(self.path).name

    def channel(self, parameter: str) -> tuple[Datapoint, ...]:
        for name, samples in (
            ("S11", self.s11),
            ("S21", self.s21),
            ("S12", self.s12),
            ("S22", self.s22),
        ):
            if parameter == name:
                return samples
        raise KeyError(parameter)

    @property
    def has_reverse(self) -> bool:
        """True when the file states S12 and S22 as well."""
        return bool(self.s12) and bool(self.s22)

    def reversed(self) -> "ReplayData":
        """The same file as measured with the DUT's ports exchanged.

        SIM-02: a complete two-port file describes both orientations, so a
        reverse capture can be *derived* rather than invented.  Ports
        swapped means the driven port's reflection becomes the other
        port's reflection and the forward transmission becomes the reverse
        one -- the same re-labelling RF-04 performs on a physical reverse
        sweep.  A one-port file cannot be reversed, because the reverse
        channels were never in it.
        """
        if not self.has_reverse:
            raise ReplayChannelError(
                f"{self.name} states no S12/S22, so it cannot be replayed"
                " in the reverse orientation without inventing them"
            )
        return replace(
            self,
            s11=self.s22,
            s21=self.s12,
            s12=self.s21,
            s22=self.s11,
            reversed_from=self.path,
        )

    def is_exact(self, freq: float) -> bool:
        """True when ``freq`` is one of the file's own sample frequencies."""
        index = bisect_left(self.frequencies, freq)
        return index < len(self.frequencies) and self.frequencies[index] == freq

    def value_at(self, parameter: str, freq: float) -> complex:
        """Sample ``parameter`` at ``freq``, exactly or interpolated.

        Raises ``ReplayOutOfSpanError`` rather than extrapolating.
        """
        samples = self.channel(parameter)
        if not samples:
            raise ReplayChannelError(
                f"{self.name} provides no {parameter} channel"
            )
        freqs = self.frequencies
        index = bisect_left(freqs, freq)
        if index < len(freqs) and freqs[index] == freq:
            dp = samples[index]
            return complex(dp.re, dp.im)
        if index == 0 or index >= len(freqs):
            raise ReplayOutOfSpanError(
                f"{freq:.0f} Hz is outside the replay range {self.start} Hz"
                f" to {self.stop} Hz of {self.name}; replay does not"
                " extrapolate beyond the file"
            )
        lower, upper = index - 1, index
        f_lower, f_upper = freqs[lower], freqs[upper]
        t = (freq - f_lower) / (f_upper - f_lower)
        low_dp, high_dp = samples[lower], samples[upper]
        return complex(
            low_dp.re + t * (high_dp.re - low_dp.re),
            low_dp.im + t * (high_dp.im - low_dp.im),
        )

    def summary(self) -> str:
        """One-line, human-readable description of the replayed file."""
        channels = "S11 and S21" if self.has_s21 else "S11 only"
        text = (
            f"{self.kind} · {len(self.s11)} points ·"
            f" {self.start} Hz to {self.stop} Hz · {channels}"
        )
        if self.reversed_from:
            text += " · ports exchanged"
        if self.has_reverse:
            text += " · S12/S22 available for reversal"
        if self.converted:
            text += f" · renormalized from {self.source_impedance:g} ohm"
        if self.ignored_parameters:
            text += f" · {', '.join(self.ignored_parameters)} ignored"
        return text

    def provenance_text(self) -> str:
        """Replay source recorded in snapshots, exports and reports."""
        channels = "S11 and S21" if self.has_s21 else "S11 only (no S21)"
        text = (
            f"replay of {self.path} ({self.kind}, {channels},"
            " complex linear interpolation, no extrapolation)"
        )
        if self.reversed_from:
            text += (
                "; ports exchanged (the file's S22/S12 presented as S11/S21)"
            )
        if self.converted:
            text += (
                f"; renormalized from the file's"
                f" {self.source_impedance:g} ohm to"
                f" {self.reference_impedance:g} ohm"
            )
        if self.ignored_parameters:
            text += (
                f"; {', '.join(self.ignored_parameters)} from the file are"
                " not replayed"
            )
        return text


def _validate_layout(
    dataset: Touchstone, ports: int, filename: str
) -> tuple[str, ...]:
    present = dataset.present_parameters()
    if "S11" not in present:
        raise ReplayError(
            f"{filename}: no S11 samples found; a replay needs at least a"
            " one-port S11 column"
        )
    expected = {
        1: {("S11",)},
        2: {("S11", "S21"), ("S11", "S21", "S12", "S22")},
    }[ports]
    if present not in expected:
        names = ", ".join(present) or "none"
        wanted = " or ".join(", ".join(e) for e in sorted(expected))
        raise ReplayError(
            f"{filename}: {len(present)} S-parameter column(s)"
            f" ({names}) cannot be replayed as a {ports}-port file;"
            f" expected {wanted}"
        )
    return present


def load_replay_file(path: str | Path) -> ReplayData:
    """Load and validate ``path``; raise ``ReplayError`` when unusable.

    Validation covers the file type, size, emptiness, option line, column
    layout, S11 presence, channel lengths, frequency ordering (strictly
    ascending, positive, no duplicates), finite sample values and the
    reference impedance.
    """
    file = Path(path).expanduser()
    ports = SUPPORTED_SUFFIXES.get(file.suffix.lower())
    if ports is None:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ReplayError(
            f"{file.name}: unsupported file type; choose a Touchstone file"
            f" ({supported})"
        )
    if not file.is_file():
        raise ReplayError(f"{file} does not exist or is not a file")
    try:
        size = file.stat().st_size
    except OSError as exc:
        raise ReplayError(f"{file} cannot be read: {exc}") from exc
    if size == 0:
        raise ReplayError(f"{file.name} is empty; it holds no samples")
    if size > MAX_REPLAY_BYTES:
        raise ReplayError(
            f"{file.name} is {size} bytes; at most {MAX_REPLAY_BYTES} bytes"
            " can be replayed"
        )
    try:
        with file.open(encoding="utf-8") as infile:
            text = infile.read(MAX_REPLAY_BYTES + 1)
    except OSError as exc:
        raise ReplayError(f"{file} cannot be read: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ReplayError(
            f"{file.name} is not valid UTF-8 text: {exc}"
        ) from exc
    if len(text) > MAX_REPLAY_BYTES:
        raise ReplayError(
            f"{file.name} exceeds the {MAX_REPLAY_BYTES} byte replay limit"
        )
    if not text.strip():
        raise ReplayError(f"{file.name} is empty; it holds no samples")

    filename = file.name
    dataset = Touchstone(str(file))
    try:
        dataset.loads_strict(text)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ReplayError(
            f"{filename}: not a usable Touchstone 1.x file: {exc}"
        ) from exc

    if dataset.opts.parameter != "s":
        raise ReplayError(
            f"{filename} holds"
            f" {dataset.opts.parameter.upper()}-parameters; only"
            " S-parameters can be replayed"
        )

    present = _validate_layout(dataset, ports, filename)

    reference = float(dataset.opts.resistance)
    _check_finite(reference, "reference impedance", filename)
    if reference <= 0:
        raise ReplayError(
            f"{filename}: reference impedance must be positive,"
            f" not {reference:g}"
        )
    converted = not math.isclose(reference, REFERENCE_IMPEDANCE, rel_tol=1e-9)
    if converted:
        # SIM-02: a non-50 ohm file is accepted only through an explicit,
        # recorded conversion.  A two-port file would need all four terms to
        # renormalize (S21 alone cannot be converted), so only one-port
        # data is converted; the alternative -- converting S11 and passing
        # S21 through untouched -- would mix two reference impedances in
        # one snapshot, which is exactly the silent corruption to avoid.
        if ports != 1:
            raise ReplayReferenceImpedanceError(
                f"{filename} is referenced to {reference:g} ohm, but the"
                f" simulated device is a {REFERENCE_IMPEDANCE:g} ohm system."
                f" Renormalizing a {ports}-port file needs all four"
                " S-parameters and this file does not carry them in a form"
                " replay preserves; export it at 50 ohm first (for example"
                " with `scikit-rf`'s `Network.renormalize`)."
            )

    s11 = list(dataset.s11)
    if not s11:
        raise ReplayError(f"{filename} contains no samples")
    if len(s11) > MAX_REPLAY_POINTS:
        raise ReplayError(
            f"{filename} holds {len(s11)} samples; at most"
            f" {MAX_REPLAY_POINTS} can be replayed"
        )

    previous = -1
    for index, dp in enumerate(s11):
        if dp.freq <= 0:
            raise ReplayError(
                f"{filename}: sample {index} has non-positive frequency"
                f" {dp.freq} Hz"
            )
        if dp.freq <= previous:
            raise ReplayError(
                f"{filename}: duplicate or non-ascending frequency"
                f" {dp.freq} Hz at sample {index}; each frequency may appear"
                " only once"
            )
        previous = dp.freq
    freqs = tuple(dp.freq for dp in s11)

    frozen_s11 = _validate_channel("S11", s11, freqs, filename)
    frozen_s21: tuple[Datapoint, ...] = ()
    if "S21" in present:
        frozen_s21 = _validate_channel(
            "S21", list(dataset.s21), freqs, filename
        )
    source_impedance = reference
    if converted:
        gamma = [
            renormalize_gamma(reference, REFERENCE_IMPEDANCE, dp.z)
            for dp in frozen_s11
        ]
        frozen_s11 = tuple(
            Datapoint(dp.freq, value.real, value.imag)
            for dp, value in zip(frozen_s11, gamma, strict=True)
        )
        logger.info(
            "Renormalized replay source %s from %g ohm to %g ohm",
            file,
            reference,
            REFERENCE_IMPEDANCE,
        )
        reference = REFERENCE_IMPEDANCE

    frozen_s12: tuple[Datapoint, ...] = ()
    frozen_s22: tuple[Datapoint, ...] = ()
    if "S12" in present and "S22" in present:
        # A file that states both reverse channels describes a complete
        # two-port, so RF-04's reversal can derive a reverse capture from
        # it.  They are kept but never published as forward channels.
        frozen_s12 = _validate_channel(
            "S12", list(dataset.s12), freqs, filename
        )
        frozen_s22 = _validate_channel(
            "S22", list(dataset.s22), freqs, filename
        )

    # A reverse pair that was kept for RF-04 reversal is reported as
    # available, not as ignored; only what the file holds and replay cannot
    # use at all is listed here.
    carried = set(REPLAYABLE_PARAMETERS)
    if frozen_s12 and frozen_s22:
        carried |= {"S12", "S22"}
    ignored = tuple(p for p in PARAMETERS if p not in carried and p in present)
    kind = "S1P" if ports == 1 else "S2P"
    logger.info(
        "Loaded replay source %s: %s, %d points",
        file,
        kind,
        len(frozen_s11),
    )
    return ReplayData(
        path=str(file),
        kind=kind,
        reference_impedance=reference,
        s11=frozen_s11,
        s21=frozen_s21,
        ignored_parameters=ignored,
        s12=frozen_s12,
        s22=frozen_s22,
        source_impedance=source_impedance,
        converted=converted,
    )


@dataclass(frozen=True)
class ReplayScenario:
    """Callable scenario replaying one Touchstone file.

    ``scenario_key``/``display_name`` are read by ``SimulatedDevice`` for
    provenance, so a replayed sweep is labelled as simulated *and* as
    derived from a file.
    """

    scenario_key = "replay"
    display_name = "Touchstone replay"

    data: ReplayData

    @property
    def supported_channels(self) -> tuple[str, ...]:
        if self.data.has_s21:
            return (CHANNEL_S11, CHANNEL_S21)
        return (CHANNEL_S11,)

    def __call__(self, freq: float) -> ScenarioResult:
        s11 = self.data.value_at("S11", freq)
        # The S21 slot is never published for a one-port file: the device
        # rejects the channel before sampling (see ``supported_channels``).
        s21 = (
            self.data.value_at("S21", freq)
            if self.data.has_s21
            else complex(0.0, 0.0)
        )
        return ScenarioResult(s11, s21)

    def preflight(self, start: int, stop: int) -> str | None:
        """Return a user-facing problem, or ``None`` when a range is usable.

        Called by the worker for every segment it is about to request, so a
        span that would need extrapolation is rejected before a single
        sample is read.
        """
        if stop < start:
            return (
                f"The sweep span is invalid: {start} Hz to {stop} Hz."
                " The stop frequency must not be below the start frequency."
            )
        data = self.data
        if start < data.start or stop > data.stop:
            return (
                f"The sweep {start} Hz to {stop} Hz is outside the replay"
                f" range {data.start} Hz to {data.stop} Hz of {data.name}."
                " Replay never extrapolates; set the sweep inside the"
                " file's own span."
            )
        return None

    def provenance_text(self) -> str:
        return self.data.provenance_text()

    def sweep_span(self) -> SourceSpan:
        """The file's own span, as a capability source.

        The text is reused verbatim in rejection messages, so a planner
        names the range the user has to stay inside.
        """
        return SourceSpan(
            self.data.start,
            self.data.stop,
            name=self.data.name,
            label="the replay range",
        )

    def capability(self, vna=None) -> SweepCapability:
        """The sweep capability of replaying this file.

        The span is the file's own range (never extrapolated), channels are
        the ones the file really holds, and the file's reference impedance
        is the one every sample is stated in.  When ``vna`` is given, its
        declared limits bound the result as well, so a planned range is
        never wider than the device that will request it.
        """
        span = self.sweep_span()
        if vna is None:
            return SweepCapability.from_vna(
                ReplayDeviceLimits(self.data), source=span
            )
        return SweepCapability.from_vna(vna, source=span)


class ReplayDeviceLimits:
    """Declared limits of a replay source, in the shape ``from_vna`` reads.

    A file holds no Nyquist grid, so there is no maximum point count to
    declare; only the span, the channels and the impedance are known.
    """

    def __init__(self, data: ReplayData) -> None:
        self.supported_channels = (
            (CHANNEL_S11, CHANNEL_S21) if data.has_s21 else (CHANNEL_S11,)
        )
        self.reference_impedance = data.reference_impedance
