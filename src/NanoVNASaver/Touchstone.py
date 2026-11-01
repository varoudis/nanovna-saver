#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
#  Copyright (C) 2020,2021 NanoVNA-Saver Authors
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
import cmath
import io
import logging
import math
from operator import attrgetter
from typing import Callable, ClassVar, List

from scipy.interpolate import interp1d

from .RFTools import Datapoint

logger = logging.getLogger(__name__)


def _is_finite(value) -> bool:
    """True when ``value`` is a real number Touchstone can carry."""
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


#: Canonical Touchstone column order for two-port data.
PARAMETERS: tuple[str, ...] = ("S11", "S21", "S12", "S22")

#: Supported file types, expressed as their number of S-parameter columns.
PARAMETERS_BY_COUNT: dict[int, tuple[str, ...]] = {
    1: ("S11",),
    4: PARAMETERS,
}

#: Parameters a file type cannot do without.
REQUIRED_BY_COUNT: dict[int, tuple[str, ...]] = {
    1: ("S11",),
    4: ("S11", "S21"),
}

#: Optional two-port parameters that may be written as zero placeholders.
PLACEHOLDER_CANDIDATES: tuple[str, ...] = ("S12", "S22")

#: Port description used in export comments.
PORTS_BY_COUNT: dict[int, str] = {1: "1-port", 4: "2-port"}

#: Prefix of the provenance comments this module writes.  Comments carrying
#: it are dropped before a new set is generated, so re-exporting the same
#: object does not stack stale notes.
EXPORT_NOTE_PREFIX = "! NanoVNASaver export:"

#: Leading text of the marker written when the data was simulated (SIM-02.2).
SIMULATED_MARKER = "SIMULATED DATA"


class TouchstoneExportError(Exception):
    """A dataset cannot be written as a consistent Touchstone file.

    Raised before any byte is written, so a rejected export never truncates
    or creates the output file.
    """


class UnsupportedParameterCountError(TouchstoneExportError, ValueError):
    """``parameter_count`` is not a supported Touchstone column count."""


class MissingParameterError(TouchstoneExportError, ValueError):
    """A channel required by the requested file type holds no samples."""


class ChannelLengthMismatchError(TouchstoneExportError, ValueError):
    """Two channels hold a different number of samples."""


class FrequencyMismatchError(TouchstoneExportError, LookupError):
    """Two channels do not share one frequency grid.

    Also a ``LookupError`` so callers written against the previous
    ``LookupError("Frequencies of sdata not correlated")`` keep working.
    """


class NonFiniteValueError(TouchstoneExportError, ValueError):
    """A sample that would be written is NaN or infinite."""


class Options:
    # Fun fact: In Touchstone 1.1 spec all params are optional unordered.
    # Just the line has to start with "#"
    UNIT_TO_FACTOR: ClassVar[dict[str, int]] = {
        "ghz": 10**9,
        "mhz": 10**6,
        "khz": 10**3,
        "hz": 10**0,
    }
    VALID_UNITS = UNIT_TO_FACTOR.keys()
    VALID_PARAMETERS = "syzgh"
    VALID_FORMATS = ("ma", "db", "ri")

    def __init__(
        self,
        unit: str = "GHZ",
        parameter: str = "S",
        t_format: str = "ma",
        resistance: int = 50,
    ):
        # set defaults
        assert unit.lower() in Options.VALID_UNITS
        assert parameter.lower() in Options.VALID_PARAMETERS
        assert t_format.lower() in Options.VALID_FORMATS
        assert resistance > 0
        self.unit = unit.lower()
        self.parameter = parameter.lower()
        self.format = t_format.lower()
        self.resistance = resistance

    @property
    def factor(self) -> int:
        return Options.UNIT_TO_FACTOR[self.unit]

    def __str__(self) -> str:
        return (
            f"# {self.unit} {self.parameter} {self.format} r {self.resistance}"
        ).upper()

    def parse(self, line: str):
        if not line.startswith("#"):
            raise TypeError(f"Not an option line: {line}")
        punit = pparam = pformat = presist = False
        params = iter(line[1:].lower().split())
        for p in params:
            if p in Options.VALID_UNITS and not punit:
                self.unit = p
                punit = True
            elif p in Options.VALID_PARAMETERS and not pparam:
                self.parameter = p
                pparam = True
            elif p in Options.VALID_FORMATS and not pformat:
                self.format = p
                pformat = True
            elif p == "r" and not presist:
                rstr = next(params)
                try:
                    self.resistance = int(rstr)
                except ValueError:
                    # Keep the exact value: the writer emits the reference
                    # impedance of the measurement, which need not be an
                    # integer.
                    logger.warning("Non integer resistance value: %s", rstr)
                    value = float(rstr)
                    self.resistance = (
                        int(value) if value.is_integer() else value
                    )
            else:
                raise TypeError(f"Illegal option line: {line}")


class Touchstone:
    FIELD_ORDER = ("11", "21", "12", "22")

    def __init__(self, filename: str = ""):
        self.filename = filename
        self.sdata: List[List[Datapoint]] = [
            [],
            [],
            [],
            [],
        ]  # at max 4 data pairs
        self.comments: list[str] = []
        #: Provenance of simulated data (SIM-02.2).  Kept as explicit fields
        #: rather than a comment so a reload can restore them structurally.
        self.simulated: bool = False
        self.simulated_scenario: str = ""
        self.opts = Options()
        self._interp: dict[str, dict[str, Callable]] = {}

    @property
    def s11(self) -> list[Datapoint]:
        return self.s("11")

    @s11.setter
    def s11(self, value: list[Datapoint]):
        self.sdata[0] = value

    @property
    def s12(self) -> list[Datapoint]:
        return self.s("12")

    @s12.setter
    def s12(self, value: list[Datapoint]):
        self.sdata[2] = value

    @property
    def s21(self) -> list[Datapoint]:
        return self.s("21")

    @s21.setter
    def s21(self, value: list[Datapoint]):
        self.sdata[1] = value

    @property
    def s22(self) -> list[Datapoint]:
        return self.s("22")

    @s22.setter
    def s22(self, value: list[Datapoint]):
        self.sdata[3] = value

    @property
    def r(self) -> int:
        return self.opts.resistance

    def s(self, name: str) -> list[Datapoint]:
        return self.sdata[Touchstone.FIELD_ORDER.index(name)]

    def s_freq(self, name: str, freq: int) -> Datapoint:
        return Datapoint(
            freq,
            float(self._interp[name]["real"](freq)),
            float(self._interp[name]["imag"](freq)),
        )

    def swap(self):
        self.sdata = [self.s22, self.s12, self.s21, self.s11]

    def min_freq(self) -> int:
        return self.s("11")[0].freq

    def max_freq(self) -> int:
        return self.s("11")[-1].freq

    def gen_interpolation(self):
        for i in Touchstone.FIELD_ORDER:
            freq = []
            real = []
            imag = []

            for dp in self.s(i):
                freq.append(dp.freq)
                real.append(dp.re)
                imag.append(dp.im)

            self._interp[i] = {
                "real": interp1d(
                    freq,
                    real,
                    kind="slinear",
                    bounds_error=False,
                    fill_value=(real[0], real[-1]),
                ),
                "imag": interp1d(
                    freq,
                    imag,
                    kind="slinear",
                    bounds_error=False,
                    fill_value=(imag[0], imag[-1]),
                ),
            }

    def gen_interpolation_s11(self):
        freq = []
        real = []
        imag = []
        for dp in self.s("11"):
            freq.append(dp.freq)
            real.append(dp.re)
            imag.append(dp.im)

        self._interp["11"] = {
            "real": interp1d(
                freq,
                real,
                kind="slinear",
                bounds_error=False,
                fill_value=(real[0], real[-1]),
            ),
            "imag": interp1d(
                freq,
                imag,
                kind="slinear",
                bounds_error=False,
                fill_value=(imag[0], imag[-1]),
            ),
        }

    def _parse_comments(self, fp) -> str:
        for ln in fp:
            line = ln.strip()
            if line.startswith("!"):
                logger.info(line)
                self.comments.append(line)
                continue
            return line
        return ""

    def _append_line_data(self, freq: int, data: list[str]):
        data_list = iter(self.sdata)
        vals = iter(data)
        for v in vals:
            if self.opts.format == "ri":
                next(data_list).append(
                    Datapoint(freq, float(v), float(next(vals)))
                )
            if self.opts.format == "ma":
                z = cmath.rect(float(v), math.radians(float(next(vals))))
                next(data_list).append(Datapoint(freq, z.real, z.imag))
            if self.opts.format == "db":
                z = cmath.rect(
                    10 ** (float(v) / 20), math.radians(float(next(vals)))
                )
                next(data_list).append(Datapoint(freq, z.real, z.imag))

    def load(self):
        logger.info("Attempting to open file %s", self.filename)
        try:
            with open(self.filename, encoding="utf-8") as infile:
                self.loads(infile.read())
        except IOError as e:
            logger.exception("Failed to open %s: %s", self.filename, e)

    def loads(self, s: str):
        """Parse touchstone 1.1 string input
        appends to existing sdata if Touchstone object exists
        """
        try:
            self._loads(s)
        except TypeError as e:
            logger.exception("Failed to parse %s: %s", self.filename, e)

    def loads_strict(self, s: str):
        """Parse like :meth:`loads` but propagate every parse error.

        Callers that must decide for themselves whether a dataset is
        usable (the replay scenario) need the original ``TypeError``,
        ``ValueError`` and ``IndexError`` instead of a logged, partial
        parse.
        """
        self._loads(s)

    def _loads(self, s: str):
        need_reorder = False
        with io.StringIO(s) as file:
            opts_line = self._parse_comments(file)
            self.opts.parse(opts_line)

            prev_freq = 0.0
            prev_len = 0
            for ln in file:
                line = ln.strip()
                # ignore empty lines (even if not specified)
                if line == "":
                    continue
                # accept comment lines after header
                if line.startswith("!"):
                    logger.warning("Comment after header: %s", line)
                    self.comments.append(line)
                    continue

                # ignore comments at data end
                data = line.split("!")[0].split()
                freq, data = round(float(data[0]) * self.opts.factor), data[1:]
                data_len = len(data)
                if data_len % 2 != 0:
                    raise TypeError("Data values aren't pairs: " + line)

                # consistency checks
                if freq <= prev_freq:
                    logger.warning("Frequency not ascending: %s", line)
                    need_reorder = True
                prev_freq = freq

                if prev_len == 0:
                    prev_len = data_len
                elif data_len != prev_len:
                    raise TypeError(f"Inconsistent number of pairs: {line}")

                self._append_line_data(freq, data)
            if need_reorder:
                logger.warning("Reordering data")
                for datalist in self.sdata:
                    datalist.sort(key=attrgetter("freq"))
        self._restore_simulated_marker()

    def _restore_simulated_marker(self) -> None:
        """Recover simulation provenance from the file's own comments."""
        for comment in self.comments:
            if SIMULATED_MARKER not in comment:
                continue
            self.simulated = True
            start = comment.find("(scenario ")
            if start != -1:
                end = comment.find(")", start)
                if end != -1:
                    self.simulated_scenario = comment[
                        start + len("(scenario ") : end
                    ].split(")")[0]
            return

    def channel(self, parameter: str) -> list[Datapoint]:
        """Return the samples of ``parameter`` (``"S11"`` ... ``"S22"``)."""
        return self.sdata[PARAMETERS.index(parameter)]

    def present_parameters(self) -> tuple[str, ...]:
        """Parameters holding at least one sample, in canonical order."""
        return tuple(p for p in PARAMETERS if self.channel(p))

    def placeholder_parameters(
        self, parameter_count: int = 4
    ) -> tuple[str, ...]:
        """Optional parameters an export would fill with zero placeholders.

        Only two-port files have optional columns; S11 (and S21 for a
        two-port file) must be present or the export is rejected.  A
        one-port export has no placeholders at all.
        """
        if parameter_count != 4:
            return ()
        return tuple(p for p in PLACEHOLDER_CANDIDATES if not self.channel(p))

    def _validated_channels(
        self, parameter_count: int
    ) -> dict[str, list[Datapoint]]:
        """Return the channels to write, filling absent optional ones.

        Every check happens before the caller can open an output file.
        Absent optional channels are replaced with zero-valued placeholders
        on the S11 frequency grid; present channels are never modified.
        """
        if parameter_count not in PARAMETERS_BY_COUNT:
            supported = ", ".join(str(c) for c in PARAMETERS_BY_COUNT)
            raise UnsupportedParameterCountError(
                f"Unsupported parameter count {parameter_count!r}:"
                f" supported counts are {supported}"
                " (1 for a one-port, 4 for a two-port file)"
            )
        missing = [
            p for p in REQUIRED_BY_COUNT[parameter_count] if not self.channel(p)
        ]
        if missing:
            raise MissingParameterError(
                f"No data for required S-parameter(s) {', '.join(missing)}"
                f" in a {PORTS_BY_COUNT[parameter_count]} export"
            )

        reference = self.channel("S11")
        channels: dict[str, list[Datapoint]] = {}
        for parameter in PARAMETERS_BY_COUNT[parameter_count]:
            samples = self.channel(parameter)
            if not samples:
                channels[parameter] = [
                    Datapoint(dp.freq, 0.0, 0.0) for dp in reference
                ]
                continue
            if len(samples) != len(reference):
                raise ChannelLengthMismatchError(
                    f"Channel {parameter} holds {len(samples)} samples"
                    f" but S11 holds {len(reference)}"
                )
            for index, (ref, dp) in enumerate(
                zip(reference, samples, strict=True)
            ):
                if dp.freq != ref.freq:
                    raise FrequencyMismatchError(
                        "Frequencies of sdata not correlated:"
                        f" S11 index {index} is {ref.freq} Hz,"
                        f" {parameter} index {index} is {dp.freq} Hz"
                    )
                if not (_is_finite(dp.re) and _is_finite(dp.im)):
                    raise NonFiniteValueError(
                        f"Channel {parameter} sample {index} at {dp.freq} Hz"
                        f" is not finite: re={dp.re!r}, im={dp.im!r}"
                    )
            channels[parameter] = list(samples)
        return channels

    def _option_line(self) -> str:
        """Option line for the data actually written (hertz, real/imag)."""
        resistance = float(self.opts.resistance)
        if not math.isfinite(resistance) or resistance <= 0:
            raise TouchstoneExportError(
                "Reference impedance must be a positive finite number,"
                f" not {self.opts.resistance!r}"
            )
        if resistance.is_integer():
            value = str(int(resistance))
        else:
            value = f"{resistance:g}"
        return f"# HZ S RI R {value}"

    def _comment_lines(self, parameter_count: int) -> list[str]:
        """Stored provenance comments followed by the export notes."""
        lines: list[str] = []
        for comment in self.comments:
            text = comment.strip()
            if not text or text.startswith(EXPORT_NOTE_PREFIX):
                continue
            lines.append(text if text.startswith("!") else f"! {text}")

        written = PARAMETERS_BY_COUNT[parameter_count]
        carried = tuple(p for p in written if self.channel(p))
        if self.simulated:
            # Survives save/reopen: a reader of the file itself can tell that
            # these numbers came from the built-in demonstration device.
            scenario = (
                f" (scenario {self.simulated_scenario})"
                if self.simulated_scenario
                else ""
            )
            lines.append(
                f"{EXPORT_NOTE_PREFIX} {SIMULATED_MARKER}{scenario}: produced"
                " by the built-in simulated device, not a physical"
                " measurement"
            )
        placeholders = tuple(p for p in written if p not in carried)
        lines.append(
            f"{EXPORT_NOTE_PREFIX} {PORTS_BY_COUNT[parameter_count]}"
            f" Touchstone file; S-parameters written: {', '.join(written)}"
        )
        lines.append(
            f"{EXPORT_NOTE_PREFIX} measured (present in source data):"
            f" {', '.join(carried)}"
        )
        if placeholders:
            names = ", ".join(placeholders)
            lines.append(
                f"{EXPORT_NOTE_PREFIX} zero placeholders, NOT measured: {names}"
            )
            lines.append(
                f"{EXPORT_NOTE_PREFIX} incomplete two-port characterization:"
                f" {names} contain synthesized zeros, so this file is not a"
                " complete two-port measurement; Touchstone consumers may"
                " ignore comment lines"
            )
        else:
            lines.append(
                f"{EXPORT_NOTE_PREFIX} nothing synthesized; every written"
                " S-parameter is present in the source data"
            )
        if parameter_count == 1:
            lines.append(
                f"{EXPORT_NOTE_PREFIX} reflection-only file;"
                " S21, S12 and S22 are not included"
            )
        return lines

    def save(self, parameter_count: int = 1) -> None:
        """Write the dataset to ``self.filename`` as a Touchstone 1.1 file.

        The dataset is validated and serialized first; the file is only
        opened once a complete, consistent output exists.

        Args:
            parameter_count: number of S-parameters the file must hold.
                ``1`` writes a one-port ``.s1p`` file (S11 only).  ``4``
                writes a two-port ``.s2p`` file carrying all four columns
                S11, S21, S12, S22.  S12 and S22 are optional and are
                written as zero placeholders when the source data has no
                such channel (see ``saves``).

        Raises:
            TouchstoneExportError: the dataset is inconsistent; nothing is
                written.  See ``saves`` for the specific subclasses.
        """
        data = self.saves(parameter_count)
        logger.info("Attempting to open file %s for writing", self.filename)
        with open(self.filename, "w", encoding="utf-8") as outfile:
            outfile.write(data)

    def saves(self, parameter_count: int = 1) -> str:
        """Return the dataset as a Touchstone 1.1 string.

        The option line is always ``# HZ S RI R <r>``, where ``<r>`` is
        ``self.opts.resistance``; samples are always real/imaginary pairs
        in hertz, whatever the units and format of the loaded file.
        ``self.comments`` is serialized above the option line, followed by
        notes stating which parameters were taken from the data and which
        were synthesized.

        Args:
            parameter_count: number of S-parameters the file must hold.
                ``1`` for a one-port ``.s1p`` file (S11 only), ``4`` for a
                two-port ``.s2p`` file with the full S11, S21, S12, S22
                column set.  S12 and S22 are optional: absent channels are
                written as zero placeholders and reported in the comments.
                S11, and S21 for a two-port file, are required.

        Raises:
            UnsupportedParameterCountError: ``parameter_count`` is not 1 or
                4.
            MissingParameterError: S11 (or S21 for two-port) is empty.
            ChannelLengthMismatchError: two channels differ in length.
            FrequencyMismatchError: two channels use different frequency
                grids; also a ``LookupError``.
            NonFiniteValueError: a written sample is NaN or infinite.
            TouchstoneExportError: common base class of all of the above.
        """
        channels = self._validated_channels(parameter_count)
        written = PARAMETERS_BY_COUNT[parameter_count]
        lines = self._comment_lines(parameter_count)
        lines.append(self._option_line())
        for index, dp_s11 in enumerate(channels["S11"]):
            fields = [f"{dp_s11.freq}"]
            for parameter in written:
                dp = channels[parameter][index]
                fields.append(f"{dp.re} {dp.im}")
            lines.append(" ".join(fields))
        return "".join(line + "\n" for line in lines)
