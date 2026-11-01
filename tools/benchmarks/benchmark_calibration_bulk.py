#!/usr/bin/env python
"""Benchmark scalar vs bulk calibration correction.

Reproducible, dependency-free (beyond the application's own deps) benchmark
for the correction path in :mod:`NanoVNASaver.Calibration`:

* ``scalar``: the original per-point loop calling ``correct11``/``correct21``
  and ``correct_delay`` — the reference oracle, one SciPy ``interp1d`` call
  per channel per datapoint.
* ``bulk*fused``: an **arrays-only prototype**.  It converts once on the way
  in, then computes calibration and offset delay on the same arrays before
  converting once on the way out.  This fuses steps the shipping code keeps
  separate, so it is a lower bound on conversion cost, *not* production.
* ``worker``: the **actual shipping path** — ``SweepWorker.applyCalibration``
  with a live-style calibration and offset delay.  It converts to arrays,
  corrects both channels, converts back to ``Datapoint``, then converts again
  per channel to apply the offset delay and converts back again.  This is the
  number to quote for real sweeps.

Every size reports the maximum absolute difference between the scalar
oracle and **both** bulk results, so the speed figures can be read together
with their numerical agreement.  The conversion microbenchmarks measure one
channel each (``Datapoint`` list to arrays, and arrays to ``Datapoint``
list); a sweep pays both in each direction when an offset delay is active.

No timing thresholds live here; nothing in the test suite depends on the
numbers printed.

Usage::

    python tools/benchmarks/benchmark_calibration_bulk.py
    python tools/benchmarks/benchmark_calibration_bulk.py --points 1000 10000
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from NanoVNASaver.Calibration import (  # noqa: E402
    Calibration,
    arrays_to_datapoints,
    correct_delay,
    correct_delay_bulk,
    datapoints_to_arrays,
)
from NanoVNASaver.RFTools import Datapoint  # noqa: E402
from NanoVNASaver.SweepWorker import SweepWorker  # noqa: E402

DEFAULT_POINTS = (1_000, 10_000, 100_000)
CAL_START = 1_000_000
CAL_STOP = 900_000_000
CAL_STEP = 1_000_000
DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 7
OFFSET_DELAY = 3.3e-11


def build_calibration() -> Calibration:
    """A deterministic, frequency-varying two-port calibration.

    The standards are not ideal: the short is inductive, the open capacitive,
    the load slightly offset, and the through/isolation terms vary with
    frequency.  That exercises the interpolation between calibration points
    instead of pinning the correction to constant coefficients.
    """
    calibration = Calibration()
    for index, freq in enumerate(range(CAL_START, CAL_STOP + 1, CAL_STEP)):
        scale = freq / CAL_STOP
        ripple = math.sin(index * 0.37)
        short = 0.98 * complex(math.cos(-0.9 * scale), math.sin(-0.9 * scale))
        opened = 0.95 * complex(math.cos(1.4 * scale), math.sin(1.4 * scale))
        load = 0.02 * complex(math.cos(2.1 * scale), math.sin(2.1 * scale))
        through = (
            complex(0.55 * scale, 0.4 * scale) + complex(0.02, -0.03) * ripple
        )
        thrurefl = complex(0.01 * scale, -0.02 * scale)
        isolation = complex(1e-4 * scale, -5e-5 * scale)
        calibration.dataset.insert(
            "short", Datapoint(freq, short.real, short.imag)
        )
        calibration.dataset.insert(
            "open", Datapoint(freq, opened.real, opened.imag)
        )
        calibration.dataset.insert(
            "load", Datapoint(freq, load.real, load.imag)
        )
        calibration.dataset.insert(
            "through", Datapoint(freq, through.real, through.imag)
        )
        calibration.dataset.insert(
            "thrurefl", Datapoint(freq, thrurefl.real, thrurefl.imag)
        )
        calibration.dataset.insert(
            "isolation", Datapoint(freq, isolation.real, isolation.imag)
        )
    calibration.calc_corrections()
    return calibration


def build_sweep(count: int) -> tuple[list[Datapoint], list[Datapoint]]:
    """A realistic sweep grid with a resonant, lossy DUT on both channels."""
    step = (CAL_STOP - CAL_START) / max(count - 1, 1)
    raw11: list[Datapoint] = []
    raw21: list[Datapoint] = []
    for index in range(count):
        freq = int(round(CAL_START + index * step))
        resonance = 0.12 + 0.4 / (1 + ((freq - 4.2e8) / 6.0e7) ** 2)
        phase11 = -2 * math.pi * freq * 2.4e-9
        s11 = resonance * complex(math.cos(phase11), math.sin(phase11))
        phase21 = -2 * math.pi * freq * 5.1e-9
        magnitude21 = 0.8 - 0.5 / (1 + (freq / 3.0e8) ** 2)
        s21 = magnitude21 * complex(math.cos(phase21), math.sin(phase21))
        raw11.append(Datapoint(freq, s11.real, s11.imag))
        raw21.append(Datapoint(freq, s21.real, s21.imag))
    return raw11, raw21


def scalar_correction(calibration, raw11, raw21, delay):
    """The pre-bulk worker pipeline, point by point."""
    data11 = [
        correct_delay(calibration.correct11(point), delay, reflect=True)
        for point in raw11
    ]
    data21 = [
        correct_delay(calibration.correct21(point21, point11), delay)
        for point21, point11 in zip(raw21, raw11, strict=True)
    ]
    return data11, data21


def bulk_arrays(calibration, freq, values11, values21, delay):
    """Bulk correction on arrays that are already NumPy (no conversion)."""
    raw11 = values11.copy()
    corrected11 = calibration.correct11_bulk(freq, values11)
    corrected21 = calibration.correct21_bulk(freq, values21, raw11)
    corrected11 = correct_delay_bulk(freq, corrected11, delay, reflect=True)
    corrected21 = correct_delay_bulk(freq, corrected21, delay)
    return corrected11, corrected21


def bulk_fused_prototype(calibration, raw11, raw21, delay):
    """Arrays-only prototype: convert once in, once out.

    This is **not** the shipping path.  It keeps the calibration correction
    and the offset delay on the same arrays, while
    ``SweepWorker.applyCalibration`` converts back to ``Datapoint`` after
    the calibration step and converts again to apply the delay.  The
    difference is a lower bound on what production saves; the ``worker``
    column is the honest end-to-end number.
    """
    freq, values11 = datapoints_to_arrays(raw11)
    _, values21 = datapoints_to_arrays(raw21)
    corrected11, corrected21 = bulk_arrays(
        calibration, freq, values11, values21, delay
    )
    return (
        arrays_to_datapoints(freq, corrected11),
        arrays_to_datapoints(freq, corrected21),
    )


def time_call(function, *args, warmup: int, repeats: int) -> float:
    """Median wall-clock seconds of ``function(*args)``."""
    for _ in range(warmup):
        function(*args)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(*args)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def max_difference(left, right) -> float:
    return max(abs(a.z - b.z) for a, b in zip(left, right, strict=True))


def make_worker(calibration, delay):
    """A ``SweepWorker`` bound to ``calibration`` for the shipping path.

    Only the attributes ``applyCalibration`` reads are set, so the benchmark
    needs no Qt application and no device.
    """
    worker = SweepWorker.__new__(SweepWorker)
    worker.app = SimpleNamespace(calibration=calibration)
    worker.offsetDelay = delay
    worker._capture = None
    return worker


def measure(calibration, count: int, warmup: int, repeats: int) -> dict:
    raw11, raw21 = build_sweep(count)
    freq, values11 = datapoints_to_arrays(raw11)
    _, values21 = datapoints_to_arrays(raw21)
    scalar11, scalar21 = scalar_correction(
        calibration, raw11, raw21, OFFSET_DELAY
    )
    fused11, fused21 = bulk_fused_prototype(
        calibration, raw11, raw21, OFFSET_DELAY
    )
    worker = make_worker(calibration, OFFSET_DELAY)
    worker11, worker21 = worker.applyCalibration(raw11, raw21)

    scalar_time = time_call(
        scalar_correction,
        calibration,
        raw11,
        raw21,
        OFFSET_DELAY,
        warmup=warmup,
        repeats=repeats,
    )
    array_time = time_call(
        bulk_arrays,
        calibration,
        freq,
        values11,
        values21,
        OFFSET_DELAY,
        warmup=warmup,
        repeats=repeats,
    )
    fused_time = time_call(
        bulk_fused_prototype,
        calibration,
        raw11,
        raw21,
        OFFSET_DELAY,
        warmup=warmup,
        repeats=repeats,
    )
    worker_time = time_call(
        worker.applyCalibration,
        raw11,
        raw21,
        warmup=warmup,
        repeats=repeats,
    )
    to_array_time = time_call(
        datapoints_to_arrays, raw11, warmup=warmup, repeats=repeats
    )
    to_points_time = time_call(
        arrays_to_datapoints,
        np.arange(count, dtype=np.int64),
        np.zeros(count, dtype=np.complex128),
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "points": count,
        "scalar_s": scalar_time,
        "arrays_s": array_time,
        "fused_s": fused_time,
        "worker_s": worker_time,
        "to_array_s": to_array_time,
        "to_points_s": to_points_time,
        "max_diff_fused_11": max_difference(scalar11, fused11),
        "max_diff_fused_21": max_difference(scalar21, fused21),
        "max_diff_worker_11": max_difference(scalar11, worker11),
        "max_diff_worker_21": max_difference(scalar21, worker21),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=list(DEFAULT_POINTS),
        help="sweep sizes to benchmark (default: 1000 10000 100000)",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args(argv)

    calibration = build_calibration()
    print(
        f"python {platform.python_version()} / numpy {np.__version__} /"
        f" {platform.machine()} {platform.system()}"
    )
    print(
        f"calibration: {calibration.size()} points,"
        f" {CAL_START}-{CAL_STOP} Hz, offset delay {OFFSET_DELAY * 1e12:.1f} ps"
    )
    print(f"warmup {args.warmup}, repeats {args.repeats} (median reported)")
    header = (
        "points | scalar ms | bulk arrays ms | fused prototype ms"
        " | ACTUAL worker ms | speedup(worker)"
        " | to-array ms | to-points ms"
        " | max|Δ| fused 11/21 | max|Δ| worker 11/21"
    )
    print(header)
    print("-" * len(header))
    for count in args.points:
        result = measure(calibration, count, args.warmup, args.repeats)
        speedup = result["scalar_s"] / result["worker_s"]
        print(
            f"{count:>6} |"
            f" {result['scalar_s'] * 1e3:9.1f} |"
            f" {result['arrays_s'] * 1e3:13.1f} |"
            f" {result['fused_s'] * 1e3:18.1f} |"
            f" {result['worker_s'] * 1e3:16.1f} |"
            f" {speedup:14.1f}x |"
            f" {result['to_array_s'] * 1e3:11.2f} |"
            f" {result['to_points_s'] * 1e3:12.2f} |"
            f" {result['max_diff_fused_11']:.3e}"
            f"/{result['max_diff_fused_21']:.3e} |"
            f" {result['max_diff_worker_11']:.3e}"
            f"/{result['max_diff_worker_21']:.3e}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
