# VERIFY-01 — packaged-artifact acceptance (local portion)

Windows and Linux packaging are **not** verified here: only macOS arm64 was
available. Those checklist items stay open in the plan. This file records
what was actually executed on the artifact, so it is never confused with a
source-tree test run.

## Artifact

| Item | Value |
| --- | --- |
| Command | `uv build --wheel --out-dir /tmp/nvs-artifact` |
| Artifact | `nanovnasaver-0.7.3.post1.dev44+gbb8e3b038.d20261101-py3-none-any.whl` |
| Build SHA | `bb8e3b0` (git), version carries `.d20261101` |
| Contents | 145 `.py` files; every new module present (`Reversal`, `Matching`, `PhaseDerivatives`, `ReferenceImpedance`, `SweepLog`, `ChartLimits`, `CalibrationKits`, `TuningHistory`, `TDRGating`, `PdfReport`, `ScanCapability`, `CombinedScan`, `cli`) |
| Environment | macOS 26 (arm64), CPython 3.12.9, `uv` 0.12.17 |

The declared floor is `requires-python = ">=3.10"`, so the suite was run on
every interpreter that range covers, not just the development one:

| Interpreter | `pytest tests` |
| --- | --- |
| CPython 3.10.21 | **2321 passed**, 2 warnings |
| CPython 3.11.16 | **2321 passed**, 2 warnings |
| CPython 3.12.9 (development) | **2321 passed**, 2 warnings |
| CPython 3.13.15 | **2321 passed**, 2 warnings |
| CPython 3.14.7 | **2327 passed** (after the phantom-reflection fix below) |

CPython 3.14 is inside the declared range (`>=3.10`, no upper bound). It
originally failed two tests, and the cause was a real defect rather than a
test: `ButterworthLowPass.response` computed `sqrt(1 - |S21|^2)`, and 3.14's
arithmetic left a `2.2e-16` residual where 3.12 left exactly `0`. The square
root amplified that to a `1.5e-8` phantom reflection. The residual is now
treated as a matched port. The tests assert bit-exact repeatability within one
interpreter and per-component agreement across interpreters, which is the
property that actually holds.

The built wheel was additionally installed into a fresh 3.10 environment
outside the checkout and swept the `filter` scenario (exit 0).

Source-tree tests are not evidence for the artifact, so the artifact was
installed **outside the checkout** into a fresh environment:

```sh
uv venv /tmp/nvs-clean --python 3.12
uv pip install --python /tmp/nvs-clean/bin/python /tmp/nvs-artifact/*.whl
```

Verified that the installed package resolves to
`/tmp/nvs-clean/lib/python3.12/site-packages/NanoVNASaver` — not to the
checkout — before running anything.

## Executed checks

| Check | Result |
| --- | --- |
| `import NanoVNASaver`, `NanoVNASaver.Reversal`, `NanoVNASaver.cli` from the installed copy | ok |
| `NanoVNASaver --version` (console script) | `NanoVNASaver 0.7.3.post1.dev44+gbb8e3b038.d20261101`, exit 0 |
| `NanoVNASaver --help` (console script) | full option list, exit 0 |
| `python -m NanoVNASaver.cli --help` | full option list |
| Packaged sweep: `--scenario filter --start 13000000 --stop 15000000 --points 51 --touchstone out.s2p --json out.json --log sweep.csv` | exit 0, three artifacts written |
| `out.s2p` content | `# HZ S RI R 50`, 51 rows, 9 columns, header comments naming measured (`S11, S21`) and synthesised (`S12, S22`) columns |
| `sweep.csv` | header row with 18 columns, one verdict row, `outcome=measured`, `verdict=unavailable`, `reason=no limit rules were evaluated` |
| `out.json` | snapshot + `summary` with `passed/failed/unavailable` counts |
| Offline path: `python -m NanoVNASaver.cli --input out.s2p` | `2 channel(s), reference 50 ohm, measured`, exit 0 |
| Missing input `--input /tmp/nonexistent.s2p` | `error: cannot read ...: no such file`, **exit 2** |
| Usage error `--bogus` | argparse message, **exit 2** |
| Limit failure: `--scenario filter --limits strict.nvlimits` (max −100 dB insertion loss) | `FAIL IL`, `0 pass, 1 fail, 0 unavailable`, **exit 1** |
| One-port device asked for S21: `--scenario antenna` | `error: this device does not provide S21; it offers S11`, **exit 3** (UNAVAILABLE, never a pass) |

The two `2`s are the documented contract, not a defect: `EXIT_INVALID = 2`
covers both an unparsable command line and an input that cannot be read
(`cli.py` raises `CliError(..., EXIT_INVALID)` for both). `EXIT_FAILED = 1`
is reserved for a *measured* run that failed its limits — which is why the
sweep above, with no limit set, reports `verdict=unavailable` and exits 0
rather than pretending to have passed.

All four codes were exercised on the installed wheel, not the checkout:
`0` (clean sweep), `1` (limit failure), `2` (unreadable input), `3`
(a channel the device cannot provide). The distinction that matters is
`1` versus `3`: "this failed" and "this could not be evaluated" are
different outcomes and never share a code.

## Platform items that remain unverified

Each is unchecked in the plan, with what it would take:

* **Windows / Linux build commands and artifacts** — `uv run task
  build-pkg-win` / `build-pkg-linux` in an isolated environment on each OS;
  then the same packaged-workflow checks above against the produced
  artifact. Not attempted: no such environment here.
* **Packaged offline workflow on Windows and Linux** — the macOS run above
  is the template; the Windows/Linux runs are the missing evidence.
* **macOS launch from Finder, outside the checkout, without a development
  environment** — the wheel was installed into a clean environment outside
  the checkout and driven headlessly, but a Finder launch of
  `NanoVNASaver.app` and a machine with no `uv`/`git`/`pytest` were not
  available.
* **Packaged file dialogs, icons, replay fixtures, optional dependencies** —
  the CLI paths were exercised; the GUI dialogs were not opened from the
  packaged artifact.
* **Signing / notarization / checksums / release packaging** — out of scope
  until distribution is selected; nothing was published.

## Reproduce

```sh
uv build --wheel --out-dir /tmp/nvs-artifact
uv venv /tmp/nvs-clean --python 3.12
uv pip install --python /tmp/nvs-clean/bin/python /tmp/nvs-artifact/*.whl
cd /tmp && QT_QPA_PLATFORM=offscreen /tmp/nvs-clean/bin/python -m NanoVNASaver.cli \
  --scenario filter --points 51 --output-dir /tmp/nvs-run \
  --touchstone out.s2p --json out.json --log sweep.csv
```
