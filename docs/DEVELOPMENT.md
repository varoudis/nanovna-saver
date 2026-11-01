# Development Process

## Local Development Environment

Prerequerements:

1. Python > 3.10. Using [pyenv](https://github.com/pyenv/pyenv) / [pyenv-win](https://github.com/pyenv-win/pyenv-win) may be requered to test comaptibility with different python versions.
2. [Astral UV v1.8.x](https://docs.astral.sh/uv/). May be installed via [pipx](https://pipx.pypa.io/latest/installation/).

## Local Run and Debugging

Following commands to prepare local env and run application or run tests:

* `uv sync` - prepare virtual environment and install prod and dev dependencies
* `uv run ui-compile` - compile Qt resources. Should be executed before first `uv run NanoVNASaver` run
* `uv run NanoVNASaver` - run local instance
* `uv run task test` - run unit tests in current venv
* `uv run task test-cov` - run unit tests in current venv and display test coverage report
* `uv run task test-full` - run unit tests for all avialable python versions
* `uv run task clean` - remove temporally artifacts (dist, build, *.pyc etc)
* `uv run task lint` - check sources and try to fix issues

`uv run task test-full` drives `tox`, whose matrix mirrors the Python
versions advertised in `pyproject.toml`. Keep the two in step: a version in
the classifiers must actually be tested.

## Related Documentation

* `DEVELOPMENT_PLAN.md` - the milestone plan and per-task acceptance
  criteria for measurement snapshots, sweeps, calibration coverage, traces,
  sessions, presets, and reports.
* `OFFLINE_WALKTHROUGH.md` - the end-to-end simulator walkthrough covering
  every user-facing workflow, including what it deliberately does not verify.
* `SESSION_FORMAT.md` - the versioned `.nvsession` format, with migration
  examples for contributors.

## Development Routines

A few usefull commands:

* 🚧 TODO: - update dependencies in `pyproject.toml` respecting specified version constraints.
* 🚧 TODO: - update dependencies ignoring in `pyproject.toml` specified version constraints.
* `uv lock` - update dependency lock file respecting specified version constraints.

## UI Development

UI layout of Widget, Dialogs and Window stored in `*.ui` files and may be edited by QT Designer. Please use `uv run pyside6-designer` or `uv run task ui-designer` to launch QT Designer. Please note

* UI layout should be creatd and stored via UI files
  * fields name should follow python_snake_style_naming scheme
  * elements should have meaningful names and initial values
  * elements which are not going to be changed during runtime should have `_` prefix
* Icons, images and other resources should be stored as part of `main.qrc` file.
  * compiled resource files (e.g. `main_rc.py`) should not be put under git version control
  * ui and resource files may be compiled manually via `uv run task compile` or will be executed automatically during `uv build` phase.

## Calibration Coverage

`CalibrationCoverage.py` answers one narrow question: *for which measured
frequencies does the loaded calibration actually supply a correction?* The
answer is shown on the frequency charts, in the sweep control, and in reports.

What it establishes:

* the calibrated start and stop frequencies, taken from the calibration file;
* whether a given frequency is fully covered, only interpolated between
  calibration points, outside the calibrated span (endpoint-held), or
  uncorrected entirely;
* which S-parameters the calibration can correct (a one-port calibration
  cannot correct S21);
* a stable identity for the calibration, so a session can prove that the
  displayed corrected data came from the same calibration that is loaded now.

What it can **not** establish, and never implies:

* **age.** Coverage reports the span in the file, not when the standards were
  measured or how much drift has accumulated since.
* **connector state.** Repeated mating, cable movement and connector wear are
  invisible to the file. A physically damaged calibration reports full
  coverage.
* **measurement uncertainty.** Coverage is a span statement. It is not a
  tolerance, an error bar, or a confidence interval, and the UI must never be
  presented as one.
* **validity of the standards.** An unknown, wrong, or out-of-specification
  load/short/open produces a calibration that looks complete.

Coverage reporting is purely informational. It does not alter the correction
mathematics: `tests/test_calibration_regression.py` pins corrected samples as
bit-identical before and after coverage queries, across every bundled
calibration fixture. If a coverage change ever moves a corrected value, that
test fails.

## Simulation

`Hardware/Simulated.py` provides a device model that needs no serial port. It
is selected in the connection panel ("Simulated device"), and every artifact
derived from it is marked:

* the window title gains `[SIMULATED: <scenario>]`;
* Touchstone exports carry a `! NanoVNASaver export: SIMULATED DATA ...`
  comment, which is read back on import;
* snapshot provenance records `simulated` and the scenario key, and sessions
  and reports surface the same labels.

Simulated data must never be presented as a physical measurement. Verification
against a real NanoVNA is a separate, physical step: the automated suite proves
the simulator's model, not any device's accuracy.

## Build Process

### Python packages

A `uv build` may be used to prepare python packages. Version would be taken from git repository.

### Windows Exeutables

Pre requirements:

* Python 3.10-3.12
* uv (see `Local Development Environment` for more details)

Execute `uv run task build-pkg-win` to get `dist\nanovna-sever*.exe` file.

### Linux Executables

Pre requirements:

* Python 3.10-3.12
* uv (see `Local Development Environment` for more details)
* Some X11 packages preinstalled

An example for Ubuntu 22.04:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt install -y python3.11 python3-pip python3.11-venv \
    python3.11-dev \
    '^libxcb.*-dev' libx11-xcb-dev \
    libglu1-mesa-dev libxrender-dev libxi-dev \
    libxkbcommon-dev libxkbcommon-x11-dev
```

Execute `uv run task build-pkg-linux` to get `dist/nanovna-sever*` file.

### Linux Flatpack

🚧 TODO: describe me

### MacOS Package

Pre requirements:

* Python 3.10-3.12
* uv (see `Local Development Environment` for more details)
* PyQt >=6.4 installed (`brew install pyqt`)

Execute `uv run task build-pkg-macos` to get `dist/NanoVNASaver.app*.exe` file.

## Publish Process

Please configure pypi credentials, create git tag and execute `uv publish` command. Please refer [UV :: publish](https://docs.astral.sh/uv/guides/publish/) for more details.

🚧 TODO: describe git tag creation and next version selection
🚧 TODO: build and publish platform specific artefacts: deb, native bin, flatpack packages
