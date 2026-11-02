# Installation Instructions

NanoVNA-Saver is a Python 3.10+ application. There are three ways to get a
working copy: run it from a source checkout with `uv`, install the published
package with `pipx`, or use a prebuilt binary from the release page.

> **Verification note.** The commands marked *verified* were run on this
> project's development machine: macOS 26.6.2 (arm64), CPython 3.12.9 in a
> `uv`-managed `.venv`, `uv` 0.12.17, PySide6 6.8.2.1. Linux and Windows
> commands are taken from the repository's CI and packaging configuration but
> have **not** been executed here; they are labelled *unverified*. Nothing in
> this file claims a physical-VNA or platform acceptance test.

## Requirements

* Python 3.10, 3.11, 3.12 or 3.13.
* A working Qt platform plugin. PySide6 ships the Qt libraries it needs, so
  you do **not** install Qt separately - and in particular you should not
  install a Homebrew or distro Qt/PyQt for this application.
* No VNA hardware is needed to run, test or evaluate the application: the
  built-in simulator covers the device-facing paths (see
  [OFFLINE_WALKTHROUGH.md](OFFLINE_WALKTHROUGH.md)).

## From a source checkout with uv (recommended)

This is the workflow the CI uses, so it is the one that is kept working.

1. Install `uv`: <https://docs.astral.sh/uv/getting-started/installation/>.

2. Clone and install the locked dependency set:

   ```sh
   git clone https://github.com/NanoVNA-Saver/nanovna-saver
   cd nanovna-saver
   uv sync --locked
   ```

   *Verified:* `uv sync --locked` completed with no changes ("Checked 50
   packages"). `--locked` means `uv.lock` is authoritative: the resolved
   versions are the ones the project tests against, and the lock file is not
   updated behind your back. If the lock file is out of date, `uv sync`
   fails instead of silently re-resolving.

3. Compile the Qt resources and UI files:

   ```sh
   uv run task ui-compile
   ```

   *Verified.* This is the only supported invocation. It runs `pyside6-uic`
   for the `.ui` files and `pyside6-rcc` for `main.qrc`, and it rewrites the
   tracked generated files `src/NanoVNASaver/Windows/ui/about.py` and
   `main_rc.py`. If you do not want those files modified in your working
   tree, restore them afterwards with
   `git checkout -- src/NanoVNASaver/Windows/ui/about.py src/NanoVNASaver/Windows/ui/main_rc.py`.

   `uv run ui-compile` and `uv run task compile` do **not** exist and fail.
   The generated modules are also checked in, so a checkout starts without
   this step unless you changed a `.ui` or `.qrc` file.

4. Run it:

   ```sh
   uv run NanoVNASaver
   uv run NanoVNASaver-gui      # same application via the GUI entry point
   uv run NanoVNASaver --version
   ```

   *Verified:* `NanoVNASaver 0.7.3.post1.dev44+gbb8e3b038`.

   On Linux without a display, or on a headless CI runner, set
   `QT_QPA_PLATFORM=offscreen`; the application then constructs its windows
   without a visible screen.

## With pipx

`pipx` installs the application into its own virtual environment and puts the
`NanoVNASaver` command on your `PATH`.

```sh
pipx install git+https://github.com/NanoVNA-Saver/nanovna-saver
```

For a specific release:

```sh
pipx install git+https://github.com/NanoVNA-Saver/nanovna-saver@v0.7.3
```

*Unverified here:* `pipx` is not installed on the verification machine, and
package-index access was blocked, so this path was not executed. The entry
points it relies on are `NanoVNASaver` (console) and `NanoVNASaver-gui` (GUI);
both are declared in `pyproject.toml` and both are present in a built wheel.

Upgrade with `pipx upgrade NanoVNASaver`, uninstall with
`pipx uninstall NanoVNASaver`.

## Prebuilt binaries

Current releases for Windows and Linux are published at
<https://github.com/NanoVNA-Saver/nanovna-saver/releases/latest>.

* **Windows** - the release workflow (`release_win.yml`) builds
  `dist/nanovna-saver.exe` with PyInstaller and uploads it. *Unverified
  here* (no Windows machine).
* **Linux** - the release workflow (`release_linux.yml`) builds
  `dist/nanovna-saver` with PyInstaller and uploads it. *Unverified here.*
* **macOS** - there is no macOS release workflow. Build the application
  locally with the script in the next section.

Prebuilt binaries are convenience artifacts. They are not code-signed or
notarized, they are built from one Python version, and they cannot be
diagnosed from this repository's issue tracker as easily as a source
checkout. If you hit a Qt or plugin problem, reproduce it from a source
checkout first.

## Building a macOS application

`build-macos-app.sh` builds the bundle in an isolated environment: it creates
its own temporary virtual environment, syncs the locked dependencies into it
with `uv sync --locked`, runs PyInstaller, and leaves your existing virtual
environments untouched.

1. Install the build prerequisites. Qt is **not** one of them - PySide6 comes
   from the locked dependencies.

   ```sh
   brew install uv
   ```

   Any CPython in the 3.10 - 3.13 range will do; the script searches for one
   and you can also point it at a specific interpreter:

   ```sh
   PYTHON=/path/to/python3.13 ./build-macos-app.sh
   ```

2. Run the build from any working directory:

   ```sh
   ./build-macos-app.sh
   ```

   *Verified:* finished in about 44 s and printed
   `NanoVNASaver 0.7.3.post1.dev44+gbb8e3b038.d20261101` from the packaged
   executable as its own self-check.

3. Artifacts (all under `dist/`, which is git-ignored):

   | Path | Content |
   | --- | --- |
   | `dist/NanoVNASaver.app` | the bundle (~157 MB on this machine) |
   | `dist/NanoVNASaver.app-arm64.tar.gz` | the whole bundle, tarred (~60 MB) |

   The archive name carries `uname -m`, so an Intel build produces
   `NanoVNASaver.app-x86_64.tar.gz`. Keep the bundle intact: the inner
   executable depends on the other files in the bundle and will not run if
   copied out on its own.

4. Open the result:

   ```sh
   open ./dist/NanoVNASaver.app
   ```

   The bundle is signed ad-hoc, not notarized. The first launch may be
   refused by Gatekeeper; allow it via *System Settings > Privacy & Security
   > Open Anyway*.

`uv run task build-pkg-macos` is the equivalent PyInstaller command from
`pyproject.toml` (also *verified*: it produces `dist/NanoVNASaver/`,
`dist/NanoVNASaver.app` and the same tarball). Use the script when you want
the isolated environment and the packaged-executable self-check; use the task
when you are already inside a synced environment.

The build regenerates `src/NanoVNASaver/Windows/ui/about.py` and `main_rc.py`
and restores the previous contents afterwards, whether it succeeded or
failed.

## Versioning: what the version string tells you

Versions come from `setuptools-scm` (`version_scheme = "no-guess-dev"`, local
scheme `node-and-date`) over the git history, so the string is derived, never
edited by hand:

```sh
git describe            # v0.7.3-44-gbb8e3b0-dirty
uv run NanoVNASaver --version
```

* An editable development install reports `0.7.3.post1.dev44+gbb8e3b038`.
* A freshly built artifact adds the build date:
  `0.7.3.post1.dev44+gbb8e3b038.d20261101`.
* A commit-identified string ends in `+g<sha>`; `+dirty` appears when the
  working tree had uncommitted changes at build time. A dirty build is
  therefore visible in `--version` rather than silently indistinguishable
  from a clean one.

At runtime the version is read from the installed package metadata
(`importlib.metadata.version("NanoVNASaver")`) and falls back to `"unknown"`
when the metadata is missing. If you see `unknown`, the application is being
run from a source directory that was never installed - use `uv sync
--locked` rather than running the interpreter directly.

## Source checkout vs packaged build: what to verify

These are two different questions, and a release checklist should answer
both.

**Source checkout (developer loop):**

```sh
uv sync --locked
uv run task ui-compile
QT_QPA_PLATFORM=offscreen uv run pytest tests -q
uv run ruff check --no-fix src tests
uv run ruff format --check src tests
```

The first four are *verified* here; the full test suite runs, with failures
possible in modules that other in-progress work is editing. Lint and format
checks are non-mutating in CI, which is why `--no-fix` and `--check` are used
there.

**Packaged build (release check):**

```sh
QT_QPA_PLATFORM=offscreen ./dist/NanoVNASaver.app/Contents/MacOS/NanoVNASaver --version
QT_QPA_PLATFORM=offscreen uv run python -c "import sys; sys.path.insert(0, 'src'); \
from PySide6 import QtWidgets; app = QtWidgets.QApplication([]); \
from NanoVNASaver.NanoVNASaver import NanoVNASaver; \
print('startup ok:', NanoVNASaver().windowTitle())"
```

The first command proves the frozen interpreter can import every bundled
module and reported the running version. The second is the same startup
smoke test that CI runs on Windows and macOS. Passing both means bundled Qt
resources resolve; it says nothing about measurement accuracy, and nothing
about a physical instrument.

## Building a wheel or sdist

```sh
uv build
```

*Verified:* produces
`dist/nanovnasaver-<version>-py3-none-any.whl` and the matching
`.tar.gz`. The in-tree build backend
(`src/tools/setuptools_wrapper.py`) tries to recompile the Qt resources in
its isolated build environment. That step prints a `FileNotFoundError` for
`pyside6-uic` because the build isolation environment does not contain
PySide6; it is non-fatal, and the checked-in generated modules are what ends
up in the wheel (verify with `unzip -l` - `NanoVNASaver/Windows/ui/about.py`
and `main_rc.py` are present).

Publishing is `uv publish` after a tag; the repository has no `tox -e
publish` step.

## Linux notes

The CI test job installs the Qt runtime libraries before running the suite:

```sh
sudo apt-get update
sudo apt-get install --no-install-recommends -y \
  libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 \
  libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0
```

*Unverified here* (macOS machine), but this is exactly what the GitHub
Actions runner uses. The pure-Python wheel works on any distribution that
provides those libraries; `uv run task build-pkg-linux` produces the
one-file `dist/nanovna-saver` binary.

A Flatpak manifest (`io.github.zarath.nanovna-saver.yml`) is checked in.

## MacPorts

A MacPorts port maintained by @ra1nb0w:

1. Install MacPorts following the
   [install guide](https://www.macports.org/install.php).
2. `sudo port install NanoVNASaver`
3. Run it as `NanoVNASaver` from a shell, or as
   `/Applications/MacPorts/NanoVNASaver.app`.

*Unverified here;* this packaging is maintained outside the repository.

## What is not supported

* **`make deb` / `make rpm`** - these targets have been removed from
  `Makefile`: they invoked `python3 setup.py`, and this project has no
  `setup.py`. Do not follow older instructions that build Debian or RPM
  packages this way; use `uv build` and the `build-pkg-*` tasks instead.
* **Installing with `pip install .` after a `get-pip.py` bootstrap** - the
  project is `uv`-managed; the supported installation paths are the ones
  above. A bare `pip install` into a system interpreter bypasses `uv.lock`
  and will pick PySide6/NumPy/SciPy versions the tests were not run against.
* **`tox -e docs`** - see [CONTRIBUTING.md](CONTRIBUTING.md); the local docs
  build is `make docs` (or `make -C docs html`).
