Contributing
============

Thanks for taking the time to contribute! Contributions are what make an
open-source project work, and they are greatly appreciated.

When contributing to this repository, please first discuss the change you
wish to make via an issue, email, or any other method with the owners of this
repository before making a change. Please note we have a
[code of conduct](CODE_OF_CONDUCT.md); follow it in all your interactions
with the project.

If you only want to *use* NanoVNA-Saver, read
[INSTALLATION.md](INSTALLATION.md) instead - this page is for people changing
the code or the documentation.

 What has actually been verified
--------------------------------

Every command in this document was executed on the development machine:
macOS 26.6.2 (arm64), CPython 3.12.9 in a `uv`-managed `.venv`, `uv` 0.12.17,
PySide6 6.8.2.1, NumPy 2.2.2, SciPy 1.15.1. Commands that were **not** run
there are labelled *unverified* and keep the platform they belong to, so the
document never implies a test that did not happen.

| Area | macOS arm64 (this repo's host) | Linux | Windows |
| --- | --- | --- | --- |
| `uv sync --locked` | verified | CI only, unverified | CI only, unverified |
| `uv run task ui-compile` | verified | CI only, unverified | CI only, unverified |
| `uv run NanoVNASaver --version` | verified | unverified | unverified |
| `uv run pytest tests -q` | command verified; suite not green during this audit (see below) | CI matrix 3.10-3.13 | CI smoke only |
| `uv run ruff check/format` | verified | CI (`ubuntu-latest`) | unverified |
| `uv build` (wheel + sdist) | verified | unverified | unverified |
| `./build-macos-app.sh` | verified | n/a | n/a |
| `uv run task build-pkg-macos` | verified | n/a | n/a |
| `uv run task build-pkg-linux` | n/a | unverified | n/a |
| `uv run task build-pkg-win` | n/a | n/a | unverified |
| `make -C docs html` | verified | unverified | unverified |
| `tox -e docs` | fails here (no network) | unverified | unverified |
| physical VNA measurement | not verified | not verified | not verified |

"CI only" means the command is present in `.github/workflows/checks.yml` and
runs on the project's CI runners; it has not been reproduced on a local
machine of that platform during this audit. No command in this repository has
been validated against a physical NanoVNA by the documentation work.

The test command itself was verified: pytest collects and runs the suite with
`QT_QPA_PLATFORM=offscreen`, and single modules pass cleanly (for example
`tests/test_limits.py` -> 95 passed, `tests/test_calibration_bulk.py` -> 65
passed). A whole-suite run during this audit was **not** green: it reported
failures and errors in modules that concurrent work was editing at the time
(a later run could not even collect three modules because
`Controls/SerialControl.py` imported `list_serial_ports` from
`Hardware/Hardware.py` while that file was mid-edit). Treat a red suite as
"check whether the tree you have is mid-change" before assuming a
documentation-relevant regression.

 Development environment setup
------------------------------

### 1. Clone the repo

```sh
git clone https://github.com/NanoVNA-Saver/nanovna-saver
cd nanovna-saver
```

### 2. Install `uv`, then the locked dependency set

Install `uv` from <https://docs.astral.sh/uv/getting-started/installation/>
and sync:

```sh
uv sync --locked
```

`--locked` is deliberate. `uv.lock` is the contract: the versions in it are
the ones the project is tested against, and a build that cannot honour the
lock is an error rather than a silent re-resolution. It also means you do not
install Qt yourself - PySide6 (6.8.2.1 in the current lock) comes from the
lock file, and a Homebrew or distro PyQt is the wrong stack.

Python 3.10 through 3.13 are supported (`requires-python = ">=3.10"`, CI
matrix and `tox.ini` env list both cover 3.10-3.13). `uv` will fetch a
suitable interpreter if you do not have one.

### 3. Compile the Qt resources

```sh
uv run task ui-compile
```

This runs `pyside6-uic` on the `.ui` files and `pyside6-rcc` on `main.qrc`.
There is exactly one correct invocation: `uv run ui-compile` and `uv run task
compile` do not exist and fail. The generated modules
(`src/NanoVNASaver/Windows/ui/about.py`, `main_rc.py`) are checked in, so you
only need this after changing a `.ui` or `.qrc` file - but running it is
harmless, and CI runs it before the tests.

It rewrites those tracked files. To keep your working tree clean afterwards:

```sh
git checkout -- src/NanoVNASaver/Windows/ui/about.py src/NanoVNASaver/Windows/ui/main_rc.py
```

### 4. Run the application

```sh
uv run NanoVNASaver
```

On a headless machine, prefix commands with `QT_QPA_PLATFORM=offscreen`.
Without a VNA, tick **Simulated device (demo)** in the connection panel; see
[OFFLINE_WALKTHROUGH.md](OFFLINE_WALKTHROUGH.md).

 Testing and linting
--------------------

The CI gates are the ones to reproduce locally.

```sh
# full suite (Qt needs an offscreen platform on a headless machine)
QT_QPA_PLATFORM=offscreen uv run pytest tests -q

# one module while iterating
uv run pytest tests/test_limits.py -q
uv run task test -- tests/test_limits.py -q

# non-mutating lint, exactly as CI runs it
uv run ruff check --no-fix src tests
uv run ruff format --check src tests
```

Notes that save time:

* `ruff check` without `--no-fix` **edits** files (`fix = true` in
  `pyproject.toml`). CI uses `--no-fix` on purpose, so a lint failure is
  reported rather than repaired.
* Line length is 80 and the target is `py311`; `ruff format` is the
  formatter, so run `uv run ruff format src tests` before committing.
* Tests must not touch the developer's real settings. UI tests construct
  widgets under `QT_QPA_PLATFORM=offscreen` and use temporary settings
  directories.
* `uv run task test` runs `pytest .` from the repository root; prefer the
  explicit `pytest tests` form when you want only the suite.
* `uv run task test-full` runs the whole tox matrix. `tox -e docs` needs
  package-index access for `docs/requirements.txt` and fails without network;
  use `make -C docs html` instead (below).
* `pre-commit run --all-files` (also `uv run task lint`) installs hook
  environments from the network on first use, so it fails offline. The ruff
  commands above are the same checks and work offline.

Writing tests that stay useful:

* Prefer analytic, independently computed expected values over recorded
  goldens you cannot re-derive.
* Assert what the code guarantees (channels, grids, labels, error types),
  not incidental formatting.
* Do not encode timing numbers or hardware behaviour as assertions.

 Building the documentation
---------------------------

```sh
make -C docs html
```

*Verified:* the build succeeds and writes `docs/_build/html`. It needs
`sphinx-build` on `PATH`; the virtual environment does not include Sphinx, so
this uses whatever Sphinx your system provides (7.3.7 here). Set
`PYTHONPATH=src` if your `sphinx-build` cannot import the package for the
autodoc step.

`tox -e docs` is the CI-shaped equivalent and is *unverified here*: it runs
`uv pip install -r docs/requirements.txt`, which needs package-index access.
If you are offline, use `make -C docs html`.

`docs/api/` and `docs/_build/` are generated and git-ignored; never edit or
commit them.

 Documentation conventions
--------------------------

* Public feature documentation lives in `docs/*.md` and is indexed by
  [FEATURES.md](FEATURES.md). When you add a feature with a user-visible
  workflow, add one short section there and link the detailed document.
* Every command in the documentation must exist. If you change a `taskipy`
  task, script name or workflow step, update the documents that mention it in
  the same change.
* State the platform a claim was verified on, or label it unverified. Do not
  describe hardware accuracy, calibration validity, or platform packaging as
  tested when it was only reasoned about.
* Preserve the project's honesty rules when documenting results: absent
  channels are absent, not zero; simulated data is labelled as simulated;
  smoothing and averaging change a measurement rather than improving it;
  UNAVAILABLE is never a pass.

 Packaging and verification
---------------------------

Local development loop (source checkout):

```sh
uv sync --locked
uv run task ui-compile
QT_QPA_PLATFORM=offscreen uv run pytest tests -q
uv run ruff check --no-fix src tests
uv run ruff format --check src tests
```

Release-shaped build:

```sh
uv build                        # wheel + sdist in dist/
./build-macos-app.sh            # macOS bundle + tarball in dist/
```

`uv build` uses the in-tree backend `src/tools/setuptools_wrapper.py`, which
attempts to recompile resources in its isolated build environment. That step
prints a non-fatal `FileNotFoundError` for `pyside6-uic` (PySide6 is not in
the isolated environment) and the checked-in generated modules are what ship.

`./build-macos-app.sh` builds in a private temporary virtualenv, so your own
environment is untouched; `uv run task build-pkg-macos` does the same
PyInstaller run inside your environment. Both produce `dist/NanoVNASaver.app`
and `dist/NanoVNASaver.app-<arch>.tar.gz`. `dist/` is git-ignored.

Version strings are derived from git by `setuptools-scm`
(`version_scheme = "no-guess-dev"`, local scheme `node-and-date`):
`git describe` gives `v0.7.3-44-gbb8e3b0-dirty`, an editable install reports
`0.7.3.post1.dev44+gbb8e3b038`, and a fresh build adds the build date
(`...d20261101`). A dirty tree shows up as `+dirty` instead of being
silently dropped. Never hand-edit a version.

Verify a packaged build separately from a source checkout:

```sh
QT_QPA_PLATFORM=offscreen ./dist/NanoVNASaver.app/Contents/MacOS/NanoVNASaver --version
```

and the platform-independent startup smoke test that CI runs:

```sh
QT_QPA_PLATFORM=offscreen uv run python -c "import sys; sys.path.insert(0, 'src'); \
from PySide6 import QtWidgets; app = QtWidgets.QApplication([]); \
from NanoVNASaver.NanoVNASaver import NanoVNASaver; \
print('startup ok:', NanoVNASaver().windowTitle())"
```

Both prove imports and bundled Qt resources resolve. Neither proves
measurement accuracy, and neither is a platform acceptance test.

 Issues and feature requests
----------------------------

You've found a bug in the source code, a mistake in the documentation or
maybe you'd like a new feature? Take a look at [GitHub
Discussions](https://github.com/NanoVNA-Saver/nanovna-saver/discussions) to
see if it's already being discussed. You can help us by [submitting an issue
on GitHub](https://github.com/NanoVNA-Saver/nanovna-saver/issues). Before you
create an issue, make sure to search the issue archive - your issue may have
already been addressed!

Please try to create bug reports that are:

- *Reproducible.* Include steps to reproduce the problem.
- *Specific.* Include as much detail as possible: which version, what
  environment, etc.
- *Unique.* Do not duplicate existing opened issues.
- *Scoped to a Single Bug.* One bug per report.

**Even better: Submit a pull request with a fix or new feature!**

### How to submit a Pull Request

1. Search our repository for open or closed
   [Pull Requests](https://github.com/NanoVNA-Saver/nanovna-saver/pulls)
   that relate to your submission. You don't want to duplicate effort.
2. Fork the project.
3. Create your feature branch (`git checkout -b feat/amazing_feature`).
4. Commit your changes (`git commit -m 'feat: add amazing_feature'`).
   NanoVNA-Saver uses [conventional
   commits](https://www.conventionalcommits.org), so please follow the
   specification in your commit messages.
5. Push to the branch (`git push origin feat/amazing_feature`).
6. [Open a Pull Request](https://github.com/NanoVNA-Saver/nanovna-saver/compare?expand=1).

Before you open it, make sure `uv run ruff check --no-fix src tests`,
`uv run ruff format --check src tests` and the test suite pass locally, and
that any documentation you touched still describes real commands.
