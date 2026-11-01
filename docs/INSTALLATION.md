# Installation Instructions

## Installation and Use with pipx
For installation it is recommended to use pipx instead of pip.  Most installations nowadays require the usage of a virtual environment when using pip, pipx automatically arranges this for you.

Installing the latest release with pipx install. e.g.:

    pipx install git+https://github.com/NanoVNA-Saver/nanovna-saver

Or if you want to install a specific version use:

    pipx install git+https://github.com/NanoVNA-Saver/nanovna-saver@v0.7.3

Once completed run with the following command: `NanoVNASaver`

## Installing via DEB for Debian (and Ubuntu)
🚧 TODO: this is currently broken

The installation has the benefit that it allows you to run the program from anywhere, because the
main program is found via the regular `$PATH` and the modules are located in the Python module path.

If you're using a debian based distro you should consider to build your own `*.deb` package.
This has the advantage that NanoVNASaver can be installed and uninstalled cleanly in the system.

For this you need to install `python3-stdeb` - the module for converting Python code and modules into a Debian package:

    apt install python3-stdeb

Then you can build the package via:

    make deb

This package can be installed the usual way with

    sudo dpkg -i nanovnasaver....deb
or

   sudo apt install ./nanovnasaver....deb

### Installing via RPM (experimental)
🚧 TODO: this is currently broken

`make rpm` builds an (untested) rpm package that can be installed on your system the usual way.

## Automated MacOS Build Script

`build-macos-app.sh` produces an unsigned `NanoVNASaver.app` plus a
`NanoVNASaver.app-<arch>.tar.gz` archive in `dist/`. It can be run from any
working directory; all paths are resolved relative to the repository.

1. Install the build prerequisites (Qt is **not** needed from Homebrew -
   PySide6 comes from the project's locked dependencies):

        brew install uv python@3.13

2. Run the automated build script:

        ./build-macos-app.sh

3. Open the completed app:

        open ./dist/NanoVNASaver.app

   or double-click on NanoVNASaver.app in finder.

Build notes:

* Each run uses a private temporary virtualenv built with the first supported
  Python it finds (3.10 - 3.13); your existing environments are not touched.
  Pick an interpreter explicitly with
  `PYTHON=/path/to/python3.13 ./build-macos-app.sh`.
* Dependencies come from `uv.lock` (`uv sync --locked`), so the pinned
  PyInstaller matches the pinned NumPy/SciPy. Resolving unpinned versions can
  produce a bundle that aborts on startup with a missing NumPy/SciPy submodule.
* The build regenerates `src/NanoVNASaver/Windows/ui/about.py` and
  `main_rc.py`; the previous contents are restored when it finishes, whether
  it succeeded or failed.
* The bundle is signed ad-hoc, not notarized. Gatekeeper may refuse the first
  launch; allow it via System Settings > Privacy & Security > "Open Anyway".
* The archive contains the whole `.app`; keep the bundle intact rather than
  copying the inner executable elsewhere.

## MacPorts

Via a MacPorts distribution maintained by @ra1nb0w.

1. Install MacPorts following the [install guide](https://www.macports.org/install.php)

2. Install NanoVNASaver :

        sudo port install NanoVNASaver

3. Now you can run the software from shell `NanoVNASaver` or run as app
   `/Applications/MacPorts/NanoVNASaver.app`

## Homebrew

1. Install Homebrew from <https://brew.sh/> (This will ask for your password)

        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

2. Python :

        brew install python

3. Pip :<br/>
    Download the get-pip.py file and run it to install pip

        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py

4. NanoVNASaver Installation : <br/>
    clone the source code to the nanovna-saver folder

        git clone https://github.com/NanoVNA-Saver/nanovna-saver
        cd nanovna-saver

5. Install local pip packages

        python3 -m pip install .

6. Run nanovna-saver folder by:

        NanoVNASaver
