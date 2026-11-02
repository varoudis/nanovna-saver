.. role:: raw-html-m2r(raw)
   :format: html

.. image:: https://img.shields.io/github/v/release/NanoVNA-Saver/nanovna-saver.svg
   :target: https://github.com/NanoVNA-Saver/nanovna-saver/releases/latest
   :alt: Latest Release

.. image:: https://img.shields.io/github/license/NanoVNA-Saver/nanovna-saver.svg
   :target: https://github.com/NanoVNA-Saver/nanovna-saver/blob/master/LICENSE.txt
   :alt: License

.. image:: https://img.shields.io/github/downloads/NanoVNA-Saver/nanovna-saver/total.svg
   :target: https://github.com/NanoVNA-Saver/nanovna-saver/releases/
   :alt: Downloads

.. image:: https://img.shields.io/github/downloads/NanoVNA-Saver/nanovna-saver/latest/total
   :target: https://github.com/NanoVNA-Saver/nanovna-saver/releases/latest
   :alt: GitHub Releases

.. image:: https://img.shields.io/badge/paypal-donate-yellow.svg
   :target: https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=T8KTGVDQF5K6E&item_name=NanoVNASaver+Development&currency_code=EUR&source=url
   :alt: Donate

NanoVNASaver
============

A multiplatform tool to save Touchstone files from the NanoVNA,
sweep frequency spans in segments to gain more than 101 data
points, and generally display and analyze the resulting data.


* Copyright 2019, 2020 Rune B. Broberg
* Copyright 2020ff NanoVNA-Saver Authors

It's developed in **Python 3 (>=3.10)** using **pyside6**, **numpy** and
**scipy**.


Introduction
------------

This software connects to a NanoVNA and extracts the data for
display on a computer and allows saving the sweep data to Touchstone files.


Current features
^^^^^^^^^^^^^^^^


* Reading data from a NanoVNA -- Compatible devices: NanoVNA, NanoVNA-H,
  NanoVNA-H4, NanoVNA-F, AVNA via Teensy, LiteVNA
* Reading data from a TinySA
* Splitting a frequency range into multiple segments to increase resolution
  (tried up to >10k points)
* Averaging repeated sweeps and segmenting a frequency span; averaged and
  smoothed traces are separate, labelled results, not "more accurate" data
* Displaying data on multiple chart types, such as Smith, LogMag, Phase and
  VSWR-charts, for both S11 and S21
* Displaying markers, and the impedance, VSWR, Q, equivalent
  capacitance/inductance etc. at these locations
* Displaying customizable frequency bands as reference, for example amateur
  radio bands
* Exporting and importing 1-port and 2-port Touchstone files
* TDR function (measurement of cable length) - including impedance display
* Filter analysis functions for low-pass, high-pass, band-pass and band-stop
  filters
* Display of both an active and a reference trace
* Live updates of data from the NanoVNA, including for multi-segment sweeps
* In-application calibration, including compensation for non-ideal calibration
  standards
* Customizable display options, including "dark mode"
* Exporting images of plotted values
* Explicit measurement limits: versioned JSON limit sets with PASS, FAIL and
  UNAVAILABLE verdicts that can be frozen into a report
* Antenna tuning workflow: overview sweep, marker on the measured minimum,
  zoom, repeat and one-port export
* Sessions (``.nvsession``): a portable, versioned record of traces, markers,
  charts, sweep settings and provenance, restorable without a device
* Touchstone replay as a simulator scenario, for reproducing a known dataset
  without hardware
* Provenance on every capture (source, simulated status, device, version,
  calibration and processing identity) that follows data into sessions,
  reports and Touchstone comments

Screenshot
^^^^^^^^^^


.. image:: https://i.imgur.com/ZoFsV2V.png
   :target: https://i.imgur.com/ZoFsV2V.png
   :alt: Screenshot of version 0.1.4


Running the application
-----------------------

The application is a Python 3.10+ program and needs no VNA hardware to start:
tick ``Simulated device (demo)`` in the connection panel to explore it. Qt is
not installed separately - PySide6 comes from the project's locked
dependencies.

From a source checkout:

.. code-block:: sh

   git clone https://github.com/NanoVNA-Saver/nanovna-saver
   cd nanovna-saver
   uv sync --locked
   uv run task ui-compile
   uv run NanoVNASaver

``uv run task ui-compile`` is the only working resource-compilation command.
A package install with ``pipx install
git+https://github.com/NanoVNA-Saver/nanovna-saver`` also works.

Installation
------------

Binary releases
^^^^^^^^^^^^^^^

Current binary releases for Windows and Linux are published as GitHub release
assets. macOS builds are produced locally with ``./build-macos-app.sh``.

`Detailed installation instructions <docs/INSTALLATION.md>`_

.. Note::

   Release binaries are built by the ``release_win.yml`` and
   ``release_linux.yml`` workflows; there is no macOS release workflow, so a
   macOS bundle is built from source. Binaries are unsigned and are not a
   substitute for reproducing a problem from a source checkout.


Using the software
------------------

Connect your NanoVNA to a serial port, and enter this serial port in the serial
port box.  If the NanoVNA is connected before the application starts, it should
be automatically detected. Otherwise, click "Rescan". Click "Connect to device"
to connect.

The app can split a span into several segments to collect more frequency
samples. Enter the number of segments to be done in the "Segments" box. Each
device segment is 101 data points and takes roughly 1.5 seconds. More segments
mean a finer grid, not a more accurate measurement.

Frequencies are entered in Hz, or suffixed with k or M.  Scientific notation
(6.5e6 for 6.5MHz) also works.

Markers can be manually entered, or controlled using the mouse. For mouse
control, select the active marker using the radio buttons, or hold "shift"
while clicking to drag the nearest marker. The marker readout boxes show the
actual frequency where values are measured.  Marker readouts can be hidden
using the "hide data" button when not needed.

Display settings are available under "Display setup". These allow changing the
chart colours, the application font size and which graphs are displayed.  The
settings are saved between program starts.

Calibration
^^^^^^^^^^^

*Before using NanoVNA-Saver, please ensure that the device itself is in a
reasonable calibration state.*

A calibration of both ports across the entire frequency span, saved to save
slot 0, is sufficient.  If the NanoVNA is completely uncalibrated, its readings
may be outside the range accepted by the application.

In-application calibration is available, either assuming ideal standards or
with relevant standard correction. To manually calibrate, sweep each standard
in turn and press the relevant button in the calibration window.
For assisted calibration, press the "Calibration Assistant" button.  If desired,
enter a note in the provided field describing the conditions under which the
calibration was performed.

Calibration results may be saved and loaded using the provided buttons at the
bottom of the window.  Notes are saved and loaded along with the calibration
data.


.. image:: https://i.imgur.com/p94cxOX.png
   :target: https://i.imgur.com/p94cxOX.png
   :alt: Screenshot of Calibration Window


Users of known characterized calibration standard sets can enter the data for
these, and save the sets.

After pressing *Apply*\ , the calibration is immediately applied to the latest
sweep data.

! *Currently, load capacitance is unsupported* !

TDR
^^^

To get accurate TDR measurements, calibrate the device, and attach the cable to
be measured at the calibration plane - i.e. at the same position where the
calibration load would be attached.  Open the "Time Domain Reflectometry"
window, and select the correct cable type, or manually enter a propagation
factor.

Measuring inductor core permeability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The permeability (mu) of cores can be measured using a one-port measurement.
Put one or more windings on a core of known dimensions and use the "S11 mu"
plot from the "Display Setup".  The core dimensions (cross section area in mm2,
effective length in mm) and number of windings can be set in the context menu
for the plot (right click on the plot).

Measurement limits
^^^^^^^^^^^^^^^^^^

The "Limits" window loads versioned JSON limit sets and checks captured
traces against them.  A limit set describes frequency bands and the bounds a
measurement has to satisfy inside them, for example a maximum S11 VSWR, a
minimum S11 return loss in dB, or a maximum S21 insertion loss in dB.  Bounds
are inclusive and are only evaluated at measured samples.

Each check reports PASS, FAIL or UNAVAILABLE; the last means the question
could not be decided (missing S-parameter channel, band not covered by the
sweep, incomplete acquisition or an invalid sample) and is never treated as a
pass.  Report previews and HTML/CSV exports freeze the selected limit set
together with its results, so a later measurement or a different selection
cannot change an already built report.

The file format is documented in `docs/LIMIT_SETS.md <docs/LIMIT_SETS.md>`_,
with working examples in ``tests/data/example.nvlimits`` (schema version 1)
and ``tests/data/example_v2.nvlimits``.

More features
^^^^^^^^^^^^^

`docs/FEATURES.md <docs/FEATURES.md>`_ is the index of the newer workflows,
each with its own document:

* `Touchstone replay <docs/REPLAY.md>`_ - run against a recorded ``s1p`` or
  ``s2p`` instead of a device, with no extrapolation.
* `Antenna tuning <docs/ANTENNA_TUNING.md>`_ - guided overview, minimum,
  zoom, repeat and one-port export.
* `Sessions <docs/SESSION_FORMAT.md>`_ - a versioned ``.nvsession`` archive of
  traces, markers, charts, sweep settings and provenance.
* `Processing pipeline <docs/PROCESSING.md>`_ - opt-in, ordered corrections
  recorded in provenance; smoothing and averaging change a measurement rather
  than improving it.
* `Bulk calibration correction <docs/CALIBRATION_BULK.md>`_ - the per-segment
  correction path used by the sweep worker.
* `Offline walkthrough <docs/OFFLINE_WALKTHROUGH.md>`_ - the whole application
  exercised with the built-in simulator.

Current restrictions, stated so no feature description overstates them: only
S11 and S21 are acquired (S12 and S22 are never measured, which is different
from measuring zero); replaying an ``s1p`` file provides S11 only, with no
fabricated S21; an S2P export writes a channel the measurement never had as an
explicit zero placeholder named in the file comments; PDF reports are not
implemented (HTML and CSV are); TDR shows a single trace.
Provenance: every capture records source, simulated status, device,
application version, calibration and processing identity, and simulated data
is labelled ``[SIMULATED: <scenario>]`` in the window title and with
``SIMULATED DATA`` in exported Touchstone comments.

Latest Changes
^^^^^^^^^^^^^^

See Notes in github release notes.

* Using pyside6
* Use uv and pyproject.toml

Contributing
------------

First off, thanks for taking the time to contribute! Contributions are what
make the open-source community such an amazing place to learn, inspire, and
create. Any contributions you make will benefit everybody else and are
**greatly appreciated**.

Please read `our contribution guidelines <docs/CONTRIBUTING.md>`_. They
contain the environment setup, the test and lint commands, the documentation
build, the packaging steps and an explicit list of what has and has not been
verified on which platform. A short reStructuredText summary is also part of
the Sphinx documentation.

License
-------

This software is licensed under version 3 of the GNU General Public License. It
comes with NO WARRANTY.

You can use it, commercially as well. You may make changes to the code, but I
(and the license) ask that you give these changes back to the community.

References
----------


* Ohan Smit wrote an introduction to using the application:
  [https://zs1sci.com/blog/nanovnasaver/]
* HexAndFlex wrote a 3-part (thus far) series on Getting Started with the
  NanoVNA:
  [https://hexandflex.com/2019/08/31/getting-started-with-the-nanovna-part-1/]
  - Part 3 is dedicated to NanoVNASaver:
  [https://hexandflex.com/2019/09/15/getting-started-with-the-nanovna-part-3-pc-software/]
* Gunthard Kraus did documentation in English and German:
  [http://www.gunthard-kraus.de/fertig_NanoVNA/English/]
  [http://www.gunthard-kraus.de/fertig_NanoVNA/Deutsch/]

Acknowledgements
----------------

Original application by Rune B. Broberg (5Q5R)

Contributions and changes by Holger Müller (DG5DBH), David Hunt and others.

TDR inspiration shamelessly stolen from the work of Salil (VU2CWA) at
https://nuclearrambo.com/wordpress/accurately-measuring-cable-length-with-nanovna/

TDR cable types by Larry Goga.

Bugfixes and Python installation work by Ohan Smit.

Thanks to everyone who have tested, commented and inspired.  Particular thanks
go to the alpha testing crew who suffer the early instability of new versions.

This software is available free of charge. If you read all this way, and you
*still* want to support it, you may donate to the developer using the button
below:


.. image:: https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif
   :target: https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=T8KTGVDQF5K6E&item_name=NanoVNASaver+Development&currency_code=EUR&source=url
   :alt: Paypal

