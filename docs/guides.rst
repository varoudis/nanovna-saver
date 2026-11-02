.. _guides:

Workflow guides
===============

The workflow documentation is maintained as Markdown beside the code, so the
same file is readable in the repository, on the hosting site and here.
Selecting a guide downloads the exact Markdown file from this build.

Feature guides
--------------

* :download:`Feature index <FEATURES.md>` - what exists, and the two-port and
  replay restrictions that apply to parts of it.
* :download:`Touchstone replay <REPLAY.md>` - sweeping against a recorded
  file instead of a device.
* :download:`Antenna tuning <ANTENNA_TUNING.md>` - guided overview, minimum
  search and the limits it respects.
* :download:`Sessions <SESSION_FORMAT.md>` - the versioned ``.nvsession``
  archive.
* :download:`Processing pipeline <PROCESSING.md>` - the ordered, opt-in
  corrections and what each one changes.
* :download:`Bulk calibration correction <CALIBRATION_BULK.md>` - per-segment
  calibration correction.
* :download:`Calibration age and verification <CALIBRATION_HEALTH.md>` - what
  is recorded with a calibration and what a verification does and does not
  show.
* :download:`Limit sets <LIMIT_SETS.md>` - the ``.nvlimits`` document,
  metrics, segments and slopes.
* :download:`Phase derivatives <PHASE_DERIVATIVES.md>` - the diagnostics and
  the units of each plot.

Operating and automation
------------------------

* :download:`Offline walkthrough <OFFLINE_WALKTHROUGH.md>` - the whole
  application exercised against the simulator, with no hardware.
* :download:`Automation <AUTOMATION.md>` - the Qt-free API, the command line,
  test sequences and the verdict log.
* :download:`Reports <REPORTS.md>` - HTML, CSV and PDF export from one
  frozen model, and why wide tables print as labelled rows.

Building and contributing
-------------------------

* :download:`Installation <INSTALLATION.md>`
* :download:`Contributing <CONTRIBUTING.md>`
* :download:`Packaging verification <VERIFY_PACKAGING.md>` - what was run
  against the built wheel, and what still needs Windows, Linux or a device.
