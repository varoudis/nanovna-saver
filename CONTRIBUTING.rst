============
Contributing
============

Welcome to the ``nanovna-saver`` contributor's guide.

This reStructuredText page is the version that appears in the Sphinx build.
The maintained, complete guide - environment setup, test and lint commands,
documentation build, packaging and the verification matrix - is
`CONTRIBUTING.md <CONTRIBUTING.md>`_ (``docs/CONTRIBUTING.md`` in the
repository). Read that first; the policy below is the summary it inherits.

Please note that all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. See the
`code of conduct <CODE_OF_CONDUCT.md>`_.


Quick start
===========

The project is ``uv``-managed and the lock file is authoritative::

    git clone https://github.com/NanoVNA-Saver/nanovna-saver
    cd nanovna-saver
    uv sync --locked
    uv run task ui-compile
    QT_QPA_PLATFORM=offscreen uv run pytest tests -q
    uv run ruff check --no-fix src tests
    uv run ruff format --check src tests

``uv run task ui-compile`` is the only working resource-compilation command;
``uv run ui-compile`` and ``uv run task compile`` do not exist. The
generated modules are checked in, so this step is only required after
changing a ``.ui`` or ``.qrc`` file, but CI runs it before the tests.

Do not install Qt, PyPI packages or ``tox`` environments by hand: PySide6 and
every other dependency come from ``uv.lock``, and a Homebrew or distro Qt is
the wrong stack.

Documentation
=============

Public documentation lives in ``docs/*.md`` and is indexed by
``docs/FEATURES.md``. Build it locally with::

    make -C docs html

The output appears in ``docs/_build/html`` and can be previewed with::

    python3 -m http.server --directory 'docs/_build/html'

``tox -e docs`` requires package-index access for ``docs/requirements.txt``;
it fails offline. ``docs/api/`` and ``docs/_build/`` are generated and must
never be committed.

Every command in the documentation must exist, and every platform claim must
name the platform it was verified on or be labelled unverified.

Code contributions
==================

Submit an issue
---------------

Before you work on any non-trivial code contribution it is best to first
create a report in the `issue tracker`_ to start a discussion on the subject.
This often provides additional considerations and avoids unnecessary work.

Create a branch
---------------

Create a branch to hold your changes::

    git checkout -b feat/my-feature

Never work on the main branch. Add docstrings to new functions, modules and
classes, especially if they are part of a public API. Add unit tests and
documentation when your contribution adds a feature rather than only fixing a
bug.

Style
-----

``ruff`` is the only formatter and linter::

    uv run ruff format src tests
    uv run ruff check --no-fix src tests

Line length is 80 and the target is ``py311``. ``ruff check`` without
``--no-fix`` edits files (``fix = true`` in ``pyproject.toml``), so reproduce
CI by passing ``--no-fix``. ``pre-commit`` is available but installs its hook
environments from the network and therefore fails offline.

Commit messages use `conventional commits`_::

    git commit -m 'feat: add amazing_feature'

Push the branch and open a pull request::

    git push -u origin feat/my-feature

Maintainer tasks
================

Releases are driven by ``setuptools-scm`` over the git history - never edit a
version by hand:

#. Make sure the test and lint commands above pass.
#. Tag the current commit on the main branch, e.g. ``v1.2.3``, and push the
   tag.
#. Remove stale artifacts with ``rm -rf dist build``.
#. Build distributions with ``uv build`` and check that the version in
   ``dist`` matches the tag and carries neither a git hash nor ``+dirty``.
#. Publish with ``uv publish``.
#. On macOS, build the application bundle separately with
   ``./build-macos-app.sh`` and check ``./dist/NanoVNASaver.app``'s
   ``--version`` output.

Windows and Linux binaries are produced by the tag-triggered
``release_win.yml`` and ``release_linux.yml`` workflows. There is no macOS
release workflow; the local script is the only macOS build path.

Troubleshooting
===============

#. Fetch all tags from the upstream repository before building a version
   string: ``git describe --abbrev=0 --tags`` should return the version you
   expect. A fork must push its tags too.
#. If the application reports the version ``unknown``, it is being run from an
   uninstalled source tree; run ``uv sync --locked`` and start it with
   ``uv run NanoVNASaver``.
#. If ``uv sync --locked`` fails because the lock file is out of date, do not
   work around it with a plain ``uv sync`` in a release branch - update the
   lock in its own commit.
#. ``Pytest can drop you``_ into an interactive session on failure with
   ``uv run pytest tests -q --pdb``.
#. If Qt cannot connect to a display, set ``QT_QPA_PLATFORM=offscreen``;
   tests and the CI smoke job run that way.

.. _conventional commits: https://www.conventionalcommits.org
.. _issue tracker: https://github.com/NanoVNA-Saver/nanovna-saver/issues
.. _Pytest can drop you: https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest
