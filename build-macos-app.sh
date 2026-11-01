#!/usr/bin/env bash
#
# Builds an unsigned, locally usable NanoVNASaver.app and a .tar.gz of it on
# macOS, from any working directory.
#
# Requirements:
#   - macOS
#   - uv (https://docs.astral.sh/uv/), e.g. brew install uv
#   - a CPython 3.10 - 3.13 interpreter (set PYTHON=/path/to/python3.13 to pick)
#
# PySide6 (and Qt) come from the locked dependencies inside a throwaway build
# environment; no Homebrew Qt or PyQt is used. Dependencies are installed from
# uv.lock so the pinned PyInstaller matches the pinned NumPy/SciPy.
#
# Usage:
#   ./build-macos-app.sh [-h|--help]
#
# Environment overrides:
#   PYTHON  interpreter used for the build environment
#
# Output:
#   dist/NanoVNASaver.app
#   dist/NanoVNASaver.app-<arch>.tar.gz

set -euo pipefail

readonly APP_NAME="NanoVNASaver"
readonly EXE_NAME="NanoVNASaver"
readonly ICON="${APP_NAME}_48x48.icns"
readonly BUNDLE_ID="io.github.nanovnasaver.nanovna-saver"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

info() {
    printf '==> %s\n' "$*"
}

usage() {
    sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
    "") ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
esac

# --- repository root, so the script works from any working directory ---------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${SCRIPT_DIR}"
cd "${ROOT}"

# --- preflight ---------------------------------------------------------------
[ "$(uname -s)" = "Darwin" ] || die "this script builds the macOS app bundle and must run on macOS"

for required in pyproject.toml uv.lock "src/${APP_NAME}/__main__.py" "${ICON}"; do
    [ -e "${required}" ] || die "missing ${required}; run this script from a full source checkout"
done

command -v uv >/dev/null 2>&1 || die "uv not found; install it with 'brew install uv'"
command -v tar >/dev/null 2>&1 || die "tar not found"

python_supported() {
    "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' \
        >/dev/null 2>&1
}

if [ -n "${PYTHON:-}" ]; then
    [ -x "${PYTHON}" ] || die "PYTHON=${PYTHON} is not executable"
    python_supported "${PYTHON}" \
        || die "PYTHON=${PYTHON} is outside the supported range (3.10 - 3.13)"
    PYTHON_BIN="${PYTHON}"
else
    PYTHON_BIN=""
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "${candidate}" >/dev/null 2>&1 && python_supported "${candidate}"; then
            PYTHON_BIN="$(command -v "${candidate}")"
            break
        fi
    done
    [ -n "${PYTHON_BIN}" ] \
        || die "no Python 3.10 - 3.13 found; install one or set PYTHON=/path/to/python3.13"
fi

ARCH="$(uname -m)"
DIST_DIR="${ROOT}/dist"
STAGING="${ROOT}/build/macos-app"

# --- throwaway build environment --------------------------------------------
# The virtualenv and the snapshot of the generated sources live in one private
# temporary directory that is removed on exit, so pre-existing environments are
# never reused or modified. PyInstaller stages into ${STAGING}, the script-owned
# path that is replaced on every run.
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nanovna-macos-build.XXXXXX")"
VENV_DIR="${WORK_DIR}/env"
VENV_PY="${VENV_DIR}/bin/python"
GENERATED_BACKUP="${WORK_DIR}/generated"
GENERATED_FILES=()

# ui_compile (run by the in-tree build backend during install as well as below)
# rewrites these generated sources in the working tree. Snapshot them before
# the first command that can trigger it and restore them on exit, so the tree
# is left exactly as found - including files that were already modified.
collect_generated_files() {
    local spec_file spec_dir spec_base
    while IFS= read -r -d '' spec_file; do
        spec_dir="$(dirname -- "${spec_file}")"
        spec_base="$(basename -- "${spec_file}")"
        case "${spec_base}" in
            *.ui) GENERATED_FILES+=("${spec_dir}/${spec_base%.ui}.py") ;;
            *.qrc) GENERATED_FILES+=("${spec_dir}/${spec_base%.qrc}_rc.py") ;;
        esac
    done < <(find "${ROOT}/src" -type f \( -name '*.ui' -o -name '*.qrc' \) -print0)
}

SNAPSHOT_COMPLETE=0
snapshot_generated_files() {
    local generated relative
    for generated in "${GENERATED_FILES[@]}"; do
        relative="${generated#"${ROOT}/"}"
        [ -f "${generated}" ] || continue
        mkdir -p "${GENERATED_BACKUP}/$(dirname -- "${relative}")"
        cp -p "${generated}" "${GENERATED_BACKUP}/${relative}"
    done
    SNAPSHOT_COMPLETE=1
}

cleanup() {
    local status=$? generated relative kept failed=0
    trap - EXIT
    for generated in "${GENERATED_FILES[@]}"; do
        relative="${generated#"${ROOT}/"}"
        kept="${GENERATED_BACKUP}/${relative}"
        if [ -f "${kept}" ]; then
            cp -p "${kept}" "${generated}" || failed=1
        elif [ "${SNAPSHOT_COMPLETE}" -eq 1 ]; then
            rm -f "${generated}" || failed=1
        fi
    done
    if [ "${failed}" -ne 0 ]; then
        printf 'error: could not restore generated files; their previous contents are kept in %s\n' \
            "${GENERATED_BACKUP}" >&2
        [ "${status}" -ne 0 ] || status=1
    else
        rm -rf "${WORK_DIR}"
    fi
    exit "${status}"
}

trap cleanup EXIT
collect_generated_files
snapshot_generated_files

info "build environment: ${VENV_DIR} (temporary)"
info "interpreter: ${PYTHON_BIN} ($("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())'))"

# A dedicated UV_PROJECT_ENVIRONMENT keeps the build out of the checkout's .venv.
# --locked refuses to rewrite uv.lock and fails if it is stale.
#
# --reinstall-package forces the project itself to be rebuilt instead of reusing
# a cached editable wheel. uv keys that wheel on its declared build inputs, not
# on Git state, so a cached wheel from another checkout or commit would install
# a stale setuptools_scm version and the bundle would misreport its commit. Only
# NanoVNASaver is reinstalled; every locked dependency is still cache-served.
UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync --locked --python "${PYTHON_BIN}" \
    --reinstall-package "${APP_NAME}"

[ -x "${VENV_PY}" ] || die "expected interpreter missing after sync: ${VENV_PY}"
"${VENV_PY}" -c 'import PyInstaller' >/dev/null 2>&1 \
    || die "PyInstaller is missing from the build environment; check uv.lock"

# pyside6-uic / pyside6-rcc must come from the build environment.
export PATH="${VENV_DIR}/bin:${PATH}"

# --- compile Qt resources ---------------------------------------------------
info "compiling Qt UI and resource files"
"${VENV_PY}" -m src.tools.ui_compile

# --- version from installed package metadata --------------------------------
VERSION="$("${VENV_PY}" -c 'from importlib.metadata import version; print(version("NanoVNASaver"))')"
[ -n "${VERSION}" ] || die "could not read the NanoVNASaver version from package metadata"
info "NanoVNASaver version: ${VERSION}"

# --- build ------------------------------------------------------------------
rm -rf "${STAGING}"
mkdir -p "${STAGING}" "${DIST_DIR}"

info "building ${APP_NAME}.app"
# Absolute paths: PyInstaller reads the generated spec from ${STAGING}, so
# relative icon/script paths would resolve against the staging directory.
"${VENV_PY}" -m PyInstaller \
    --onedir \
    --window \
    --clean \
    -y \
    --name "${EXE_NAME}" \
    --paths "${ROOT}/src" \
    --icon "${ROOT}/${ICON}" \
    --recursive-copy-metadata "${APP_NAME}" \
    --osx-bundle-identifier "${BUNDLE_ID}" \
    --distpath "${STAGING}/dist" \
    --workpath "${STAGING}/work" \
    --specpath "${STAGING}" \
    "${ROOT}/src/${APP_NAME}/__main__.py"

APP_PATH="${STAGING}/dist/${APP_NAME}.app"
APP_EXE="${APP_PATH}/Contents/MacOS/${EXE_NAME}"
[ -d "${APP_PATH}" ] || die "PyInstaller did not produce ${APP_PATH}"
[ -x "${APP_EXE}" ] || die "packaged executable missing: ${APP_EXE}"

info "checking the packaged executable reports its version"
PACKAGED_VERSION="$(QT_QPA_PLATFORM=offscreen "${APP_EXE}" --version 2>&1)" || {
    printf '%s\n' "${PACKAGED_VERSION}" >&2
    die "packaged executable failed to start (import/resource error)"
}
case "${PACKAGED_VERSION}" in
    *"${VERSION}"*) ;;
    *) printf '%s\n' "${PACKAGED_VERSION}" >&2
       die "packaged executable reported a version that does not match ${VERSION}" ;;
esac
printf '    %s\n' "${PACKAGED_VERSION}"

# --- publish ----------------------------------------------------------------
info "publishing to ${DIST_DIR}"
rm -rf "${DIST_DIR}/${APP_NAME}.app"
mv "${APP_PATH}" "${DIST_DIR}/${APP_NAME}.app"

ARCHIVE="${DIST_DIR}/${APP_NAME}.app-${ARCH}.tar.gz"
rm -f "${ARCHIVE}"
tar -C "${DIST_DIR}" -zcf "${ARCHIVE}" "${APP_NAME}.app"

info "done"
printf '    app:     %s\n' "${DIST_DIR}/${APP_NAME}.app"
printf '    archive: %s\n' "${ARCHIVE}"
printf '    open it: open "%s"\n' "${DIST_DIR}/${APP_NAME}.app"
printf '    note: unsigned (ad-hoc) build; Gatekeeper may refuse the first launch,\n'
printf '          allow it via System Settings > Privacy & Security > Open Anyway.\n'