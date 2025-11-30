#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./install.sh                    # install into active Python env, build Python bindings (default)
#   ./install.sh /some/path         # install into /some/path, build Python bindings
#   ./install.sh /some/path off     # install into /some/path, NO Python bindings
#   ./install.sh "" off             # install into Python env, NO Python bindings

INSTALL_PREFIX="${1:-}"
WITH_PYTHON_ARG="${2:-on}"

# Normalize second arg to ON/OFF
case "${WITH_PYTHON_ARG,,}" in
    on|yes|y|1)
        BUILD_PYTHON_BINDINGS="ON"
        ;;
    off|no|n|0)
        BUILD_PYTHON_BINDINGS="OFF"
        ;;
    *)
        echo "[ERROR] Second argument (WITH_PYTHON) must be one of: on/off/yes/no/1/0"
        exit 1
        ;;
esac

# If no explicit install prefix was passed, use the active Python env's purelib
if [ -z "$INSTALL_PREFIX" ]; then
    # Prefer 'python' if available, fall back to 'python3'
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "[ERROR] Could not find 'python' or 'python3' in PATH."
        echo "        Either install Python or pass an explicit install prefix as the first argument."
        exit 1
    fi

    INSTALL_PREFIX="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    echo "[INFO] No install prefix specified. Using Python env site-packages:"
    echo "       $INSTALL_PREFIX"
else
    echo "[INFO] Using custom install prefix: $INSTALL_PREFIX"
    echo "[INFO] After installation, ensure Python can find DataStates-LLM by adding:"
    echo "       export PYTHONPATH=\"$INSTALL_PREFIX:\$PYTHONPATH\""
    echo "       # (assuming the 'datastates' package will live under: $INSTALL_PREFIX/datastates)"
fi

echo "[INFO] BUILD_PYTHON_BINDINGS = $BUILD_PYTHON_BINDINGS"

# Configure and build
cmake -B build \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DBUILD_PYTHON_BINDINGS="$BUILD_PYTHON_BINDINGS"

cmake --build build -j"$(nproc)"
cmake --install build

echo "[INFO] DataStates-LLM installation complete."
