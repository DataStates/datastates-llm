#!/bin/bash
set -e
set -o pipefail

# Determine Python site-packages install location
INSTALL_PATH=$(python3 -c "
import os, sysconfig, site
purelib = sysconfig.get_paths().get('purelib')
if 'site-packages' in purelib and ('conda' not in purelib and 'envs' not in purelib and 'venv' not in purelib):
    purelib = site.getusersitepackages()
print(purelib)
")

echo "[INFO] Installing DataStates-LLM into: $INSTALL_PATH"

# Step 1: Configure and build the C++ core
cmake -B build -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH"
cmake --build build -j$(nproc)
cmake --install build

echo "[INFO] C++ core installed to $INSTALL_PATH"

# Step 2: Install Python package for LLMs
cd llm
pip install .
cd ..

echo "[INFO] DataStates-LLM installation complete."
