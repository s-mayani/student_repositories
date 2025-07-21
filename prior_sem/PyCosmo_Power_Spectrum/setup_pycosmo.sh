#!/bin/bash
# setup_pycosmo.sh - Activate PyCosmo venv and load GSL 2.8

# Prevent running as executable
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: This script must be sourced, not executed."
  echo "Use: source setup_pycosmo.sh"
  exit 1
fi

# Define virtual environment path (absolute)
VENV_DIR="$(pwd)/venv_pycosmo"

# Detect or set GSL 2.8 path
if [ -n "$GSL_BASE_PYCOSMO" ]; then
  export GSL_DIR="$GSL_BASE_PYCOSMO/lib"
elif GSL_BASE_PYCOSMO=$(echo /cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/gsl-2.8-*/lib 2>/dev/null | head -n1); then
  export GSL_DIR="$GSL_BASE_PYCOSMO"
elif [ -d "$HOME/gsl-2.8/lib" ]; then
  export GSL_DIR="$HOME/gsl-2.8/lib"
else
  echo "WARNING: Could not locate GSL 2.8 library directory. Please set GSL_BASE_PYCOSMO environment variable if needed."
  export GSL_DIR=""
fi

# Export library paths, only if GSL_DIR is set
if [ -n "$GSL_DIR" ]; then
  export LD_LIBRARY_PATH="$GSL_DIR:$LD_LIBRARY_PATH"
fi

# Force correct libstdc++ from GCC 12.2.0
GCC_LIB_DIR=$(dirname $(g++ -print-file-name=libstdc++.so.6))
export LD_LIBRARY_PATH="$GCC_LIB_DIR:$LD_LIBRARY_PATH"

if ! strings "$GCC_LIB_DIR/libstdc++.so.6" | grep -q GLIBCXX_3.4.29; then
  echo "Error: GLIBCXX_3.4.29 not found in $GCC_LIB_DIR/libstdc++.so.6"
  return 1
fi

# Create 'initial_conditions' directory if it does not exist
INIT_DIR="$(pwd)/initial_conditions"
if [ ! -d "$INIT_DIR" ]; then
  mkdir "$INIT_DIR"
fi

# Check if virtual environment exists; if not, create it using virtualenv
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment not found at $VENV_DIR"
  echo "Creating virtual environment using virtualenv..."

  # Ensure ~/.local/bin is in PATH so user-installed virtualenv is found
  export PATH="$HOME/.local/bin:$PATH"

  # Check if virtualenv is available; if not, try to install it
  if ! command -v virtualenv >/dev/null 2>&1; then
  echo "[setup_pycosmo.sh] 'virtualenv' not found. Attempting to install it..."

  # Try installing virtualenv using Python's built-in pip module
  if ! python3 -m ensurepip --version >/dev/null 2>&1; then
  echo "Error: ensurepip is not available. Cannot install pip/virtualenv."
  return 1
  fi

  echo "[setup_pycosmo.sh] Installing 'virtualenv' via python3 -m pip..."
  python3 -m pip install --user virtualenv

  # Add ~/.local/bin to PATH if not already there
  export PATH="$HOME/.local/bin:$PATH"

  # Recheck availability
  if ! command -v virtualenv >/dev/null 2>&1; then
    echo "Error: virtualenv installation failed or is still not in PATH."
    return 1
  fi

  echo "[setup_pycosmo.sh] 'virtualenv' successfully installed."
  fi


  virtualenv -p python3.9 "$VENV_DIR"

  if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment with virtualenv."
    return 1
  fi

  echo "Virtual environment created with virtualenv."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo "[setup_pycosmo.sh] PyCosmo environment activated with GSL 2.8."
