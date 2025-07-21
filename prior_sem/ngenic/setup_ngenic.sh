#!/bin/bash
# Activate N-GenIC virtual environment and load GSL 2.7.1 + FFTW

# Define the virtual environment path
VENV_DIR="$(pwd)/venv_ngenic"

# Locate GSL 2.7.1
if [ -n "$GSL_BASE_NGENIC" ]; then
    GSL_DIR="$GSL_BASE_NGENIC/lib"
elif GSL_DIR=$(ls -d /cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-12.2.0/gsl-2.7.1-*/lib 2>/dev/null | head -n1); then
    :
elif [ -d "$HOME/gsl-2.7.1/lib" ]; then
    GSL_DIR="$HOME/gsl-2.7.1/lib"
else
    echo "❌ ERROR: Could not locate GSL 2.7.1 library directory."
    echo "👉 Set GSL_BASE_NGENIC to the base directory containing GSL (e.g., /path/to/gsl-2.7.1)"
    return 1  # Important: `return`, not `exit`, since script is sourced
fi

# Set FFTW location
FFTW_DIR="$HOME/fftw2_mpi"

# Add libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$FFTW_DIR/lib:$GSL_DIR:$LD_LIBRARY_PATH"
export PATH="$FFTW_DIR/bin:$PATH"

# Create the virtual environment if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "[setup_ngenic.sh] Creating N-GenIC virtual environment..."
    python3 -m venv --without-pip "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo "✅ [setup_ngenic.sh] Environment activated:"
echo "   GSL dir:   $GSL_DIR"
echo "   FFTW dir:  $FFTW_DIR"
