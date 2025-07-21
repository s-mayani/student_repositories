# === Configuration ===
VENV        := ./venv_pycosmo
PARAM_FILE  := pycosmo_input_32.param
CACHE_DIR   := $(HOME)/_cache/sympy2c
ROOT_DIR    := $(shell pwd)

# === Platform Detection ===
UNAME_S     := $(shell uname -s)
UNAME_R     := $(shell uname -r)
PLATFORM    := unknown

ifeq ($(UNAME_S),Linux)
  ifneq (,$(findstring Microsoft,$(UNAME_R)))
    PLATFORM := wsl
  else
    PLATFORM := linux
  endif
endif

ifeq ($(UNAME_S),Darwin)
  PLATFORM := mac
endif

# === Targets ===

.PHONY: all venv setup install_deps clear_cache test test_full run phase_space clean help

all: setup

# Create virtual environment with Python 3.9 if not exists
venv:
	@echo "Setting up virtual environment in $(VENV)..."
	@if [ ! -d "$(VENV)" ]; then \
		virtualvenv -p python3.9 $(VENV); \
	fi

# Install system and Python dependencies
install_deps: venv
	@echo "Installing/upgrading Python build tools..."
	@$(VENV)/bin/python -m pip install --upgrade pip setuptools wheel

	@echo "Installing system dependencies..."
	@if command -v apt-get >/dev/null; then \
		echo "Detected apt-based system."; \
		if command -v sudo >/dev/null; then \
			sudo apt-get update; \
			sudo apt-get install -y build-essential libgsl-dev autoconf automake libtool gfortran; \
			sudo apt-get install -y texlive-latex-base texlive-fonts-recommended dvipng cm-super; \
		else \
			echo "sudo not found; skipping apt installs."; \
		fi; \
	elif command -v brew >/dev/null; then \
		echo "Detected macOS with Homebrew."; \
		brew install gsl gcc; \
	else \
		echo "Unknown system. Please install GSL and gfortran manually."; \
	fi

	@echo "Installing Python dependencies..."
	@$(VENV)/bin/pip install -r requirements.txt

# Full environment setup
setup: venv install_deps
	@echo "Setup complete. Run 'source setup_pycosmo.sh' before continuing."

# Run basic tests
test:
	@echo "Running tests..."
	PYTHON_SYMPY2C_USE_SHARED_GSL=1 $(VENV)/bin/python -m pytest tests/ || echo "Tests failed"

# Clear sympy2c cache and rerun tests
test_full: install_deps clear_cache
	@echo "Running full test suite (after clearing cache)..."
	PYTHON_SYMPY2C_USE_SHARED_GSL=1 $(VENV)/bin/python -m pytest tests/

# Run with custom parameter file
run:
	@if [ -z "$(PARAM)" ]; then \
		echo "Error: No parameter file specified. Use PARAM=your_param_file.param"; \
		exit 1; \
	fi
	@echo "Running power spectrum generator with $(PARAM)..."
	$(VENV)/bin/python -m power_spectrum_generation.custom_power_spectrum $(PARAM)

# Clear sympy2c cache
clear_cache:
	@echo "Clearing cache at $(CACHE_DIR)..."
	rm -rf $(CACHE_DIR)

# Clean all generated files
clean:
	@echo "Cleaning virtualenv and bytecode..."
	rm -rf build *.o $(VENV) __pycache__
	find . -name "*.pyc" -delete

# Help message
help:
	@echo "Usage:"
	@echo "  make setup                # create venv and install deps"
	@echo "  make run PARAM=...       # run PyCosmo generator"
	@echo "  make test / test_full    # run tests"
	@echo "  make clean               # remove artifacts and venv"
