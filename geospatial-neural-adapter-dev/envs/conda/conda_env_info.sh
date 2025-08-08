#!/bin/bash
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_BASE_PATH="${CONDA_DIR}/../.."

DEFAULT_PYTHON_VERSION="3.10.13"
POETRY_VERSION="2.0.0"

COLOR_MAP_PATH="${CONDA_DIR}/../.bin/color_map.sh"
EXIT_CODE_PATH="${CONDA_DIR}/../.bin/exit_code.sh"
