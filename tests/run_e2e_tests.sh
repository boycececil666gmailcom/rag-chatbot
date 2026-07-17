#!/bin/bash

# Set script directory to run independent of execution location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Define log_step function for beautiful output
log_step() {
    echo -e "\n\033[1;96m========================================================\033[0m"
    echo -e "\033[1;92m>>> $1 [$(basename "${BASH_SOURCE[0]}")] $2\033[0m"
    echo -e "\033[1;96m========================================================\033[0m\n"
}

# Step 1: Initialize local python environment
log_step "[1/2]" "Setting Up Python Virtual Environment & Dependencies"

# Create venv if not exists
if [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    python -m venv ../venv || python3 -m venv ../venv || exit 1
fi

# Resolve virtual environment python and pip executable paths
if [ -f "../venv/Scripts/python.exe" ]; then
    VENV_PYTHON="../venv/Scripts/python.exe"
    VENV_PIP="../venv/Scripts/pip.exe"
elif [ -f "../venv/Scripts/python" ]; then
    VENV_PYTHON="../venv/Scripts/python"
    VENV_PIP="../venv/Scripts/pip"
elif [ -f "../venv/bin/python" ]; then
    VENV_PYTHON="../venv/bin/python"
    VENV_PIP="../venv/bin/pip"
else
    VENV_PYTHON="python"
    VENV_PIP="pip"
fi

# Install dependencies if requirements.txt is present
# if [ -f "../requirements.txt" ]; then
#     echo "Checking/installing dependencies..."
#     "$VENV_PIP" install -r ../requirements.txt
# fi

# Configure Target Live API Gateway URL (defaults to Kubernetes port-forward on port 8080)
export GATEWAY_URL=${GATEWAY_URL:-"http://localhost:8080"}

# Verify live gateway connectivity before running tests
echo "Verifying API Gateway connectivity at ${GATEWAY_URL}..."
if ! curl -sf "${GATEWAY_URL}/health" >/dev/null; then
    echo -e "\033[1;31mError: Gateway at ${GATEWAY_URL} is unreachable!\033[0m"
    echo "Please ensure you have deployed to Kubernetes (e.g. via bootstrap.sh) and that port-forwarding is running on port 8080."
    exit 1
fi

# Step 2: Run End-to-End Tests
log_step "[2/2]" "Running End-to-End Tests"
"$VENV_PYTHON" -m pytest test_api_e2e.py -v

