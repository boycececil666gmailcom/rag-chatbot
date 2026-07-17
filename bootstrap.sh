#!/bin/bash

# Load configurations from .env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "========================================================"
echo "[1/2] Setting Up Python Virtual Environment"
echo "========================================================"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
    VENV_PYTHON="venv/Scripts/python"
    VENV_PIP="venv/Scripts/pip"
else
    source venv/bin/activate
    VENV_PYTHON="venv/bin/python"
    VENV_PIP="venv/bin/pip"
fi

echo "========================================================"
echo "[2/2] Installing Dependencies and Starting Server"
echo "========================================================"

$VENV_PIP install -r requirements.txt

echo "Starting backend server..."
$VENV_PYTHON -m src.main
