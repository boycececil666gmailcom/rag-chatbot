#!/bin/bash

# Load configurations from .env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | tr -d '\r' | xargs)
fi

echo "========================================================"
echo "[1/2] Activating Python Virtual Environment"
echo "========================================================"
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "========================================================"
echo "[2/2] Running Unit Tests"
echo "========================================================"
python -m pytest tests/test_api_unit.py -v
