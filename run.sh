#!/bin/bash

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
else
    source venv/bin/activate
fi

echo "========================================================"
echo "[2/2] Installing Dependencies and Starting Server"
echo "========================================================"
pip install -r requirements.txt

echo "Starting backend server..."
python -m src.main
