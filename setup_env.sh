#!/bin/bash

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo -e "\n\033[1;96m========================================================\033[0m"
    echo -e "\033[1;92m>>> [1/2] [\$(basename \"\${BASH_SOURCE[0]}\")] Creating Virtual Environment\033[0m"
    echo -e "\033[1;96m========================================================\033[0m\n"
    python -m venv venv || python3 -m venv venv || exit 1
fi

# Activate python virtual environment
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate 2>/dev/null

# Install dependencies if requirements.txt is present
if [ -f "requirements.txt" ]; then
    echo -e "\n\033[1;96m========================================================\033[0m"
    echo -e "\033[1;92m>>> [2/2] [\$(basename \"\${BASH_SOURCE[0]}\")] Installing Dependencies\033[0m"
    echo -e "\033[1;96m========================================================\033[0m\n"
    pip install -r requirements.txt
fi
