#!/bin/bash
echo "============================================================"
echo "  MIDAS GDP Forecasting Project - Setup"
echo "  Andreou, Ghysels, Kourtellos (2013) Replication"
echo "============================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo "Please install Python 3.9+ first"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "     Virtual environment already exists, skipping creation."
else
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "     Done!"
fi

echo
echo "[2/4] Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi
echo "     Done!"

echo
echo "[3/4] Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo "     Done!"

echo
echo "[4/4] Installing dependencies..."
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi
echo "     Done!"

echo
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo
echo "To activate the environment manually, run:"
echo "    source .venv/bin/activate"
echo
echo "To run Jupyter Notebook:"
echo "    jupyter notebook main.ipynb"
echo
echo "Or open the project in VS Code and select the .venv interpreter."
echo "============================================================"
