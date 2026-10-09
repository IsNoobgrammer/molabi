#!/bin/sh
# Build dist/molabi.app (macOS) or dist/molabi (Linux). First time:
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pyinstaller pillow
set -e
cd "$(dirname "$0")"
.venv/bin/python pool.py test
.venv/bin/python -m PyInstaller --noconfirm --windowed --name molabi --icon molabi.ico --add-data "ui:ui" app.py
echo "Built dist/molabi"
