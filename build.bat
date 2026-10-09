@echo off
rem Build dist\molabi.exe (one file, no console). First time: uv venv .venv && uv pip install --python .venv\Scripts\python.exe -r requirements.txt pyinstaller pillow
cd /d %~dp0
.venv\Scripts\python.exe pool.py test || exit /b 1
.venv\Scripts\python.exe -m PyInstaller --noconfirm --onefile --windowed --name molabi --icon molabi.ico --add-data "ui;ui" app.py || exit /b 1
if not exist dist\config.json if exist config.json copy config.json dist\ >nul
echo Built dist\molabi.exe
