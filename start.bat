@echo off
cd /d "%~dp0"

echo Checking for Python 3.12...
py -3.12 --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 3.12 not found. Installing it now, this may take a few minutes...
    winget install --id Python.Python.3.12 -e --silent
    echo Please close this window and re-run start.bat once the install finishes.
    pause
    exit
)

echo Checking for Python 3.10...
py -3.10 --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 3.10 not found. Installing it now, this may take a few minutes...
    winget install --id Python.Python.3.10 -e --silent
    echo Please close this window and re-run start.bat once the install finishes.
    pause
    exit
)

echo Setting up main app environment...
if not exist venv_main (
    py -3.12 -m venv venv_main
    venv_main\Scripts\pip install -r requirements.txt
)

echo Setting up neural sidecar environment...
if not exist venv_neural (
    py -3.10 -m venv venv_neural
    venv_neural\Scripts\pip install -r neural_env\requirements.txt
)

echo Starting neural sidecar...
cd neural_env
start /B ..\venv_neural\Scripts\python.exe rvc_server.py
cd ..

timeout /t 8 >nul

echo Starting main app...
start /B venv_main\Scripts\python.exe server.py

timeout /t 2 >nul
start http://localhost:8000

echo.
echo Both servers running. Close this window to stop them.
pause