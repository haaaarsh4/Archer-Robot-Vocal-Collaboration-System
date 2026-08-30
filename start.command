#!/bin/bash
cd "$(dirname "$0")"

if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing it now, this needs your Mac password..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

if ! command -v python3.12 &> /dev/null; then
    echo "Python 3.12 not found. Installing it now..."
    brew install python@3.12
fi

if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10 not found. Installing it now..."
    brew install python@3.10
fi

echo "Setting up main app environment..."
if [ ! -d "venv_main" ]; then
    python3.12 -m venv venv_main
    ./venv_main/bin/pip install -r requirements.txt
fi

echo "Setting up neural sidecar environment..."
if [ ! -d "venv_neural" ]; then
    python3.10 -m venv venv_neural
    ./venv_neural/bin/pip install -r neural_env/requirements.txt
fi

echo "Starting neural sidecar..."
(cd neural_env && ../venv_neural/bin/python rvc_server.py) &
SIDECAR_PID=$!

echo "Waiting for the sidecar to finish loading..."
until curl -s http://127.0.0.1:8801/health > /dev/null; do sleep 1; done
sleep 3

echo "Starting main app..."
./venv_main/bin/python server.py &
MAIN_PID=$!

sleep 2
open http://localhost:8000

echo ""
echo "Both servers running. Press Ctrl+C to stop everything."
trap "kill $SIDECAR_PID $MAIN_PID" EXIT
wait