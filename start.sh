#!/bin/bash

# Ensure we're in the virtual environment
source .venv/bin/activate

# Set environment variables if needed
export MODEL_PATH="./save/humanml_enc_512_50steps/model000200000.pt"
export GUIDANCE_PARAM=2.5
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export FLASK_DEBUG=1
export FLASK_APP=backend/main.py

# Start the Flask development server
echo "Starting Flask development server..."
python -m flask run --port 3000