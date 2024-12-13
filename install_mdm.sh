#!/bin/bash

set -e  # Exit on error

# Pull latest changes
echo "Pulling latest changes..."
git pull origin main

# Function definitions
cleanup() {
    if [ $? -ne 0 ]; then
        echo "Installation failed!"
        rm -f humanml_enc_512_50steps.zip
        deactivate 2>/dev/null || true
    fi
}

check_disk_space() {
    required_space=5  # in GB, adjust as needed
    available_space=$(df -PH . | awk 'NR==2 {print $4}' | sed 's/G//')
    if (( $(echo "$available_space < $required_space" | bc -l) )); then
        echo "Insufficient disk space. Need at least ${required_space}GB"
        exit 1
    fi
}

# Set up trap for cleanup
trap cleanup EXIT

# Initial checks
echo "Check for Command Line Tools"

# Check for Command Line Tools
if ! command -v xcode-select &> /dev/null || ! xcode-select -p &> /dev/null; then
    echo "Installing Command Line Tools..."
    xcode-select --install
    echo "Waiting for Command Line Tools installation to complete..."
    echo "Please complete the installation dialog if it appears..."
    until xcode-select -p &> /dev/null; do
        sleep 5
    done
fi

echo "Check for Homebrew"

# Check for Homebrew and install if missing
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

echo "Check for Python 3.11"

# Unlink any other Python versions first
brew unlink python@3.12 &>/dev/null || true
brew unlink python@3.10 &>/dev/null || true

# Install Python 3.11 specifically
if ! brew list python@3.11 &>/dev/null; then
    echo "Installing Python 3.11..."
    brew install python@3.11
fi

# Force link Python 3.11
brew link --force python@3.11

# Ensure we're using Python 3.11
PYTHON_CMD=$(brew --prefix python@3.11)/bin/python3.11
echo "Using Python: $($PYTHON_CMD --version)"

echo "Check for pip"
# Ensure pip is installed and updated
echo "Installing/upgrading pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD
$PYTHON_CMD -m pip install --upgrade pip

echo "Check for virtual environment"
# Ask about rebuilding virtual environment
if [ -d ".venv" ]; then
    read -p "Virtual environment already exists. Do you want to rebuild it? (y/N): " rebuild_venv
    if [[ $rebuild_venv =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf .venv
    fi
fi

echo "Check for virtual environment"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.11..."
    $PYTHON_CMD -m venv .venv
    echo "Virtual environment created. Activating..."
    source .venv/bin/activate
    
    # Install essential build tools first
    echo "Installing build dependencies..."
    python -m pip install --upgrade pip
    python -m pip install setuptools wheel build numpy cython

    echo "Installing chumpy-py311..."
    # Create a temporary directory for chumpy installation
    temp_dir=$(mktemp -d)
    git clone https://github.com/Grant-CP/chumpy-py311.git "$temp_dir"
    cd "$temp_dir"
    # Install in development mode with required build dependencies
    python setup.py develop
    cd -
    rm -rf "$temp_dir"

    echo "Installing dependencies from requirements.txt..."
    
    # Now install the rest of the requirements
    pip install -r requirements.txt
    
    echo "Installing PyTorch..."
    pip install torch torchvision torchaudio
    
    echo "Installing additional dependencies..."
    pip install h5py
else
    echo "Virtual environment already exists, activating it..."
    source .venv/bin/activate
fi

echo "Check for spacy model"
# Only download spacy model if not already installed
if ! python -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null; then
    echo "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
fi

echo "Check for directories"
# Create directories if they don't exist
echo "Setting up directories..."
for dir in "pretrained_models" "dataset" "combined_motions_discovery" "body_models"; do
    [ ! -d "$dir" ] && mkdir -p "$dir"
done

# Download and extract the model if not already present
if [ ! -d "pretrained_models/humanml_enc_512_50steps" ]; then
    echo "Downloading pre-trained model..."
    wget "https://www.dropbox.com/s/l1fporynk9f2vmp/humanml_enc_512_50steps.zip?dl=1" -O humanml_enc_512_50steps.zip
    echo "Extracting model..."
    unzip -n humanml_enc_512_50steps.zip -d pretrained_models/
    rm humanml_enc_512_50steps.zip
else
    echo "Pre-trained model already exists, skipping download..."
fi

# Install evaluator dependencies only if requested and not already installed
if [ "$1" = "--with-eval" ]; then
    if [ ! -d "prepare" ]; then
        echo "Installing evaluator dependencies..."
        mkdir -p prepare && cd prepare
        
        for script in "download_smpl_files.sh" "download_glove.sh" "download_t2m_evaluators.sh"; do
            if [ ! -f "$script" ]; then
                curl -O "https://raw.githubusercontent.com/GuyTevet/motion-diffusion-model/main/prepare/$script"
                chmod +x "$script"
                ./"$script" &
            fi
        done
        wait
        cd ..
    else
        echo "Evaluator dependencies already installed, skipping..."
    fi
fi

read -p "Please enter your Anthropic API key (or press Enter to skip): " anthropic_key
if [ -n "$anthropic_key" ]; then
    # Add to .env file
    echo "Adding Anthropic API key to .env file..."
    echo "ANTHROPIC_API_KEY='$anthropic_key'" >> .env
    echo "Anthropic API key has been set."
else
    echo "Skipping Anthropic API key setup. You can set it later by adding ANTHROPIC_API_KEY to .env file"
fi

# Installation complete message
echo "Installation complete! The virtual environment is activated."

# Install and build frontend
echo "Installing frontend dependencies..."
cd frontend
npm install
echo "Building frontend..."
npm run build
cd ..

echo "You can now run the server with: ./start.sh"

# Make start script executable
chmod +x start.sh

# Ask user if they want to start the server
read -p "Do you want to start the server now? (y/N): " start_server
if [[ $start_server =~ ^[Yy]$ ]]; then
    ./start.sh
fi