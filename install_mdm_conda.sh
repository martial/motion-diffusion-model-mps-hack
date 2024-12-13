#!/bin/bash

set -x  # Enable debug output
set -e  # Exit on error

# Initialize conda for the current shell
eval "$(conda shell.bash hook)"

# Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "Conda is already installed, skipping installation..."
else
    # Install Miniconda
    echo "Installing Miniconda..."
    MINICONDA_PATH="$HOME/miniconda"

    # Download and install Miniconda
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh
    bash miniconda.sh -b -p $MINICONDA_PATH
    rm miniconda.sh

    # Add Miniconda to PATH
    export PATH="$MINICONDA_PATH/bin:$PATH"

    # Initialize conda
    conda init bash

    # Reload shell configuration
    source ~/.bash_profile || source ~/.bashrc
fi

# Remove existing environment if it exists
conda deactivate 2>/dev/null || true
conda env remove -n mdm -y 2>/dev/null || true

# Create fresh conda environment
echo "Creating conda environment..."
conda create -n mdm python=3.9 -y

# Activate the environment
echo "Activating conda environment..."
conda activate mdm

# Add conda-forge channel for more packages
conda config --add channels conda-forge
conda config --set channel_priority flexible

# Install dependencies through conda one by one to resolve conflicts
echo "Installing dependencies..."

# Install numpy first to ensure it's properly linked
conda install -y "numpy<2.0.0"

# Install matplotlib 3.5.0 and its dependencies
conda install -y matplotlib=3.5.0

# Install PyTorch with pip instead of conda for better version control
conda run -n mdm python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Verify installations
python -c "import numpy; print(f'Numpy version: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

# Install other dependencies
conda install -y jupyter
conda install -y spacy=3.6.1

# Install pip packages using conda environment's pip
echo "Installing pip packages in conda environment..."
conda run -n mdm python -m pip install smplx
conda run -n mdm python -m pip install git+https://github.com/Grant-CP/chumpy-py311.git
conda run -n mdm python -m pip install git+https://github.com/openai/CLIP.git

# Install spacy language model
conda run -n mdm python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p save
mkdir -p dataset
mkdir -p combined_motions_discovery

# Download the 50-step model if not already present
if [ ! -d "save/humanml_enc_512_50steps" ]; then
    echo "Downloading pre-trained model..."
    wget "https://www.dropbox.com/s/l1fporynk9f2vmp/humanml_enc_512_50steps.zip?st=3lzc5a8b&dl=1" -O humanml_enc_512_50steps.zip
    unzip -n humanml_enc_512_50steps.zip -d save/  # -n flag to never overwrite existing files
    rm humanml_enc_512_50steps.zip
else
    echo "Pre-trained model already exists, skipping download..."
fi

# Get HumanML3D data if not already present
if [ ! -d "dataset/HumanML3D" ]; then
    echo "Downloading HumanML3D dataset..."
    # Clean up any existing partial clone
    rm -rf HumanML3D
    git clone https://github.com/EricGuo5513/HumanML3D.git
    unzip -n ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
    mkdir -p dataset
    cp -r HumanML3D/HumanML3D dataset/
    rm -rf HumanML3D  # Clean up the temporary clone
else
    echo "HumanML3D directory already exists, skipping clone..."
fi

# Download SMPL files if not already present
if [ ! -d "body_models/smpl" ]; then
    echo "Downloading SMPL files..."
    mkdir -p body_models
    cd body_models/
    gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
    unzip smpl.zip
    rm smpl.zip
    cd ..
    echo "SMPL files downloaded"
else
    echo "SMPL files already exist, skipping download..."
fi

# Install evaluator dependencies only if requested
if [ "$1" = "--with-eval" ]; then
    echo "Installing evaluator dependencies..."
    mkdir -p prepare
    cd prepare
    
    # Download dependency scripts
    echo "Downloading dependency scripts..."
    curl -O https://raw.githubusercontent.com/GuyTevet/motion-diffusion-model/main/prepare/download_smpl_files.sh
    curl -O https://raw.githubusercontent.com/GuyTevet/motion-diffusion-model/main/prepare/download_glove.sh
    curl -O https://raw.githubusercontent.com/GuyTevet/motion-diffusion-model/main/prepare/download_t2m_evaluators.sh

    # Make scripts executable
    chmod +x download_smpl_files.sh
    chmod +x download_glove.sh
    chmod +x download_t2m_evaluators.sh

    # Execute dependency scripts
    echo "Downloading additional dependencies..."
    ./download_smpl_files.sh
    ./download_glove.sh
    ./download_t2m_evaluators.sh

    cd ..
fi

echo "Installation complete! Activate the environment with: source .venv/bin/activate" 