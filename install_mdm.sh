#!/bin/bash

set -e  # Exit on error

# Pull latest changes
echo "Pulling latest changes..."
git_output=$(git pull origin main)

# Check if changes were pulled
if [[ "$git_output" != "Already up to date." ]]; then
    echo "New changes were pulled from the repository."
    read -p "Would you like to restart the installation script to apply these changes? (y/N): " restart
    if [[ $restart =~ ^[Yy]$ ]]; then
        echo "Restarting installation script..."
        exec "$0" "$@"
        exit 0
    fi
fi

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

# Check for Homebrew and install if missing
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

brew install node 
brew install ffmpeg

# Install required system packages
for package in "python@3.10" "wget" "unzip"; do
    if ! brew list $package &>/dev/null; then
        echo "Installing $package..."
        brew install $package
    fi
done

# Install pip if not already installed
if ! command -v pip &> /dev/null; then
    echo "Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
fi




# Install UV if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add UV to PATH temporarily for this session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Add to appropriate shell config file for future sessions
    shell_name=$(basename "$SHELL")
    case "$shell_name" in
        "bash")
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
            ;;
        "zsh")
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
            ;;
        "fish")
            echo 'set -gx PATH "$HOME/.local/bin" $PATH' >> ~/.config/fish/config.fish
            ;;
    esac
    
    # Add to .profile for broader compatibility
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile
    source ~/.profile
    
    echo "UV path has been added to shell config. It will be permanent after shell restart."
fi

# Ask about rebuilding virtual environment
if [ -d ".venv" ]; then
    read -p "Virtual environment already exists. Do you want to rebuild it? (y/N): " rebuild_venv
    if [[ $rebuild_venv =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf .venv
    fi
fi


if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.10 --seed
    echo "Virtual environment created. Activating..."
    source .venv/bin/activate
    
    # Ensure pip is installed in the virtual environment
    echo "Ensuring pip is installed in virtual environment..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
    
    echo "Installing dependencies from requirements.txt..."
    uv pip install -r requirements.txt --no-build-isolation
    
    echo "Installing PyTorch..."
    uv pip install --no-deps torch torchvision torchaudio

    uv pip install spacy
    

else
    echo "Virtual environment already exists, activating it..."
    source .venv/bin/activate
fi

# Only download spacy model if not already installed
if ! python -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null; then
    echo "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
fi

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

# Download and extract the model if not already present
if [ ! -d "bodymodels" ]; then
    echo "Downloading body models..."
    wget "https://www.dropbox.com/scl/fi/pxxub0k45ad4l4j0o9jxx/body_models.zip?rlkey=lrhyoo75lf2t390okue40apf3&dl=1" -O bodymodels.zip
    echo "Extracting model..."
    unzip -n bodymodels.zip 
    rm bodymodels.zip
else
    echo "Pre-trained model already exists, skipping download..."
fi

# Download and extract the model if not already present
if [ ! -d "dataset/HumanML3D" ]; then
    echo "Downloading HumanML3D dataset..."
    wget "https://www.dropbox.com/scl/fi/z64eg54olpxx0hp2pjhp5/HumanML3D.zip?rlkey=ruzx2ybgi0ycdmm4djic1d9ea&dl=1" -O HumanML3D.zip
    echo "Extracting model..."
    unzip -n HumanML3D.zip -d dataset/
    rm HumanML3D.zip
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