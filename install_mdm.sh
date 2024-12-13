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

check_internet() {
    echo "Checking internet connection..."
    if ! ping -c 1 google.com &> /dev/null; then
        echo "No internet connection detected!"
        exit 1
    fi
}

check_architecture() {
    arch=$(uname -m)
    if [[ $arch != "x86_64" && $arch != "arm64" ]]; then
        echo "Unsupported architecture: $arch"
        exit 1
    fi
    
    if [[ $arch == "arm64" ]]; then
        echo "Apple Silicon detected - ensuring Rosetta 2 is installed..."
        if ! pkgutil --pkg-info com.apple.pkg.RosettaUpdateAuto > /dev/null 2>&1; then
            softwareupdate --install-rosetta --agree-to-license
        fi
    fi
}

check_tmp_permissions() {
    if ! [ -w /tmp ]; then
        echo "Error: No write permission in /tmp directory"
        exit 1
    fi
}

# Set up trap for cleanup
trap cleanup EXIT

# Initial checks
echo "Running initial system checks..."
check_internet
check_architecture
check_tmp_permissions
check_disk_space

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

# Install required system packages
for package in "python@3.10" "wget" "unzip"; do
    if ! brew list $package &>/dev/null; then
        echo "Installing $package..."
        brew install $package
    fi
done

# Install pip if not already installed
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
fi

# Ensure pip is up to date
python3 -m pip install --upgrade pip


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

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --seed
    echo "Virtual environment created. Activating..."
    source .venv/bin/activate
    
    # Ensure pip is installed in the virtual environment
    echo "Ensuring pip is installed in virtual environment..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
    
    echo "Installing dependencies from requirements.txt..."
    uv pip install -r requirements.txt
    
    echo "Installing PyTorch..."
    uv pip install --no-deps torch torchvision torchaudio
    
    echo "Installing additional dependencies..."
    uv pip install h5py
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

# Add UV shell completion only if not already added
shell_name=$(basename "$SHELL")
completion_check=""
case "$shell_name" in
    "bash")
        completion_check="grep -q 'uv generate-shell-completion bash' ~/.bashrc"
        completion_file=~/.bashrc
        completion_cmd='eval "$(uv generate-shell-completion bash)"'
        ;;
    "zsh")
        completion_check="grep -q 'uv generate-shell-completion zsh' ~/.zshrc"
        completion_file=~/.zshrc
        completion_cmd='eval "$(uv generate-shell-completion zsh)"'
        ;;
    "fish")
        completion_check="grep -q 'uv generate-shell-completion fish' ~/.config/fish/config.fish"
        completion_file=~/.config/fish/config.fish
        completion_cmd='uv generate-shell-completion fish | source'
        ;;
esac

if [ -n "$completion_check" ] && ! eval "$completion_check"; then
    echo "$completion_cmd" >> "$completion_file"
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