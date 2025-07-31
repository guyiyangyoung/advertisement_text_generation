#!/bin/bash

echo "Setting up Qwen-3 8B Fine-tuning Environment"
echo "============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check Python version (should be 3.8+)
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "Python version: $python_version ✓"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
python3 -c "
import torch
import transformers
import peft
import datasets
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'PEFT version: {peft.__version__}')
print(f'Datasets version: {datasets.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
"

echo ""
echo "Setup completed successfully!"
echo ""
echo "Usage Instructions:"
echo "==================="
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Place your clean_data.csv file in the current directory"
echo "3. Run data preprocessing: python data_preprocessing.py"
echo "4. Start fine-tuning: python qwen_finetune.py"
echo "5. Generate advertising copy: python inference.py --chapters_text 'your_text'"
echo ""
echo "For more options, run: python <script_name>.py --help"