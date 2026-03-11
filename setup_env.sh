#!/bin/bash
# Setup conda environment for spatial-omics-fusion
# Works on both macOS (CPU/MPS) and Linux (CUDA)

ENV_NAME="spatial-omics"

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch (detect CUDA)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected — installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No CUDA — installing CPU/MPS PyTorch"
    pip install torch torchvision torchaudio
fi

# Install PyTorch Geometric
pip install torch-geometric

# Install bio + ML packages
pip install scanpy squidpy anndata

# Install utilities
pip install scikit-learn matplotlib seaborn
pip install pyyaml tqdm

echo ""
echo "Done! Activate with: conda activate $ENV_NAME"
