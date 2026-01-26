#!/bin/bash
# Script to run DLRM v3 training on CPU with Gloo backend
# Usage: ./run_dlrm_v3_cpu.sh [dataset] [mode]
# Example: ./run_dlrm_v3_cpu.sh movielens-1m train-eval

set -e

# Default values
DATASET="${1:-movielens-1m}"
MODE="${2:-train-eval}"

# Distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1

# Conda environment
CONDA_ENV="torchRecCPU"
CONDA_PATH="/opt/miniforge"

echo "=============================================="
echo " DLRM v3 CPU Training with Gloo Backend"
echo "=============================================="
echo "Dataset: ${DATASET}"
echo "Mode: ${MODE}"
echo "World Size: ${WORLD_SIZE}"
echo "Conda Env: ${CONDA_ENV}"
echo "=============================================="

# Activate conda environment
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Verify environment
echo ""
echo "Environment verification:"
python -c "
import torch
import torchrec
import fbgemm_gpu
import torch.distributed as dist

print(f'  PyTorch: {torch.__version__}')
print(f'  TorchRec: {torchrec.__version__}')
print(f'  FBGEMM: {fbgemm_gpu.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  Gloo available: {dist.is_gloo_available()}')
"

echo ""
echo "Starting training..."
echo "=============================================="

# Navigate to repo directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Run training
python -m generative_recommenders.dlrm_v3.train.train_ranker \
    --dataset "${DATASET}" \
    --mode "${MODE}"

echo ""
echo "=============================================="
echo " Training completed!"
echo "=============================================="
