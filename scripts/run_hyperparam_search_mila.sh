#!/bin/bash
#SBATCH --job-name=conspec_hypersearch
#SBATCH --output=logs/hypersearch_%j.out
#SBATCH --error=logs/hypersearch_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

set -e  # Exit on error
set -x  # Print commands

# Mila-specific modules
module load python/3.9
module load cuda/11.3
module load libffi

# Add local bin to PATH
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

# Navigate to project directory
cd ~/MemorySchema

# Create logs and results directories
mkdir -p logs hyperparam_search_results

# Display GPU information
echo "=========================================="
echo "Starting ConSpec Hyperparameter Search on Mila"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Working directory: $(pwd)"
echo "Python version: $(python3.9 --version)"
echo "PyTorch available: $(python3.9 -c 'import torch; print(torch.__version__)')"
echo "=========================================="
echo ""

# Run hyperparameter search for all environments
echo "Launching hyperparameter search..."
python3.9 -u metrics_experiments/hyperparameter_search_conspec.py \
    --config configs/conspec_hyperparam_search.yaml \
    --env all

echo "=========================================="
echo "Hyperparameter search completed at: $(date)"
echo "Results saved in ~/MemorySchema/hyperparam_search_results/"
echo "=========================================="

# Display GPU report
nvidia-smi

echo "Job finished successfully!"

