#!/bin/bash
#SBATCH --job-name=conspec_training
#SBATCH --output=logs/conspec_%j.out
#SBATCH --error=logs/conspec_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

# Navigate to project directory
cd $SLURM_TMPDIR
cp -r ~/MemorySchema .
cd MemorySchema

# Create logs directory if it doesn't exist
mkdir -p logs

# No need for venv, using system packages installed with --user flag

# Login to wandb (make sure you've run 'wandb login' before submitting)
# wandb login <your-api-key>

# Display GPU information
echo "=========================================="
echo "Starting ConSpec training on Mila cluster"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Wandb Entity: mandanasmi"
echo "Wandb Project: schema-learning"
echo "Dashboard: https://wandb.ai/mandanasmi/schema-learning"
echo "Working directory: $(pwd)"
echo "Python version: $(python3.9 --version)"
echo "PyTorch available: $(python3.9 -c 'import torch; print(torch.__version__)' 2>&1 || echo 'NOT INSTALLED')"
echo "=========================================="
echo ""

# Run training with hyperparameter search on GPU
echo "Launching training script..."
python3.9 -u train_conspec_mila.py \
    --env all \
    --episodes 5000 \
    --min-prototypes 3 \
    --max-prototypes 7 \
    --checkpoint-dir checkpoints \
    --save-interval 500

echo "=========================================="
echo "Training completed at: $(date)"

# Copy results back to home directory
cp -r checkpoints ~/MemorySchema/
cp -r logs ~/MemorySchema/

echo "Results copied to ~/MemorySchema/checkpoints/"

