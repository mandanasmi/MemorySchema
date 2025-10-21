#!/bin/bash
#SBATCH --job-name=conspec_training
#SBATCH --output=logs/conspec_%j.out
#SBATCH --error=logs/conspec_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

# Mila-specific modules
module load python/3.9
module load cuda/11.3

# Navigate to project directory
cd $SLURM_TMPDIR
cp -r ~/MemorySchema .
cd MemorySchema

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Install additional dependencies if needed
pip install wandb --quiet

# Login to wandb (make sure you've run 'wandb login' before submitting)
# wandb login <your-api-key>

# Run training with hyperparameter search
echo "Starting ConSpec training on Mila cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=========================================="

python train_conspec_mila.py \
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

