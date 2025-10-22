#!/bin/bash
# Simple script to launch hyperparameter search on Mila
# Run this after SSHing to Mila

echo "=========================================="
echo "Launching Hyperparameter Search on Mila"
echo "=========================================="
echo ""

# Navigate to project directory
cd ~/MemorySchema || { echo "Error: MemorySchema directory not found"; exit 1; }
echo "✓ In directory: $(pwd)"

# Checkout correct branch
echo "Checking out conspec branch..."
git checkout conspec
git pull origin conspec
echo "✓ Branch: $(git branch --show-current)"
echo ""

# Load required modules
echo "Loading modules..."
module load python/3.9
module load cuda/11.3
module load libffi
echo "✓ Modules loaded"
echo ""

# Verify Python
echo "Python version: $(python3.9 --version)"
echo ""

# Create logs directory
mkdir -p logs
echo "✓ Logs directory created"
echo ""

# Check if wandb is logged in
echo "Checking wandb status..."
if wandb status &>/dev/null; then
    echo "✓ Wandb is logged in"
else
    echo "⚠ Wandb not logged in - attempting login..."
    wandb login
fi
echo ""

# Check critical dependencies
echo "Verifying dependencies..."
python3.9 -c "import torch; print('  ✓ PyTorch:', torch.__version__)" || echo "  ✗ PyTorch not found"
python3.9 -c "import wandb; print('  ✓ Wandb:', wandb.__version__)" || echo "  ✗ Wandb not found"
python3.9 -c "import yaml; print('  ✓ PyYAML: OK')" || { echo "  ✗ PyYAML not found - installing..."; pip install --user pyyaml; }
python3.9 -c "from environments.key_door_goal_base import create_key_door_goal_env; print('  ✓ Environments module: OK')" || echo "  ✗ Environments module failed"
echo ""

# Submit the job
echo "Submitting hyperparameter search job..."
JOB_ID=$(sbatch scripts/run_hyperparam_search_mila.sh | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo "✓ Job submitted successfully!"
    echo ""
    echo "Job ID: $JOB_ID"
    echo ""
    echo "=========================================="
    echo "Job Monitoring Commands"
    echo "=========================================="
    echo ""
    echo "Check status:"
    echo "  squeue -u \$USER"
    echo ""
    echo "View live output:"
    echo "  tail -f logs/hypersearch_${JOB_ID}.out"
    echo ""
    echo "View errors:"
    echo "  tail -f logs/hypersearch_${JOB_ID}.err"
    echo ""
    echo "Cancel job:"
    echo "  scancel $JOB_ID"
    echo ""
    echo "Wandb dashboard:"
    echo "  https://wandb.ai/mandanasmi/schema-learning"
    echo ""
else
    echo "✗ Job submission failed!"
    echo "Check the error above and try manual submission:"
    echo "  sbatch scripts/run_hyperparam_search_mila.sh"
fi

echo "=========================================="

