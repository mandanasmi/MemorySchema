#!/bin/bash
# Setup and run ConSpec training on Mila cluster

echo "=============================================="
echo "ConSpec Training Setup for Mila Cluster"
echo "=============================================="

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p figures

# Install wandb if not already installed
if ! python -c "import wandb" &> /dev/null; then
    echo "Installing wandb..."
    pip install wandb
fi

# Check if wandb is logged in
if ! wandb verify &> /dev/null; then
    echo ""
    echo "  Wandb not configured. Please run:"
    echo "    wandb login"
    echo "    (Get your API key from https://wandb.ai/authorize)"
    echo ""
    read -p "Would you like to login to wandb now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wandb login
    fi
fi

# Make the batch script executable
chmod +x run_conspec_mila.sh

echo ""
echo " Setup complete!"
echo ""
echo "To submit the job to Mila cluster:"
echo "    sbatch run_conspec_mila.sh"
echo ""
echo "To check job status:"
echo "    squeue -u $USER"
echo ""
echo "To view live logs:"
echo "    tail -f logs/conspec_<job_id>.out"
echo ""
echo "To cancel a job:"
echo "    scancel <job_id>"
echo ""
echo "=============================================="

