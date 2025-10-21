#!/bin/bash
# Setup Weights & Biases for ConSpec training

echo "=========================================="
echo "Weights & Biases Setup for ConSpec"
echo "=========================================="
echo ""

# Check if wandb is installed
if ! python -c "import wandb" &> /dev/null; then
    echo "Installing wandb..."
    pip install wandb
    echo "✅ Wandb installed"
else
    echo "✅ Wandb already installed"
fi

echo ""
echo "Your wandb entity (username): samieima@mila.quebec"
echo "Project name: conspec-key-door-goal-mila"
echo ""

# Check if already logged in
if wandb verify &> /dev/null; then
    echo "✅ Already logged in to wandb"
    wandb verify
else
    echo "Please login to Weights & Biases:"
    echo ""
    echo "1. Get your API key from: https://wandb.ai/authorize"
    echo "2. Run the command below and paste your API key:"
    echo ""
    wandb login
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Your experiments will be logged to:"
echo "  https://wandb.ai/samieima/conspec-key-door-goal-mila"
echo ""
echo "To run training with wandb:"
echo "  python train_conspec_mila.py --env all --episodes 5000 --min-prototypes 3 --max-prototypes 7"
echo ""
echo "To disable wandb (optional):"
echo "  python train_conspec_mila.py --no-wandb --env all --episodes 5000"
echo ""

