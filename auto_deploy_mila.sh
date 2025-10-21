#!/bin/bash
# Automated deployment script for Mila cluster

echo "=============================================="
echo "ConSpec Training - Automated Mila Deployment"
echo "=============================================="
echo ""

# SSH connection (adjust if needed)
MILA_HOST="mila"
MILA_USER="samieima"
REPO_URL="https://github.com/mandanasmi/MemorySchema.git"
PROJECT_DIR="MemorySchema"

echo "Connecting to Mila cluster..."
echo "Host: $MILA_HOST"
echo "User: $MILA_USER"
echo ""

# Create deployment commands
ssh $MILA_HOST << 'ENDSSH'

echo "Connected to Mila cluster"
echo "=============================================="

# Navigate to home
cd ~

# Clone or update repository
if [ -d "MemorySchema" ]; then
    echo "Repository exists, updating..."
    cd MemorySchema
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/mandanasmi/MemorySchema.git
    cd MemorySchema
fi

echo ""
echo "=============================================="
echo "Setting up environment..."
echo "=============================================="

# Load modules
module load python/3.9
module load cuda/11.3

# Create venv if doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.9 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install scikit-learn wandb --quiet

echo "Dependencies installed"

# Create directories
mkdir -p logs checkpoints figures

# Make scripts executable
chmod +x run_conspec_mila.sh setup_and_run_mila.sh setup_wandb.sh setup_mila.sh

echo ""
echo "=============================================="
echo "Checking wandb status..."
echo "=============================================="

# Check if wandb is logged in
if wandb verify 2>/dev/null; then
    echo "Wandb is already logged in"
else
    echo ""
    echo "WARNING: Wandb not logged in!"
    echo "Please run: wandb login"
    echo "Get API key from: https://wandb.ai/authorize"
    echo ""
    echo "After logging in, rerun this script or submit manually:"
    echo "  sbatch run_conspec_mila.sh"
    exit 1
fi

echo ""
echo "=============================================="
echo "Submitting ConSpec training job..."
echo "=============================================="

# Submit the job
JOB_ID=$(sbatch run_conspec_mila.sh | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u samieima"
    echo "  tail -f logs/conspec_${JOB_ID}.out"
    echo ""
    echo "Wandb Dashboard:"
    echo "  https://wandb.ai/mandanasmi/schema-learning"
    echo ""
    echo "=============================================="
    echo "Training Configuration:"
    echo "  - 3 environments (single, double, triple)"
    echo "  - 5 prototype values (3, 4, 5, 6, 7)"
    echo "  - 15 total training runs"
    echo "  - 5000 episodes per run"
    echo "  - GPU-accelerated"
    echo "  - Checkpoints every 500 episodes"
    echo "=============================================="
else
    echo "ERROR: Job submission failed!"
    echo "Check the error above and submit manually:"
    echo "  sbatch run_conspec_mila.sh"
fi

ENDSSH

echo ""
echo "=============================================="
echo "Deployment script completed"
echo "=============================================="

