#!/bin/bash
# Submit ConSpec job to Mila after wandb login

echo "=============================================="
echo "Submit ConSpec Job to Mila"
echo "=============================================="
echo ""
echo "This script will:"
echo "1. Login to wandb (you'll need your API key)"
echo "2. Submit the training job"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# SSH to Mila and run commands
ssh mila << 'ENDSSH'

cd ~/MemorySchema
source venv/bin/activate

# Login to wandb
echo "=============================================="
echo "Logging in to Wandb..."
echo "=============================================="
echo ""
echo "Please paste your API key from:"
echo "https://wandb.ai/authorize"
echo ""

wandb login

# Verify login
if wandb verify 2>/dev/null; then
    echo ""
    echo "Wandb login successful!"
    echo ""
    
    # Submit job
    echo "=============================================="
    echo "Submitting ConSpec training job..."
    echo "=============================================="
    
    JOB_ID=$(sbatch run_conspec_mila.sh | awk '{print $4}')
    
    echo ""
    echo "Job submitted: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  ssh mila"
    echo "  squeue -u samieima"
    echo "  tail -f ~/MemorySchema/logs/conspec_${JOB_ID}.out"
    echo ""
    echo "Wandb Dashboard:"
    echo "  https://wandb.ai/mandanasmi/schema-learning"
    echo ""
else
    echo ""
    echo "Wandb login failed. Please try again manually:"
    echo "  ssh mila"
    echo "  cd ~/MemorySchema"
    echo "  source venv/bin/activate"
    echo "  wandb login"
    echo "  sbatch run_conspec_mila.sh"
fi

ENDSSH

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="

