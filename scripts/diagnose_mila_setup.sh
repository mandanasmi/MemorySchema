#!/bin/bash
# Diagnostic script for Mila cluster setup
# Run this on Mila to check if everything is configured correctly

echo "=========================================="
echo "Mila Cluster Diagnostic for MemorySchema"
echo "=========================================="
echo ""

# Check current directory
echo "1. Current Directory:"
pwd
echo ""

# Check if MemorySchema exists
echo "2. MemorySchema Directory:"
if [ -d "$HOME/MemorySchema" ]; then
    echo "   ✓ Found: $HOME/MemorySchema"
    cd $HOME/MemorySchema
    echo "   Current branch: $(git branch --show-current)"
else
    echo "   ✗ NOT FOUND: $HOME/MemorySchema"
fi
echo ""

# Check Python version
echo "3. Python Version:"
module load python/3.9 2>/dev/null
python3.9 --version
echo ""

# Check required modules
echo "4. Required Modules:"
for mod in python/3.9 cuda/11.3 libffi; do
    if module is-loaded $mod 2>/dev/null; then
        echo "   ✓ Loaded: $mod"
    else
        echo "   ✗ Not loaded: $mod"
        echo "     Loading..."
        module load $mod 2>/dev/null
    fi
done
echo ""

# Check Python packages
echo "5. Python Packages:"
packages=("torch" "wandb" "yaml" "gymnasium" "minigrid" "numpy" "sklearn")
for pkg in "${packages[@]}"; do
    if python3.9 -c "import $pkg" 2>/dev/null; then
        version=$(python3.9 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        echo "   ✓ $pkg: $version"
    else
        echo "   ✗ $pkg: NOT INSTALLED"
    fi
done
echo ""

# Check custom imports
echo "6. Custom Module Imports:"
cd $HOME/MemorySchema 2>/dev/null
if python3.9 -c "from environments.key_door_goal_base import create_key_door_goal_env" 2>/dev/null; then
    echo "   ✓ environments.key_door_goal_base"
else
    echo "   ✗ environments.key_door_goal_base - FAILED"
fi

if python3.9 -c "from methods.conspec_minigrid import ConSpec" 2>/dev/null; then
    echo "   ✓ methods.conspec_minigrid"
else
    echo "   ✗ methods.conspec_minigrid - FAILED"
fi
echo ""

# Check config file
echo "7. Configuration Files:"
files=("configs/conspec_hyperparam_search.yaml" "scripts/run_hyperparam_search_mila.sh" "metrics_experiments/hyperparameter_search_conspec.py")
for file in "${files[@]}"; do
    if [ -f "$HOME/MemorySchema/$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file - NOT FOUND"
    fi
done
echo ""

# Check wandb status
echo "8. Wandb Status:"
if command -v wandb &> /dev/null; then
    wandb status 2>&1 || echo "   Wandb not logged in"
else
    echo "   ✗ Wandb command not found"
fi
echo ""

# Check recent jobs
echo "9. Recent Slurm Jobs:"
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed -S $(date -d '7 days ago' +%Y-%m-%d) 2>/dev/null | head -10
echo ""

# Check logs directory
echo "10. Logs Directory:"
if [ -d "$HOME/MemorySchema/logs" ]; then
    echo "   ✓ Logs directory exists"
    echo "   Recent logs:"
    ls -lht $HOME/MemorySchema/logs/ 2>/dev/null | head -5
else
    echo "   ✗ Logs directory not found"
    echo "   Creating logs directory..."
    mkdir -p $HOME/MemorySchema/logs
fi
echo ""

# Check GPU availability
echo "11. GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "   No GPU allocated currently"
else
    echo "   nvidia-smi not available (normal if not in a job)"
fi
echo ""

echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="
echo ""
echo "To run hyperparameter search:"
echo "  cd ~/MemorySchema"
echo "  sbatch scripts/run_hyperparam_search_mila.sh"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To view logs:"
echo "  tail -f logs/hypersearch_*.out"

