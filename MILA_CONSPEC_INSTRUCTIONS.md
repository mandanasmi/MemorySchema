# Running ConSpec Training on Mila Cluster

## Quick Start

### 1. SSH into Mila
```bash
ssh mila
```

### 2. Navigate to your project directory
```bash
cd ~/MemorySchema
```

### 3. Activate virtual environment
```bash
source venv/bin/activate
```

### 4. Install wandb (if not already installed)
```bash
pip install wandb
```

### 5. Login to Weights & Biases
```bash
wandb login
```
Get your API key from: https://wandb.ai/authorize

**Your wandb configuration:**
- Entity (username): `samieima@mila.quebec` → `samieima`
- Project: `schema-learning`
- GitHub: `mandanasmi`
- Dashboard URL: https://wandb.ai/mandanasmi/schema-learning

Or use the setup script:
```bash
./setup_wandb.sh
```

### 6. Setup and submit the job
```bash
chmod +x setup_and_run_mila.sh
./setup_and_run_mila.sh
sbatch run_conspec_mila.sh
```

## What the Training Does

The training script will:

1. **Train on 3 environments**: Single, Double, Triple (Chain, Parallel, Sequential graphs)
2. **Search over prototypes**: Tests 3, 4, 5, 6, and 7 prototypes for each environment
3. **Save checkpoints**: Every 500 episodes + best model for each configuration
4. **Resume training**: Can resume from checkpoints if interrupted
5. **Log to wandb**: All metrics tracked in Weights & Biases
6. **Total configurations**: 3 environments × 5 prototype values = 15 training runs

## Monitoring Your Job

### Check job status
```bash
squeue -u $USER
```

### View live output
```bash
tail -f logs/conspec_<job_id>.out
```

### View live errors
```bash
tail -f logs/conspec_<job_id>.err
```

### View wandb dashboard
Go to: https://wandb.ai/mandanasmi/schema-learning

All your experiments will be tracked here in real-time!

GitHub Repository: https://github.com/mandanasmi/MemorySchema

## Managing Jobs

### Cancel a job
```bash
scancel <job_id>
```

### Cancel all your jobs
```bash
scancel -u $USER
```

### Check job details
```bash
scontrol show job <job_id>
```

## Checkpoints and Results

All results are saved in the `checkpoints/` directory:

- **Latest checkpoints**: `checkpoint_{env}_proto{N}_latest.pth`
- **Periodic checkpoints**: `checkpoint_{env}_proto{N}_ep{episode}.pth`
- **Best models**: `best_model_{env}_proto{N}.pth`
- **Training summary**: `training_summary.json`

### Checkpoint Contents

Each checkpoint contains:
- Policy model state
- ConSpec encoder state
- Prototype network state
- Training statistics (rewards, success rates, etc.)
- Memory buffers (success/failure trajectories)
- Episode number for resumption

## Resuming Training

Training automatically resumes from the latest checkpoint if available.

To start fresh (ignore checkpoints):
```bash
python train_conspec_mila.py --no-resume --env all --episodes 5000 --min-prototypes 3 --max-prototypes 7
```

To resume specific configuration:
```bash
python train_conspec_mila.py --env single --episodes 5000 --min-prototypes 5 --max-prototypes 5
```

## Hyperparameter Search

The training searches over different numbers of prototypes (3-7) to find the optimal configuration for each environment.

### Customizing the Search

Edit `run_conspec_mila.sh` or run directly:

```bash
python train_conspec_mila.py \
    --env single \              # or 'double', 'triple', 'all'
    --episodes 5000 \           # number of episodes per configuration
    --min-prototypes 3 \        # minimum prototypes to try
    --max-prototypes 7 \        # maximum prototypes to try
    --checkpoint-dir checkpoints \
    --save-interval 500         # save checkpoint every N episodes
```

## Expected Training Time

- **Per configuration**: ~2-4 hours (5000 episodes)
- **Total for all 15 configurations**: ~30-60 hours
- The job is configured for 24 hours, so you may need to resubmit or increase time limit

## Results Analysis

After training, check the summary:

```bash
cat checkpoints/training_summary.json
```

This shows the best success rate for each environment-prototype combination.

### Finding the Best Configuration

```python
import json

with open('checkpoints/training_summary.json', 'r') as f:
    summary = json.load(f)

for result in summary['results']:
    print(f"{result['env_type']}-proto{result['num_prototypes']}: "
          f"Best={result['best_success_rate']:.3f}, "
          f"Final={result['final_success_rate']:.3f}")
```

## Loading Trained Models

```python
from train_conspec_mila import SimplePolicy
from conspec_minigrid import ConSpec
import torch

# Load best model for single environment with 5 prototypes
checkpoint = torch.load('checkpoints/best_model_single_proto5.pth')

# Create policy
policy = SimplePolicy(checkpoint['obs_shape'], checkpoint['num_actions'])
policy.load_state_dict(checkpoint['policy_state_dict'])
policy.eval()

# Use for inference
# ... (see load_optimal_policy.py for examples)
```

## Troubleshooting

### Job fails immediately
- Check logs: `cat logs/conspec_<job_id>.err`
- Verify virtual environment is activated
- Check GPU availability: `nvidia-smi`

### Out of memory
- Reduce batch size in the code
- Request more memory: Edit `#SBATCH --mem=32G` in run_conspec_mila.sh

### Job times out
- Increase time limit: Edit `#SBATCH --time=48:00:00` in run_conspec_mila.sh
- Reduce number of episodes or configurations

### Wandb not logging
- Verify wandb login: `wandb verify`
- Check network connectivity
- Use `--no-wandb` flag to disable

## Interactive Development

For testing before submitting batch job:

```bash
# Request interactive GPU node
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=2:00:00

# Activate environment
source venv/bin/activate

# Run a short test
python train_conspec_mila.py --env single --episodes 100 --min-prototypes 5 --max-prototypes 5

# Exit interactive session
exit
```

## File Structure

```
MemorySchema/
├── checkpoints/              # Saved models and checkpoints
│   ├── checkpoint_*.pth      # Training checkpoints
│   ├── best_model_*.pth      # Best models
│   └── training_summary.json # Results summary
├── logs/                     # Slurm job logs
│   ├── conspec_*.out         # Standard output
│   └── conspec_*.err         # Error output
├── train_conspec_mila.py     # Main training script
├── run_conspec_mila.sh       # Slurm batch script
└── setup_and_run_mila.sh     # Setup helper script
```

## Support

For Mila-specific issues:
- Documentation: https://docs.mila.quebec/
- Slack: #mila-support
- Email: support@mila.quebec

For code/ConSpec issues:
- Check the code documentation
- Review ConSpec paper: https://arxiv.org/pdf/2210.05845.pdf
