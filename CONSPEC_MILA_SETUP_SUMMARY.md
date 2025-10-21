# ConSpec Training on Mila Cluster - Setup Complete! ✅

## What Has Been Created

### 1. Main Training Script: `train_conspec_mila.py`
- **Checkpoint Support**: Automatically saves and resumes from checkpoints
- **Hyperparameter Search**: Tests 3-7 prototypes for each environment
- **Wandb Integration**: Full experiment tracking and logging
- **Best Model Saving**: Saves best performing model for each configuration

### 2. Mila Cluster Scripts
- **`run_conspec_mila.sh`**: Slurm batch script to submit jobs
- **`setup_and_run_mila.sh`**: Setup helper script
- **`MILA_CONSPEC_INSTRUCTIONS.md`**: Comprehensive documentation

### 3. Additional Training Scripts
- **`train_conspec_wandb.py`**: Wandb-enabled training (general use)
- **`train_conspec_all_environments.py`**: Train on all three environments
- **`load_optimal_policy.py`**: Load and test saved models

## Quick Start on Mila

```bash
# 1. SSH to Mila
ssh mila

# 2. Navigate to project
cd ~/MemorySchema

# 3. Activate virtual environment
source venv/bin/activate

# 4. Setup wandb
./setup_wandb.sh
# Or manually:
# pip install wandb
# wandb login  # API key from: https://wandb.ai/authorize

# Your wandb dashboard:
# https://wandb.ai/samieima/schema-learning
# GitHub: https://github.com/mandanasmi/MemorySchema

# 6. Submit the job
chmod +x run_conspec_mila.sh setup_and_run_mila.sh
sbatch run_conspec_mila.sh
```

## What the Training Does

### Experiments
- **3 Environments**: Single (Chain), Double (Parallel), Triple (Sequential)
- **5 Prototype Values**: 3, 4, 5, 6, 7 prototypes
- **Total Configurations**: 15 (3 envs × 5 prototypes)
- **Episodes per Config**: 5000 (configurable)

### Checkpointing
- **Auto-save**: Every 500 episodes
- **Resume capability**: Automatically resumes from latest checkpoint
- **Best models**: Saves best performing model for each config
- **Memory preservation**: Saves ConSpec success/failure memory

### Output Files (in `checkpoints/` directory)
```
checkpoints/
├── checkpoint_single_proto3_latest.pth       # Latest checkpoint for resumption
├── checkpoint_single_proto3_ep500.pth        # Periodic checkpoint
├── checkpoint_single_proto3_ep1000.pth       # Periodic checkpoint
├── best_model_single_proto3.pth              # Best model for this config
├── checkpoint_single_proto4_latest.pth       # ... and so on for all configs
├── training_summary.json                     # Summary of all results
```

## Monitoring Training

### Check Job Status
```bash
squeue -u $USER
```

### View Live Logs
```bash
tail -f logs/conspec_<job_id>.out
```

### Wandb Dashboard
Visit: https://wandb.ai/samieima/schema-learning

See all your experiments in real-time:
- Episode rewards and success rates
- Intrinsic rewards from ConSpec
- Memory buffer statistics
- Policy and value losses
- Best success rates per configuration

GitHub: https://github.com/mandanasmi/MemorySchema

## Key Features

### 1. Checkpoint Resumption
The training can be interrupted and resumed seamlessly:
```python
# Automatically loads from: checkpoints/checkpoint_{env}_proto{N}_latest.pth
# Restores:
#   - Policy weights
#   - Optimizer state
#   - ConSpec encoder/prototypes
#   - Training statistics
#   - Memory buffers
#   - Episode number
```

### 2. Hyperparameter Search Results
After training completes, check `checkpoints/training_summary.json`:
```json
{
  "results": [
    {
      "env_type": "single",
      "num_prototypes": 5,
      "best_success_rate": 0.85,
      "final_success_rate": 0.82
    },
    ...
  ]
}
```

### 3. Wandb Tracking
All metrics logged in real-time:
- Episode rewards
- Success rates
- Best success rate
- Intrinsic rewards
- Memory buffer sizes
- Policy/value losses
- Epsilon (exploration rate)

## Expected Results

### Success Rate Goals
- **Single Environment**: Target >70% success rate
- **Double Environment**: Target >60% success rate  
- **Triple Environment**: Target >50% success rate

### Training Time
- **Per configuration**: ~2-4 hours (5000 episodes on GPU)
- **Total (all 15 configs)**: ~30-60 hours
- **Job time limit**: 24 hours (may need resubmission)

## Using Trained Models

### Load Best Model
```python
import torch
from train_conspec_mila import SimplePolicy

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model_single_proto5.pth')

# Create and load policy
policy = SimplePolicy(checkpoint['obs_shape'], checkpoint['num_actions'])
policy.load_state_dict(checkpoint['policy_state_dict'])
policy.eval()

print(f"Best success rate: {checkpoint['best_success_rate']:.3f}")
```

### Test the Policy
```python
from key_door_goal_base import create_key_door_goal_env

env = create_key_door_goal_env('single', size=10)
obs, _ = env.reset()

for step in range(100):
    with torch.no_grad():
        logits, _ = policy(obs['image'])
        action = torch.argmax(torch.softmax(logits, dim=-1)).item()
    
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        print(f"Episode finished in {step} steps with reward {reward}")
        break
```

## Customizing Training

### Change Number of Episodes
Edit `run_conspec_mila.sh`:
```bash
python train_conspec_mila.py \
    --episodes 10000 \  # Change this
    ...
```

### Train Specific Environment
```bash
python train_conspec_mila.py --env single --episodes 5000 --min-prototypes 5 --max-prototypes 5
```

### Disable Wandb
```bash
python train_conspec_mila.py --no-wandb --env all --episodes 5000
```

### Start Fresh (Ignore Checkpoints)
```bash
python train_conspec_mila.py --no-resume --env all --episodes 5000
```

## Troubleshooting

### Job Fails
Check error logs:
```bash
cat logs/conspec_<job_id>.err
```

### Out of Memory
Edit `run_conspec_mila.sh`:
```bash
#SBATCH --mem=32G  # Increase from 16G
```

### Timeout
Edit `run_conspec_mila.sh`:
```bash
#SBATCH --time=48:00:00  # Increase from 24 hours
```

## Files Created

| File | Purpose |
|------|---------|
| `train_conspec_mila.py` | Main training script with checkpointing |
| `run_conspec_mila.sh` | Slurm batch submission script |
| `setup_and_run_mila.sh` | Setup helper script |
| `MILA_CONSPEC_INSTRUCTIONS.md` | Detailed usage instructions |
| `train_conspec_wandb.py` | Wandb training (general use) |
| `train_conspec_all_environments.py` | Multi-env training |
| `load_optimal_policy.py` | Model loading utilities |

## Next Steps

1. **Push to GitHub**: 
   ```bash
   git push origin main
   ```
   (Note: May need to use Git LFS for large PNG files)

2. **SSH to Mila**:
   ```bash
   ssh mila
   ```

3. **Clone/Pull Repository**:
   ```bash
   cd ~
   git clone <your-repo-url> MemorySchema
   # or
   cd ~/MemorySchema && git pull
   ```

4. **Setup and Run**:
   ```bash
   cd ~/MemorySchema
   source venv/bin/activate
   ./setup_and_run_mila.sh
   sbatch run_conspec_mila.sh
   ```

5. **Monitor Progress**:
   - Check Slurm: `squeue -u $USER`
   - Check Wandb: https://wandb.ai/
   - Check logs: `tail -f logs/conspec_*.out`

## Summary

✅ **Checkpoint system**: Saves every 500 episodes + best models  
✅ **Resume capability**: Auto-resumes from latest checkpoint  
✅ **Hyperparameter search**: Tests 3-7 prototypes per environment  
✅ **Wandb integration**: Real-time experiment tracking  
✅ **Mila-ready**: Slurm scripts configured for cluster  
✅ **15 total experiments**: 3 environments × 5 prototype values  
✅ **Production-ready**: Can run for days without data loss  

Your ConSpec training system is now fully configured for the Mila cluster with automatic checkpointing, hyperparameter search, and experiment tracking! 🚀
