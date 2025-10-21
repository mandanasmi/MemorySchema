# Weights & Biases Quick Reference

## Your Wandb Configuration

**Entity (Username):** `samieima@mila.quebec` → `samieima`  
**Project Name:** `schema-learning`  
**GitHub:** `mandanasmi`  
**Dashboard URL:** https://wandb.ai/mandanasmi/schema-learning  
**Repository:** https://github.com/mandanasmi/MemorySchema

## Setup on Mila

```bash
# 1. Install wandb
pip install wandb

# 2. Login (get API key from https://wandb.ai/authorize)
wandb login

# Or use the setup script
./setup_wandb.sh
```

## What Gets Tracked

All training runs will automatically log to wandb:

### Real-time Metrics
- **Episode Metrics**
  - `episode_reward`: Reward per episode
  - `episode_length`: Steps per episode
  - `success`: Binary success indicator
  - `success_rate`: Rolling average success rate

- **ConSpec Metrics**
  - `avg_intrinsic_reward`: Intrinsic rewards from ConSpec
  - `memory_success`: Number of successful trajectories in memory
  - `memory_failure`: Number of failed trajectories in memory

- **Training Metrics**
  - `epsilon`: Exploration rate
  - `policy_loss`: Policy gradient loss
  - `value_loss`: Value function loss
  - `best_success_rate`: Best success rate achieved

### Per Environment-Prototype Configuration
Each combination of environment and prototype count gets its own wandb run:
- `single_proto3` → Single environment with 3 prototypes
- `single_proto4` → Single environment with 4 prototypes
- ... (15 total runs)

## Viewing Your Experiments

### Dashboard
Go to: https://wandb.ai/mandanasmi/schema-learning

### Features Available
1. **Runs Table**: Compare all 15 configurations
2. **Charts**: Plot any metric over time
3. **Compare**: Side-by-side comparison of runs
4. **Reports**: Create custom analysis reports
5. **Artifacts**: Download saved models
6. **Logs**: View console output

## Quick Commands

### Check if logged in
```bash
wandb verify
```

### View current user
```bash
wandb whoami
```

### List projects
```bash
wandb projects
```

### Pull run data
```bash
wandb pull <run-id>
```

## Example Queries in Dashboard

### Find Best Configuration
Filter by:
- `best_success_rate` (descending)
- Group by: `env_type`

### Compare Prototype Counts
- X-axis: Episode
- Y-axis: Success rate
- Group by: `num_prototypes`
- Color by: `env_type`

### Track Memory Growth
- Plot: `memory_success` and `memory_failure`
- Over: Episode number
- To see how ConSpec learns from experience

## Useful Wandb Features

### 1. Real-time Monitoring
Watch your training progress live while the job runs on Mila

### 2. Automatic Comparisons
Wandb automatically creates comparison views for all your runs

### 3. Hyperparameter Sweeps
Already configured! The script runs 15 configs automatically

### 4. Model Artifacts
Best models are saved as wandb artifacts:
```python
import wandb
run = wandb.init()
artifact = run.use_artifact('model_single_proto5:latest')
artifact_dir = artifact.download()
```

### 5. Custom Visualizations
Create custom charts in the dashboard:
- Success Rate vs Prototypes
- Memory Size vs Episode
- Loss curves comparison

## Integration in Your Code

The wandb integration is already set up in:
- `train_conspec_mila.py` - Main Mila training script
- `train_conspec_wandb.py` - General wandb training

### Logged Automatically
```python
wandb.log({
    'episode': episode,
    'episode_reward': episode_reward,
    'success_rate': recent_success_rate,
    'best_success_rate': best_success_rate,
    'avg_intrinsic_reward': np.mean(episode_intrinsic_rewards),
    'memory_success': len(conspec.memory.success_memory),
    'memory_failure': len(conspec.memory.failure_memory),
})
```

### Model Saving
```python
# Best models are logged as artifacts
artifact = wandb.Artifact(f'model_{env_type}', type='model')
artifact.add_file(f'checkpoints/optimal_policy_{env_type}_env.pth')
wandb.log_artifact(artifact)
```

## Troubleshooting

### Not seeing runs
- Check you're logged in: `wandb verify`
- Check project name matches: `conspec-key-door-goal-mila`
- Check entity: `samieima`

### Runs not uploading
- Check network connectivity on Mila
- May need to wait - logs upload at end of run
- Check `~/.wandb/` for error logs

### Multiple accounts
```bash
# Logout
wandb logout

# Login with specific account
wandb login
```

## After Training Completes

### Download Results
```bash
# Download all run data
wandb pull <run-id>

# Or use Python API
import wandb
api = wandb.Api()
runs = api.runs("mandanasmi/schema-learning")
for run in runs:
    print(f"{run.name}: {run.summary['best_success_rate']}")
```

### Create Report
In the wandb dashboard:
1. Click "Reports" 
2. "Create Report"
3. Add charts comparing all 15 configurations
4. Share with collaborators

## Summary

✅ **Entity configured**: `samieima`  
✅ **Project created**: `schema-learning`  
✅ **GitHub**: `mandanasmi`  
✅ **Auto-logging enabled**: All metrics tracked  
✅ **Model artifacts**: Best models saved  
✅ **Dashboard ready**: https://wandb.ai/mandanasmi/schema-learning  
✅ **Repository**: https://github.com/mandanasmi/MemorySchema

Your experiments will be beautifully tracked and easy to analyze! 📊
