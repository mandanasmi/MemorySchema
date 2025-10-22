# Troubleshooting Guide for Mila Cluster

## Quick Diagnostics

### Step 1: Run Diagnostic Script

```bash
# SSH to Mila
ssh mila
cd ~/MemorySchema
bash scripts/diagnose_mila_setup.sh
```

This will check:
- Directory structure
- Python and module availability
- Package installations
- Recent job history
- Log files

## Common Issues and Solutions

### Issue 1: Job Ended Immediately

**Check job status:**
```bash
# See recent jobs
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed -S $(date -d '7 days ago' +%Y-%m-%d)

# Check specific job details
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS
```

**Exit codes:**
- `0`: Success
- `1`: General error
- `127`: Command not found
- `137`: Out of memory (killed)
- `CANCELLED`: Job was cancelled
- `TIMEOUT`: Exceeded time limit

### Issue 2: Missing Dependencies

**Solution:**
```bash
# Load required modules
module load python/3.9 cuda/11.3 libffi

# Install missing packages
pip install --user torch torchvision
pip install --user wandb scikit-learn
pip install --user pyyaml

# Verify installations
python3.9 -c "import torch; import wandb; import yaml; print('All packages OK')"
```

### Issue 3: Import Errors

**Check if environments can be imported:**
```bash
cd ~/MemorySchema
python3.9 -c "from environments.key_door_goal_base import create_key_door_goal_env"
```

**If this fails:**
```bash
# Check Python path
echo $PYTHONPATH

# Add to path if needed
export PYTHONPATH=$HOME/MemorySchema:$PYTHONPATH

# Or modify the script to add path
```

### Issue 4: Wandb Not Logged In

**Check status:**
```bash
wandb status
```

**Login:**
```bash
wandb login
# Enter your API key when prompted
```

**Find your API key:**
1. Go to https://wandb.ai/authorize
2. Copy the API key
3. Paste when prompted

### Issue 5: Config File Not Found

**Check if file exists:**
```bash
ls -la ~/MemorySchema/configs/conspec_hyperparam_search.yaml
```

**If missing:**
```bash
cd ~/MemorySchema
git pull origin conspec  # or dag-gflownet
```

### Issue 6: Out of Memory

**Symptoms:**
- Job state: `OUT_OF_MEMORY` or `OOM`
- Exit code: 137

**Solutions:**
1. Reduce batch size in config:
   ```yaml
   batch_size:
     type: choice
     values: [16, 32, 64]  # Remove 128, 256
   ```

2. Request more memory in Slurm script:
   ```bash
   #SBATCH --mem=64G  # Instead of 32G
   ```

3. Reduce number of parallel environments:
   ```python
   # In hyperparameter_search_conspec.py
   env = create_key_door_goal_env(env_type, size=10)  # Smaller environments
   ```

### Issue 7: Time Limit Exceeded

**Symptoms:**
- Job state: `TIMEOUT`
- Exit code: TIMEOUT

**Solutions:**
1. Increase time limit:
   ```bash
   #SBATCH --time=72:00:00  # 72 hours instead of 48
   ```

2. Reduce number of trials:
   ```yaml
   search:
     n_trials: 30  # Instead of 50
   ```

3. Reduce episodes per trial:
   ```yaml
   training:
     episodes: 3000  # Instead of 5000
   ```

### Issue 8: GPU Not Available

**Check GPU allocation:**
```bash
# In a job
nvidia-smi

# If no GPU, check Slurm request
scontrol show job <JOB_ID>
```

**Fix Slurm script:**
```bash
#SBATCH --gres=gpu:1  # Request 1 GPU
```

### Issue 9: No Output in Logs

**Check if logs directory exists:**
```bash
mkdir -p ~/MemorySchema/logs
```

**Check Slurm script paths:**
```bash
#SBATCH --output=logs/hypersearch_%j.out
#SBATCH --error=logs/hypersearch_%j.err
```

**Verify paths are relative to working directory:**
```bash
cd ~/MemorySchema  # Must be here when submitting
sbatch scripts/run_hyperparam_search_mila.sh
```

### Issue 10: Permission Denied

**Symptoms:**
- Cannot write to directory
- Cannot execute script

**Solutions:**
```bash
# Make script executable
chmod +x scripts/run_hyperparam_search_mila.sh

# Fix directory permissions
chmod 755 ~/MemorySchema
chmod 755 ~/MemorySchema/logs
```

## Viewing Logs in Real-Time

### Monitor Running Job

```bash
# Get job ID
squeue -u $USER

# Tail the output log
tail -f ~/MemorySchema/logs/hypersearch_<JOB_ID>.out

# Check error log
tail -f ~/MemorySchema/logs/hypersearch_<JOB_ID>.err
```

### Check Completed Job Logs

```bash
# List recent logs
ls -lht ~/MemorySchema/logs/

# View full log
cat ~/MemorySchema/logs/hypersearch_<JOB_ID>.out

# View errors only
cat ~/MemorySchema/logs/hypersearch_<JOB_ID>.err
```

## Testing Before Full Run

### Quick Test Run

```bash
# On Mila
cd ~/MemorySchema

# Test with minimal iterations
python3.9 scripts/quick_hyperparam_test.py
```

### Interactive Job for Debugging

```bash
# Request interactive session
salloc --gres=gpu:1 --mem=16G --time=1:00:00

# Once allocated, run commands manually
module load python/3.9 cuda/11.3 libffi
cd ~/MemorySchema
python3.9 scripts/quick_hyperparam_test.py
```

## Resubmitting Jobs

### After Fixing Issues

```bash
cd ~/MemorySchema

# Pull latest changes if needed
git pull origin conspec

# Ensure modules are loaded
module load python/3.9 cuda/11.3 libffi

# Submit job
sbatch scripts/run_hyperparam_search_mila.sh

# Verify submission
squeue -u $USER
```

### Monitor Progress

```bash
# Check job status
watch -n 10 'squeue -u $USER'

# Or use scontrol
scontrol show job <JOB_ID>

# Check resource usage
sacct -j <JOB_ID> --format=JobID,Elapsed,MaxRSS,State
```

## Getting Help

### Mila Specific

If issues persist:

1. **Check Mila documentation:**
   - https://docs.mila.quebec/

2. **Contact Mila support:**
   - Email: support@mila.quebec
   - Include job ID and error logs

3. **Check Slurm status:**
   ```bash
   sinfo  # Cluster status
   squeue --start  # Estimated start times
   ```

### Debug Mode

Add debug output to your script:

```bash
# In run_hyperparam_search_mila.sh, add:
set -x  # Print all commands
set -e  # Exit on error

# Add verbose logging
python3.9 -u metrics_experiments/hyperparameter_search_conspec.py \
    --config configs/conspec_hyperparam_search.yaml \
    --env all \
    2>&1 | tee logs/debug.log
```

## Quick Command Reference

```bash
# Job Management
sbatch <script>              # Submit job
squeue -u $USER              # Check your jobs
scancel <JOB_ID>             # Cancel job
scontrol show job <JOB_ID>   # Job details

# Resource Checking
sinfo                        # Cluster info
squeue --start               # Job start times
sacct -u $USER               # Job history

# Module Management
module avail                 # Available modules
module list                  # Loaded modules
module load <module>         # Load module
module purge                 # Unload all

# File Management
du -sh ~/MemorySchema        # Directory size
df -h $HOME                  # Disk usage
quota -s                     # Your quota
```

---

**Last Updated**: October 22, 2025

