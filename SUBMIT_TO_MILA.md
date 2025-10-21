# Submit ConSpec Training to Mila Cluster

## Current Status
 **No jobs running on Mila yet** - You need to submit the job from the Mila cluster

## Step-by-Step Guide to Submit Job on Mila

### Step 1: Push Code to GitHub
```bash
# On your local machine
cd ~/MemorySchema
git add -A
git commit -m "Ready for Mila cluster deployment"
git push origin main
```

### Step 2: SSH to Mila
```bash
ssh samieima@mila.quebec
# Or if configured:
ssh mila
```

### Step 3: Clone or Pull Repository on Mila
```bash
# If first time:
cd ~
git clone https://github.com/mandanasmi/MemorySchema.git
cd MemorySchema

# If already cloned:
cd ~/MemorySchema
git pull origin main
```

### Step 4: Setup Virtual Environment on Mila
```bash
cd ~/MemorySchema

# Create virtual environment if not exists
python3.9 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio
pip install scikit-learn wandb
```

### Step 5: Login to Wandb on Mila
```bash
# Still on Mila cluster
wandb login
# Paste your API key from: https://wandb.ai/authorize
```

### Step 6: Create Necessary Directories
```bash
mkdir -p logs checkpoints figures
```

### Step 7: Submit the Job
```bash
# Make scripts executable
chmod +x run_conspec_mila.sh setup_and_run_mila.sh

# Submit the batch job
sbatch run_conspec_mila.sh
```

You should see output like:
```
Submitted batch job 123456
```

### Step 8: Verify Job is Running
```bash
# Check your jobs
squeue -u samieima

# Should show something like:
# JOBID  PARTITION  NAME              USER      ST  TIME  NODES
# 123456 main       conspec_training  samieima  R   0:01  1
```

## Monitoring Your Job on Mila

### Check Job Status
```bash
squeue -u samieima
```

### View Live Output
```bash
# Find the job ID from squeue, then:
tail -f logs/conspec_<job_id>.out

# Example:
tail -f logs/conspec_123456.out
```

### View Errors (if any)
```bash
tail -f logs/conspec_<job_id>.err
```

### Check GPU Usage
```bash
# On the compute node where your job is running
ssh <node_name>
nvidia-smi
```

### View Wandb Dashboard
Open in browser: https://wandb.ai/mandanasmi/schema-learning

## What the Job Will Do

1. **Train on 3 Environments**
   - Single (Chain graph)
   - Double (Parallel graph)
   - Triple (Sequential graph)

2. **Test 5 Prototype Configurations**
   - 3, 4, 5, 6, 7 prototypes each

3. **Total: 15 Training Runs**
   - Each run: 5000 episodes
   - Checkpoints saved every 500 episodes
   - Best models saved automatically

4. **Results Saved in:**
   - `checkpoints/` - All model checkpoints
   - `logs/` - Slurm output logs
   - Wandb dashboard - Real-time metrics

## Job Management

### Cancel Job
```bash
scancel <job_id>
```

### Cancel All Your Jobs
```bash
scancel -u samieima
```

### Check Job Details
```bash
scontrol show job <job_id>
```

### Check Job Resource Usage
```bash
sacct -j <job_id> --format=JobID,JobName,MaxRSS,Elapsed
```

## If Job Fails

### Check Error Logs
```bash
cat logs/conspec_<job_id>.err
```

### Common Issues

1. **Out of Memory**
   - Edit `run_conspec_mila.sh`: Change `#SBATCH --mem=16G` to `#SBATCH --mem=32G`

2. **Timeout**
   - Edit `run_conspec_mila.sh`: Change `#SBATCH --time=24:00:00` to `#SBATCH --time=48:00:00`

3. **GPU Not Available**
   - Change partition: `#SBATCH --partition=long`
   - Or try different GPU: `#SBATCH --gres=gpu:rtx8000:1`

4. **Module Load Failures**
   - Check available modules: `module avail python`
   - Adjust in script if needed

## After Job Completes

### Copy Results to Local Machine
```bash
# On your local machine
scp -r samieima@mila.quebec:~/MemorySchema/checkpoints ~/MemorySchema/
scp -r samieima@mila.quebec:~/MemorySchema/logs ~/MemorySchema/
```

### View Results Summary
```bash
# On Mila
cat ~/MemorySchema/checkpoints/training_summary.json

# Or download and view locally
```

## Alternative: Interactive Session

For testing before submitting batch job:

```bash
# On Mila
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=2:00:00

# Wait for allocation, then:
source ~/MemorySchema/venv/bin/activate
cd ~/MemorySchema

# Run a quick test
python train_conspec_mila.py --env single --episodes 100 --min-prototypes 5 --max-prototypes 5

# Exit when done
exit
```

## Quick Command Summary

```bash
# On Mila cluster:
cd ~/MemorySchema
source venv/bin/activate
sbatch run_conspec_mila.sh          # Submit job
squeue -u samieima                   # Check status
tail -f logs/conspec_*.out          # View logs
scancel <job_id>                    # Cancel if needed
```

## Expected Timeline

- **Setup on Mila**: 10-15 minutes
- **Per training run**: 2-4 hours on GPU
- **Total (15 runs)**: 30-60 hours
- **Job will run**: 24 hours (may need resubmission for remaining runs)

## Wandb Dashboard URL

**Your Dashboard:** https://wandb.ai/mandanasmi/schema-learning

All 15 training runs will appear here as they execute!

---

**Remember:** The job only runs on Mila after you submit it with `sbatch` from the Mila cluster!

