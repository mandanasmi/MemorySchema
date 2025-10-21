# Launch ConSpec Training on Mila Cluster

## Important Note

The SSH host key for Mila has changed. You'll need to manually SSH to Mila and run the commands below.

## Step-by-Step Launch Instructions

### Step 1: Fix SSH Host Key (One-time)

```bash
# On your local machine
ssh-keygen -R mila.quebec
# This removes the old host key
```

### Step 2: SSH to Mila

```bash
ssh samieima@mila.quebec
# Answer 'yes' when asked about the new host key
```

### Step 3: Setup Repository on Mila

Once logged into Mila, run these commands:

```bash
# Navigate to home directory
cd ~

# Clone repository if first time
git clone https://github.com/mandanasmi/MemorySchema.git

# OR pull latest if already cloned
cd ~/MemorySchema
git pull origin main

# Go to project directory
cd ~/MemorySchema
```

### Step 4: Setup Virtual Environment

```bash
# Load required modules
module load python/3.9
module load cuda/11.3

# Create virtual environment
python3.9 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install scikit-learn for analysis
pip install scikit-learn
```

### Step 5: Login to Wandb

```bash
# Install wandb (if not already in requirements)
pip install wandb

# Login to wandb
wandb login

# When prompted, paste your API key from:
# https://wandb.ai/authorize
```

### Step 6: Create Directories

```bash
mkdir -p logs
mkdir -p checkpoints
mkdir -p figures
```

### Step 7: Make Scripts Executable

```bash
chmod +x run_conspec_mila.sh
chmod +x setup_and_run_mila.sh
chmod +x setup_wandb.sh
chmod +x setup_mila.sh
chmod +x monitor_training.sh
```

### Step 8: Submit the Job

```bash
# Submit the ConSpec training job
sbatch run_conspec_mila.sh
```

You should see output like:
```
Submitted batch job 1234567
```

### Step 9: Verify Job is Running

```bash
# Check your job queue
squeue -u samieima

# Expected output (example):
# JOBID    PARTITION  NAME              USER      ST  TIME  NODES
# 1234567  main       conspec_training  samieima  R   0:05  1
```

### Step 10: Monitor Training

```bash
# View live output (replace JOB_ID with your actual job ID)
tail -f logs/conspec_JOB_ID.out

# View errors (if any)
tail -f logs/conspec_JOB_ID.err

# Check GPU usage
nvidia-smi  # (only works on the compute node)
```

### Step 11: View Wandb Dashboard

Open in browser: **https://wandb.ai/mandanasmi/schema-learning**

All 15 training runs will appear here as they execute!

## What the Job Does

### Training Configuration
- **Environments:** Single, Double, Triple (3 total)
- **Prototypes:** 3, 4, 5, 6, 7 (5 values)
- **Total Runs:** 15 configurations
- **Episodes per run:** 5000
- **Device:** GPU (CUDA)
- **Checkpoints:** Saved every 500 episodes

### Expected Runtime
- **Per configuration:** 2-4 hours on GPU
- **Total:** 30-60 hours for all 15 runs
- **Job time limit:** 24 hours (may need resubmission)

### Output Files

All saved in `checkpoints/`:
```
checkpoints/
├── checkpoint_single_proto3_latest.pth    # Resume from here
├── checkpoint_single_proto3_ep500.pth     # Periodic save
├── checkpoint_single_proto3_ep1000.pth    # Periodic save
├── best_model_single_proto3.pth           # Best model
├── ... (repeated for all 15 configurations)
└── training_summary.json                   # All results
```

## Monitoring Commands

```bash
# Check job status
squeue -u samieima

# Check detailed job info
scontrol show job JOB_ID

# View recent output
tail -n 50 logs/conspec_JOB_ID.out

# Check resource usage
sacct -j JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,State

# List all your jobs
squeue -u samieima --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
```

## Managing Jobs

```bash
# Cancel a job
scancel JOB_ID

# Cancel all your jobs
scancel -u samieima

# Hold a job (pause)
scontrol hold JOB_ID

# Release a held job
scontrol release JOB_ID
```

## If Job Doesn't Start

### Check Queue Position
```bash
squeue -u samieima --start
```

### Check GPU Availability
```bash
sinfo -p main --Format=NodeList,Gres,GresUsed
```

### Try Different Partition
Edit `run_conspec_mila.sh`:
```bash
#SBATCH --partition=long
```

## After Job Completes

### Download Results to Local Machine

```bash
# On your local machine (not on Mila)
scp -r samieima@mila.quebec:~/MemorySchema/checkpoints ~/MemorySchema/
scp -r samieima@mila.quebec:~/MemorySchema/logs ~/MemorySchema/
```

### View Results Summary

```bash
# On Mila
cat ~/MemorySchema/checkpoints/training_summary.json
```

## Troubleshooting

### Job Failed Immediately
```bash
# Check error log
cat logs/conspec_JOB_ID.err

# Common issues:
# - Module not found: Check venv is activated
# - CUDA error: Check GPU availability
# - Wandb error: Run 'wandb login' again
```

### Out of Memory
Edit `run_conspec_mila.sh`:
```bash
#SBATCH --mem=32G  # Increase from 16G
```

### Job Timeout
Edit `run_conspec_mila.sh`:
```bash
#SBATCH --time=48:00:00  # Increase from 24 hours
```

### Wandb Not Logging
```bash
# Verify wandb login
wandb verify

# Check wandb status
wandb status

# Re-login if needed
wandb login
```

## Quick Copy-Paste Commands

```bash
# 1. Fix SSH (on local machine)
ssh-keygen -R mila.quebec

# 2. SSH to Mila
ssh samieima@mila.quebec

# 3. Setup (on Mila)
cd ~
git clone https://github.com/mandanasmi/MemorySchema.git || (cd ~/MemorySchema && git pull)
cd ~/MemorySchema
module load python/3.9 cuda/11.3
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
wandb login
mkdir -p logs checkpoints
chmod +x *.sh

# 4. Submit job
sbatch run_conspec_mila.sh

# 5. Monitor
squeue -u samieima
```

## Alternative: Interactive Testing

Before submitting the full batch job, test interactively:

```bash
# Request interactive session with GPU
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=2:00:00

# Once allocated:
cd ~/MemorySchema
source venv/bin/activate

# Run a quick test
python train_conspec_mila.py --env single --episodes 100 --min-prototypes 5 --max-prototypes 5

# If successful, exit and submit batch job
exit
sbatch run_conspec_mila.sh
```

## Expected Output

When you run `squeue -u samieima`, you should see:

```
JOBID    PARTITION  NAME              USER      ST  TIME  NODES  NODELIST(REASON)
1234567  main       conspec_training  samieima  R   2:15  1      cn-g001
```

- **ST = R:** Running
- **ST = PD:** Pending (waiting for resources)
- **ST = CG:** Completing

## Success Indicators

1. Job appears in `squeue -u samieima`
2. Log file created in `logs/conspec_JOB_ID.out`
3. Wandb shows active run at https://wandb.ai/mandanasmi/schema-learning
4. Checkpoints being created in `checkpoints/`

## Summary

You need to manually:
1. Fix SSH host key: `ssh-keygen -R mila.quebec`
2. SSH to Mila: `ssh samieima@mila.quebec`
3. Clone/pull repository
4. Setup venv and install dependencies
5. Login to wandb
6. Submit job: `sbatch run_conspec_mila.sh`
7. Monitor: `squeue -u samieima`

The job will train ConSpec on all 3 environments with 5 different prototype configurations (15 total runs) and track everything in wandb!

