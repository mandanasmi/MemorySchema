# MemorySchema Setup on Mila Cluster

This guide will help you set up the MemorySchema repository on the Mila cluster.

## Prerequisites

1. Access to the Mila cluster
2. SSH access configured
3. Git access to the repository

## Quick Setup

### Step 1: Connect to Mila Cluster

```bash
ssh mila
```

### Step 2: Clone and Setup Repository

You have two options:

#### Option A: Use the automated setup script (Recommended)

```bash
# Download and run the setup script
curl -O https://raw.githubusercontent.com/mandanasamiei/MemorySchema/main/setup_mila.sh
chmod +x setup_mila.sh
./setup_mila.sh
```

#### Option B: Manual setup

```bash
# Clone the repository
git clone https://github.com/mandanasamiei/MemorySchema.git
cd MemorySchema

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install additional packages for Mila cluster
pip install jupyter matplotlib seaborn

# Test the setup
python test_setup.py
```

## Running the Project

### Activate Environment

```bash
# Quick activation
source ~/MemorySchema/activate_env.sh

# Or manual activation
cd ~/MemorySchema
source venv/bin/activate
```

### Run Demo

```bash
# Using the run script
~/MemorySchema/run_demo.sh

# Or manually
cd ~/MemorySchema
source venv/bin/activate
python demo.py
```

### Run Other Scripts

```bash
# Test environment
python environment.py

# Run structured environment
python structured_test.py

# Run visualizations
python structured_visualizer.py
```

## Mila Cluster Specific Tips

### Interactive Sessions

```bash
# Request an interactive session with GPU
srun --gres=gpu:1 --pty bash

# Request interactive session with specific time limit
srun --time=2:00:00 --gres=gpu:1 --pty bash
```

### Batch Jobs

Create a job script `run_memory_schema.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=memory_schema
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

cd ~/MemorySchema
source venv/bin/activate
python demo.py
```

Submit with:
```bash
sbatch run_memory_schema.sh
```

### Monitor Jobs

```bash
# Check your jobs
squeue -u $USER

# Check GPU usage
nvidia-smi

# Check job details
scontrol show job <job_id>
```

## Environment Details

The setup includes:

- **Python 3.9+** virtual environment
- **MiniGrid 3.0.0** for reinforcement learning environments
- **Gymnasium 1.1.1** for RL interface
- **NumPy 2.0.2** for numerical computations
- **Pygame 2.6.1** for rendering
- **Additional packages**: jupyter, matplotlib, seaborn

## Project Structure

```
~/MemorySchema/
├── venv/                    # Virtual environment
├── environment.py           # Main environment
├── demo.py                  # Demo script
├── structured_env.py        # Structured environment
├── structured_visualizer.py # Visualization tools
├── requirements.txt         # Dependencies
├── setup_mila.sh           # Setup script
├── activate_env.sh         # Quick activation
└── run_demo.sh             # Quick run script
```

## Troubleshooting

### Common Issues

1. **Permission denied**: Make sure scripts are executable
   ```bash
   chmod +x setup_mila.sh activate_env.sh run_demo.sh
   ```

2. **Package conflicts**: Recreate virtual environment
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **GPU not available**: Check GPU allocation
   ```bash
   nvidia-smi
   squeue -u $USER
   ```

4. **Display issues**: For GUI applications, use X11 forwarding
   ```bash
   ssh -X mila
   ```

### Getting Help

- Check Mila cluster documentation
- Use `module avail` to see available software modules
- Contact Mila support for cluster-specific issues

## Next Steps

1. **Explore the environments**: Try different environment configurations
2. **Run experiments**: Use the structured environments for your research
3. **Visualize results**: Use the visualization tools to analyze performance
4. **Scale up**: Use batch jobs for larger experiments

## File Locations

- **Repository**: `~/MemorySchema/`
- **Virtual environment**: `~/MemorySchema/venv/`
- **Activation script**: `~/MemorySchema/activate_env.sh`
- **Run script**: `~/MemorySchema/run_demo.sh`
