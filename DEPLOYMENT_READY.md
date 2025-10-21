# 🚀 ConSpec Schema Learning - Ready for Mila Deployment

## ✅ Repository Cleaned and Ready!

All redundant files have been removed. The repository is now production-ready for Mila cluster deployment.

## Your Configuration

**Mila Email:** samieima@mila.quebec  
**Wandb Entity:** mandanasmi  
**Wandb Project:** schema-learning  
**GitHub:** mandanasmi  

**Wandb Dashboard:** https://wandb.ai/mandanasmi/schema-learning  
**GitHub Repository:** https://github.com/mandanasmi/MemorySchema

## 📋 Quick Deployment Checklist

### On Mila Cluster (Run these commands):

```bash
# 1. SSH to Mila
ssh samieima@mila.quebec

# 2. Clone repository (if first time)
cd ~
git clone https://github.com/mandanasmi/MemorySchema.git
cd MemorySchema

# Or pull latest (if already cloned)
cd ~/MemorySchema
git pull origin main

# 3. Setup virtual environment
python3.9 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For GPU support (recommended):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Login to wandb
wandb login
# Paste API key from: https://wandb.ai/authorize

# 6. Create directories
mkdir -p logs checkpoints figures

# 7. Make scripts executable
chmod +x *.sh

# 8. Submit job
sbatch run_conspec_mila.sh

# You should see: "Submitted batch job XXXXXX"
```

### Monitor Job

```bash
# Check job status
squeue -u samieima

# View live logs
tail -f logs/conspec_*.out

# View wandb dashboard
# https://wandb.ai/mandanasmi/schema-learning
```

## 📁 Repository Structure (Cleaned)

### Core Environment Files
- `key_door_goal_base.py` - Environment definitions
- `key_door_goal_visualizer.py` - Visualization system
- `key_door_goal_demo.py` - Environment demo

### ConSpec Implementation
- `conspec_minigrid.py` - Main ConSpec implementation

### Training Scripts
- `train_conspec_mila.py` - **Main script for Mila** (with checkpointing & hyperparameter search)
- `train_conspec_wandb.py` - General wandb training
- `train_conspec_all_environments.py` - Multi-environment training
- `quick_test_conspec.py` - Quick testing

### Testing Scripts
- `test_conspec.py` - ConSpec functionality tests
- `test_key_door_goal.py` - Environment tests

### Utilities
- `load_optimal_policy.py` - Model loading utilities
- `monitor_training.sh` - Training monitoring

### Mila Cluster Scripts
- `run_conspec_mila.sh` - **Slurm batch script** (GPU-enabled)
- `setup_and_run_mila.sh` - Setup helper
- `setup_mila.sh` - Mila environment setup
- `setup_wandb.sh` - Wandb configuration

### Documentation
- `README.md` - Main project documentation
- `MILA_SETUP.md` - Mila cluster setup guide
- `MILA_CONSPEC_INSTRUCTIONS.md` - Complete usage instructions
- `SUBMIT_TO_MILA.md` - Step-by-step submission guide
- `CONSPEC_INTEGRATION.md` - ConSpec implementation details
- `CONSPEC_MILA_SETUP_SUMMARY.md` - Quick reference
- `WANDB_QUICK_REFERENCE.md` - Wandb usage guide
- `PROJECT_INFO.md` - Project information
- `deploy_to_mila.txt` - Copy-paste deployment commands
- `DEPLOYMENT_READY.md` - This file

### Configuration
- `requirements.txt` - All Python dependencies
- `.gitignore` - Git ignore rules

## 🎯 What the Training Will Do

### Experiments (15 total)
- **3 Environments:** Single, Double, Triple
- **5 Prototype Values:** 3, 4, 5, 6, 7
- **5000 Episodes** per configuration

### GPU Training
- **Requested:** 1 GPU
- **Memory:** 16GB
- **CPUs:** 4
- **Time Limit:** 24 hours

### Checkpointing
- **Auto-save:** Every 500 episodes
- **Best models:** Saved automatically
- **Resume:** Can resume from any checkpoint

### Wandb Tracking
All metrics logged to: https://wandb.ai/mandanasmi/schema-learning

## 📊 Expected Output

### Checkpoints Directory
```
checkpoints/
├── checkpoint_single_proto3_latest.pth
├── checkpoint_single_proto3_ep500.pth
├── checkpoint_single_proto3_ep1000.pth
├── best_model_single_proto3.pth
├── ... (for all 15 configurations)
└── training_summary.json
```

### Wandb Dashboard
- 15 separate runs (one per configuration)
- Real-time metrics and charts
- Model artifacts
- Comparison tools

## 🔍 Removed Redundant Files

The following files were removed to clean up the repository:

1. ❌ `demo_conspec.py` → Replaced by `quick_test_conspec.py`
2. ❌ `train_conspec.py` → Replaced by `train_conspec_mila.py`
3. ❌ `train_and_analyze_conspec.py` → Replaced by `train_conspec_all_environments.py`
4. ❌ `SETUP.md` → Merged into `MILA_SETUP.md`
5. ❌ `KEY_DOOR_GOAL_README.md` → Content in main `README.md`
6. ❌ `conspec_requirements.txt` → Merged into `requirements.txt`

## ⚡ Quick Start (Copy-Paste on Mila)

```bash
cd ~
git clone https://github.com/mandanasmi/MemorySchema.git
cd MemorySchema
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
wandb login
mkdir -p logs checkpoints
sbatch run_conspec_mila.sh
squeue -u samieima
```

## 📞 Support Files

- `SUBMIT_TO_MILA.md` - Detailed submission guide
- `deploy_to_mila.txt` - Command reference
- `MILA_CONSPEC_INSTRUCTIONS.md` - Complete instructions

## ✨ Status

✅ **Repository cleaned**  
✅ **Wandb configured** (entity: mandanasmi)  
✅ **GPU-enabled batch script**  
✅ **Checkpoint system ready**  
✅ **All code pushed to GitHub**  
✅ **Documentation complete**  
✅ **Ready for Mila deployment**

**Next Step:** SSH to Mila and run the commands above! 🚀

---

**Last Updated:** 2025-10-21  
**Status:** ✅ Production Ready  
**Action Required:** Submit job on Mila cluster

