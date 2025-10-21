# Schema Learning - ConSpec Project Information

## Project Details

**Project Name:** Schema Learning  
**Description:** ConSpec (Contrastive Retrospection) training on Key-Door-Goal environments for understanding graph dependencies in reinforcement learning

## Your Information

**Mila Email:** samieima@mila.quebec  
**Wandb Username:** samieima  
**GitHub Username:** mandanasmi  

## Links

**GitHub Repository:** https://github.com/mandanasmi/MemorySchema  
**Wandb Dashboard:** https://wandb.ai/samieima/schema-learning  
**Wandb Project:** schema-learning

## Project Structure

### Environments
1. **Single Environment (Chain Graph)**
   - Blue key → Blue door → Green goal
   - Tests sequential dependency learning

2. **Double Environment (Parallel Graph)**
   - Blue key → Blue door AND Green key → Green door → Goal
   - Tests parallel exploration with equal priority

3. **Triple Environment (Sequential Graph)**
   - Blue/Green keys → Green door → Green goal + Purple key → Purple door → Purple goal
   - Tests complex multi-stage dependencies

### Hyperparameter Search
- **Prototypes tested:** 3, 4, 5, 6, 7
- **Total configurations:** 15 (3 environments × 5 prototype values)
- **Episodes per config:** 5000 (configurable)

## Quick Start Commands

### On Mila Cluster
```bash
# 1. SSH to Mila
ssh mila

# 2. Navigate to project
cd ~/MemorySchema

# 3. Activate virtual environment
source venv/bin/activate

# 4. Setup wandb
./setup_wandb.sh

# 5. Submit training job
sbatch run_conspec_mila.sh
```

### Monitor Training
```bash
# Check job status
squeue -u $USER

# View live logs
tail -f logs/conspec_*.out

# View wandb dashboard
# https://wandb.ai/samieima/schema-learning
```

## Key Features

✅ **Automatic Checkpointing** - Saves every 500 episodes  
✅ **Resume Capability** - Never start from scratch  
✅ **Wandb Integration** - Real-time experiment tracking  
✅ **Hyperparameter Search** - Finds optimal prototype count  
✅ **Best Model Saving** - Saves top-performing models  
✅ **Memory Preservation** - ConSpec success/failure buffers saved  

## File Structure

```
MemorySchema/
├── checkpoints/                  # Saved models and checkpoints
├── logs/                         # Slurm job logs
├── figures/                      # Visualizations
├── train_conspec_mila.py         # Main training script (Mila)
├── train_conspec_wandb.py        # Wandb training script
├── run_conspec_mila.sh           # Slurm batch script
├── setup_wandb.sh                # Wandb setup helper
├── MILA_CONSPEC_INSTRUCTIONS.md  # Complete usage guide
├── WANDB_QUICK_REFERENCE.md      # Wandb reference
└── README.md                     # Project README
```

## Output Files

### Checkpoints (in `checkpoints/`)
- `checkpoint_{env}_proto{N}_latest.pth` - Latest for resumption
- `checkpoint_{env}_proto{N}_ep{episode}.pth` - Periodic saves
- `best_model_{env}_proto{N}.pth` - Best performing model
- `training_summary.json` - Results summary

### Each Checkpoint Contains
- Policy model weights
- Optimizer state
- ConSpec encoder/prototype networks
- Training statistics
- Memory buffers
- Episode number

## Wandb Metrics Tracked

- `episode_reward` - Reward per episode
- `episode_length` - Steps per episode
- `success_rate` - Rolling success rate
- `best_success_rate` - Best rate achieved
- `avg_intrinsic_reward` - ConSpec intrinsic rewards
- `memory_success` - Successful trajectories stored
- `memory_failure` - Failed trajectories stored
- `epsilon` - Exploration rate
- `policy_loss` - Policy gradient loss
- `value_loss` - Value function loss

## Expected Results

### Success Rate Targets
- **Single Environment:** >70%
- **Double Environment:** >60%
- **Triple Environment:** >50%

### Training Time
- **Per configuration:** ~2-4 hours on GPU
- **Total (15 configs):** ~30-60 hours
- **Job time limit:** 24 hours

## Citation

If you use this work, please cite:

**ConSpec Paper:**
```bibtex
@article{sun2023conspec,
  title={Contrastive Retrospection: Learning from Success and Failure},
  author={Sun, Chen and Yang, Wannan and Jiralerspong, Thomas and Malenfant, Dane and Alsbury-Nealy, Benjamin and Bengio, Yoshua and Richards, Blake},
  journal={arXiv preprint arXiv:2210.05845},
  year={2023}
}
```

**Repository:**
```
GitHub: https://github.com/mandanasmi/MemorySchema
Wandb: https://wandb.ai/samieima/schema-learning
```

## Contact

**Email:** samieima@mila.quebec  
**GitHub:** @mandanasmi  
**Wandb:** @samieima  

## Documentation Files

1. **MILA_CONSPEC_INSTRUCTIONS.md** - Complete Mila cluster guide
2. **WANDB_QUICK_REFERENCE.md** - Wandb usage and features
3. **CONSPEC_MILA_SETUP_SUMMARY.md** - Quick setup reference
4. **CONSPEC_INTEGRATION.md** - ConSpec implementation details
5. **README.md** - Project overview and environments

## Version Control

**Branch:** main  
**Latest Commit:** Update wandb project name to 'schema-learning'  
**Status:** Ready for deployment on Mila cluster

---

**Last Updated:** 2025-10-21  
**Status:** ✅ Ready for Training
