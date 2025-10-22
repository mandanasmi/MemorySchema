# MemorySchema Project Structure

## Overview

This document describes the organized structure of the MemorySchema project, designed for learning schemas using reinforcement learning and generative modeling.

## Directory Structure

```
MemorySchema/
├── environments/              # Environment implementations
│   ├── __init__.py
│   ├── key_door_goal_base.py       # Base classes and 3 environments
│   ├── key_door_goal_visualizer.py # Visualization tools
│   ├── key_door_goal_demo.py       # Demo and testing
│   └── test_key_door_goal.py       # Environment tests
│
├── methods/                   # Algorithm implementations
│   ├── __init__.py
│   ├── conspec_minigrid.py         # ConSpec algorithm
│   └── dag_gflownet/               # DAG-GFlowNet (on dag-gflownet branch)
│
├── metrics_experiments/       # Training, experiments, and analysis
│   ├── __init__.py
│   ├── train_conspec_all_environments.py  # Train on all envs
│   ├── train_conspec_mila.py              # Mila cluster training
│   ├── train_conspec_wandb.py             # Training with wandb
│   └── load_optimal_policy.py             # Load trained models
│
├── scripts/                   # Utility scripts
│   ├── setup_mila.sh                 # Mila cluster setup
│   ├── setup_wandb.sh                # Wandb configuration
│   ├── run_conspec_mila.sh           # Slurm job script
│   ├── monitor_training.sh           # Training monitoring
│   ├── auto_deploy_mila.sh           # Automated deployment
│   ├── submit_job_mila.sh            # Job submission
│   ├── quick_test_conspec.py         # Quick ConSpec test
│   └── test_conspec.py               # ConSpec unit tests
│
├── configs/                   # Configuration files
│   └── (future: hyperparameters, experiment configs)
│
├── docs/                      # Documentation
│   ├── CONSPEC_INTEGRATION.md
│   ├── CONSPEC_MILA_SETUP_SUMMARY.md
│   ├── DAG_GFLOWNET_INTEGRATION.md   # (on dag-gflownet branch)
│   ├── DAG_GFLOWNET_SUMMARY.md       # (on dag-gflownet branch)
│   ├── DEPLOYMENT_READY.md
│   ├── MILA_CONSPEC_INSTRUCTIONS.md
│   ├── MILA_SETUP.md
│   ├── PROJECT_INFO.md
│   ├── WANDB_QUICK_REFERENCE.md
│   ├── LAUNCH_ON_MILA.md
│   ├── SUBMIT_TO_MILA.md
│   ├── COPY_PASTE_MILA_COMMANDS.txt
│   └── deploy_to_mila.txt
│
├── figures/                   # Visualizations and plots
│   ├── graph_dependency_environments.png
│   ├── equal_priority_graph_environments.png
│   ├── conspec_performance_summary.png
│   └── conspec_critical_features_*.png
│
├── checkpoints/              # Model checkpoints (gitignored)
├── logs/                     # Training logs (gitignored)
├── wandb/                    # Wandb runs (gitignored)
├── venv/                     # Virtual environment (gitignored)
│
├── requirements.txt          # Python dependencies
├── README.md                 # Main project README
├── PROJECT_STRUCTURE.md      # This file
├── LICENSE                   # MIT License
└── .gitignore               # Git ignore rules
```

## Module Descriptions

### 1. Environments (`environments/`)

Contains all environment implementations and related utilities.

#### Key Files:
- **`key_door_goal_base.py`**: Core environment classes
  - `BaseKeyDoorGoalEnv`: Base class with common functionality
  - `SingleKeyDoorGoalEnv`: Chain graph (key → door → goal)
  - `DoubleKeyDoorGoalEnv`: Parallel graph (2 parallel paths)
  - `TripleKeyDoorGoalEnv`: Complex graph (multiple dependencies)

- **`key_door_goal_visualizer.py`**: Visualization tools
  - Custom rendering for keys, doors, goals
  - Dynamic legends based on environment objects
  - PDF/PNG export capabilities

- **`key_door_goal_demo.py`**: Demo script
  - Interactive environment demonstration
  - Manual control testing

- **`test_key_door_goal.py`**: Environment tests
  - Unit tests for environment functionality
  - Validation of graph dependencies

#### Usage:
```python
from environments.key_door_goal_base import SingleKeyDoorGoalEnv
from environments.key_door_goal_visualizer import KeyDoorGoalVisualizer

env = SingleKeyDoorGoalEnv()
visualizer = KeyDoorGoalVisualizer(env)
```

### 2. Methods (`methods/`)

Algorithm implementations for learning and structure discovery.

#### Current Implementations:

**ConSpec (Contrastive Retrospection)**
- File: `conspec_minigrid.py`
- Purpose: Intrinsic motivation and critical feature learning
- Components:
  - `EncoderConSpec`: CNN + LSTM encoder
  - `PrototypeNetwork`: Prototype-based memory
  - `ConSpecMemory`: Experience replay
  - `ConSpec`: Main algorithm class

**DAG-GFlowNet** (on `dag-gflownet` branch)
- Location: `methods/dag_gflownet/`
- Purpose: Bayesian structure learning
- Components: Graph environment, GFlowNet policy, scoring functions

#### Usage:
```python
from methods.conspec_minigrid import ConSpec, EncoderConSpec

encoder = EncoderConSpec(...)
conspec = ConSpec(encoder, num_prototypes=5)
```

### 3. Metrics & Experiments (`metrics_experiments/`)

Training scripts, experiment management, and performance analysis.

#### Training Scripts:

- **`train_conspec_all_environments.py`**: 
  - Train on all three environments
  - Hyperparameter search
  - Performance comparison
  - Critical feature visualization

- **`train_conspec_mila.py`**:
  - Optimized for Mila cluster
  - Checkpoint management
  - Resume training capability
  - Multi-configuration runs

- **`train_conspec_wandb.py`**:
  - Weights & Biases integration
  - Real-time metric tracking
  - Experiment organization

#### Analysis Scripts:

- **`load_optimal_policy.py`**:
  - Load trained models
  - Evaluate performance
  - Generate rollouts

#### Usage:
```bash
# Train locally
python metrics_experiments/train_conspec_wandb.py --env single --episodes 1000

# Train on Mila
sbatch scripts/run_conspec_mila.sh
```

### 4. Scripts (`scripts/`)

Utility scripts for setup, deployment, and monitoring.

#### Setup Scripts:
- `setup_mila.sh`: Mila cluster environment setup
- `setup_wandb.sh`: Wandb authentication and configuration

#### Deployment Scripts:
- `run_conspec_mila.sh`: Slurm batch script for GPU jobs
- `auto_deploy_mila.sh`: Automated deployment workflow
- `submit_job_mila.sh`: Interactive job submission

#### Monitoring:
- `monitor_training.sh`: Real-time training progress

#### Testing:
- `quick_test_conspec.py`: Quick ConSpec functionality test
- `test_conspec.py`: Comprehensive unit tests

### 5. Configs (`configs/`)

Configuration files for experiments and hyperparameters.

**Future additions:**
- Experiment configurations (YAML/JSON)
- Hyperparameter search spaces
- Environment variants
- Model architectures

### 6. Documentation (`docs/`)

Comprehensive project documentation.

#### Categories:

**Setup & Deployment:**
- `MILA_SETUP.md`: Mila cluster setup guide
- `DEPLOYMENT_READY.md`: Deployment checklist
- `LAUNCH_ON_MILA.md`: Job launch instructions
- `COPY_PASTE_MILA_COMMANDS.txt`: Quick command reference

**Algorithm Integration:**
- `CONSPEC_INTEGRATION.md`: ConSpec integration details
- `CONSPEC_MILA_SETUP_SUMMARY.md`: Quick ConSpec reference
- `DAG_GFLOWNET_INTEGRATION.md`: DAG-GFlowNet integration
- `DAG_GFLOWNET_SUMMARY.md`: DAG-GFlowNet quick reference

**Project Info:**
- `PROJECT_INFO.md`: Comprehensive project information
- `WANDB_QUICK_REFERENCE.md`: Wandb usage guide

## Branch Structure

### `main` - Main Development
- Core environments
- ConSpec implementation
- Training infrastructure
- Documentation

### `conspec` - ConSpec Focus
- Same as main
- Dedicated to ConSpec experiments
- Training runs on Mila

### `dag-gflownet` - Structure Learning
- Includes `methods/dag_gflownet/`
- DAG-GFlowNet integration
- Causal structure discovery

### `mila-cluster-backup` - Backup
- Original Mila cluster state
- Archived for reference

## Workflow Examples

### 1. Train ConSpec Locally
```bash
# Activate environment
source venv/bin/activate

# Run training with wandb
python metrics_experiments/train_conspec_wandb.py \
    --env single \
    --episodes 1000 \
    --wandb-project schema-learning
```

### 2. Deploy to Mila
```bash
# SSH to Mila
ssh mila

# Clone and setup
git clone https://github.com/mandanasmi/MemorySchema.git
cd MemorySchema
bash scripts/setup_mila.sh

# Submit job
sbatch scripts/run_conspec_mila.sh
```

### 3. Monitor Training
```bash
# View live logs
bash scripts/monitor_training.sh

# Check job status
squeue -u $USER

# View wandb dashboard
# https://wandb.ai/mandanasmi/schema-learning
```

### 4. Load and Evaluate Model
```python
from metrics_experiments.load_optimal_policy import load_checkpoint
from environments.key_door_goal_base import SingleKeyDoorGoalEnv

# Load model
policy, conspec = load_checkpoint("checkpoints/best_model.pth")

# Evaluate
env = SingleKeyDoorGoalEnv()
obs = env.reset()
# ... run episodes
```

## Development Guidelines

### Adding New Environments
1. Create class in `environments/key_door_goal_base.py`
2. Inherit from `BaseKeyDoorGoalEnv`
3. Add visualization support in `key_door_goal_visualizer.py`
4. Add tests in `test_key_door_goal.py`

### Adding New Methods
1. Create new file in `methods/`
2. Follow naming convention: `{method_name}_minigrid.py`
3. Add integration tests in `scripts/`
4. Document in `docs/`

### Running Experiments
1. Create config in `configs/` (optional)
2. Use existing training scripts or create new in `metrics_experiments/`
3. Log to wandb for tracking
4. Save checkpoints in `checkpoints/`
5. Generate visualizations in `figures/`

## File Naming Conventions

- **Environments**: `{env_name}_*.py`
- **Methods**: `{method_name}_minigrid.py`
- **Training**: `train_{method}_{variant}.py`
- **Scripts**: `{action}_{target}.sh` or `{action}_{target}.py`
- **Docs**: `{TOPIC}_{DESCRIPTION}.md` (uppercase)
- **Tests**: `test_{module}.py`

## Dependencies

All dependencies are specified in `requirements.txt`:
- **Core**: gymnasium, minigrid, numpy
- **ConSpec**: torch, scikit-learn
- **DAG-GFlowNet**: jax, dm-haiku, optax, pgmpy (on dag-gflownet branch)
- **Utilities**: matplotlib, wandb, pandas

## Git Workflow

```bash
# Work on features
git checkout -b feature-name
# ... make changes ...
git commit -m "Description"
git push origin feature-name

# Switch between branches
git checkout main          # Main development
git checkout conspec       # ConSpec experiments
git checkout dag-gflownet  # Structure learning
```

## Resources

- **Repository**: https://github.com/mandanasmi/MemorySchema
- **Wandb Dashboard**: https://wandb.ai/mandanasmi/schema-learning
- **Mila Docs**: https://docs.mila.quebec/

---

**Last Updated**: October 22, 2025

