# MemorySchema Project Status

**Last Updated**: October 22, 2025

## Overview

MemorySchema is a comprehensive project for learning schemas using reinforcement learning and causal structure discovery with key-door-goal environments.

## Current Status: Production Ready

### ✅ Completed Components

#### 1. **Environments** (`environments/`)
- ✅ Three key-door-goal environments representing different graph structures
- ✅ Modular base class with inheritance
- ✅ Proper reward shaping (keys: +0.5, doors: +0.5, goals: +0.7-1.0)
- ✅ Comprehensive visualizations with dynamic legends
- ✅ Graph dependency representations (chain, parallel, hierarchical)

**Environments:**
1. **Single**: Chain graph (Key → Door → Goal)
2. **Double**: Parallel graph (Two independent paths to goal)
3. **Triple**: Hierarchical graph (Sequential with multiple goals)

#### 2. **ConSpec Implementation** (`methods/`)
- ✅ ConSpec algorithm adapted for MiniGrid
- ✅ CNN+LSTM encoder for spatial-temporal features
- ✅ Prototype-based memory system
- ✅ Intrinsic reward computation
- ✅ Critical feature visualization

#### 3. **Hyperparameter Search** (`metrics_experiments/`, `configs/`)
- ✅ Comprehensive search infrastructure
- ✅ 9 hyperparameters with proper search ranges
- ✅ Random/grid/bayesian search support
- ✅ Wandb integration for tracking
- ✅ Automatic result analysis and visualization
- ✅ Mila cluster batch scripts

**Search Parameters:**
- num_prototypes: [3, 5, 7, 10]
- learning_rate: log-uniform [0.0001, 0.01]
- intrinsic_reward_scale: uniform [0.1, 1.0]
- gamma: [0.95, 0.98, 0.99]
- epsilon_decay: uniform [0.99, 0.999]
- hidden_dim: [64, 128, 256]
- lstm_hidden_size: [64, 128, 256]
- batch_size: [32, 64, 128, 256]
- memory_size: [1000, 5000, 10000]

#### 4. **DAG-GFlowNet Integration** (`dag-gflownet` branch)
- ✅ Original JAX-based DAG-GFlowNet (`dag_gflownet_src/`)
- ✅ RL-enhanced DAG-GFlowNet (`rl_dag_gflownet/`)
- ✅ Trajectory data support
- ✅ Posterior analysis tools
- ✅ GNN and Transformer architectures

#### 5. **Documentation** (`docs/`)
- ✅ Comprehensive setup guides
- ✅ Hyperparameter search documentation
- ✅ DAG-GFlowNet integration guide
- ✅ Project structure documentation
- ✅ Deployment instructions for Mila

#### 6. **Experiment Infrastructure**
- ✅ Wandb integration (project: schema-learning, entity: mandanasmi)
- ✅ Mila cluster scripts with GPU support
- ✅ Automatic checkpointing
- ✅ Result analysis tools
- ✅ Visualization generation

## Branch Structure

### `main` Branch
- Core environments
- ConSpec implementation
- Hyperparameter search
- Documentation
- Base infrastructure

### `conspec` Branch
- Main branch + ConSpec-specific optimizations
- Training scripts for Mila
- Performance optimizations
- Experiment configurations

### `dag-gflownet` Branch
- Main branch content
- DAG-GFlowNet implementations (original + RL-enhanced)
- Causal structure learning
- Posterior analysis tools
- Integration with ConSpec trajectories

## Repository Structure

```
MemorySchema/
├── environments/              # Key-door-goal environments
│   ├── key_door_goal_base.py
│   ├── key_door_goal_visualizer.py
│   └── key_door_goal_demo.py
│
├── methods/                   # Algorithm implementations
│   └── conspec_minigrid.py   # ConSpec for MiniGrid
│
├── metrics_experiments/       # Training and experiments
│   ├── hyperparameter_search_conspec.py
│   ├── analyze_hyperparam_results.py
│   └── train_conspec_all_environments.py
│
├── scripts/                   # Utility scripts
│   ├── run_hyperparam_search_mila.sh
│   ├── quick_hyperparam_test.py
│   └── test_rewards.py
│
├── configs/                   # Configuration files
│   └── conspec_hyperparam_search.yaml
│
├── docs/                      # Documentation
│   ├── HYPERPARAMETER_SEARCH.md
│   ├── RL_DAG_GFLOWNET_INTEGRATION.md
│   ├── PROJECT_STRUCTURE.md
│   └── LAUNCH_ON_MILA.md
│
├── figures/                   # Visualizations
│   ├── equal_priority_graph_environments.png
│   └── conspec_critical_features_*.png
│
├── dag_gflownet_src/         # Original DAG-GFlowNet (dag-gflownet branch)
└── rl_dag_gflownet/          # RL-enhanced DAG-GFlowNet (dag-gflownet branch)
```

## Key Features

### 1. **Modular Environment System**
```python
from environments.key_door_goal_base import create_key_door_goal_env
env = create_key_door_goal_env('single')  # or 'double', 'triple'
```

### 2. **Reward Shaping**
- Key pickup: +0.5
- Door opening: +0.5
- Intermediate goal: +0.7
- Final goal: +1.0
- No step penalties

### 3. **Comprehensive Visualization**
- Dynamic legends based on environment objects
- Color-coded keys, doors, and goals
- Clean, publication-ready plots
- Individual and combined visualizations

### 4. **Hyperparameter Optimization**
- 50 trials per environment (150 total)
- ~40-50 hours on Mila cluster
- Automatic best configuration identification
- Comprehensive analysis and reporting

### 5. **Causal Structure Learning**
- Two DAG-GFlowNet implementations
- Learn posterior over graphs from trajectories
- Compare with ground truth structures
- Visualization of learned posteriors

## Running Experiments

### Local Testing

```bash
# Test environments
python environments/key_door_goal_demo.py

# Test reward structure
python scripts/test_rewards.py

# Quick hyperparameter search test
python scripts/quick_hyperparam_test.py
```

### Mila Cluster

```bash
# SSH to Mila
ssh mila
cd ~/MemorySchema

# Pull latest changes
git checkout conspec  # or dag-gflownet
git pull origin conspec

# Submit hyperparameter search
sbatch scripts/run_hyperparam_search_mila.sh

# Monitor
squeue -u $USER
tail -f logs/hypersearch_*.out
```

### Analysis

```bash
# Analyze hyperparameter search results
python metrics_experiments/analyze_hyperparam_results.py

# View visualizations
open figures/reward_distributions.png
open figures/hyperparameter_impact.png
```

## Wandb Dashboard

- **Project**: schema-learning
- **Entity**: mandanasmi
- **URL**: https://wandb.ai/mandanasmi/schema-learning

**Tracked Metrics:**
- Episode rewards
- Success rates
- Intrinsic rewards
- Hyperparameter configurations
- Critical features

## Next Steps

### Immediate (Ready to Run)

1. **Run Hyperparameter Search on Mila**
   ```bash
   sbatch scripts/run_hyperparam_search_mila.sh
   ```

2. **Analyze Results**
   - Check wandb dashboard
   - Run analysis script
   - Identify best configurations

3. **Train with Best Hyperparameters**
   - Use best configs for final training
   - Generate publication-quality results

### Short-term (Implementation Needed)

1. **Trajectory Collection from ConSpec**
   - Create script to save trajectories during training
   - Format: episode, step, state variables, action, reward

2. **Trajectory to DAG Format Conversion**
   - Extract relevant state variables
   - Create trajectory matrix
   - Save as CSV for DAG-GFlowNet

3. **DAG-GFlowNet Training**
   - Train on single environment first
   - Validate against ground truth
   - Scale to all environments

### Long-term (Research Goals)

1. **Posterior Analysis**
   - Compare learned graphs with ground truth
   - Analyze edge probabilities
   - Identify causal dependencies

2. **Schema Discovery**
   - Use learned structures for transfer learning
   - Test generalization to new environments
   - Analyze prototype representations

3. **Publication**
   - Comprehensive experiments
   - Statistical analysis
   - Visualization and results

## Dependencies

### Core
- Python 3.9
- gymnasium 1.1.1
- minigrid 3.0.0
- numpy 2.0.2

### ConSpec
- torch >= 2.0.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0

### DAG-GFlowNet
- jax >= 0.4.0
- jraph >= 0.0.6
- dm-haiku >= 0.0.9
- optax >= 0.1.0
- pgmpy >= 0.1.19
- networkx >= 2.8.0

### Tracking
- wandb >= 0.15.0

## Known Issues

### ✅ Fixed
- ~~Reward structure not working properly~~ → Fixed with reward shaping
- ~~ConSpec observation format mismatch~~ → Fixed dict handling
- ~~Emoji encoding issues on Mila~~ → Removed all emojis
- ~~Graph dependencies not clear~~ → Redesigned environments

### Current
- None critical

## Performance Expectations

### Environment Complexity
- **Single**: Simplest, ~1000 episodes to solve
- **Double**: Medium, ~2000 episodes to solve
- **Triple**: Complex, ~3000 episodes to solve

### Training Time
- **Single**: ~20 min/trial on GPU
- **Double**: ~25 min/trial on GPU
- **Triple**: ~30 min/trial on GPU

### Success Metrics
- **Single**: >70% success rate is good
- **Double**: >60% success rate is good
- **Triple**: >50% success rate is good

## Contact & Resources

- **GitHub**: mandanasmi
- **Wandb**: mandanasmi
- **Email**: samieima@mila.quebec
- **Cluster**: Mila (tamia.alliancecan.ca)

## Citation

If using this codebase:

```
@misc{memoryschema2025,
  title={MemorySchema: Learning Schemas with Reinforcement Learning and Causal Discovery},
  author={Samiei, Mandana},
  year={2025},
  publisher={GitHub},
  url={https://github.com/mandanasmi/MemorySchema}
}
```

---

**Status**: All core components implemented and tested. Ready for large-scale experiments on Mila cluster.

