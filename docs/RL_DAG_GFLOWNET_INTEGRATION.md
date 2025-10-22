# RL-DAG-GFlowNet Integration

## Overview

This document describes the integration of the RL-enhanced DAG-GFlowNet implementation into the MemorySchema project. This is your fork with RL-specific modifications for learning causal structure from trajectory data.

## Repository Information

- **Source**: [mandanasmi/rl-dag-gflownet](https://github.com/mandanasmi/rl-dag-gflownet)
- **Location**: `rl_dag_gflownet/`
- **Base**: JAX + Haiku implementation of DAG-GFlowNet
- **Branch**: `dag-gflownet`

## Key Features

### 1. **RL-Enhanced Structure Learning**
- Modified DAG-GFlowNet for learning from RL trajectories
- Integration with ConSpec-generated data
- Posterior estimation over causal graphs

### 2. **Implementation Components**

```
rl_dag_gflownet/
├── dag_gflownet/           # Core implementation
│   ├── env.py             # GFlowNet environment
│   ├── gflownet.py        # Main GFlowNet algorithm
│   ├── nets/              # Neural network architectures
│   │   ├── gnn/          # Graph Neural Network version
│   │   └── transformer/  # Transformer-based version
│   ├── scores/           # Scoring functions (BDe, BGe)
│   ├── posterior_analysis/  # Analysis tools
│   └── utils/            # Utilities and helpers
├── experiments/          # Experiment scripts
├── train.py             # Main training script
└── traj_matrix.csv      # Example trajectory data
```

### 3. **Neural Network Architectures**

#### Graph Neural Networks (GNN)
- Uses Jraph for graph operations
- Message passing over DAG structures
- Node and edge embeddings

#### Transformer
- Attention-based architecture
- Better for capturing long-range dependencies
- Self-attention over graph structures

### 4. **Scoring Functions**

- **BDe (Bayesian Dirichlet equivalent)**: For discrete data
- **BGe (Bayesian Gaussian equivalent)**: For continuous data
- Custom scoring for trajectory data

## Integration with MemorySchema

### Goal

Learn the posterior distribution over causal graph structures from:
1. Agent trajectories in key-door-goal environments
2. ConSpec-identified critical features
3. State-action-reward sequences

### Workflow

```
ConSpec Training → Trajectory Data → RL-DAG-GFlowNet → Posterior Graphs
                                                      ↓
                    Compare with Ground Truth ← Environment Structure
```

## Setup

### Dependencies

```bash
# Core JAX dependencies
jax>=0.4.0
jraph>=0.0.6
dm-haiku>=0.0.9
optax>=0.1.0

# Graph and structure learning
pgmpy>=0.1.19
networkx>=2.8.0

# Data and utilities
numpy
scipy
pandas
scikit-learn
wandb
tqdm
```

### Installation on Mila

```bash
# SSH to Mila
ssh mila
cd ~/MemorySchema
git checkout dag-gflownet

# Load modules
module load python/3.9 cuda/11.3

# Install JAX with CUDA
pip install --user jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install jraph (graph neural network library)
pip install --user jraph

# Install other dependencies
cd rl_dag_gflownet
pip install --user -r requirements.txt
```

## Key Differences from Original

Your `rl-dag-gflownet` fork includes:

1. **RL Trajectory Support**
   - Modified data loading for trajectory matrices
   - `traj_matrix.csv` format for sequential data

2. **Posterior Analysis Tools**
   - `posterior_analysis/` directory
   - Visualization of learned posteriors
   - Edge presence analysis
   - Convergence monitoring

3. **Wandb Integration**
   - Enhanced logging utilities
   - Experiment tracking
   - Visualization support

4. **Experiment Scripts**
   - `experiments/run_conspec.sh` - ConSpec integration
   - `experiments/run_blicket.sh` - Blicket game experiments
   - Multiple configuration options

## Usage

### Training on Trajectory Data

```bash
cd rl_dag_gflownet

# Train on trajectory data
python train.py \
    --batch_size 256 \
    --num_variables 5 \
    --num_samples 100 \
    --trajectory_file traj_matrix.csv
```

### Integration with ConSpec

```python
# Pseudo-code for integration
from environments.key_door_goal_base import create_key_door_goal_env
from methods.conspec_minigrid import ConSpec
from rl_dag_gflownet.dag_gflownet import DAGGFlowNet

# 1. Collect ConSpec trajectories
env = create_key_door_goal_env('single')
conspec = ConSpec(...)

# Collect episodes
trajectories = collect_conspec_episodes(env, conspec, n_episodes=1000)

# 2. Convert to DAG-GFlowNet format
traj_matrix = convert_to_dag_format(trajectories)

# 3. Train DAG-GFlowNet
dag_gflownet = DAGGFlowNet(...)
posterior = train_dag_gflownet(traj_matrix)

# 4. Analyze learned structure
compare_with_ground_truth(posterior, true_graph)
```

### Analyzing Results

```bash
cd rl_dag_gflownet/dag_gflownet/posterior_analysis

# Visualize learned graphs
python vis_graph.py --posterior_file posterior_files/kd3-10k-posterior-1000.npy

# Analyze convergence
python convergence.py --log_dir ../../../logs
```

## Experiment Scripts

### Available Scripts

1. **`run_conspec.sh`** - ConSpec integration experiments
2. **`run_blicket.sh`** - Blicket game (causal learning) experiments
3. **`run.sh`** - General training
4. **`run_job.sh`** - Cluster job submission

### Example Usage

```bash
cd rl_dag_gflownet/experiments

# Run ConSpec experiment
bash run_conspec.sh

# Or submit as cluster job
sbatch run_job.sh
```

## Data Format

### Trajectory Matrix Format

The `traj_matrix.csv` file should contain:

```csv
episode,step,var1,var2,var3,var4,var5,action,reward
0,0,0,0,0,0,0,1,0.0
0,1,1,0,0,0,0,2,0.5
...
```

Where:
- `episode`: Episode number
- `step`: Step within episode
- `var1-varN`: State variables (keys, doors, positions)
- `action`: Action taken
- `reward`: Reward received

### Converting ConSpec Data

To convert ConSpec trajectories to this format:

```python
import pandas as pd
import numpy as np

def convert_conspec_to_dag_format(episodes):
    """Convert ConSpec episodes to DAG-GFlowNet format"""
    rows = []
    
    for ep_idx, episode in enumerate(episodes):
        for step_idx, (state, action, reward) in enumerate(episode):
            # Extract relevant state variables
            has_key = state['carrying'] is not None
            door_open = check_door_status(state)
            at_goal = check_goal_status(state)
            
            row = {
                'episode': ep_idx,
                'step': step_idx,
                'has_key': int(has_key),
                'door_open': int(door_open),
                'at_goal': int(at_goal),
                'action': action,
                'reward': reward
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv('traj_matrix.csv', index=False)
    return df
```

## Posterior Analysis

### Tools Available

1. **Edge Presence Analysis**
   - `avg_presence_edges_kd3_10k.png` - Average edge probabilities
   - Shows which edges are most likely in the posterior

2. **Graph Visualization**
   - `graph_avg_presence.png` - Visual representation of posterior
   - Node sizes/colors indicate confidence

3. **Convergence Monitoring**
   - Track how posterior estimate changes over training
   - Identify when learning has converged

### Loading Posteriors

```python
import numpy as np

# Load posterior samples
posterior = np.load('posterior_files/kd3-10k-posterior-1000.npy')

# Analyze edge probabilities
edge_probs = posterior.mean(axis=0)  # Average over samples
print(f"Edge probabilities:\n{edge_probs}")

# Threshold to get most likely graph
threshold = 0.5
predicted_graph = (edge_probs > threshold).astype(int)
```

## Integration Steps

### 1. Data Collection

```bash
# Train ConSpec and collect trajectories
python metrics_experiments/train_conspec_all_environments.py \
    --save-trajectories \
    --output-dir trajectories/
```

### 2. Data Conversion

```bash
# Convert trajectories to DAG format
python scripts/convert_trajectories_to_dag.py \
    --input trajectories/ \
    --output rl_dag_gflownet/traj_matrix.csv
```

### 3. Train DAG-GFlowNet

```bash
cd rl_dag_gflownet

# Train on converted data
python train.py \
    --data traj_matrix.csv \
    --num_variables 5 \
    --batch_size 256 \
    --num_iterations 10000
```

### 4. Analyze Results

```bash
# Analyze learned structure
python dag_gflownet/posterior_analysis/vis_graph.py

# Compare with ground truth
python scripts/compare_learned_graph.py \
    --posterior rl_dag_gflownet/posterior.npy \
    --ground_truth environments/graph_structures.json
```

## Expected Outcomes

### Single Environment (Chain Graph)

**True Graph**: Key → Door → Goal

**Expected Posterior**:
- Strong edge: Key → Door (>0.8 probability)
- Strong edge: Door → Goal (>0.8 probability)
- Weak/no edge: Key → Goal (confounded)

### Double Environment (Parallel Graph)

**True Graph**: 
```
Blue Key → Blue Door ↘
                      → Goal
Green Key → Green Door ↗
```

**Expected Posterior**:
- Parallel paths identified
- Independence between blue/green paths
- Common effect at goal

### Triple Environment (Complex Graph)

**True Graph**:
```
Green Key → Green Door → Green Goal
                      ↓
                  Purple Key → Purple Door → Purple Goal
```

**Expected Posterior**:
- Sequential dependencies
- Hierarchical structure
- Multiple reward paths

## Troubleshooting

### JAX/Jraph Issues

```bash
# Reinstall JAX with CUDA
pip uninstall jax jaxlib
pip install --user jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --user jraph
```

### Memory Issues

- Reduce `batch_size`
- Reduce `num_envs`
- Use smaller networks

### Convergence Issues

- Increase training iterations
- Adjust learning rate
- Check data quality

## Next Steps

1. **Create trajectory collection script**
2. **Implement conversion from ConSpec to DAG format**
3. **Train on single environment first**
4. **Validate against ground truth**
5. **Scale to all three environments**
6. **Analyze posterior distributions**

## References

- **Paper**: Deleu et al., 2022 - Bayesian Structure Learning with Generative Flow Networks
- **Original Repo**: https://github.com/tristandeleu/jax-dag-gflownet
- **Your Fork**: https://github.com/mandanasmi/rl-dag-gflownet

---

**Last Updated**: October 22, 2025

