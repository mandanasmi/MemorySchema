# DAG-GFlowNet Integration

## Overview

This document describes the integration of DAG-GFlowNet (Deleu et al., 2022) into the MemorySchema project for learning posterior distributions over causal graphs.

## Repository Information

- **Source:** [tristandeleu/jax-dag-gflownet](https://github.com/tristandeleu/jax-dag-gflownet)
- **Paper:** [Bayesian Structure Learning with Generative Flow Networks (2022)](https://arxiv.org/abs/2202.13903)
- **Framework:** JAX
- **License:** MIT

## What is DAG-GFlowNet?

DAG-GFlowNet is a Bayesian structure learning algorithm based on Generative Flow Networks (GFlowNets). It learns to sample Directed Acyclic Graphs (DAGs) by sequentially adding edges, with the probability of sampling a particular DAG being proportional to a reward function (typically the posterior probability given data).

### Key Features:
1. **Sequential Graph Construction:** Builds graphs one edge at a time
2. **Bayesian Approach:** Learns posterior distribution over graph structures
3. **GFlowNet Framework:** Uses energy-based models for diverse sampling
4. **Gym Environment:** Implements graph building as a reinforcement learning environment

## Integration with MemorySchema Environments

### Goal
Use DAG-GFlowNet to learn the posterior distribution over causal graph structures that could have generated the trajectories observed in our key-door-goal environments.

### Environment Types
1. **Single (Chain Graph):** Blue key → Blue door → Green goal
2. **Double (Parallel Graph):** (Blue key → Blue door) ∥ (Green key → Green door) → Green goal
3. **Triple (Complex Graph):** Multiple dependencies with two goals

## Installation

### Requirements
The DAG-GFlowNet code has been cloned into `dag_gflownet_src/` and requires:

```bash
# Core JAX dependencies
jax
dm-haiku
optax

# Graph and structure learning
pgmpy
networkx

# Data processing
numpy
scipy
pandas
scikit-learn

# Environment
gym>=0.22.0

# Utilities
tqdm
```

### Setup on Mila Cluster

```bash
# Load modules
module load python/3.9 cuda/11.3

# Install JAX with CUDA support
pip install --user jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install DAG-GFlowNet requirements
cd ~/MemorySchema/dag_gflownet_src
pip install --user -r requirements.txt
```

## Planned Integration Steps

### 1. Data Collection from Environments
- Collect trajectories from trained agents in each environment
- Extract state-action-reward sequences
- Identify key events: key pickups, door openings, goal reaching

### 2. Convert Trajectories to Graph Data
- Map environment states to graph variables
- Create observational data from agent behavior
- Format data for DAG-GFlowNet input

### 3. Train DAG-GFlowNet
- Train on collected trajectory data
- Learn posterior over possible causal structures
- Compare learned graphs to true environment structures

### 4. Analysis
- Evaluate whether DAG-GFlowNet recovers:
  - Chain structure in Environment 1
  - Parallel structure in Environment 2
  - Complex dependencies in Environment 3
- Analyze posterior samples for diversity and accuracy

## Directory Structure

```
MemorySchema/
├── dag_gflownet_src/          # Cloned DAG-GFlowNet repository
│   ├── dag_gflownet/          # Core implementation
│   ├── data/                  # Data utilities
│   ├── train.py              # Training script
│   └── requirements.txt      # Dependencies
├── key_door_goal_base.py      # Environment implementations
├── train_dag_gflownet.py      # Integration training script (to be created)
└── collect_trajectories.py   # Trajectory collection (to be created)
```

## Next Steps

1. Create trajectory collection script
2. Implement data conversion from trajectories to graph format
3. Adapt DAG-GFlowNet training for our environments
4. Run experiments on Mila cluster
5. Analyze and visualize learned graph posteriors

## References

- Deleu, T., Góis, A., Emezue, C., Rankawat, M., Lacoste-Julien, S., Bauer, S., & Bengio, Y. (2022). Bayesian Structure Learning with Generative Flow Networks. arXiv preprint arXiv:2202.13903.
- Bengio, Y., et al. (2021). Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation.

## Citation

```bibtex
@article{deleu2022daggflownet,
    title={{Bayesian Structure Learning with Generative Flow Networks}},
    author={Deleu, Tristan and G{\'o}is, Ant{\'o}nio and Emezue, Chris and Rankawat, Mansi and Lacoste-Julien, Simon and Bauer, Stefan and Bengio, Yoshua},
    journal={arXiv preprint},
    year={2022}
}
```

