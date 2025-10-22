# DAG-GFlowNet Integration Summary

## Successfully Integrated!

The DAG-GFlowNet repository has been successfully cloned and integrated into the `dag-gflownet` branch.

## What Was Done

### 1. Repository Setup
- **Source:** [tristandeleu/jax-dag-gflownet](https://github.com/tristandeleu/jax-dag-gflownet)
- **Cloned to:** `dag_gflownet_src/`
- **Branch:** `dag-gflownet`
- **Status:** Committed and pushed to GitHub

### 2. Files Added

#### Core DAG-GFlowNet Code
```
dag_gflownet_src/
├── dag_gflownet/              # Core implementation
│   ├── env.py                # Graph building environment
│   ├── gflownet.py           # GFlowNet algorithm
│   ├── nets/                 # Neural network architectures
│   │   ├── attention.py      # Attention mechanisms
│   │   ├── gflownet.py       # GFlowNet networks
│   │   └── transformers.py   # Transformer models
│   ├── scores/               # Scoring functions
│   │   ├── bde_score.py      # BDe score
│   │   ├── bge_score.py      # BGe score
│   │   └── priors.py         # Prior distributions
│   └── utils/                # Utilities
│       ├── graph.py          # Graph operations
│       ├── data.py           # Data handling
│       └── sampling.py       # Sampling utilities
├── train.py                  # Training script
└── requirements.txt          # Dependencies
```

#### Documentation
- `DAG_GFLOWNET_INTEGRATION.md` - Comprehensive integration guide
- `DAG_GFLOWNET_SUMMARY.md` - This summary
- Updated `README.md` with branch structure
- Updated `.gitignore` for DAG-GFlowNet files

### 3. Dependencies Added

```
# JAX ecosystem
jax>=0.4.0
dm-haiku>=0.0.9
optax>=0.1.0

# Causal structure learning
pgmpy>=0.1.19
networkx>=2.8.0

# Data processing
pandas>=1.5.0
scipy>=1.9.0
tqdm>=4.64.0
```

## What is DAG-GFlowNet?

**DAG-GFlowNet** learns to sample Directed Acyclic Graphs (DAGs) by building them sequentially, one edge at a time. It uses Generative Flow Networks (GFlowNets) to learn a policy that samples graphs proportional to a reward (typically the Bayesian posterior).

### Key Capabilities:
- **Bayesian Structure Learning:** Learn posterior distributions over graph structures
- **Diverse Sampling:** Generate multiple plausible graph structures
- **Sequential Construction:** Build graphs incrementally
- **Causal Discovery:** Identify causal relationships from data

## Integration with MemorySchema

### Goal
Use DAG-GFlowNet to learn the posterior distribution over causal graphs that could generate the observed behavior in our key-door-goal environments.

### Environment Graph Structures

1. **Environment 1 (Single):** Chain Graph
   ```
   Blue Key → Blue Door → Green Goal
   ```

2. **Environment 2 (Double):** Parallel Graph
   ```
   Blue Key → Blue Door ↘
                         → Green Goal
   Green Key → Green Door ↗
   ```

3. **Environment 3 (Triple):** Complex Dependencies
   ```
   Blue Key → Blue Door ↘
                        → Green Goal
   Green Key → Green Door ↗
   
   Purple Key → Purple Door → Purple Goal
   ```

### Next Steps

1. **Data Collection**
   - Collect trajectories from trained agents
   - Extract state sequences and action choices
   - Identify critical events (key pickups, door interactions)

2. **Data Conversion**
   - Convert environment states to graph variables
   - Create observational dataset for structure learning
   - Format for DAG-GFlowNet input

3. **Training**
   - Train DAG-GFlowNet on trajectory data
   - Learn posterior over causal structures
   - Sample and analyze discovered graphs

4. **Evaluation**
   - Compare learned graphs to true environment structures
   - Measure posterior accuracy
   - Analyze diversity of sampled graphs

## Running on Mila Cluster

### Installation
```bash
# SSH to Mila
ssh mila
cd ~/MemorySchema

# Switch to dag-gflownet branch
git checkout dag-gflownet

# Load modules
module load python/3.9 cuda/11.3

# Install JAX with CUDA
pip install --user jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install dependencies
pip install --user -r requirements.txt
```

### Test Installation
```bash
# Test DAG-GFlowNet basic functionality
cd dag_gflownet_src
python train.py --batch_size 256 erdos_renyi_lingauss --num_variables 5 --num_edges 5 --num_samples 100
```

## References

### Paper
Deleu, T., Góis, A., Emezue, C., Rankawat, M., Lacoste-Julien, S., Bauer, S., & Bengio, Y. (2022). 
**Bayesian Structure Learning with Generative Flow Networks.** 
*arXiv preprint arXiv:2202.13903.*
- Paper: https://arxiv.org/abs/2202.13903

### Implementation
- Repository: https://github.com/tristandeleu/jax-dag-gflownet
- License: MIT
- Framework: JAX + Haiku + Optax

### Related Work
- Bengio, Y., et al. (2021). Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation.

## Branch Information

- **Current Branch:** `dag-gflownet`
- **URL:** https://github.com/mandanasmi/MemorySchema/tree/dag-gflownet
- **Status:** ✅ Ready for development

### Other Branches
- `main` - Core environments and infrastructure
- `conspec` - ConSpec algorithm (currently running on Mila)
- `mila-cluster-backup` - Backup of original Mila state

## Files Structure

```
MemorySchema/
├── key_door_goal_base.py          # Environment implementations
├── key_door_goal_visualizer.py    # Visualization tools
├── key_door_goal_demo.py          # Demo script
├── dag_gflownet_src/              # DAG-GFlowNet implementation
├── DAG_GFLOWNET_INTEGRATION.md    # Integration guide
├── DAG_GFLOWNET_SUMMARY.md        # This file
└── requirements.txt               # All dependencies
```

## Success Metrics

The integration will be successful if DAG-GFlowNet can:

1. ✅ Recover the chain structure from Environment 1
2. ✅ Identify parallel structure in Environment 2
3. ✅ Discover complex dependencies in Environment 3
4. ✅ Generate diverse posterior samples
5. ✅ Provide uncertainty estimates over graph structures

---

**Status:** ✅ Integration Complete - Ready for trajectory collection and training!

