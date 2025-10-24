# Complete Pipeline: ConSpec → Expert Trajectories → DAG-GFlowNet

This pipeline implements the complete workflow for learning causal structure from reinforcement learning trajectories.

## Overview

The pipeline consists of three main steps:

1. **ConSpec Training & Analysis** - Train ConSpec to identify critical features/prototypes
2. **Expert Trajectory Collection** - Collect successful trajectories with binary state features
3. **Causal Structure Learning** - Use DAG-GFlowNet to learn posterior over causal graphs

## Pipeline Steps

### Step 1: ConSpec Training (`step1_conspec_analysis.py`)

**Purpose**: Train ConSpec to learn critical features and prototypes from the environment.

**Inputs**:
- Environment type (`single`, `double`, `triple`)
- Training parameters (episodes, prototypes, etc.)

**Outputs**:
- Trained ConSpec model
- Prototype features
- Training metrics and visualizations
- Episode trajectories

**Key Features**:
- Records intrinsic rewards, cosine similarities
- Identifies critical states/features
- Saves best-performing policy

### Step 2: Expert Trajectory Collection (`step2_expert_trajectories.py`)

**Purpose**: Collect expert trajectories from the best policy, recording binary state features.

**Inputs**:
- Trained policy from Step 1
- Environment configuration

**Outputs**:
- CSV with binary trajectory data
- Feature timeline visualizations
- Temporal analysis

**Binary Features Tracked**:
- `has_blue_key`, `has_green_key`, `has_purple_key`
- `blue_door_open`, `green_door_open`, `purple_door_open`
- `at_green_goal`, `at_purple_goal`
- `task_complete`

### Step 3: Causal Structure Learning (`step3_learn_causal_structure.py`)

**Purpose**: Learn posterior distribution over causal graphs using DAG-GFlowNet.

**Inputs**:
- Binary trajectory data from Step 2
- Prior specifications

**Outputs**:
- Posterior samples from uniform and temporal priors
- Graph visualizations
- Comparison analysis

**Priors Compared**:
- **Uniform Prior**: No temporal bias
- **Temporal Prior**: Favors edges from earlier to later features

## Usage

### Run Complete Pipeline

```bash
# Run full pipeline for single environment
python pipeline/run_full_pipeline.py --env single

# Run with custom parameters
python pipeline/run_full_pipeline.py \
    --env double \
    --step1_episodes 3000 \
    --step1_prototypes 7 \
    --step2_trajectories 500 \
    --step3_iterations 30000 \
    --output_dir my_results
```

### Run Individual Steps

```bash
# Step 1: ConSpec training
python pipeline/step1_conspec_analysis.py \
    --env single \
    --episodes 5000 \
    --num_prototypes 5 \
    --output_dir results/step1

# Step 2: Expert trajectory collection
python pipeline/step2_expert_trajectories.py \
    --env single \
    --step1_results results/step1 \
    --num_trajectories 1000 \
    --output_dir results/step2

# Step 3: Causal structure learning
python pipeline/step3_learn_causal_structure.py \
    --env single \
    --expert_data results/step2/single_expert_data_*.csv \
    --num_iterations 50000 \
    --output_dir results/step3
```

### Skip Steps (Resume Pipeline)

```bash
# Skip Step 1, use existing results
python pipeline/run_full_pipeline.py \
    --env single \
    --skip_step1 \
    --step1_dir existing_results/step1

# Skip Steps 1 and 2, use existing expert data
python pipeline/run_full_pipeline.py \
    --env single \
    --skip_step1 \
    --skip_step2 \
    --expert_data existing_data.csv
```

## Expected Results

### Environment 1 (Single - Chain Graph)
**True Structure**: Key → Door → Goal

**Expected Posteriors**:
- Strong edge: Key → Door (>0.8 probability)
- Strong edge: Door → Goal (>0.8 probability)
- Weak edge: Key → Goal (confounded)

### Environment 2 (Double - Parallel Graph)
**True Structure**:
```
Blue Key → Blue Door ↘
                      → Goal
Green Key → Green Door ↗
```

**Expected Posteriors**:
- Two independent paths identified
- No spurious dependencies between blue/green
- Common effect at goal

### Environment 3 (Triple - Hierarchical Graph)
**True Structure**:
```
Green Key → Green Door → Green Goal
                      ↓
                  Purple Key → Purple Door → Purple Goal
```

**Expected Posteriors**:
- Sequential dependencies captured
- Hierarchical structure learned
- Multiple reward paths distinguished

## Output Structure

```
pipeline_results/
├── single_20241022_143022/          # Timestamped run
│   ├── step1/                       # ConSpec results
│   │   ├── single_metrics_*.pkl
│   │   ├── single_conspec_*.pt
│   │   ├── single_summary_*.json
│   │   └── single_conspec_metrics.png
│   ├── step2/                       # Expert trajectories
│   │   ├── single_expert_data_*.csv
│   │   ├── single_analysis_*.pkl
│   │   └── single_feature_timeline.png
│   ├── step3/                       # Causal structure
│   │   ├── single_uniform_posterior_*.npy
│   │   ├── single_temporal_posterior_*.npy
│   │   ├── single_analysis_*.pkl
│   │   ├── single_summary_*.json
│   │   └── single_prior_comparison.png
│   └── pipeline_summary.json        # Overall summary
```

## Dependencies

### Core Dependencies
- `torch` - ConSpec implementation
- `jax` - DAG-GFlowNet implementation
- `pandas` - Data handling
- `matplotlib` - Visualizations
- `networkx` - Graph operations

### Environment Dependencies
- `gymnasium` - RL environments
- `minigrid` - Grid world environments

### DAG-GFlowNet Dependencies
- `jraph` - Graph neural networks
- `optax` - Optimization
- `pgmpy` - Probabilistic graphical models

## Configuration

### Step 1 Parameters
- `num_episodes`: Training episodes (default: 5000)
- `num_prototypes`: Number of prototypes (default: 5)
- `hidden_dim`: Encoder hidden dimension (default: 128)
- `lstm_hidden_size`: LSTM hidden size (default: 128)
- `intrinsic_reward_scale`: Intrinsic reward scaling (default: 0.5)

### Step 2 Parameters
- `num_trajectories`: Trajectories to collect (default: 1000)
- `min_successful`: Minimum successful trajectories (default: 100)
- `epsilon`: Exploration rate (default: 0.0 for greedy)

### Step 3 Parameters
- `num_iterations`: DAG-GFlowNet training iterations (default: 50000)
- `temporal_strength`: Strength of temporal prior (default: 1.0)
- `batch_size`: Training batch size (default: 32)
- `lr`: Learning rate (default: 1e-5)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch size or number of iterations
3. **No Successful Trajectories**: Increase training episodes in Step 1
4. **JAX Issues**: Install JAX with CUDA support if using GPU

### Performance Tips

1. **GPU Usage**: DAG-GFlowNet benefits from GPU acceleration
2. **Parallel Environments**: Increase `num_envs` for faster training
3. **Checkpointing**: Results are saved incrementally
4. **Visualization**: Check generated plots for debugging

## Examples

### Quick Test Run
```bash
# Small test run
python pipeline/run_full_pipeline.py \
    --env single \
    --step1_episodes 1000 \
    --step2_trajectories 100 \
    --step3_iterations 10000
```

### Production Run
```bash
# Full production run
python pipeline/run_full_pipeline.py \
    --env triple \
    --step1_episodes 10000 \
    --step1_prototypes 7 \
    --step2_trajectories 2000 \
    --step3_iterations 100000 \
    --step3_temporal_strength 2.0
```

### Batch Run All Environments
```bash
# Run all three environments
for env in single double triple; do
    python pipeline/run_full_pipeline.py \
        --env $env \
        --output_dir batch_results \
        --seed 42
done
```

## Citation

If you use this pipeline in your research:

```bibtex
@misc{conspec_daggflownet_pipeline2025,
  title={Learning Causal Structure from RL Trajectories: ConSpec to DAG-GFlowNet Pipeline},
  author={Samiei, Mandana},
  year={2025},
  publisher={GitHub},
  url={https://github.com/mandanasmi/MemorySchema}
}
```

---

**Last Updated**: October 22, 2025
