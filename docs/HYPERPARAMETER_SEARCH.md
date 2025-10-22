# ConSpec Hyperparameter Search

## Overview

This document describes the hyperparameter search infrastructure for ConSpec across all three key-door-goal environments.

## Search Space

### Hyperparameters

| Parameter | Type | Range/Values | Description |
|-----------|------|--------------|-------------|
| `num_prototypes` | Choice | [3, 5, 7, 10] | Number of prototypes in ConSpec memory |
| `learning_rate` | Log-uniform | [0.0001, 0.01] | Learning rate for policy optimization |
| `intrinsic_reward_scale` | Uniform | [0.1, 1.0] | Scaling factor for intrinsic rewards |
| `gamma` | Choice | [0.95, 0.98, 0.99] | Discount factor |
| `epsilon_decay` | Uniform | [0.99, 0.999] | Epsilon decay rate for exploration |
| `hidden_dim` | Choice | [64, 128, 256] | Hidden layer dimension |
| `lstm_hidden_size` | Choice | [64, 128, 256] | LSTM hidden state size |
| `batch_size` | Choice | [32, 64, 128, 256] | Training batch size |
| `memory_size` | Choice | [1000, 5000, 10000] | ConSpec memory buffer size |

### Training Configuration

- **Episodes**: 5000 per trial
- **Max Steps**: 400 per episode
- **Evaluation Frequency**: Every 500 episodes
- **Checkpoint Frequency**: Every 1000 episodes

### Search Configuration

- **Method**: Random search
- **Trials**: 50 per environment (150 total)
- **Optimization Metric**: Mean reward (last 100 episodes)
- **Optimization Direction**: Maximize

## Usage

### Local Testing

Quick test with reduced trials:

```bash
# Test on single environment with 3 trials
python scripts/quick_hyperparam_test.py
```

### Full Search (Local)

```bash
# Search all environments
python metrics_experiments/hyperparameter_search_conspec.py \
    --config configs/conspec_hyperparam_search.yaml \
    --env all

# Search specific environment
python metrics_experiments/hyperparameter_search_conspec.py \
    --env single \
    --n-trials 50 \
    --episodes 5000
```

### Mila Cluster

#### Setup

```bash
# SSH to Mila
ssh mila
cd ~/MemorySchema

# Pull latest changes
git pull origin conspec

# Ensure dependencies are installed
module load python/3.9
python3.9 -m pip install --user pyyaml
```

#### Submit Job

```bash
# Submit hyperparameter search job
sbatch scripts/run_hyperparam_search_mila.sh

# Monitor job
squeue -u $USER
tail -f logs/hypersearch_<job_id>.out
```

#### Job Configuration

The Slurm script (`run_hyperparam_search_mila.sh`) is configured for:
- **Time**: 48 hours
- **Memory**: 32GB
- **GPU**: 1 GPU
- **CPUs**: 4

## Results Analysis

### Analyze Results

```bash
# Run full analysis
python metrics_experiments/analyze_hyperparam_results.py \
    --results-dir hyperparam_search_results \
    --output hyperparam_search_report.md
```

### Output Files

#### Results Directory

```
hyperparam_search_results/
├── search_single_<timestamp>.json     # Single env results
├── search_double_<timestamp>.json     # Double env results
├── search_triple_<timestamp>.json     # Triple env results
└── search_summary_<timestamp>.json    # Complete summary
```

#### Result Format

```json
{
  "environment": "single",
  "trial": 0,
  "hyperparameters": {
    "num_prototypes": 5,
    "learning_rate": 0.0023,
    "intrinsic_reward_scale": 0.45,
    ...
  },
  "results": {
    "mean_reward": 1.85,
    "success_rate": 0.73,
    "max_reward": 2.0
  },
  "timestamp": "2025-10-22T..."
}
```

### Visualizations

The analysis generates:

1. **Reward Distributions** (`figures/reward_distributions.png`)
   - Histograms of mean rewards per environment
   - Shows distribution and variance

2. **Hyperparameter Impact** (`figures/hyperparameter_impact.png`)
   - Scatter plots showing hyperparameter vs. reward
   - Identifies important parameters

3. **Best Configs Comparison** (`figures/best_configs_comparison.png`)
   - Bar chart comparing best configurations
   - Mean reward and success rate

### Report

Markdown report includes:
- Best configurations per environment
- Hyperparameter correlations
- Performance metrics
- Visualization references

## Configuration File

### Structure (`configs/conspec_hyperparam_search.yaml`)

```yaml
environments:
  - single
  - double
  - triple

hyperparameters:
  num_prototypes:
    type: choice
    values: [3, 5, 7, 10]
  
  learning_rate:
    type: loguniform
    min: 0.0001
    max: 0.01
  # ... more parameters

training:
  episodes: 5000
  max_steps_per_episode: 400
  
search:
  method: random
  n_trials: 50
  optimization_metric: mean_reward
  
logging:
  wandb_project: schema-learning
  wandb_entity: mandanasmi
```

### Customization

Modify the config file to:
- Add/remove hyperparameters
- Change search ranges
- Adjust training duration
- Change search method (random/grid/bayesian)

## Wandb Integration

### Tracking

Each trial is logged to wandb with:
- Hyperparameters as config
- Episode rewards
- Success rates
- Intrinsic rewards
- Final metrics

### Dashboard

View results at:
- **Project**: schema-learning
- **Entity**: mandanasmi
- **URL**: https://wandb.ai/mandanasmi/schema-learning

### Tags

- Environment name (`single`, `double`, `triple`)
- `hyperparam_search`

## Best Practices

### 1. Start Small

```bash
# Quick test first
python scripts/quick_hyperparam_test.py
```

### 2. Monitor Progress

```bash
# Check logs
tail -f logs/hypersearch_*.out

# Check wandb dashboard
# https://wandb.ai/mandanasmi/schema-learning
```

### 3. Analyze Intermediate Results

```bash
# Analyze partial results while job runs
python metrics_experiments/analyze_hyperparam_results.py
```

### 4. Use Checkpoints

Results are saved after each trial, so you can:
- Resume interrupted searches
- Analyze partial results
- Combine multiple runs

## Expected Runtime

### Per Trial

- **Single**: ~20 minutes
- **Double**: ~25 minutes
- **Triple**: ~30 minutes

### Full Search (50 trials × 3 envs)

- **Total**: ~40-50 hours
- **Mila**: Runs in parallel on GPU

## Troubleshooting

### Out of Memory

Reduce:
- `batch_size`
- `hidden_dim`
- `lstm_hidden_size`
- `memory_size`

### Slow Convergence

Increase:
- `learning_rate`
- `intrinsic_reward_scale`
Decrease:
- `epsilon_decay`

### Poor Results

Check:
- Reward structure (see `scripts/test_rewards.py`)
- Environment behavior
- Training duration

## Advanced Usage

### Custom Search Space

Edit `configs/conspec_hyperparam_search.yaml`:

```yaml
hyperparameters:
  my_new_param:
    type: uniform
    min: 0.0
    max: 1.0
```

### Grid Search

```yaml
search:
  method: grid  # Instead of random
```

### Bayesian Optimization

```yaml
search:
  method: bayesian
  n_trials: 100
```

(Requires additional dependencies: `optuna` or `scikit-optimize`)

## File Organization

```
MemorySchema/
├── configs/
│   └── conspec_hyperparam_search.yaml    # Search configuration
├── metrics_experiments/
│   ├── hyperparameter_search_conspec.py  # Main search script
│   └── analyze_hyperparam_results.py     # Analysis script
├── scripts/
│   ├── quick_hyperparam_test.py          # Quick test
│   └── run_hyperparam_search_mila.sh     # Slurm script
├── hyperparam_search_results/            # Results (gitignored)
└── docs/
    └── HYPERPARAMETER_SEARCH.md          # This file
```

## References

- ConSpec: Contrastive Retrospection for intrinsic motivation
- Hyperparameter optimization: Random search vs grid search
- Wandb: Experiment tracking and visualization

---

**Last Updated**: October 22, 2025

