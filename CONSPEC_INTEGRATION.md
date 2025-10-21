# ConSpec Integration with Key-Door-Goal Environments

## Overview

This document describes the successful integration of ConSpec (Contrastive Retrospection) with our key-door-goal environments. ConSpec is a reinforcement learning algorithm that uses contrastive learning to improve exploration by learning from both successful and failed trajectories.

## What is ConSpec?

ConSpec is based on the paper "Contrastive Retrospection: Learning from Success and Failure" (https://arxiv.org/pdf/2210.05845.pdf). It works by:

1. **Learning Prototypes**: Creating prototype representations of successful and failed trajectories
2. **Contrastive Learning**: Using contrastive loss to distinguish between success and failure patterns
3. **Intrinsic Rewards**: Computing intrinsic rewards based on novelty and similarity to prototypes
4. **Memory System**: Maintaining separate memory buffers for successful and failed trajectories

## Implementation

### Files Created

1. **`conspec_minigrid.py`**: Main ConSpec implementation adapted for MiniGrid environments
2. **`train_conspec.py`**: Training script that integrates ConSpec with PPO
3. **`test_conspec.py`**: Test suite to verify ConSpec functionality
4. **`demo_conspec.py`**: Demo script showing ConSpec in action
5. **`conspec_requirements.txt`**: Dependencies for ConSpec training

### Key Components

#### ConSpecEncoder
- CNN-based encoder for processing MiniGrid observations
- LSTM for temporal processing
- Handles MiniGrid's dictionary observation format

#### PrototypeNetwork
- Learns prototype representations
- Projects features using MLP
- Computes similarities to prototypes

#### ConSpecMemory
- Stores success and failure trajectories
- Samples balanced batches for training
- Maintains separate buffers

#### ConSpec Main Class
- Integrates all components
- Computes intrinsic rewards
- Manages training process

## Integration with Our Environments

### Environment Compatibility

ConSpec works with all three of our key-door-goal environments:

1. **Single Environment (Chain Graph)**
   - Blue key → Blue door → Green goal
   - Sequential dependency structure

2. **Double Environment (Parallel Graph)**
   - Blue key → Blue door AND Green key → Green door
   - Parallel dependency structure with equal priority

3. **Triple Environment (Sequential Graph)**
   - Multi-stage sequential progression
   - Complex dependency structure

### Key Features

- **Observation Handling**: Properly processes MiniGrid's dictionary observations
- **Intrinsic Rewards**: Computes novelty-based intrinsic rewards
- **Memory Management**: Stores and learns from trajectory experiences
- **Graph Dependencies**: Works with different graph dependency structures

## Usage

### Basic Usage

```python
from conspec_minigrid import ConSpec
from key_door_goal_base import create_key_door_goal_env

# Create environment
env = create_key_door_goal_env('single', size=10)
obs_shape = (env.height, env.width, 3)
num_actions = env.action_space.n

# Create ConSpec
conspec = ConSpec(obs_shape, num_actions, device='cpu')

# Use in training loop
obs, _ = env.reset()
action = env.action_space.sample()
next_obs, reward, done, _, _ = env.step(action)

# Compute intrinsic reward
intrinsic_reward = conspec.compute_intrinsic_reward(
    [obs], [action], [reward], [done]
)

# Update ConSpec
conspec.train_step([obs], [action], [reward], [done])
```

### Running Tests

```bash
# Test ConSpec functionality
python test_conspec.py

# Run demo
python demo_conspec.py

# Train with ConSpec (requires more setup)
python train_conspec.py
```

## Results

The integration is successful and demonstrates:

- ✅ ConSpec successfully processes MiniGrid observations
- ✅ Intrinsic rewards are computed based on observation novelty
- ✅ Memory system stores success and failure trajectories
- ✅ Works on all three graph dependency environments
- ✅ Compatible with different environment sizes and complexities

## Next Steps

1. **Full Training**: Run complete training experiments with ConSpec
2. **Hyperparameter Tuning**: Optimize ConSpec parameters for our environments
3. **Comparison**: Compare ConSpec performance with baseline methods
4. **Analysis**: Analyze how ConSpec handles different graph dependency structures

## References

- Original ConSpec Paper: https://arxiv.org/pdf/2210.05845.pdf
- Original ConSpec Repository: https://github.com/sunchipsster1/ConSpec
- MiniGrid Documentation: https://minigrid.farama.org/

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- Gymnasium >= 1.1.1
- MiniGrid >= 3.0.0
