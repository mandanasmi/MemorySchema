# Key-Door-Goal RL Environments

This module provides a comprehensive collection of reinforcement learning environments based on the key-door-goal concept. These environments are built using the MiniGrid framework and offer various complexities and challenges for RL agents.

## 🎯 Overview

The key-door-goal concept is a fundamental puzzle in reinforcement learning where an agent must:
1. **Find and collect keys** of specific colors
2. **Open doors** using the corresponding colored keys
3. **Navigate to the goal** to complete the task

## 🏗️ Environment Types

### 1. Basic Key-Door-Goal Environment
- **File**: `key_door_goal_envs.py` → `BasicKeyDoorGoalEnv`
- **Complexity**: Simple
- **Layout**: Single key, single door, single goal
- **Grid Size**: 8x8 (configurable)
- **Mission**: "Find the key, open the door, and reach the goal"

### 2. Multi-Key-Door Environment
- **File**: `key_door_goal_envs.py` → `MultiKeyDoorEnv`
- **Complexity**: Medium
- **Layout**: Multiple color-coded key-door pairs
- **Grid Size**: 10x10 (configurable)
- **Mission**: "Collect N keys to open N doors and reach the goal"

### 3. Sequential Key-Door Environment
- **File**: `key_door_goal_envs.py` → `SequentialKeyDoorEnv`
- **Complexity**: Medium-High
- **Layout**: Keys and doors must be collected/opened in sequence
- **Grid Size**: 12x12 (configurable)
- **Mission**: "Open N doors in sequence to reach the goal"

### 4. Maze Key-Door Environment
- **File**: `key_door_goal_envs.py` → `MazeKeyDoorEnv`
- **Complexity**: High
- **Layout**: Complex maze with multiple keys and doors
- **Grid Size**: 15x15 (configurable)
- **Mission**: "Navigate the maze, collect keys, open doors, and reach the goal"

### 5. Lava Key-Door Environment
- **File**: `key_door_goal_envs.py` → `LavaKeyDoorEnv`
- **Complexity**: High
- **Layout**: Dangerous lava patches to avoid
- **Grid Size**: 10x10 (configurable)
- **Mission**: "Avoid lava, collect the key, open the door, and reach the goal"

##  Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Activate virtual environment (if using)
source venv/bin/activate
```

### Basic Usage
```python
from key_door_goal_envs import create_key_door_goal_env

# Create a basic environment
env = create_key_door_goal_env("basic", size=8)

# Reset environment
obs, info = env.reset()

# Take actions
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Close environment
env.close()
```

### Using Gymnasium Registration
```python
import gymnasium as gym

# Create environment using gymnasium
env = gym.make('MiniGrid-BasicKeyDoorGoal-v0')
obs, info = env.reset()
```

##  Actions

All environments use the standard MiniGrid action space:
- **0**: Turn Left
- **1**: Turn Right
- **2**: Move Forward
- **3**: Pick Up
- **4**: Drop
- **5**: Toggle (open door)
- **6**: Done

##  Visualization

### Using the Visualizer
```python
from key_door_goal_visualizer import KeyDoorGoalVisualizer

# Create visualizer
visualizer = KeyDoorGoalVisualizer()

# Visualize single environment
env = create_key_door_goal_env("basic")
visualizer.visualize_env(env, "My Environment")

# Compare all environments
visualizer.compare_environments()

# Visualize episode
visualizer.visualize_episode(env, max_steps=10)
```

### Visual Elements
- **🔴 Agent**: Red triangle with direction arrow
- **🟢 Goal**: Green square
- ** Keys**: Colored squares (blue, green, red, yellow, purple, orange)
- **🚪 Doors**: Colored rectangles (matching key colors)
- **🟫 Walls**: Brown rectangles
- **🌋 Lava**: Red-orange squares (dangerous!)
- **⬜ Floor**: Light beige squares

##  Testing

### Run Tests
```bash
# Test all environments
python test_key_door_goal.py

# Test individual environment
python key_door_goal_envs.py
```

### Demo Script
```bash
# Run interactive demo
python key_door_goal_demo.py
```

##  Environment Comparison

| Environment | Grid Size | Keys | Doors | Complexity | Special Features |
|-------------|-----------|------|-------|------------|------------------|
| Basic | 8x8 | 1 | 1 | Low | Simple layout |
| Multi | 10x10 | 3 | 3 | Medium | Color matching |
| Sequential | 12x12 | 4 | 4 | Medium-High | Order matters |
| Maze | 15x15 | 3 | 3 | High | Complex navigation |
| Lava | 10x10 | 1 | 1 | High | Danger avoidance |

##  Customization

### Creating Custom Environments
```python
from key_door_goal_envs import BasicKeyDoorGoalEnv

class MyCustomEnv(BasicKeyDoorGoalEnv):
    def __init__(self, **kwargs):
        super().__init__(size=12, **kwargs)
    
    def _gen_grid(self, width, height):
        # Custom grid generation
        super()._gen_grid(width, height)
        # Add custom elements
```

### Environment Parameters
```python
# Basic environment
env = create_key_door_goal_env("basic", size=10, max_steps=500)

# Multi-key environment
env = create_key_door_goal_env("multi", size=12, num_pairs=4)

# Sequential environment
env = create_key_door_goal_env("sequential", size=15, num_doors=5)

# Maze environment
env = create_key_door_goal_env("maze", size=20, maze_density=0.4)

# Lava environment
env = create_key_door_goal_env("lava", size=12, lava_density=0.15)
```

## 🎯 RL Training Tips

### Reward Shaping
- **Key Collection**: +0.5 reward for picking up keys
- **Door Opening**: +0.5 reward for opening doors
- **Goal Reaching**: +1.0 reward for reaching the goal
- **Step Penalty**: -0.01 per step (encourages efficiency)

### Observation Space
- **Image**: 7x7x3 RGB observation
- **Direction**: Agent's current direction
- **Carrying**: Currently carried object

### Training Considerations
1. **Curriculum Learning**: Start with basic environment, progress to complex ones
2. **Exploration**: Use epsilon-greedy or other exploration strategies
3. **Memory**: Consider using recurrent networks for sequential tasks
4. **Reward Engineering**: Adjust rewards based on training progress

## 📁 File Structure

```
MemorySchema/
├── key_door_goal_envs.py          # Main environment implementations
├── key_door_goal_visualizer.py    # Visualization tools
├── key_door_goal_demo.py          # Interactive demo script
├── test_key_door_goal.py          # Test suite
├── KEY_DOOR_GOAL_README.md        # This documentation
└── requirements.txt               # Dependencies
```

## 🔗 Dependencies

- **gymnasium**: RL environment interface
- **minigrid**: Grid-world environment framework
- **numpy**: Numerical computations
- **matplotlib**: Visualization
- **pygame**: Rendering (optional)

## 🎉 Examples

### Simple Training Loop
```python
import gymnasium as gym
from key_door_goal_envs import create_key_door_goal_env

# Create environment
env = create_key_door_goal_env("basic")

# Training loop
for episode in range(1000):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()  # Replace with your policy
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    if episode % 100 == 0:
        print(f"Episode {episode} completed")

env.close()
```

### Environment Comparison
```python
from key_door_goal_visualizer import KeyDoorGoalVisualizer

visualizer = KeyDoorGoalVisualizer()
visualizer.compare_environments(save_path="comparison.png")
```

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Add your environment or improvements
4. Add tests
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- Built on top of the excellent [MiniGrid](https://github.com/Farama-Foundation/MiniGrid) framework
- Inspired by classic key-door-goal puzzles in RL literature
- Part of the MemorySchema project for learning schemas using RL

---

**Happy Training! **
