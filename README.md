# MemorySchema

Learning schemas using reinforcement learning and generative modelling with comprehensive key-door-goal environments.

## Overview

This repository provides a collection of reinforcement learning environments based on the key-door-goal concept, built using the MiniGrid framework. These environments are designed for learning schemas and spatial reasoning through reinforcement learning.

## Project Structure

The repository is organized into modular components:

```
MemorySchema/
├── environments/          # Environment implementations
├── methods/              # Algorithm implementations (ConSpec, DAG-GFlowNet)
├── metrics_experiments/  # Training scripts and experiments
├── scripts/             # Utility scripts for setup and deployment
├── configs/             # Configuration files
├── docs/                # Comprehensive documentation
└── figures/             # Visualizations and plots
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed documentation.

## Key Features

- **3 Modular Environment Types**: Single, Double, and Triple key-door-goal environments
- **Comprehensive Visualization**: Advanced plotting and visualization tools with detailed legends


## Quick Start

### Local Setup
```bash
# Clone the repository
git clone <repository-url>
cd MemorySchema

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Test the environments
python test_key_door_goal.py

# Run interactive demo
python key_door_goal_demo.py
```

### Mila Cluster Setup
```bash
# SSH to Mila cluster
ssh mila

# Run automated setup
curl -O https://raw.githubusercontent.com/mandanasamiei/MemorySchema/main/setup_mila.sh
chmod +x setup_mila.sh
./setup_mila.sh
```

## Repository Structure

```
MemorySchema/
 key_door_goal_base.py          # Modular environment system with base class
 key_door_goal_visualizer.py    # Advanced visualization tools with custom shapes
 key_door_goal_demo.py          # Interactive demo script
 test_key_door_goal.py          # Test suite
 requirements.txt               # Python dependencies
 setup_mila.sh                 # Mila cluster setup script
 MILA_SETUP.md                 # Mila cluster documentation
 SETUP.md                      # Local setup instructions
 LICENSE                       # MIT License
 figures/                      # Generated visualizations
     all_three_environments_corrected.png
    
```

## Environment Types

### **Modular Key-Door-Goal Environments**

1. **Single Key-Door-Goal Environment** (`single`) - **Chain Graph**
   - Blue key  Blue door  Green goal (sequential dependency)
   - Agent must pick up the key, then open the door, then reach the goal
   - 10x10 grid representing a linear dependency chain

2. **Double Key-Door-Goal Environment** (`double`) - **Parallel Graph**
   - Blue key  Blue door AND Green key  Green door (parallel dependencies)
   - Agent can explore both paths simultaneously
   - Both doors must be opened to reach the final goal
   - 10x10 grid with 3 rooms, allowing parallel exploration

3. **Triple Key-Door-Goal Environment** (`triple`) - **Sequential Graph**
   - Blue and green keys in first room, purple key in second room
   - Green door separates first and second rooms, purple door separates second and third rooms
   - Two goals: green goal in second room, teal goal in third room
   - Sequential progression: blue+green keys  green door  green goal, purple door  teal goal
   - 10x10 grid with 3-room layout

## Visualization

The repository includes comprehensive visualization tools:
- Individual environment visualization with detailed legends
- Combined visualization of all three environments
- Custom key and door shapes with schematic representations
- Dynamic legends showing all colored keys, doors, and goals present in each environment
- High-quality PNG and PDF export options

## Testing

Run the test suite to verify everything works:
```bash
python test_key_door_goal.py
```

## Documentation

- **KEY_DOOR_GOAL_README.md**: Comprehensive documentation for all environments
- **MILA_SETUP.md**: Detailed Mila cluster setup instructions
- **SETUP.md**: Local development setup guide

## Dependencies

- gymnasium>=1.1.1
- minigrid>=3.0.0
- numpy>=2.0.2
- matplotlib>=3.9.4
- pygame>=2.6.1

## Usage Examples

### Creating Environments
```python
from key_door_goal_base import create_key_door_goal_env

# Create different environment types
single_env = create_key_door_goal_env("single", size=10)
double_env = create_key_door_goal_env("double", size=10)
triple_env = create_key_door_goal_env("triple", size=10)

# Reset and interact with environment
obs, info = single_env.reset()
action = single_env.action_space.sample()
obs, reward, terminated, truncated, info = single_env.step(action)
single_env.close()
```

### Visualization
```python
from key_door_goal_visualizer import KeyDoorGoalVisualizer

visualizer = KeyDoorGoalVisualizer()
env = create_key_door_goal_env("single")
fig, ax = visualizer.visualize_env(env, "Single Key-Door-Goal Environment")
```

### Running Demos
```python
# Run the comprehensive demo
python key_door_goal_demo.py

# Test all environments
python test_key_door_goal.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of the excellent [MiniGrid](https://github.com/Farama-Foundation/MiniGrid) framework
- Part of the MemorySchema project for learning schemas using RL
- Designed for research in spatial reasoning and schema learning

---

**Ready for RL Training!**
