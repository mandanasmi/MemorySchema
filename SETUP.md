# MiniGrid Environment Setup

This project contains a custom MiniGrid environment that demonstrates the use of Key, Door, and Goal objects from the MiniGrid library.

## Setup

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Environment

```bash
# Run the basic environment test
python environment.py

# Run the demo script
python demo.py
```

## Environment Details

The custom environment (`CustomMiniGridEnv`) includes:

- **Grid Size**: 8x8 by default (configurable)
- **Objects**: 
  - Blue key that the agent must collect
  - Blue door that requires the key to open
  - Goal that the agent must reach
- **Mission**: "Find the key, open the door, and reach the goal"

## Key Components

- `environment.py`: Main environment implementation using MiniGrid's Key, Door, and Goal objects
- `demo.py`: Example script showing how to use the environment
- `requirements.txt`: Python dependencies

## Usage Example

```python
from environment import create_environment

# Create environment
env = create_environment(size=8)

# Reset and get initial observation
obs, info = env.reset()

# Take an action
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Close environment
env.close()
```

## Dependencies

- minigrid>=3.0.0
- numpy>=1.18.0
- gymnasium>=0.28.1
- pygame>=2.4.0

