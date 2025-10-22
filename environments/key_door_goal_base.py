"""
Base Key-Door-Goal Environment Classes
This module provides a base class for key-door-goal environments with common functionality.
"""

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Goal, Key, Door, Wall
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES
from gymnasium import spaces


class BaseKeyDoorGoalEnv(MiniGridEnv):
    """
    Base class for key-door-goal environments.
    Provides common functionality for creating secure key-door-goal layouts.
    """
    
    def __init__(self, size=10, max_steps=None, **kwargs):
        self.size = size
        if max_steps is None:
            max_steps = 4 * size * size
        
        mission_space = MissionSpace(mission_func=lambda: self._gen_mission())
        
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs
        )
    
    def _gen_mission(self):
        """Override in subclasses to define specific missions"""
        return "Complete the key-door-goal task"
    
    def _gen_grid(self, width, height):
        """Override in subclasses to define specific layouts"""
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.agent_pos = (1, height - 2)
        self.agent_dir = 0
    
    def _create_room_layout(self, width, height, num_rooms):
        """
        Create a basic room layout with walls separating rooms.
        
        Args:
            width: Grid width
            height: Grid height
            num_rooms: Number of rooms to create
        """
        # Create outer walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Create vertical walls between rooms
        room_width = width // num_rooms
        for i in range(1, num_rooms):
            x = room_width * i
            for y in range(1, height - 1):
                self.grid.set(x, y, Wall())
        
        return room_width
    
    def _place_key(self, x, y, color):
        """Place a key at the specified position"""
        key = Key(color)
        self.grid.set(x, y, key)
    
    def _place_door(self, x, y, color, is_locked=True):
        """Place a door at the specified position"""
        door = Door(color, is_locked=is_locked)
        self.grid.set(x, y, door)
    
    def _place_goal(self, x, y, color="green"):
        """Place a goal at the specified position"""
        goal = Goal(color)
        self.grid.set(x, y, goal)
    
    def _place_agent(self, x, y, direction=0):
        """Place the agent at the specified position"""
        self.agent_pos = (x, y)
        self.agent_dir = direction
    
    def step(self, action):
        """Execute one step in the environment"""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Check if the agent has reached a goal
        if self.grid.get(*self.agent_pos) and isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = self._reward()
            terminated = True
        
        return obs, reward, terminated, truncated, info


class SingleKeyDoorGoalEnv(BaseKeyDoorGoalEnv):
    """
    Environment 1: Chain Graph - Blue key  Blue door  Green goal
    Sequential dependency: must pick up key, then open door, then reach goal.
    """
    
    def _gen_mission(self):
        return "Chain graph: Pick up blue key, open blue door, then reach green goal"
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Create outer walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Create a wall in the middle
        for y in range(1, height - 1):
            self.grid.set(width // 2, y, Wall())
        
        # Place agent in left room
        self._place_agent(1, height - 2, 0)
        
        # Place blue key in left room (first in chain)
        self._place_key(2, height // 2, "blue")
        
        # Place blue door in middle wall (second in chain)
        self._place_door(width // 2, height // 2, "blue", is_locked=True)
        
        # Place green goal in farthest corner of right room (final in chain)
        self._place_goal(width - 2, height - 2)


class DoubleKeyDoorGoalEnv(BaseKeyDoorGoalEnv):
    """
    Environment 2: Parallel Graph - Blue key  Blue door AND Green key  Green door
    Parallel dependencies: agent can explore both paths simultaneously.
    Both doors must be opened to reach the final goal.
    """
    
    def _gen_mission(self):
        return "Parallel graph: Open both blue and green doors to reach the goal"
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Create outer walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Create a single wall in the middle to create 2 rooms
        wall_x = width // 2
        
        # Middle wall with two doors (blue and green side by side)
        for y in range(1, height - 1):
            self.grid.set(wall_x, y, Wall())
        
        # Place agent in first room
        self._place_agent(1, height - 2, 0)
        
        # Place blue key in first room (left side)
        self._place_key(2, height // 2 - 1, "blue")
        
        # Place green key in first room (right side)
        self._place_key(2, height // 2 + 1, "green")
        
        # Place blue door in middle wall (left side)
        self._place_door(wall_x, height // 2 - 1, "blue", is_locked=True)
        
        # Place green door in middle wall (right side)
        self._place_door(wall_x, height // 2 + 1, "green", is_locked=True)
        
        # Place goal in farthest corner of second room (requires both doors)
        self._place_goal(width - 2, height - 2)


class TripleKeyDoorGoalEnv(BaseKeyDoorGoalEnv):
    """
    Environment 3: Three rooms with sequential progression.
    Room 1: Blue and green keys
    Room 2: Green goal and purple key (accessible via green door)
    Room 3: Purple goal (more rewarding, accessible via purple door)
    """
    
    def _gen_mission(self):
        return "Pick up green key to open green door and reach green goal, then pick up purple key to open purple door and reach the more rewarding purple goal"
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Create outer walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Create 3-room layout for 10x10 grid
        # First vertical wall (separates room 1 from room 2)
        for y in range(1, height - 1):
            self.grid.set(3, y, Wall())
        
        # Second vertical wall (separates room 2 from room 3)
        for y in range(1, height - 1):
            self.grid.set(6, y, Wall())
        
        # Place agent in first room
        self._place_agent(1, height - 2, 0)
        
        # Room 1: Blue and green keys
        self._place_key(1, height // 2 - 1, "blue")
        self._place_key(1, height // 2 + 1, "green")
        
        # Green door in first wall (separates room 1 from room 2)
        self._place_door(3, height // 2, "green", is_locked=True)
        
        # Room 2: Green goal in corner and purple key (farther in column)
        self._place_goal(4, height - 2, "green")
        self._place_key(5, height // 2 + 1, "purple")
        
        # Purple door in second wall (separates room 2 from room 3)
        self._place_door(6, height // 2, "purple", is_locked=True)
        
        # Room 3: Purple goal in farthest corner
        self._place_goal(width - 2, height - 2, "purple")


# Factory function
def create_key_door_goal_env(env_type="single", size=10, **kwargs):
    """
    Factory function to create key-door-goal environments
    
    Args:
        env_type (str): Type of environment to create
            - "single": SingleKeyDoorGoalEnv (1 key, 1 door, 1 goal)
            - "double": DoubleKeyDoorGoalEnv (2 keys, 2 doors, 1 goal)
            - "triple": TripleKeyDoorGoalEnv (3 keys, 3 doors, 2 goals)
        size (int): Size of the grid (default: 10)
        **kwargs: Additional arguments for the environment
    
    Returns:
        BaseKeyDoorGoalEnv: The created environment
    """
    env_map = {
        "single": SingleKeyDoorGoalEnv,
        "double": DoubleKeyDoorGoalEnv,
        "triple": TripleKeyDoorGoalEnv,
    }
    
    if env_type not in env_map:
        raise ValueError(f"Unknown environment type: {env_type}. Available types: {list(env_map.keys())}")
    
    return env_map[env_type](size=size, **kwargs)


if __name__ == "__main__":
    # Test all environments
    env_types = ["single", "double", "triple"]
    
    for env_type in env_types:
        print(f"\n=== Testing {env_type.upper()} Key-Door-Goal Environment ===")
        try:
            env = create_key_door_goal_env(env_type, size=10)
            obs, info = env.reset()
            print(f"{env_type} environment created successfully")
            print(f"   Grid size: {env.width}x{env.height}")
            print(f"   Mission: {env.mission}")
            print(f"   Observation shape: {obs['image'].shape}")
            env.close()
        except Exception as e:
            print(f" {env_type} environment failed: {e}")
    
    print("\n All key-door-goal environments tested!")
