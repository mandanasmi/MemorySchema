"""
Key-Door-Goal Environment Visualizer
This module provides visualization tools for all key-door-goal environments.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import gymnasium as gym
from key_door_goal_base import (
    SingleKeyDoorGoalEnv, DoubleKeyDoorGoalEnv, TripleKeyDoorGoalEnv, 
    create_key_door_goal_env
)
from minigrid.core.world_object import Goal, Key, Door, Wall, Lava
from minigrid.core.constants import COLOR_TO_IDX, IDX_TO_COLOR


class KeyDoorGoalVisualizer:
    """Visualizer for key-door-goal environments"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = {
            'wall': '#2C2C2C',      # Dark Gray
            'floor': '#F8F8F8',     # Light Gray/White
            'lava': '#8B0000',      # Dark Red
            'agent': '#FF0000',     # Red (keep for visibility)
            'goal': '#00FF00',      # Green (keep for visibility)
            'key_blue': '#4682B4',  # Steel Blue
            'key_green': '#556B2F', # Dark Olive Green
            'key_teal': '#20B2AA',  # Light Sea Green (between blue and green)
            'key_red': '#8B0000',   # Dark Red
            'key_yellow': '#B8860B', # Dark Goldenrod
            'key_purple': '#4B0082', # Indigo
            'key_orange': '#FF8C00', # Dark Orange
            'door_blue': '#4682B4',  # Steel Blue
            'door_green': '#556B2F', # Dark Olive Green
            'door_teal': '#20B2AA',  # Light Sea Green (between blue and green)
            'door_red': '#8B0000',   # Dark Red
            'door_yellow': '#B8860B', # Dark Goldenrod
            'door_purple': '#4B0082', # Indigo
            'door_orange': '#FF8C00', # Dark Orange
            'goal_green': '#00FF00',  # Green
            'goal_blue': '#4682B4',   # Steel Blue
            'goal_red': '#FF0000',    # Red
            'goal_yellow': '#FFFF00', # Yellow
            'goal_purple': '#800080', # Purple
            'goal_orange': '#FF8C00', # Orange
            'goal_teal': '#20B2AA',   # Light Sea Green (between blue and green)
        }
    
    def visualize_env(self, env, title="Key-Door-Goal Environment", save_path=None):
        """Visualize a key-door-goal environment"""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Get environment state
        obs = env.render()
        width, height = env.width, env.height
        
        # Create grid visualization
        self._draw_grid(ax, env, width, height)
        
        # Add title
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.invert_yaxis()  # Invert y-axis to match grid coordinates
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, width, 1))
        ax.set_yticks(np.arange(-0.5, height, 1))
        ax.grid(True, alpha=0.3)
        
        # Remove axis tick labels and tick marks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # Add legend
        self._add_legend(ax, env)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        return fig, ax
    
    def _draw_grid(self, ax, env, width, height):
        """Draw the grid elements"""
        for x in range(width):
            for y in range(height):
                cell = env.grid.get(x, y)
                
                if cell is None:
                    # Empty cell (floor)
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                           facecolor=self.colors['floor'], 
                                           edgecolor='black', linewidth=0.5)
                elif isinstance(cell, Wall):
                    # Wall
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                           facecolor=self.colors['wall'], 
                                           edgecolor='black', linewidth=1)
                elif isinstance(cell, Lava):
                    # Lava
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                           facecolor=self.colors['lava'], 
                                           edgecolor='black', linewidth=1)
                elif isinstance(cell, Goal):
                    # Goal - use color if available, otherwise default to green
                    goal_color = self.colors.get(f'goal_{cell.color}', self.colors['goal'])
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                           facecolor=goal_color, 
                                           edgecolor='black', linewidth=1)
                elif isinstance(cell, Key):
                    # Key - draw as a key shape
                    # Keep original colors for keys (blue and green keys stay blue and green)
                    color_key = f"key_{cell.color}"
                    key_color = self.colors.get(color_key, self.colors['key_blue'])
                    self._draw_key(ax, x, y, key_color)
                elif isinstance(cell, Door):
                    # Door - draw as a door shape
                    # Use teal color for green doors in triple environment
                    if cell.color == 'green' and hasattr(env, '__class__') and 'Triple' in env.__class__.__name__:
                        door_color = self.colors['door_teal']
                    else:
                        color_key = f"door_{cell.color}"
                        door_color = self.colors.get(color_key, self.colors['door_blue'])
                    self._draw_door(ax, x, y, door_color, cell.is_locked)
                else:
                    # Unknown object
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                           facecolor='gray', 
                                           edgecolor='black', linewidth=1)
                
                ax.add_patch(rect)
        
        # Draw agent triangle pointing in the direction it's facing
        agent_x, agent_y = env.agent_pos
        agent_triangle = self._get_directional_triangle(agent_x, agent_y, env.agent_dir)
        ax.add_patch(agent_triangle)
    
    def _get_directional_triangle(self, x, y, direction):
        """Get directional triangle for agent"""
        # Direction: 0=right, 1=down, 2=left, 3=up
        size = 0.3
        
        if direction == 0:  # Right
            triangle = patches.Polygon([
                (x+size, y),      # Point right
                (x-size, y-size), # Bottom left
                (x-size, y+size)  # Top left
            ], facecolor=self.colors['agent'], edgecolor='black', linewidth=1)
        elif direction == 1:  # Down
            triangle = patches.Polygon([
                (x, y+size),      # Point down
                (x-size, y-size), # Top left
                (x+size, y-size)  # Top right
            ], facecolor=self.colors['agent'], edgecolor='black', linewidth=1)
        elif direction == 2:  # Left
            triangle = patches.Polygon([
                (x-size, y),      # Point left
                (x+size, y-size), # Bottom right
                (x+size, y+size)  # Top right
            ], facecolor=self.colors['agent'], edgecolor='black', linewidth=1)
        else:  # Up
            triangle = patches.Polygon([
                (x, y-size),      # Point up
                (x-size, y+size), # Bottom left
                (x+size, y+size)  # Bottom right
            ], facecolor=self.colors['agent'], edgecolor='black', linewidth=1)
        
        return triangle
    
    def _draw_key(self, ax, x, y, color):
        """Draw a simple key shape: circle, line, orthogonal lines"""
        # Key head (circle)
        key_head = patches.Circle((x, y), 0.15, 
                                facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(key_head)
        
        # Key head hole (center)
        key_hole = patches.Circle((x, y), 0.06, 
                                facecolor='white', edgecolor='black', linewidth=1)
        ax.add_patch(key_hole)
        
        # Key shaft (vertical line) - moved further down to create more space
        key_shaft = patches.Rectangle((x-0.02, y-0.45), 0.04, 0.4, 
                                    facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(key_shaft)
        
        # Key teeth (orthogonal lines extending from shaft) - right side only
        # Right side teeth (horizontal lines extending right)
        teeth1 = patches.Rectangle((x+0.02, y-0.25), 0.1, 0.02, 
                                 facecolor=color, edgecolor='black', linewidth=1)
        teeth2 = patches.Rectangle((x+0.02, y-0.35), 0.1, 0.02, 
                                 facecolor=color, edgecolor='black', linewidth=1)
        teeth3 = patches.Rectangle((x+0.02, y-0.45), 0.1, 0.02, 
                                 facecolor=color, edgecolor='black', linewidth=1)
        
        ax.add_patch(teeth1)
        ax.add_patch(teeth2)
        ax.add_patch(teeth3)
    
    def _draw_door(self, ax, x, y, color, is_locked):
        """Draw a door shape"""
        # Door frame
        door_frame = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                     facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(door_frame)
        
        # Door panel (slightly smaller)
        door_panel = patches.Rectangle((x-0.35, y-0.35), 0.7, 0.7, 
                                     facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(door_panel)
        
        # Door handle
        handle = patches.Circle((x+0.2, y), 0.08, facecolor='#C0C0C0', edgecolor='black', linewidth=1)
        ax.add_patch(handle)
        
        # Lock indicator
        if is_locked:
            # Draw a lock symbol
            lock = patches.Rectangle((x-0.1, y-0.1), 0.2, 0.2, 
                                   facecolor='darkred', edgecolor='black', linewidth=1)
            ax.add_patch(lock)
            # Lock keyhole
            keyhole = patches.Circle((x, y), 0.05, facecolor='black')
            ax.add_patch(keyhole)
    
    def _add_legend(self, ax, env=None):
        """Add simplified legend with schematic representations"""
        from matplotlib.lines import Line2D
        
        # Create custom legend elements with schematics
        legend_elements = []
        
        # Agent (triangle)
        agent_triangle = patches.Polygon([(0, 0), (-0.3, -0.3), (-0.3, 0.3)], 
                                       facecolor=self.colors['agent'], edgecolor='black', linewidth=1)
        legend_elements.append(agent_triangle)
        
        # Wall (rectangle)
        wall_rect = patches.Rectangle((0, 0), 0.6, 0.6, 
                                    facecolor=self.colors['wall'], edgecolor='black', linewidth=1)
        legend_elements.append(wall_rect)
        
        # Floor (rectangle)
        floor_rect = patches.Rectangle((0, 0), 0.6, 0.6, 
                                     facecolor=self.colors['floor'], edgecolor='black', linewidth=1)
        legend_elements.append(floor_rect)
        
        # Add all keys and doors from the environment
        keys_present = set()
        doors_present = set()
        
        if env is not None:
            for x in range(env.width):
                for y in range(env.height):
                    cell = env.grid.get(x, y)
                    if isinstance(cell, Key):
                        keys_present.add(cell.color)
                    elif isinstance(cell, Door):
                        doors_present.add(cell.color)
        
        # Add all goals from the environment
        goals_present = set()
        
        if env is not None:
            for x in range(env.width):
                for y in range(env.height):
                    cell = env.grid.get(x, y)
                    if isinstance(cell, Goal):
                        goals_present.add(cell.color)
        
        # Add goals to legend (same style as agent/wall/floor)
        for color in sorted(goals_present):
            if color == 'blue' and hasattr(env, '__class__') and 'Triple' in env.__class__.__name__:
                goal_color = self.colors['goal_teal']
                goal_label = 'Teal Goal'
            else:
                goal_color = self.colors.get(f'goal_{color}', self.colors['goal_green'])
                goal_label = f'{color.title()} Goal'
            
            goal_diamond = patches.Polygon([(0, 0.3), (0.3, 0), (0, -0.3), (-0.3, 0)], 
                                         facecolor=goal_color, edgecolor='black', linewidth=1)
            legend_elements.append(goal_diamond)
        
        # Add keys to legend
        for color in sorted(keys_present):
            # Keys in triple environment are blue and green, not teal
            key_color = self.colors.get(f'key_{color}', self.colors['key_blue'])
            key_label = f'{color.title()} Key'
            
            key_circle = Line2D([0], [0], marker='o', color=key_color, 
                               markersize=8, linestyle='None', markerfacecolor=key_color,
                               markeredgecolor='black', markeredgewidth=1)
            legend_elements.append(key_circle)
        
        # Add doors to legend
        for color in sorted(doors_present):
            if color == 'green' and hasattr(env, '__class__') and 'Triple' in env.__class__.__name__:
                door_color = self.colors['door_teal']
                door_label = 'Teal Door'
            else:
                door_color = self.colors.get(f'door_{color}', self.colors['door_blue'])
                door_label = f'{color.title()} Door'
            
            door_square = Line2D([0], [0], marker='s', color=door_color, 
                                markersize=8, linestyle='None', markerfacecolor=door_color,
                                markeredgecolor='black', markeredgewidth=1)
            legend_elements.append(door_square)
        
        # Create labels list
        labels = ['Agent', 'Wall', 'Floor']
        for color in sorted(goals_present):
            if color == 'blue' and hasattr(env, '__class__') and 'Triple' in env.__class__.__name__:
                labels.append('Teal Goal')
            else:
                labels.append(f'{color.title()} Goal')
        for color in sorted(keys_present):
            labels.append(f'{color.title()} Key')
        for color in sorted(doors_present):
            if color == 'green' and hasattr(env, '__class__') and 'Triple' in env.__class__.__name__:
                labels.append('Teal Door')
            else:
                labels.append(f'{color.title()} Door')
        
        ax.legend(legend_elements, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    def _get_key_color_for_legend(self, env):
        """Get the appropriate key color for the legend based on environment"""
        if env is None:
            return self.colors['key_blue']
        
        # Check what keys are in the environment
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if isinstance(cell, Key):
                    # Use teal for green keys in triple environment
                    if cell.color == 'green' and hasattr(env, '__class__') and 'Triple' in env.__class__.__name__:
                        return self.colors['key_teal']
                    else:
                        color_key = f"key_{cell.color}"
                        return self.colors.get(color_key, self.colors['key_blue'])
        
        return self.colors['key_blue']
    
    def _get_door_color_for_legend(self, env):
        """Get the appropriate door color for the legend based on environment"""
        if env is None:
            return self.colors['door_blue']
        
        # Check what doors are in the environment
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if isinstance(cell, Door):
                    # Use teal for green doors in triple environment
                    if cell.color == 'green' and hasattr(env, '__class__') and 'Triple' in env.__class__.__name__:
                        return self.colors['door_teal']
                    else:
                        color_key = f"door_{cell.color}"
                        return self.colors.get(color_key, self.colors['door_blue'])
        
        return self.colors['door_blue']
    
    def _create_detailed_key_schematic(self):
        """Create a detailed key schematic for the legend"""
        # Create a custom key patch that combines circle and line
        key_patch = patches.Patch(color=self.colors['key_blue'], label='Key')
        return key_patch
    
    def _create_detailed_door_schematic(self):
        """Create a detailed door schematic for the legend"""
        # Create a custom door patch
        door_patch = patches.Patch(color=self.colors['door_blue'], label='Door')
        return door_patch
    
    def compare_environments(self, env_types=None, save_path=None):
        """Compare multiple key-door-goal environments side by side"""
        if env_types is None:
            env_types = ["single", "double", "triple"]
        
        n_envs = len(env_types)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, env_type in enumerate(env_types):
            if i >= len(axes):
                break
                
            try:
                env = create_key_door_goal_env(env_type)
                obs, info = env.reset()
                
                # Draw environment
                self._draw_grid(axes[i], env, env.width, env.height)
                
                # Add title
                axes[i].set_title(f"{env_type.upper()} Environment\n{env.mission}", 
                                fontsize=12, fontweight='bold')
                axes[i].set_aspect('equal')
                axes[i].set_xlim(-0.5, env.width - 0.5)
                axes[i].set_ylim(-0.5, env.height - 0.5)
                axes[i].invert_yaxis()
                
                # Add grid lines
                axes[i].set_xticks(np.arange(-0.5, env.width, 1))
                axes[i].set_yticks(np.arange(-0.5, env.height, 1))
                axes[i].grid(True, alpha=0.3)
                
                # Add legend for this environment
                self._add_legend(axes[i], env)
                
                env.close()
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error creating {env_type}:\n{str(e)}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{env_type.upper()} Environment (Error)")
        
        # Hide unused subplots
        for i in range(n_envs, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison visualization saved to: {save_path}")
        
        plt.show()
        return fig, axes
    
    def visualize_episode(self, env, max_steps=50, save_path=None):
        """Visualize an episode of the environment with random actions"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        obs, info = env.reset()
        step = 0
        
        for i in range(min(6, max_steps)):
            if i >= len(axes):
                break
                
            # Visualize current state
            self._draw_grid(axes[i], env, env.width, env.height)
            axes[i].set_title(f"Step {step}\nAction: {env.action_space.sample()}", 
                            fontsize=12, fontweight='bold')
            axes[i].set_aspect('equal')
            axes[i].set_xlim(-0.5, env.width - 0.5)
            axes[i].set_ylim(-0.5, env.height - 0.5)
            axes[i].invert_yaxis()
            
            # Add grid lines
            axes[i].set_xticks(np.arange(-0.5, env.width, 1))
            axes[i].set_yticks(np.arange(-0.5, env.height, 1))
            axes[i].grid(True, alpha=0.3)
            
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            if terminated or truncated:
                break
        
        # Hide unused subplots
        for i in range(step, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Episode visualization saved to: {save_path}")
        
        plt.show()
        return fig, axes


def main():
    """Main function to demonstrate the visualizer"""
    visualizer = KeyDoorGoalVisualizer()
    
    print(" Key-Door-Goal Environment Visualizer")
    print("=" * 50)
    
    # Test individual environments
    env_types = ["basic", "multi", "sequential", "maze", "lava"]
    
    for env_type in env_types:
        print(f"\n Visualizing {env_type.upper()} environment...")
        try:
            env = create_key_door_goal_env(env_type)
            visualizer.visualize_env(env, title=f"{env_type.upper()} Key-Door-Goal Environment")
            env.close()
        except Exception as e:
            print(f" Error visualizing {env_type}: {e}")
    
    # Compare all environments
    print("\n Comparing all environments...")
    visualizer.compare_environments(save_path="key_door_goal_comparison.png")
    
    # Visualize episode
    print("\n  Visualizing episode...")
    env = create_key_door_goal_env("basic")
    visualizer.visualize_episode(env, max_steps=6, save_path="key_door_goal_episode.png")
    env.close()
    
    print("\n Visualization complete!")


if __name__ == "__main__":
    main()
