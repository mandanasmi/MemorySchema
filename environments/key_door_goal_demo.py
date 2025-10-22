"""
Key-Door-Goal Environments Demo
This script demonstrates all the key-door-goal environments with interactive examples.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from environments.key_door_goal_base import create_key_door_goal_env
from environments.key_door_goal_visualizer import KeyDoorGoalVisualizer
from minigrid.core.actions import Actions


class KeyDoorGoalDemo:
    """Demo class for key-door-goal environments"""
    
    def __init__(self):
        self.visualizer = KeyDoorGoalVisualizer()
        self.env_types = ["single", "double", "triple"]
    
    def run_single_demo(self):
        """Run single key-door-goal environment demo"""
        print("Single Key-Door-Goal Environment Demo")
        print("=" * 50)
        
        env = create_key_door_goal_env("single", size=8)
        obs, info = env.reset()
        
        print(f"Environment: {env.__class__.__name__}")
        print(f"Grid size: {env.width}x{env.height}")
        print(f"Mission: {env.mission}")
        print(f"Max steps: {env.max_steps}")
        print(f"Observation shape: {obs['image'].shape}")
        
        # Visualize initial state
        self.visualizer.visualize_env(env, "Single Key-Door-Goal Environment")
        
        # Run a few random steps
        print("\n Running random actions...")
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step + 1}: Action={action}, Reward={reward}, Terminated={terminated}")
            
            if terminated or truncated:
                print("Episode ended!")
                break
        
        env.close()
        print(" Single demo completed!\n")
    
    def run_double_demo(self):
        """Run double key-door environment demo"""
        print(" Double Key-Door Environment Demo")
        print("=" * 50)
        
        env = create_key_door_goal_env("double", size=10)
        obs, info = env.reset()
        
        print(f"Environment: {env.__class__.__name__}")
        print(f"Grid size: {env.width}x{env.height}")
        print(f"Mission: {env.mission}")
        
        # Visualize initial state
        self.visualizer.visualize_env(env, "Double Key-Door Environment")
        
        # Run a few random steps
        print("\n Running random actions...")
        for step in range(15):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step + 1}: Action={action}, Reward={reward}, Terminated={terminated}")
            
            if terminated or truncated:
                print("Episode ended!")
                break
        
        env.close()
        print(" Double demo completed!\n")
    
    def run_triple_demo(self):
        """Run triple key-door environment demo"""
        print(" Triple Key-Door Environment Demo")
        print("=" * 50)
        
        env = create_key_door_goal_env("triple", size=12)
        obs, info = env.reset()
        
        print(f"Environment: {env.__class__.__name__}")
        print(f"Grid size: {env.width}x{env.height}")
        print(f"Mission: {env.mission}")
        
        # Visualize initial state
        self.visualizer.visualize_env(env, "Triple Key-Door Environment")
        
        # Run a few random steps
        print("\n Running random actions...")
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step + 1}: Action={action}, Reward={reward}, Terminated={terminated}")
            
            if terminated or truncated:
                print("Episode ended!")
                break
        
        env.close()
        print(" Triple demo completed!\n")
    
    
    def run_comparison_demo(self):
        """Run comparison demo of all environments"""
        print(" Environment Comparison Demo")
        print("=" * 50)
        
        # Compare all environments
        self.visualizer.compare_environments(
            env_types=self.env_types,
            save_path="key_door_goal_comparison.png"
        )
        
        print(" Comparison demo completed!\n")
    
    def run_episode_demo(self):
        """Run episode visualization demo"""
        print("  Episode Visualization Demo")
        print("=" * 50)
        
        # Visualize episode for basic environment
        env = create_key_door_goal_env("basic", size=8)
        self.visualizer.visualize_episode(
            env, 
            max_steps=6, 
            save_path="key_door_goal_episode.png"
        )
        env.close()
        
        print(" Episode demo completed!\n")
    
    def run_interactive_demo(self):
        """Run interactive demo with user input"""
        print(" Interactive Key-Door-Goal Demo")
        print("=" * 50)
        
        # Let user choose environment
        print("Available environments:")
        for i, env_type in enumerate(self.env_types):
            print(f"  {i + 1}. {env_type.upper()}")
        
        try:
            choice = int(input("\nChoose environment (1-5): ")) - 1
            if 0 <= choice < len(self.env_types):
                env_type = self.env_types[choice]
                print(f"\nSelected: {env_type.upper()}")
                
                # Create environment
                env = create_key_door_goal_env(env_type)
                obs, info = env.reset()
                
                print(f"Mission: {env.mission}")
                print("Actions: 0=Turn Left, 1=Turn Right, 2=Move Forward, 3=Pick Up, 4=Drop, 5=Toggle")
                
                # Visualize initial state
                self.visualizer.visualize_env(env, f"{env_type.upper()} Environment")
                
                # Interactive loop
                step = 0
                while step < 50:
                    try:
                        action = int(input(f"Step {step + 1}: Enter action (0-5) or -1 to quit: "))
                        if action == -1:
                            break
                        
                        if 0 <= action <= 5:
                            obs, reward, terminated, truncated, info = env.step(action)
                            print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
                            
                            if terminated or truncated:
                                print("Episode ended!")
                                break
                            
                            step += 1
                        else:
                            print("Invalid action! Please enter 0-5 or -1 to quit.")
                    except ValueError:
                        print("Invalid input! Please enter a number.")
                
                env.close()
                print(" Interactive demo completed!")
            else:
                print("Invalid choice!")
        except ValueError:
            print("Invalid input!")
    
    def run_all_demos(self):
        """Run all demos"""
        print(" Key-Door-Goal Environments - Complete Demo")
        print("=" * 60)
        
        demos = [
            ("Single Environment", self.run_single_demo),
            ("Double Environment", self.run_double_demo),
            ("Triple Environment", self.run_triple_demo),
            ("Environment Comparison", self.run_comparison_demo),
            ("Episode Visualization", self.run_episode_demo),
        ]
        
        for demo_name, demo_func in demos:
            print(f"\n Running {demo_name}...")
            try:
                demo_func()
            except Exception as e:
                print(f" Error in {demo_name}: {e}")
        
        print("\n All demos completed!")
        print("\n Summary:")
        print("   Single Key-Door-Goal Environment")
        print("   Double Key-Door Environment")
        print("   Triple Key-Door Environment")
        print("   Environment Comparison")
        print("   Episode Visualization")
        print("\n You can now use these environments for RL training!")


def main():
    """Main function"""
    demo = KeyDoorGoalDemo()
    
    print(" Key-Door-Goal Environments Demo")
    print("=" * 40)
    print("1. Run all demos")
    print("2. Run individual demo")
    print("3. Interactive demo")
    print("4. Exit")
    
    while True:
        try:
            choice = int(input("\nChoose option (1-4): "))
            
            if choice == 1:
                demo.run_all_demos()
                break
            elif choice == 2:
                print("\nAvailable demos:")
                print("1. Single Environment")
                print("2. Double Environment")
                print("3. Triple Environment")
                print("4. Environment Comparison")
                print("5. Episode Visualization")
                
                demo_choice = int(input("Choose demo (1-5): "))
                demo_funcs = [
                    demo.run_single_demo,
                    demo.run_double_demo,
                    demo.run_triple_demo,
                    demo.run_comparison_demo,
                    demo.run_episode_demo,
                ]
                
                if 1 <= demo_choice <= 5:
                    demo_funcs[demo_choice - 1]()
                else:
                    print("Invalid choice!")
                break
            elif choice == 3:
                demo.run_interactive_demo()
                break
            elif choice == 4:
                print("Goodbye!")
                break
            else:
                print("Invalid choice! Please enter 1-4.")
        except ValueError:
            print("Invalid input! Please enter a number.")


if __name__ == "__main__":
    main()
