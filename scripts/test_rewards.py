"""
Test script to verify reward structure in environments
"""

import sys
sys.path.append('.')

from environments.key_door_goal_base import create_key_door_goal_env
from minigrid.core.actions import Actions

def test_environment_rewards(env_type="single", num_steps=100):
    """Test that environments give proper rewards"""
    print(f"\n{'='*60}")
    print(f"Testing {env_type.upper()} environment")
    print(f"{'='*60}\n")
    
    env = create_key_door_goal_env(env_type)
    obs = env.reset()
    
    total_reward = 0
    rewards_received = []
    
    for step in range(num_steps):
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward > 0:
            event = info.get('event', 'unknown')
            print(f"Step {step}: Reward = {reward:.2f}, Event = {event}")
            rewards_received.append((step, reward, event))
            total_reward += reward
        
        if terminated or truncated:
            print(f"\nEpisode terminated at step {step}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Number of rewards: {len(rewards_received)}")
    
    if len(rewards_received) == 0:
        print("WARNING: No rewards received! This might indicate a problem.")
    else:
        print("SUCCESS: Rewards are being given!")
    
    return total_reward, rewards_received

if __name__ == "__main__":
    # Test all three environments
    for env_type in ["single", "double", "triple"]:
        total, rewards = test_environment_rewards(env_type, num_steps=200)
        print(f"\nSummary for {env_type}: Total={total:.2f}, Events={len(rewards)}\n")

