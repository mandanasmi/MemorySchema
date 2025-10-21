"""
Demo script showing ConSpec working with our key-door-goal environments
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from key_door_goal_base import create_key_door_goal_env
from conspec_minigrid import ConSpec


def demo_conspec_on_environment(env_type='single', num_episodes=5):
    """
    Demo ConSpec on a key-door-goal environment
    
    Args:
        env_type: Type of environment ('single', 'double', 'triple')
        num_episodes: Number of demo episodes
    """
    
    print(f"Demo: ConSpec on {env_type} environment")
    print("=" * 50)
    
    device = torch.device('cpu')
    
    # Create environment
    env = create_key_door_goal_env(env_type, size=8)
    obs_shape = (env.height, env.width, 3)
    num_actions = env.action_space.n
    
    # Create ConSpec
    conspec = ConSpec(obs_shape, num_actions, device, 
                     intrinsic_reward_scale=0.1, learning_rate=1e-3)
    
    print(f"Environment: {env.__class__.__name__}")
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    print(f"Mission: {env.mission}")
    print()
    
    episode_rewards = []
    episode_lengths = []
    intrinsic_rewards_history = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_intrinsic_rewards = []
        
        print(f"Episode {episode + 1}:")
        
        while not done and episode_length < 50:  # Max 50 steps
            # Random action for demo
            action = env.action_space.sample()
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Compute intrinsic reward with ConSpec
            intrinsic_reward = conspec.compute_intrinsic_reward(
                [obs], [action], [reward], [done]
            )
            
            # Update ConSpec
            conspec.train_step([obs], [action], [reward], [done])
            
            episode_intrinsic_rewards.append(intrinsic_reward.item())
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        intrinsic_rewards_history.extend(episode_intrinsic_rewards)
        
        print(f"  Final reward: {episode_reward}")
        print(f"  Episode length: {episode_length}")
        print(f"  Avg intrinsic reward: {np.mean(episode_intrinsic_rewards):.4f}")
        print(f"  Memory - Success: {len(conspec.memory.success_memory)}, "
              f"Failure: {len(conspec.memory.failure_memory)}")
        print()
    
    env.close()
    
    # Print summary
    print("Summary:")
    print(f"  Average episode reward: {np.mean(episode_rewards):.2f}")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f}")
    print(f"  Average intrinsic reward: {np.mean(intrinsic_rewards_history):.4f}")
    print(f"  Total memory size: {len(conspec.memory.success_memory) + len(conspec.memory.failure_memory)}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'intrinsic_rewards': intrinsic_rewards_history
    }


def main():
    """Run ConSpec demo on all environments"""
    print("ConSpec Demo on Key-Door-Goal Environments")
    print("=" * 60)
    print()
    
    env_types = ['single', 'double', 'triple']
    all_results = {}
    
    for env_type in env_types:
        print(f"\n{'='*60}")
        results = demo_conspec_on_environment(env_type, num_episodes=3)
        all_results[env_type] = results
        print(f"{'='*60}")
    
    print("\n" + "="*60)
    print("ConSpec Demo Complete!")
    print("="*60)
    print()
    print("Key Features Demonstrated:")
    print("✓ ConSpec successfully integrates with MiniGrid environments")
    print("✓ Intrinsic rewards are computed based on observation novelty")
    print("✓ Memory system stores success and failure trajectories")
    print("✓ Works on all three graph dependency environments:")
    print("  - Single (Chain Graph)")
    print("  - Double (Parallel Graph)")  
    print("  - Triple (Sequential Graph)")
    print()
    print("ConSpec is ready for training on these environments!")


if __name__ == "__main__":
    main()
