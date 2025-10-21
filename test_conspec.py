"""
Test script to verify ConSpec works with our key-door-goal environments
"""

import torch
import numpy as np
from key_door_goal_base import create_key_door_goal_env
from conspec_minigrid import ConSpec


def test_conspec_basic():
    """Test basic ConSpec functionality"""
    print("Testing ConSpec basic functionality...")
    
    device = torch.device('cpu')
    
    # Create a simple environment
    env = create_key_door_goal_env('single', size=8)
    obs_shape = (env.height, env.width, 3)
    num_actions = env.action_space.n
    
    print(f"Environment: {env.__class__.__name__}")
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    
    # Create ConSpec
    conspec = ConSpec(obs_shape, num_actions, device)
    print("ConSpec created successfully!")
    
    # Test encoding
    obs, _ = env.reset()
    print(f"Initial observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"Observation keys: {obs.keys()}")
        print(f"Image shape: {obs['image'].shape}")
    else:
        print(f"Initial observation shape: {obs.shape}")
    
    # Test encoding
    encoded_features = conspec.encode_observations(obs)
    print(f"Encoded features shape: {encoded_features.shape}")
    
    # Test intrinsic reward computation
    if isinstance(obs, dict):
        obs_batch = [obs, obs]  # Batch of 2 dicts
    else:
        obs_batch = np.array([obs, obs])  # Batch of 2
    action_batch = np.array([0, 1])
    reward_batch = np.array([0.0, 0.0])
    done_batch = np.array([False, False])
    
    intrinsic_rewards = conspec.compute_intrinsic_reward(
        obs_batch, action_batch, reward_batch, done_batch
    )
    print(f"Intrinsic rewards shape: {intrinsic_rewards.shape}")
    print(f"Intrinsic rewards: {intrinsic_rewards}")
    
    # Test training step
    intrinsic_rewards = conspec.train_step(
        obs_batch, action_batch, reward_batch, done_batch
    )
    print(f"Training step completed successfully!")
    
    env.close()
    print("Basic ConSpec test passed!")


def test_conspec_with_episode():
    """Test ConSpec with a full episode"""
    print("\nTesting ConSpec with full episode...")
    
    device = torch.device('cpu')
    
    # Create environment
    env = create_key_door_goal_env('single', size=8)
    obs_shape = (env.height, env.width, 3)
    num_actions = env.action_space.n
    
    # Create ConSpec
    conspec = ConSpec(obs_shape, num_actions, device)
    
    # Run a short episode
    obs, _ = env.reset()
    done = False
    step = 0
    max_steps = 20
    
    episode_observations = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []
    
    while not done and step < max_steps:
        # Random action
        action = env.action_space.sample()
        
        # Take step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store data
        episode_observations.append(obs)
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_dones.append(done)
        
        # Update ConSpec every 5 steps
        if len(episode_observations) >= 5:
            obs_batch = np.array(episode_observations[-5:])
            action_batch = np.array(episode_actions[-5:])
            reward_batch = np.array(episode_rewards[-5:])
            done_batch = np.array(episode_dones[-5:])
            
            intrinsic_rewards = conspec.train_step(
                obs_batch, action_batch, reward_batch, done_batch
            )
            print(f"Step {step}: Intrinsic reward = {intrinsic_rewards.mean().item():.4f}")
        
        obs = next_obs
        step += 1
    
    print(f"Episode completed in {step} steps")
    print(f"Final reward: {episode_rewards[-1] if episode_rewards else 0}")
    print(f"Memory size - Success: {len(conspec.memory.success_memory)}, "
          f"Failure: {len(conspec.memory.failure_memory)}")
    
    env.close()
    print("Episode test passed!")


def test_all_environments():
    """Test ConSpec on all three environments"""
    print("\nTesting ConSpec on all environments...")
    
    device = torch.device('cpu')
    env_types = ['single', 'double', 'triple']
    
    for env_type in env_types:
        print(f"\nTesting {env_type} environment...")
        
        # Create environment
        env = create_key_door_goal_env(env_type, size=8)
        obs_shape = (env.height, env.width, 3)
        num_actions = env.action_space.n
        
        # Create ConSpec
        conspec = ConSpec(obs_shape, num_actions, device)
        
        # Run a few steps
        obs, _ = env.reset()
        for step in range(5):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Test encoding
            encoded = conspec.encode_observations(obs)
            
            # Test intrinsic reward
            if isinstance(obs, dict):
                obs_for_reward = [obs]
            else:
                obs_for_reward = obs.reshape(1, *obs.shape)
                
            intrinsic_reward = conspec.compute_intrinsic_reward(
                obs_for_reward, 
                np.array([action]), 
                np.array([reward]), 
                np.array([done])
            )
            
            print(f"  Step {step}: Reward={reward}, Intrinsic={intrinsic_reward.item():.4f}")
            
            obs = next_obs
            if done:
                break
        
        env.close()
        print(f"  {env_type} environment test passed!")
    
    print("All environments test passed!")


def main():
    """Run all tests"""
    print("Running ConSpec tests...")
    print("=" * 50)
    
    try:
        test_conspec_basic()
        test_conspec_with_episode()
        test_all_environments()
        
        print("\n" + "=" * 50)
        print("All ConSpec tests passed successfully!")
        print("ConSpec is ready to use with our key-door-goal environments!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
