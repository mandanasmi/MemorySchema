"""
Quick test of ConSpec on environment 1 with wandb tracking
Note: You'll need to run 'wandb login' first
"""

import torch
import numpy as np
import os

# Check if wandb is available and handle login
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✅ Wandb is installed")
    
    # Check if logged in
    try:
        wandb.login(verify=True)
        print("✅ Wandb is logged in")
    except Exception as e:
        print("⚠️  Wandb login required. Please run: wandb login")
        print("   Get your API key from: https://wandb.ai/authorize")
        WANDB_AVAILABLE = False
except ImportError:
    print("❌ Wandb not installed")
    WANDB_AVAILABLE = False

from key_door_goal_base import create_key_door_goal_env
from conspec_minigrid import ConSpec
from train_conspec_wandb import SimplePolicy, PPOTrainer

def quick_test(num_episodes=100, use_wandb=True):
    """
    Quick test run of ConSpec on single environment
    """
    print("\n" + "="*60)
    print("Quick ConSpec Test on Single Environment (Chain Graph)")
    print("="*60)
    
    device = torch.device('cpu')
    env_type = 'single'
    
    # Use wandb only if available and requested
    wandb_enabled = WANDB_AVAILABLE and use_wandb
    
    # Create environment
    env = create_key_door_goal_env(env_type, size=10)
    obs_shape = (env.height, env.width, 3)
    num_actions = env.action_space.n
    
    print(f"Environment: {env.__class__.__name__}")
    print(f"Mission: {env.mission}")
    print(f"Grid size: {env.width}x{env.height}")
    print(f"Training episodes: {num_episodes}")
    print()
    
    # Create models
    policy = SimplePolicy(obs_shape, num_actions).to(device)
    conspec = ConSpec(obs_shape, num_actions, device, 
                     intrinsic_reward_scale=0.5, num_prototypes=5)
    trainer = PPOTrainer(policy, device)
    
    # Initialize wandb if available
    wandb_run = None
    if wandb_enabled:
        try:
            wandb_run = wandb.init(
                project='schema-learning',
                entity='samieima',
                name=f'quick_test_{env_type}',
                config={
                    'env_type': env_type,
                    'num_prototypes': 5,
                    'num_episodes': num_episodes,
                    'test': True
                }
            )
            print("✅ Wandb tracking enabled")
            print(f"   Dashboard: {wandb.run.url}")
        except Exception as e:
            print(f"⚠️  Wandb initialization failed: {e}")
            print("   Continuing without wandb tracking...")
            wandb_enabled = False
    else:
        print("ℹ️  Training without wandb tracking")
    
    print()
    
    # Training metrics
    episode_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        episode_observations = []
        episode_actions = []
        episode_rewards_list = []
        episode_intrinsic_rewards = []
        episode_log_probs = []
        
        # Epsilon-greedy
        epsilon = max(0.1, 1.0 - episode / (num_episodes * 0.7))
        
        while not done and episode_length < 200:
            with torch.no_grad():
                logits, value = policy(obs['image'])
                action_probs = torch.softmax(logits, dim=-1)
                
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = torch.multinomial(action_probs, 1).item()
                
                log_prob = torch.log(action_probs[0, action] + 1e-8).item()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_observations.append(obs['image'])
            episode_actions.append(action)
            episode_rewards_list.append(reward)
            episode_log_probs.append(log_prob)
            
            # ConSpec intrinsic reward
            intrinsic_reward = conspec.compute_intrinsic_reward(
                [obs], [action], [reward], [done]
            )
            episode_intrinsic_rewards.append(intrinsic_reward.item())
            
            conspec.train_step([obs], [action], [reward], [done])
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        # Update policy
        if len(episode_observations) > 0:
            total_rewards = np.array(episode_rewards_list) + np.array(episode_intrinsic_rewards)
            trainer.update(episode_observations, episode_actions, total_rewards,
                          [done] * len(episode_actions), episode_log_probs)
        
        episode_rewards.append(episode_reward)
        if episode_reward > 0:
            success_count += 1
        
        # Log to wandb
        if wandb_enabled and wandb_run:
            recent_success = np.mean([1 if r > 0 else 0 for r in episode_rewards[-50:]]) if len(episode_rewards) >= 50 else success_count / (episode + 1)
            wandb.log({
                'episode': episode,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'success_rate': recent_success,
                'avg_intrinsic_reward': np.mean(episode_intrinsic_rewards),
                'epsilon': epsilon,
                'memory_success': len(conspec.memory.success_memory),
                'memory_failure': len(conspec.memory.failure_memory)
            })
        
        # Print progress
        if (episode + 1) % 20 == 0:
            recent_success = success_count / (episode + 1)
            avg_intrinsic = np.mean(episode_intrinsic_rewards)
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Success Rate: {recent_success:.3f}, "
                  f"Reward: {episode_reward:.1f}, "
                  f"Intrinsic: {avg_intrinsic:.4f}, "
                  f"Memory: {len(conspec.memory.success_memory)}S/{len(conspec.memory.failure_memory)}F")
    
    env.close()
    
    # Final summary
    print()
    print("="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total episodes: {num_episodes}")
    print(f"Success rate: {success_count / num_episodes:.3f}")
    print(f"Successful episodes: {success_count}")
    print(f"Failed episodes: {num_episodes - success_count}")
    print(f"ConSpec memory - Success: {len(conspec.memory.success_memory)}, "
          f"Failure: {len(conspec.memory.failure_memory)}")
    
    if wandb_enabled and wandb_run:
        print(f"\n📊 View results at: {wandb.run.url}")
        wandb.finish()
    
    return {
        'success_rate': success_count / num_episodes,
        'episode_rewards': episode_rewards
    }


if __name__ == "__main__":
    print("\n🚀 Starting Quick ConSpec Test")
    print("="*60)
    
    if not WANDB_AVAILABLE:
        print("\n⚠️  To enable wandb tracking, run:")
        print("   wandb login")
        print("   (Get API key from: https://wandb.ai/authorize)")
        print("\nContinuing with local training only...")
        input("\nPress Enter to continue...")
    
    results = quick_test(num_episodes=100)
    
    print(f"\n✅ Test complete! Success rate: {results['success_rate']:.2%}")
