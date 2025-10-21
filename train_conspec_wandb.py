"""
Train ConSpec on all three environments with Weights & Biases tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os
import pickle
import argparse
from key_door_goal_base import create_key_door_goal_env
from conspec_minigrid import ConSpec

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Install with: pip install wandb")


class SimplePolicy(nn.Module):
    """Simple policy network for the agent"""
    
    def __init__(self, obs_shape, num_actions, hidden_size=64):
        super(SimplePolicy, self).__init__()
        
        # CNN encoder
        self.conv_layers = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate conv output size
        conv_output_size = 64 * 4 * 4
        
        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, obs):
        """Forward pass"""
        # Convert to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
            
        # Add batch dimension if needed
        if len(obs.shape) == 3:  # (height, width, channels)
            obs = obs.unsqueeze(0)
            
        # Permute to (batch, channels, height, width)
        obs = obs.permute(0, 3, 1, 2)
        
        # Apply conv layers
        conv_out = self.conv_layers(obs)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        
        # Get policy and value
        logits = self.policy_head(conv_out)
        value = self.value_head(conv_out)
        
        return logits, value


class PPOTrainer:
    """Simple PPO trainer for the agent"""
    
    def __init__(self, policy, device, lr=3e-4, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        self.policy = policy
        self.device = device
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
    def update(self, observations, actions, rewards, dones, old_log_probs):
        """Update the policy using PPO"""
        # Convert to tensors
        observations = np.array(observations)
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Get current policy outputs
        logits, values = self.policy(observations)
        
        # Compute action probabilities
        action_probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get log probabilities for taken actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute policy loss (PPO)
        ratio = torch.exp(action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()
        
        # Compute value loss
        value_loss = nn.MSELoss()(values.squeeze(), rewards)
        
        # Compute entropy loss
        entropy = -(action_probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }


def train_conspec_on_environment(env_type, num_episodes=2000, device='cpu', 
                                 save_checkpoints=True, use_wandb=True, wandb_run=None):
    """
    Train ConSpec on a specific environment with wandb tracking
    """
    print(f"\n{'='*60}")
    print(f"Training ConSpec on {env_type.upper()} Environment")
    print(f"{'='*60}")
    
    # Create environment
    env = create_key_door_goal_env(env_type, size=10)
    obs_shape = (env.height, env.width, 3)
    num_actions = env.action_space.n
    
    # Create policy and ConSpec
    policy = SimplePolicy(obs_shape, num_actions).to(device)
    conspec = ConSpec(obs_shape, num_actions, device, 
                     intrinsic_reward_scale=0.5, learning_rate=1e-3)
    
    # Create trainer with adjusted hyperparameters
    trainer = PPOTrainer(policy, device, lr=3e-4, clip_ratio=0.2, 
                        value_coef=0.5, entropy_coef=0.05)
    
    # Model saving variables
    best_success_rate = 0.0
    best_policy_state = None
    best_conspec_state = None
    convergence_episodes = 0
    convergence_threshold = 0.6  # Consider converged when success rate > 60%
    patience = 30  # Episodes to wait after convergence before saving
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    intrinsic_rewards_history = []
    
    # Detailed trajectory analysis
    successful_trajectories = []
    failed_trajectories = []
    
    print(f"Environment: {env.__class__.__name__}")
    print(f"Grid size: {env.width}x{env.height}")
    print(f"Mission: {env.mission}")
    print(f"Number of actions: {num_actions}")
    print()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Episode data
        episode_observations = []
        episode_actions = []
        episode_rewards_list = []
        episode_intrinsic_rewards = []
        episode_log_probs = []
        
        # Epsilon-greedy exploration
        epsilon = max(0.1, 1.0 - episode / (num_episodes * 0.7))  # Decay epsilon
        
        while not done and episode_length < 200:  # Max 200 steps
            # Get action from policy
            with torch.no_grad():
                logits, value = policy(obs['image'])
                action_probs = torch.softmax(logits, dim=-1)
                
                # Epsilon-greedy exploration
                if np.random.random() < epsilon:
                    action = env.action_space.sample()  # Random action
                else:
                    action = torch.multinomial(action_probs, 1).item()
                
                log_prob = torch.log(action_probs[0, action] + 1e-8).item()
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store episode data
            episode_observations.append(obs['image'])
            episode_actions.append(action)
            episode_rewards_list.append(reward)
            episode_log_probs.append(log_prob)
            
            # Compute intrinsic reward with ConSpec
            intrinsic_reward = conspec.compute_intrinsic_reward(
                [obs], [action], [reward], [done]
            )
            episode_intrinsic_rewards.append(intrinsic_reward.item())
            
            # Update ConSpec
            conspec.train_step([obs], [action], [reward], [done])
            
            # Update for next step
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        # Update policy at end of episode
        if len(episode_observations) > 0:
            # Add intrinsic rewards to extrinsic rewards
            total_rewards = np.array(episode_rewards_list) + np.array(episode_intrinsic_rewards)
            losses = trainer.update(episode_observations, episode_actions, total_rewards, 
                                   [done] * len(episode_actions), episode_log_probs)
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_rate.append(1 if episode_reward > 0 else 0)
        intrinsic_rewards_history.extend(episode_intrinsic_rewards)
        
        # Store detailed trajectory for analysis
        trajectory = {
            'observations': episode_observations,
            'actions': episode_actions,
            'rewards': episode_rewards_list,
            'intrinsic_rewards': episode_intrinsic_rewards,
            'total_reward': episode_reward,
            'length': episode_length,
            'success': episode_reward > 0
        }
        
        if episode_reward > 0:
            successful_trajectories.append(trajectory)
        else:
            failed_trajectories.append(trajectory)
        
        # Calculate recent metrics
        recent_success_rate = np.mean(success_rate[-50:]) if len(success_rate) >= 50 else np.mean(success_rate)
        recent_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        recent_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths)
        recent_intrinsic = np.mean(episode_intrinsic_rewards)
        
        # Log to wandb
        if use_wandb and wandb_run is not None:
            wandb.log({
                f"{env_type}/episode_reward": episode_reward,
                f"{env_type}/episode_length": episode_length,
                f"{env_type}/success": 1 if episode_reward > 0 else 0,
                f"{env_type}/success_rate": recent_success_rate,
                f"{env_type}/avg_intrinsic_reward": recent_intrinsic,
                f"{env_type}/epsilon": epsilon,
                f"{env_type}/num_successful_trajectories": len(successful_trajectories),
                f"{env_type}/num_failed_trajectories": len(failed_trajectories),
                f"{env_type}/policy_loss": losses.get('policy_loss', 0) if len(episode_observations) > 0 else 0,
                f"{env_type}/value_loss": losses.get('value_loss', 0) if len(episode_observations) > 0 else 0,
                "episode": episode
            })
        
        # Check for convergence and save best models
        if len(success_rate) >= 50:
            # Update best model if current is better
            if recent_success_rate > best_success_rate:
                best_success_rate = recent_success_rate
                best_policy_state = policy.state_dict().copy()
                best_conspec_state = {
                    'encoder': conspec.encoder.state_dict(),
                    'prototype_net': conspec.prototype_net.state_dict(),
                    'memory': {
                        'success_memory': list(conspec.memory.success_memory),
                        'failure_memory': list(conspec.memory.failure_memory)
                    }
                }
                convergence_episodes = 0
                
                if use_wandb and wandb_run is not None:
                    wandb.log({
                        f"{env_type}/best_success_rate": best_success_rate,
                        "episode": episode
                    })
            else:
                convergence_episodes += 1
            
            # Check if converged and save models
            if (recent_success_rate >= convergence_threshold and 
                convergence_episodes >= patience and 
                save_checkpoints and 
                best_policy_state is not None):
                
                print(f"\n🎉 POLICY CONVERGED! Saving optimal models...")
                print(f"Success rate: {recent_success_rate:.3f} (threshold: {convergence_threshold})")
                
                # Create checkpoints directory
                os.makedirs('checkpoints', exist_ok=True)
                
                # Save policy model
                torch.save({
                    'model_state_dict': best_policy_state,
                    'obs_shape': obs_shape,
                    'num_actions': num_actions,
                    'success_rate': best_success_rate,
                    'episode': episode,
                    'env_type': env_type
                }, f'checkpoints/optimal_policy_{env_type}_env.pth')
                
                # Save ConSpec model
                torch.save({
                    'encoder_state_dict': best_conspec_state['encoder'],
                    'prototype_net_state_dict': best_conspec_state['prototype_net'],
                    'obs_shape': obs_shape,
                    'num_actions': num_actions,
                    'intrinsic_reward_scale': conspec.intrinsic_reward_scale,
                    'num_prototypes': conspec.num_prototypes,
                    'success_rate': best_success_rate,
                    'episode': episode,
                    'env_type': env_type
                }, f'checkpoints/optimal_conspec_{env_type}_env.pth')
                
                # Save memory separately
                with open(f'checkpoints/conspec_memory_{env_type}_env.pkl', 'wb') as f:
                    pickle.dump(best_conspec_state['memory'], f)
                
                print(f"✅ Models saved to checkpoints/ directory:")
                print(f"   - optimal_policy_{env_type}_env.pth")
                print(f"   - optimal_conspec_{env_type}_env.pth") 
                print(f"   - conspec_memory_{env_type}_env.pkl")
                print(f"   Best success rate: {best_success_rate:.3f}")
                
                # Log artifact to wandb
                if use_wandb and wandb_run is not None:
                    artifact = wandb.Artifact(f'model_{env_type}', type='model')
                    artifact.add_file(f'checkpoints/optimal_policy_{env_type}_env.pth')
                    artifact.add_file(f'checkpoints/optimal_conspec_{env_type}_env.pth')
                    wandb.log_artifact(artifact)
                
                # Stop training early if converged
                break
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}: "
                  f"Avg Reward: {recent_reward:.2f}, "
                  f"Avg Length: {recent_length:.1f}, "
                  f"Success Rate: {recent_success_rate:.2f}, "
                  f"Avg Intrinsic: {recent_intrinsic:.4f}, "
                  f"Epsilon: {epsilon:.3f}")
            print(f"  Memory - Success: {len(conspec.memory.success_memory)}, "
                  f"Failure: {len(conspec.memory.failure_memory)}")
            print(f"  Best Success Rate: {best_success_rate:.3f}")
    
    env.close()
    
    return {
        'env_type': env_type,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_rate,
        'intrinsic_rewards': intrinsic_rewards_history,
        'successful_trajectories': successful_trajectories,
        'failed_trajectories': failed_trajectories,
        'conspec': conspec,
        'best_success_rate': best_success_rate,
        'converged': best_success_rate >= convergence_threshold
    }


def main():
    """Main training function for all environments"""
    parser = argparse.ArgumentParser(description='Train ConSpec on key-door-goal environments')
    parser.add_argument('--env', type=str, default='all', choices=['all', 'single', 'double', 'triple'],
                       help='Which environment to train on')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='schema-learning',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default='samieima',
                       help='Wandb entity (username or team)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    print("ConSpec Training with Weights & Biases")
    print("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    wandb_run = None
    
    if use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                'num_episodes': args.episodes,
                'device': str(device),
                'intrinsic_reward_scale': 0.5,
                'learning_rate': 3e-4,
                'convergence_threshold': 0.6,
                'architecture': 'PPO + ConSpec',
                'env_size': '10x10'
            },
            name=f'conspec_training_{time.strftime("%Y%m%d_%H%M%S")}',
            tags=['conspec', 'key-door-goal']
        )
        print(f"Wandb initialized: {wandb.run.url}")
    else:
        print("Wandb disabled or not available")
    
    # Determine which environments to train on
    if args.env == 'all':
        env_types = ['single', 'double', 'triple']
    else:
        env_types = [args.env]
    
    all_results = {}
    
    # Train on selected environments
    for env_type in env_types:
        results = train_conspec_on_environment(
            env_type=env_type,
            num_episodes=args.episodes,
            device=device,
            save_checkpoints=True,
            use_wandb=use_wandb,
            wandb_run=wandb_run
        )
        
        all_results[env_type] = results
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for env_type in env_types:
        if env_type in all_results:
            results = all_results[env_type]
            final_success = np.mean(results['success_rate'][-100:]) if len(results['success_rate']) >= 100 else np.mean(results['success_rate'])
            converged = results.get('converged', False)
            
            print(f"\n{env_type.upper()} Environment:")
            print(f"  Final success rate: {final_success:.2f}")
            print(f"  Best success rate: {results['best_success_rate']:.2f}")
            print(f"  Total episodes: {len(results['episode_rewards'])}")
            print(f"  Converged: {'✅' if converged else '❌'}")
            print(f"  Successful trajectories: {len(results['successful_trajectories'])}")
            print(f"  Failed trajectories: {len(results['failed_trajectories'])}")
    
    print(f"\n📁 All checkpoints saved to checkpoints/ directory!")
    
    if use_wandb:
        # Log final summary to wandb
        summary_table = wandb.Table(
            columns=["Environment", "Final Success Rate", "Best Success Rate", "Converged", "Episodes"],
            data=[
                [env_type.upper(), 
                 np.mean(all_results[env_type]['success_rate'][-100:]) if len(all_results[env_type]['success_rate']) >= 100 else np.mean(all_results[env_type]['success_rate']),
                 all_results[env_type]['best_success_rate'],
                 all_results[env_type].get('converged', False),
                 len(all_results[env_type]['episode_rewards'])]
                for env_type in env_types if env_type in all_results
            ]
        )
        wandb.log({"training_summary": summary_table})
        
        print(f"📊 Results logged to wandb: {wandb.run.url}")
        wandb.finish()
    
    return all_results


if __name__ == "__main__":
    results = main()
