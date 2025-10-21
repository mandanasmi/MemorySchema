"""
Training script for ConSpec on key-door-goal environments
Integrates ConSpec with our MiniGrid environments
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os

# Import our environments and ConSpec
from key_door_goal_base import create_key_door_goal_env
from conspec_minigrid import ConSpec


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
        conv_out = conv_out.view(conv_out.size(0), -1)
        
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


def train_conspec_on_environment(env_type='single', num_episodes=1000, device='cpu'):
    """
    Train ConSpec on a key-door-goal environment
    
    Args:
        env_type: Type of environment ('single', 'double', 'triple')
        num_episodes: Number of training episodes
        device: Device to use for training
    """
    
    print(f"Training ConSpec on {env_type} environment...")
    
    # Create environment
    env = create_key_door_goal_env(env_type, size=10)
    obs_shape = (env.height, env.width, 3)  # RGB channels
    num_actions = env.action_space.n
    
    # Create policy and ConSpec
    policy = SimplePolicy(obs_shape, num_actions).to(device)
    conspec = ConSpec(obs_shape, num_actions, device, 
                     intrinsic_reward_scale=0.1, learning_rate=1e-3)
    
    # Create trainer
    trainer = PPOTrainer(policy, device)
    
    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    success_rate = deque(maxlen=100)
    
    # Training loop
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Episode data
        observations = []
        actions = []
        rewards = []
        dones = []
        old_log_probs = []
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                logits, value = policy(obs)
                action_probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs[0, action]).item()
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store data
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            old_log_probs.append(log_prob)
            
            # Update for next step
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Update ConSpec every few steps
            if len(observations) >= 10:  # Batch size
                obs_batch = np.array(observations[-10:])
                action_batch = np.array(actions[-10:])
                reward_batch = np.array(rewards[-10:])
                done_batch = np.array(dones[-10:])
                
                # Compute intrinsic rewards
                intrinsic_rewards = conspec.train_step(
                    obs_batch, action_batch, reward_batch, done_batch
                )
                
                # Add intrinsic rewards to extrinsic rewards
                total_rewards = reward_batch + intrinsic_rewards.cpu().numpy().flatten()
                rewards[-10:] = total_rewards.tolist()
        
        # Update policy at end of episode
        if len(observations) > 0:
            trainer.update(observations, actions, rewards, dones, old_log_probs)
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_rate.append(1 if episode_reward > 0 else 0)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            avg_success = np.mean(success_rate)
            
            print(f"Episode {episode}: "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.1f}, "
                  f"Success Rate: {avg_success:.2f}")
    
    env.close()
    
    return {
        'episode_rewards': list(episode_rewards),
        'episode_lengths': list(episode_lengths),
        'success_rate': list(success_rate)
    }


def plot_training_results(results, env_type, save_path=None):
    """Plot training results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Episode rewards
    axes[0].plot(results['episode_rewards'])
    axes[0].set_title(f'{env_type.title()} Environment - Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True)
    
    # Episode lengths
    axes[1].plot(results['episode_lengths'])
    axes[1].set_title(f'{env_type.title()} Environment - Episode Lengths')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Length')
    axes[1].grid(True)
    
    # Success rate
    axes[2].plot(results['success_rate'])
    axes[2].set_title(f'{env_type.title()} Environment - Success Rate')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training results saved to {save_path}")
    
    plt.show()


def main():
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train on all three environments
    env_types = ['single', 'double', 'triple']
    all_results = {}
    
    for env_type in env_types:
        print(f"\n{'='*50}")
        print(f"Training on {env_type} environment")
        print(f"{'='*50}")
        
        # Train
        results = train_conspec_on_environment(
            env_type=env_type,
            num_episodes=500,  # Reduced for demo
            device=device
        )
        
        all_results[env_type] = results
        
        # Plot results
        plot_training_results(
            results, 
            env_type, 
            save_path=f'figures/conspec_training_{env_type}.png'
        )
    
    # Save all results
    np.save('conspec_training_results.npy', all_results)
    print("\nAll training results saved to conspec_training_results.npy")


if __name__ == "__main__":
    main()
