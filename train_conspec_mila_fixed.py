"""
Train ConSpec on Mila cluster with checkpoint resumption and hyperparameter search
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import argparse
import json
from datetime import datetime
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
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
            
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
            
        obs = obs.permute(0, 3, 1, 2)
        conv_out = self.conv_layers(obs)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        
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
        observations = np.array(observations)
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        logits, values = self.policy(observations)
        action_probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ratio = torch.exp(action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()
        
        value_loss = nn.MSELoss()(values.squeeze(), rewards)
        entropy = -(action_probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state"""
        self.optimizer.load_state_dict(state_dict)
    
    def state_dict(self):
        """Get optimizer state"""
        return self.optimizer.state_dict()


def save_checkpoint(checkpoint_dir, env_type, num_prototypes, episode, policy, trainer, conspec,
                   training_stats, best_success_rate):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, 
                                  f'checkpoint_{env_type}_proto{num_prototypes}_ep{episode}.pth')
    
    torch.save({
        'episode': episode,
        'env_type': env_type,
        'num_prototypes': num_prototypes,
        'policy_state_dict': policy.state_dict(),
        'trainer_state_dict': trainer.state_dict(),
        'encoder_state_dict': conspec.encoder.state_dict(),
        'prototype_net_state_dict': conspec.prototype_net.state_dict(),
        'training_stats': training_stats,
        'best_success_rate': best_success_rate,
        'memory': {
            'success_memory': list(conspec.memory.success_memory),
            'failure_memory': list(conspec.memory.failure_memory)
        }
    }, checkpoint_path)
    
    # Also save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 
                               f'checkpoint_{env_type}_proto{num_prototypes}_latest.pth')
    torch.save({
        'episode': episode,
        'env_type': env_type,
        'num_prototypes': num_prototypes,
        'policy_state_dict': policy.state_dict(),
        'trainer_state_dict': trainer.state_dict(),
        'encoder_state_dict': conspec.encoder.state_dict(),
        'prototype_net_state_dict': conspec.prototype_net.state_dict(),
        'training_stats': training_stats,
        'best_success_rate': best_success_rate,
        'memory': {
            'success_memory': list(conspec.memory.success_memory),
            'failure_memory': list(conspec.memory.failure_memory)
        }
    }, latest_path)
    
    print(f" Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, policy, trainer, conspec, device):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    print(f"