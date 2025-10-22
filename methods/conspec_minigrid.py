"""
ConSpec (Contrastive Retrospection) implementation for MiniGrid environments
Adapted from the original ConSpec implementation to work with our key-door-goal environments.

Based on: https://github.com/sunchipsster1/ConSpec
Paper: https://arxiv.org/pdf/2210.05845.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ConSpecEncoder(nn.Module):
    """Encoder for ConSpec that processes MiniGrid observations"""
    
    def __init__(self, obs_shape, num_actions, hidden_size=64):
        super(ConSpecEncoder, self).__init__()
        
        # For MiniGrid, obs_shape is typically (width, height, channels)
        # We'll use a simple CNN encoder
        self.conv_layers = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Fixed size output
        )
        
        # Calculate the size after conv layers
        conv_output_size = 64 * 4 * 4
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(conv_output_size, hidden_size, batch_first=True)
        
        # Output layers
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
    def forward(self, obs, hidden_state=None):
        """
        Forward pass through the encoder
        
        Args:
            obs: Observation tensor of shape (batch_size, seq_len, height, width, channels) or (height, width, channels)
            hidden_state: Previous hidden state for LSTM
            
        Returns:
            encoded_features: Encoded features
            new_hidden_state: New hidden state
        """
        # Handle single observation (add batch and sequence dimensions)
        if len(obs.shape) == 3:  # (height, width, channels)
            obs = obs.unsqueeze(0).unsqueeze(0)  # (1, 1, height, width, channels)
        elif len(obs.shape) == 4:  # (batch, height, width, channels)
            obs = obs.unsqueeze(1)  # (batch, 1, height, width, channels)
        
        batch_size, seq_len = obs.shape[:2]
        
        # Reshape for conv layers: (batch_size * seq_len, height, width, channels)
        obs_reshaped = obs.reshape(-1, *obs.shape[2:])
        
        # Permute to (batch_size * seq_len, channels, height, width)
        obs_reshaped = obs_reshaped.permute(0, 3, 1, 2)
        
        # Apply conv layers
        conv_out = self.conv_layers(obs_reshaped)
        
        # Reshape back: (batch_size, seq_len, conv_output_size)
        conv_out = conv_out.reshape(batch_size, seq_len, -1)
        
        # Apply LSTM
        if hidden_state is None:
            lstm_out, new_hidden_state = self.lstm(conv_out)
        else:
            lstm_out, new_hidden_state = self.lstm(conv_out, hidden_state)
        
        return lstm_out, new_hidden_state


class PrototypeNetwork(nn.Module):
    """Network for learning prototypes in ConSpec"""
    
    def __init__(self, input_size, hidden_size, num_prototypes, device):
        super(PrototypeNetwork, self).__init__()
        
        self.num_prototypes = num_prototypes
        self.device = device
        
        # Projection network g_theta from the paper
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Initialize prototypes randomly
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_size))
        
    def forward(self, features):
        """
        Forward pass through prototype network
        
        Args:
            features: Input features of shape (batch_size, feature_dim)
            
        Returns:
            projected_features: Projected features
            similarities: Similarities to prototypes
        """
        # Project features
        projected_features = self.projection(features)
        
        # Compute similarities to prototypes
        similarities = F.cosine_similarity(
            projected_features.unsqueeze(1), 
            self.prototypes.unsqueeze(0), 
            dim=2
        )
        
        return projected_features, similarities


class ConSpecMemory:
    """Memory buffer for storing success and failure trajectories"""
    
    def __init__(self, max_size=1000):
        self.success_memory = deque(maxlen=max_size)
        self.failure_memory = deque(maxlen=max_size)
        
    def add_success(self, trajectory):
        """Add a successful trajectory to memory"""
        self.success_memory.append(trajectory)
        
    def add_failure(self, trajectory):
        """Add a failed trajectory to memory"""
        self.failure_memory.append(trajectory)
        
    def sample_batch(self, batch_size):
        """Sample a batch of trajectories from both memories"""
        batch = []
        
        # Sample from success memory
        if len(self.success_memory) > 0:
            success_samples = random.sample(
                self.success_memory, 
                min(batch_size // 2, len(self.success_memory))
            )
            batch.extend(success_samples)
            
        # Sample from failure memory
        if len(self.failure_memory) > 0:
            failure_samples = random.sample(
                self.failure_memory,
                min(batch_size // 2, len(self.failure_memory))
            )
            batch.extend(failure_samples)
            
        return batch


class ConSpec(nn.Module):
    """
    ConSpec (Contrastive Retrospection) implementation
    
    This class implements the ConSpec algorithm as described in the paper:
    "Contrastive Retrospection: Learning from Success and Failure"
    """
    
    def __init__(self, obs_shape, num_actions, device, 
                 hidden_size=64, num_prototypes=10, 
                 intrinsic_reward_scale=0.1, learning_rate=1e-3):
        super(ConSpec, self).__init__()
        
        self.device = device
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.num_prototypes = num_prototypes
        
        # Initialize encoder
        self.encoder = ConSpecEncoder(obs_shape, num_actions, hidden_size)
        self.encoder.to(device)
        
        # Initialize prototype network
        self.prototype_net = PrototypeNetwork(
            hidden_size, hidden_size * 2, num_prototypes, device
        )
        self.prototype_net.to(device)
        
        # Initialize memory
        self.memory = ConSpecMemory()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.prototype_net.parameters()),
            lr=learning_rate
        )
        
        # Initialize hidden state
        self.hidden_state = None
        
    def encode_observations(self, observations):
        """Encode observations using the encoder"""
        # Handle MiniGrid observation format (dict with 'image' key)
        if isinstance(observations, dict):
            if 'image' in observations:
                observations = observations['image']
            else:
                raise ValueError("Cannot extract image from observation dict")
        elif isinstance(observations, list):
            # If it's a batch of dicts, extract images
            if len(observations) > 0 and isinstance(observations[0], dict):
                observations = np.array([obs['image'] for obs in observations], dtype=np.float32)
            else:
                observations = np.array(observations, dtype=np.float32)
        
        # Convert to tensor if needed
        if not isinstance(observations, torch.Tensor):
            # Ensure we have a proper numpy array
            if isinstance(observations, np.ndarray) and observations.dtype == np.object_:
                # Handle object arrays that might contain dicts
                if len(observations) > 0 and isinstance(observations[0], dict):
                    observations = np.array([obs['image'] for obs in observations], dtype=np.float32)
                else:
                    observations = np.array([obs for obs in observations], dtype=np.float32)
            observations = torch.FloatTensor(observations).to(self.device)
            
        # Add sequence dimension if needed
        if len(observations.shape) == 4:  # (batch, height, width, channels)
            observations = observations.unsqueeze(1)  # (batch, 1, height, width, channels)
            
        # Encode
        encoded_features, self.hidden_state = self.encoder(observations, self.hidden_state)
        
        return encoded_features
        
    def compute_intrinsic_reward(self, observations, actions, rewards, dones):
        """
        Compute intrinsic reward using ConSpec
        
        Args:
            observations: Batch of observations (list of dicts or numpy array)
            actions: Batch of actions
            rewards: Batch of extrinsic rewards
            dones: Batch of done flags
            
        Returns:
            intrinsic_rewards: Computed intrinsic rewards
        """
        # Handle batch size
        if isinstance(observations, list):
            batch_size = len(observations)
        else:
            batch_size = observations.shape[0]
            
        intrinsic_rewards = torch.zeros(batch_size, 1).to(self.device)
        
        # Reset hidden state for each batch to avoid size mismatches
        old_hidden_state = self.hidden_state
        self.hidden_state = None
        
        # Encode observations
        encoded_features = self.encode_observations(observations)
        
        # Restore hidden state
        self.hidden_state = old_hidden_state
        
        # Get the last encoded feature for each trajectory
        last_features = encoded_features[:, -1, :]  # (batch_size, hidden_size)
        
        # Project features and compute similarities
        projected_features, similarities = self.prototype_net(last_features)
        
        # Compute intrinsic reward based on novelty
        # Higher reward for less similar features (more novel)
        max_similarities = torch.max(similarities, dim=1)[0]
        novelty_reward = 1.0 - max_similarities
        intrinsic_rewards = novelty_reward.unsqueeze(1) * self.intrinsic_reward_scale
        
        return intrinsic_rewards
        
    def update_memory(self, observations, actions, rewards, dones):
        """Update memory with new trajectories"""
        # Handle batch size
        if isinstance(observations, list):
            batch_size = len(observations)
        else:
            batch_size = observations.shape[0]
        
        for i in range(batch_size):
            trajectory = {
                'observations': observations[i],
                'actions': actions[i],
                'rewards': rewards[i],
                'done': dones[i]
            }
            
            # Classify as success or failure based on final reward
            # Handle different reward formats
            if isinstance(rewards, list):
                final_reward = rewards[i]
            elif np.isscalar(rewards):
                final_reward = rewards
            elif len(rewards.shape) == 1:
                final_reward = rewards[i]
            else:
                final_reward = rewards[i, -1]
                
            if final_reward > 0:  # Success
                self.memory.add_success(trajectory)
            else:  # Failure
                self.memory.add_failure(trajectory)
                
    def train_step(self, observations, actions, rewards, dones):
        """
        Perform one training step of ConSpec
        
        Args:
            observations: Batch of observations
            actions: Batch of actions  
            rewards: Batch of rewards
            dones: Batch of done flags
            
        Returns:
            intrinsic_rewards: Computed intrinsic rewards
        """
        # Compute intrinsic rewards
        intrinsic_rewards = self.compute_intrinsic_reward(observations, actions, rewards, dones)
        
        # Update memory
        self.update_memory(observations, actions, rewards, dones)
        
        # Train on memory if we have enough samples
        if len(self.memory.success_memory) + len(self.memory.failure_memory) > 10:
            self._train_on_memory()
            
        return intrinsic_rewards
        
    def _train_on_memory(self):
        """Train the encoder and prototypes on memory"""
        # Sample batch from memory
        memory_batch = self.memory.sample_batch(32)
        
        if len(memory_batch) < 2:
            return
            
        # Prepare data
        success_features = []
        failure_features = []
        
        for trajectory in memory_batch:
            # Encode the trajectory
            obs = trajectory['observations']
            if isinstance(obs, dict):
                obs = obs['image']
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            encoded, _ = self.encoder(obs)
            last_feature = encoded[:, -1, :]
            
            # Handle different reward formats
            rewards = trajectory['rewards']
            if np.isscalar(rewards):
                final_reward = rewards
            elif len(rewards.shape) == 1:
                final_reward = rewards[-1]
            else:
                final_reward = rewards[-1]
                
            if final_reward > 0:  # Success
                success_features.append(last_feature)
            else:  # Failure
                failure_features.append(last_feature)
                
        if len(success_features) == 0 or len(failure_features) == 0:
            return
            
        # Stack features
        success_features = torch.cat(success_features, dim=0)
        failure_features = torch.cat(failure_features, dim=0)
        
        # Compute contrastive loss
        loss = self._compute_contrastive_loss(success_features, failure_features)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def _compute_contrastive_loss(self, success_features, failure_features):
        """Compute contrastive loss between success and failure features"""
        # Project features
        success_projected, success_similarities = self.prototype_net(success_features)
        failure_projected, failure_similarities = self.prototype_net(failure_features)
        
        # Contrastive loss: success features should be similar to prototypes,
        # failure features should be dissimilar
        success_loss = -torch.mean(torch.max(success_similarities, dim=1)[0])
        failure_loss = torch.mean(torch.max(failure_similarities, dim=1)[0])
        
        total_loss = success_loss + failure_loss
        return total_loss
        
    def reset_hidden_state(self):
        """Reset the hidden state of the encoder"""
        self.hidden_state = None


def create_conspec(obs_shape, num_actions, device, **kwargs):
    """
    Factory function to create a ConSpec instance
    
    Args:
        obs_shape: Shape of observations (height, width, channels)
        num_actions: Number of actions
        device: PyTorch device
        **kwargs: Additional arguments for ConSpec
        
    Returns:
        ConSpec instance
    """
    return ConSpec(obs_shape, num_actions, device, **kwargs)
