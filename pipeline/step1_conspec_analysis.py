"""
Step 1: ConSpec Training and Analysis

Train ConSpec on an environment and record all metrics:
- Intrinsic rewards
- Prototype features
- Cosine similarities
- Critical states
- Episode trajectories
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from environments.key_door_goal_base import create_key_door_goal_env
from methods.conspec_minigrid import ConSpec, EncoderConSpec


class ConSpecAnalyzer:
    """
    Train ConSpec and analyze learned prototypes and critical features
    """
    
    def __init__(
        self,
        env_type: str,
        num_prototypes: int = 5,
        hidden_dim: int = 128,
        lstm_hidden_size: int = 128,
        intrinsic_reward_scale: float = 0.5,
        seed: int = 0
    ):
        """
        Initialize ConSpec analyzer
        
        Args:
            env_type: Environment type ('single', 'double', 'triple')
            num_prototypes: Number of prototypes for ConSpec
            hidden_dim: Hidden dimension for encoder
            lstm_hidden_size: LSTM hidden size
            intrinsic_reward_scale: Scale for intrinsic rewards
            seed: Random seed
        """
        self.env_type = env_type
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create environment
        self.env = create_key_door_goal_env(env_type, size=10)
        
        # Create ConSpec components
        obs_shape = self.env.observation_space['image'].shape
        input_channels = obs_shape[2]
        
        self.encoder = EncoderConSpec(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            lstm_hidden_size=lstm_hidden_size
        )
        
        self.conspec = ConSpec(
            encoder=self.encoder,
            num_prototypes=num_prototypes,
            intrinsic_reward_scale=intrinsic_reward_scale
        )
        
        # Storage for metrics
        self.all_metrics = {
            'episode_rewards': [],
            'intrinsic_rewards': [],
            'cosine_similarities': [],
            'prototype_usage': [],
            'success_episodes': [],
            'trajectories': []
        }
        
        print(f"Initialized ConSpecAnalyzer for {env_type} environment")
        print(f"  - Prototypes: {num_prototypes}")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - LSTM hidden: {lstm_hidden_size}")
    
    def train_episode(
        self,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        epsilon: float = 0.1,
        max_steps: int = 400
    ) -> Dict:
        """
        Train one episode and collect metrics
        
        Returns:
            Dictionary with episode metrics
        """
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_intrinsic_reward = 0
        done = False
        step = 0
        
        observations = []
        actions = []
        rewards = []
        states_info = []
        
        while not done and step < max_steps:
            obs_image = obs['image']
            observations.append(obs_image)
            
            # Record state info (for expert trajectory collection)
            state_info = {
                'has_key': obs.get('carrying') is not None,
                'step': step,
                'agent_pos': self.env.agent_pos if hasattr(self.env, 'agent_pos') else None
            }
            states_info.append(state_info)
            
            # Action selection
            if np.random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_image).unsqueeze(0)
                    features, _ = self.encoder(obs_tensor)
                    q_values = policy(features)
                    action = q_values.argmax().item()
            
            actions.append(action)
            
            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            episode_reward += reward
            step += 1
        
        # Compute intrinsic reward and cosine similarities
        if len(observations) > 0:
            intrinsic_reward = self.conspec.compute_intrinsic_reward(observations)
            episode_intrinsic_reward = float(intrinsic_reward)
            
            # Get cosine similarities to prototypes
            obs_tensor = torch.FloatTensor(np.array(observations))
            features, _ = self.encoder(obs_tensor)
            
            # Compute similarities
            similarities = []
            for i in range(len(features)):
                feature = features[i:i+1]
                sim = torch.cosine_similarity(
                    feature.unsqueeze(1),
                    self.conspec.memory.prototypes.unsqueeze(0),
                    dim=2
                )
                similarities.append(sim.detach().cpu().numpy())
            
            similarities = np.array(similarities)
            
            # Update ConSpec memory
            self.conspec.update_memory(observations, np.array(rewards))
        else:
            similarities = None
        
        # Store metrics
        metrics = {
            'episode_reward': episode_reward,
            'intrinsic_reward': episode_intrinsic_reward,
            'success': episode_reward > 0,
            'num_steps': step,
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'states_info': states_info,
            'cosine_similarities': similarities
        }
        
        return metrics
    
    def train(
        self,
        num_episodes: int = 5000,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        verbose: bool = True
    ) -> Dict:
        """
        Train ConSpec for multiple episodes
        
        Returns:
            Dictionary with all collected metrics
        """
        # Create simple policy
        feature_dim = self.encoder.lstm_hidden_size
        policy = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.env.action_space.n)
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        
        epsilon = epsilon_start
        best_reward = -float('inf')
        best_episode_idx = -1
        
        iterator = tqdm(range(num_episodes), desc='Training ConSpec') if verbose else range(num_episodes)
        
        for episode in iterator:
            # Train episode
            metrics = self.train_episode(policy, optimizer, epsilon, max_steps=400)
            
            # Store metrics
            self.all_metrics['episode_rewards'].append(metrics['episode_reward'])
            self.all_metrics['intrinsic_rewards'].append(metrics['intrinsic_reward'])
            
            if metrics['success']:
                self.all_metrics['success_episodes'].append({
                    'episode': episode,
                    'reward': metrics['episode_reward'],
                    'steps': metrics['num_steps'],
                    'trajectory': {
                        'observations': metrics['observations'],
                        'actions': metrics['actions'],
                        'rewards': metrics['rewards'],
                        'states_info': metrics['states_info'],
                        'cosine_similarities': metrics['cosine_similarities']
                    }
                })
                
                if metrics['episode_reward'] > best_reward:
                    best_reward = metrics['episode_reward']
                    best_episode_idx = episode
            
            # Update epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Logging
            if verbose and episode % 100 == 0:
                recent_rewards = self.all_metrics['episode_rewards'][-100:]
                success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
                if hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({
                        'reward': f"{np.mean(recent_rewards):.2f}",
                        'success': f"{success_rate:.2%}",
                        'epsilon': f"{epsilon:.3f}"
                    })
        
        # Analyze prototypes
        self._analyze_prototypes()
        
        print(f"\nTraining completed!")
        print(f"  - Total episodes: {num_episodes}")
        print(f"  - Successful episodes: {len(self.all_metrics['success_episodes'])}")
        print(f"  - Best episode: {best_episode_idx} (reward: {best_reward:.2f})")
        print(f"  - Final success rate: {success_rate:.2%}")
        
        return {
            'policy': policy,
            'encoder': self.encoder,
            'conspec': self.conspec,
            'best_episode_idx': best_episode_idx,
            'metrics': self.all_metrics
        }
    
    def _analyze_prototypes(self):
        """Analyze learned prototypes"""
        prototypes = self.conspec.memory.prototypes.detach().cpu().numpy()
        
        # Compute prototype statistics
        prototype_norms = np.linalg.norm(prototypes, axis=1)
        prototype_distances = np.zeros((len(prototypes), len(prototypes)))
        
        for i in range(len(prototypes)):
            for j in range(len(prototypes)):
                if i != j:
                    prototype_distances[i, j] = np.linalg.norm(prototypes[i] - prototypes[j])
        
        self.all_metrics['prototype_analysis'] = {
            'prototypes': prototypes,
            'norms': prototype_norms,
            'pairwise_distances': prototype_distances
        }
    
    def visualize_metrics(self, save_dir: str):
        """Visualize training metrics"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.all_metrics['episode_rewards'], alpha=0.3)
        window = 100
        if len(self.all_metrics['episode_rewards']) >= window:
            rolling_mean = np.convolve(
                self.all_metrics['episode_rewards'],
                np.ones(window)/window,
                mode='valid'
            )
            axes[0, 0].plot(range(window-1, len(self.all_metrics['episode_rewards'])),
                           rolling_mean, 'r-', linewidth=2, label=f'Rolling mean ({window})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Intrinsic rewards
        axes[0, 1].plot(self.all_metrics['intrinsic_rewards'], alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Intrinsic Reward')
        axes[0, 1].set_title('ConSpec Intrinsic Rewards')
        axes[0, 1].grid(alpha=0.3)
        
        # Success rate over time
        window = 100
        success_indicators = [1 if r > 0 else 0 for r in self.all_metrics['episode_rewards']]
        if len(success_indicators) >= window:
            success_rate = np.convolve(success_indicators, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(success_indicators)), success_rate, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title(f'Success Rate (rolling {window})')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(alpha=0.3)
        
        # Prototype distances
        if 'prototype_analysis' in self.all_metrics:
            distances = self.all_metrics['prototype_analysis']['pairwise_distances']
            im = axes[1, 1].imshow(distances, cmap='viridis')
            axes[1, 1].set_xlabel('Prototype')
            axes[1, 1].set_ylabel('Prototype')
            axes[1, 1].set_title('Prototype Pairwise Distances')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(save_path / f'{self.env_type}_conspec_metrics.png', dpi=150, bbox_inches='tight')
        print(f"Saved metrics visualization: {save_path / f'{self.env_type}_conspec_metrics.png'}")
        plt.close()
    
    def save_results(self, output_dir: str):
        """Save all results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = output_path / f'{self.env_type}_metrics_{timestamp}.pkl'
        with open(metrics_file, 'wb') as f:
            pickle.dump(self.all_metrics, f)
        print(f"Saved metrics: {metrics_file}")
        
        # Save ConSpec model
        conspec_file = output_path / f'{self.env_type}_conspec_{timestamp}.pt'
        torch.save({
            'encoder': self.encoder.state_dict(),
            'prototypes': self.conspec.memory.prototypes,
            'num_prototypes': self.conspec.num_prototypes
        }, conspec_file)
        print(f"Saved ConSpec model: {conspec_file}")
        
        # Save summary
        summary = {
            'env_type': self.env_type,
            'num_episodes': len(self.all_metrics['episode_rewards']),
            'num_successful': len(self.all_metrics['success_episodes']),
            'mean_reward': np.mean(self.all_metrics['episode_rewards']),
            'mean_intrinsic_reward': np.mean(self.all_metrics['intrinsic_rewards']),
            'timestamp': timestamp
        }
        
        summary_file = output_path / f'{self.env_type}_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {summary_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 1: ConSpec Training and Analysis')
    parser.add_argument('--env', type=str, required=True, choices=['single', 'double', 'triple'],
                       help='Environment type')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--num_prototypes', type=int, default=5,
                       help='Number of prototypes')
    parser.add_argument('--output_dir', type=str, default='pipeline_results/step1',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ConSpecAnalyzer(
        env_type=args.env,
        num_prototypes=args.num_prototypes,
        seed=args.seed
    )
    
    # Train
    results = analyzer.train(num_episodes=args.episodes, verbose=True)
    
    # Save results
    analyzer.save_results(args.output_dir)
    analyzer.visualize_metrics(args.output_dir)
    
    print("\nStep 1 completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

