"""
Hyperparameter Search for ConSpec on Key-Door-Goal Environments

This script performs a comprehensive hyperparameter search for ConSpec
across all three environments using different search strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import yaml
import json
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Any
import random

from environments.key_door_goal_base import create_key_door_goal_env
from methods.conspec_minigrid import ConSpec, EncoderConSpec

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Results will only be logged locally.")


class SimplePolicy(nn.Module):
    """Simple policy network for ConSpec"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        return self.network(x)


class HyperparameterSearcher:
    """Manages hyperparameter search for ConSpec"""
    
    def __init__(self, config_path: str = None):
        """Initialize searcher with configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = self._get_default_config()
        
        # Create results directory
        self.results_dir = Path("hyperparam_search_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.all_results = []
        
    def _get_default_config(self) -> Dict:
        """Get default search configuration"""
        return {
            'environments': ['single', 'double', 'triple'],
            'hyperparameters': {
                'num_prototypes': {'type': 'choice', 'values': [3, 5, 7, 10]},
                'learning_rate': {'type': 'loguniform', 'min': 0.0001, 'max': 0.01},
                'intrinsic_reward_scale': {'type': 'uniform', 'min': 0.1, 'max': 1.0},
                'gamma': {'type': 'choice', 'values': [0.95, 0.98, 0.99]},
                'epsilon_decay': {'type': 'uniform', 'min': 0.99, 'max': 0.999},
                'hidden_dim': {'type': 'choice', 'values': [64, 128, 256]},
                'lstm_hidden_size': {'type': 'choice', 'values': [64, 128, 256]},
                'batch_size': {'type': 'choice', 'values': [32, 64, 128, 256]},
                'memory_size': {'type': 'choice', 'values': [1000, 5000, 10000]},
            },
            'training': {
                'episodes': 5000,
                'max_steps_per_episode': 400,
                'eval_frequency': 500,
                'checkpoint_frequency': 1000,
            },
            'search': {
                'method': 'random',
                'n_trials': 50,
                'optimization_metric': 'mean_reward',
                'optimization_direction': 'maximize',
            },
            'logging': {
                'wandb_project': 'schema-learning',
                'wandb_entity': 'mandanasmi',
                'log_frequency': 100,
                'save_best_only': True,
            }
        }
    
    def sample_hyperparameters(self) -> Dict[str, Any]:
        """Sample a set of hyperparameters from the search space"""
        hyperparams = {}
        
        for name, config in self.config['hyperparameters'].items():
            if config['type'] == 'choice':
                hyperparams[name] = random.choice(config['values'])
            elif config['type'] == 'uniform':
                hyperparams[name] = random.uniform(config['min'], config['max'])
            elif config['type'] == 'loguniform':
                log_min = np.log(config['min'])
                log_max = np.log(config['max'])
                hyperparams[name] = np.exp(random.uniform(log_min, log_max))
        
        return hyperparams
    
    def train_with_hyperparameters(
        self,
        env_type: str,
        hyperparams: Dict[str, Any],
        trial_id: int
    ) -> Dict[str, float]:
        """Train ConSpec with specific hyperparameters"""
        
        print(f"\n{'='*70}")
        print(f"Trial {trial_id} - Environment: {env_type.upper()}")
        print(f"{'='*70}")
        print("Hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
        print()
        
        # Create environment
        env = create_key_door_goal_env(env_type)
        
        # Initialize wandb if available
        run = None
        if WANDB_AVAILABLE:
            run = wandb.init(
                project=self.config['logging']['wandb_project'],
                entity=self.config['logging']['wandb_entity'],
                name=f"{env_type}_trial_{trial_id}",
                config=hyperparams,
                tags=[env_type, 'hyperparam_search'],
                reinit=True
            )
        
        # Create ConSpec components
        obs_shape = env.observation_space['image'].shape
        input_channels = obs_shape[2]
        
        encoder = EncoderConSpec(
            input_channels=input_channels,
            hidden_dim=int(hyperparams['hidden_dim']),
            lstm_hidden_size=int(hyperparams['lstm_hidden_size'])
        )
        
        conspec = ConSpec(
            encoder=encoder,
            num_prototypes=int(hyperparams['num_prototypes']),
            intrinsic_reward_scale=hyperparams['intrinsic_reward_scale']
        )
        
        # Create policy
        feature_dim = int(hyperparams['lstm_hidden_size'])
        policy = SimplePolicy(
            input_dim=feature_dim,
            hidden_dim=int(hyperparams['hidden_dim']),
            output_dim=env.action_space.n
        )
        
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=hyperparams['learning_rate']
        )
        
        # Training parameters
        gamma = hyperparams['gamma']
        epsilon = 1.0
        epsilon_decay = hyperparams['epsilon_decay']
        epsilon_min = 0.01
        
        # Training metrics
        episode_rewards = []
        episode_successes = []
        recent_rewards = []
        
        # Training loop
        for episode in range(self.config['training']['episodes']):
            obs = env.reset()
            episode_reward = 0
            episode_intrinsic_reward = 0
            done = False
            step = 0
            
            observations = []
            actions = []
            rewards = []
            
            while not done and step < self.config['training']['max_steps_per_episode']:
                # Get observation
                obs_image = obs['image']
                observations.append(obs_image)
                
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs_image).unsqueeze(0)
                        features, _ = encoder(obs_tensor)
                        q_values = policy(features)
                        action = q_values.argmax().item()
                
                actions.append(action)
                
                # Take action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                rewards.append(reward)
                episode_reward += reward
                
                step += 1
            
            # Calculate intrinsic reward
            if len(observations) > 0:
                intrinsic_reward = conspec.compute_intrinsic_reward(observations)
                episode_intrinsic_reward = float(intrinsic_reward)
            
            # Update memory
            if len(observations) > 0:
                conspec.update_memory(observations, np.array(rewards))
            
            # Policy update (simple policy gradient)
            if len(observations) > 1:
                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + gamma * G
                    returns.insert(0, G)
                
                returns = torch.FloatTensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
                for i, obs_img in enumerate(observations[:-1]):
                    obs_tensor = torch.FloatTensor(obs_img).unsqueeze(0)
                    features, _ = encoder(obs_tensor)
                    q_values = policy(features)
                    
                    action_probs = torch.softmax(q_values, dim=-1)
                    action_prob = action_probs[0, actions[i]]
                    
                    loss = -torch.log(action_prob + 1e-8) * returns[i]
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Decay epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Track metrics
            success = episode_reward > 0
            episode_rewards.append(episode_reward)
            episode_successes.append(1 if success else 0)
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            
            # Logging
            if episode % self.config['logging']['log_frequency'] == 0:
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0
                success_rate = np.mean(episode_successes[-100:]) if len(episode_successes) >= 100 else np.mean(episode_successes)
                
                print(f"Episode {episode:4d}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Mean(100)={mean_reward:.2f}, "
                      f"Success={success_rate:.2%}, "
                      f"Epsilon={epsilon:.3f}")
                
                if WANDB_AVAILABLE and run:
                    wandb.log({
                        'episode': episode,
                        'reward': episode_reward,
                        'mean_reward_100': mean_reward,
                        'success_rate': success_rate,
                        'epsilon': epsilon,
                        'intrinsic_reward': episode_intrinsic_reward,
                    })
        
        # Calculate final metrics
        final_mean_reward = np.mean(episode_rewards[-100:])
        final_success_rate = np.mean(episode_successes[-100:])
        max_reward = np.max(episode_rewards)
        
        results = {
            'mean_reward': final_mean_reward,
            'success_rate': final_success_rate,
            'max_reward': max_reward,
            'total_episodes': len(episode_rewards),
        }
        
        print(f"\nFinal Results:")
        print(f"  Mean Reward (last 100): {final_mean_reward:.2f}")
        print(f"  Success Rate (last 100): {final_success_rate:.2%}")
        print(f"  Max Reward: {max_reward:.2f}")
        
        if WANDB_AVAILABLE and run:
            wandb.log({
                'final_mean_reward': final_mean_reward,
                'final_success_rate': final_success_rate,
                'final_max_reward': max_reward,
            })
            run.finish()
        
        env.close()
        
        return results
    
    def run_search(self, env_type: str = None):
        """Run hyperparameter search"""
        
        envs_to_search = [env_type] if env_type else self.config['environments']
        
        for env_name in envs_to_search:
            print(f"\n{'#'*70}")
            print(f"# Hyperparameter Search for {env_name.upper()} Environment")
            print(f"{'#'*70}\n")
            
            env_results = []
            best_score = float('-inf') if self.config['search']['optimization_direction'] == 'maximize' else float('inf')
            best_hyperparams = None
            
            for trial in range(self.config['search']['n_trials']):
                # Sample hyperparameters
                hyperparams = self.sample_hyperparameters()
                
                # Train and evaluate
                results = self.train_with_hyperparameters(env_name, hyperparams, trial)
                
                # Store results
                trial_result = {
                    'environment': env_name,
                    'trial': trial,
                    'hyperparameters': hyperparams,
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }
                env_results.append(trial_result)
                self.all_results.append(trial_result)
                
                # Track best
                metric_value = results[self.config['search']['optimization_metric']]
                is_better = (
                    (metric_value > best_score and self.config['search']['optimization_direction'] == 'maximize') or
                    (metric_value < best_score and self.config['search']['optimization_direction'] == 'minimize')
                )
                
                if is_better:
                    best_score = metric_value
                    best_hyperparams = hyperparams.copy()
                
                # Save intermediate results
                self._save_results(env_name, env_results)
            
            # Print best configuration
            print(f"\n{'='*70}")
            print(f"Best Configuration for {env_name.upper()}")
            print(f"{'='*70}")
            print(f"Best {self.config['search']['optimization_metric']}: {best_score:.4f}")
            print("\nBest Hyperparameters:")
            for key, value in best_hyperparams.items():
                print(f"  {key}: {value}")
            print()
    
    def _save_results(self, env_name: str, results: List[Dict]):
        """Save search results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"search_{env_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filename}")
    
    def save_summary(self):
        """Save summary of all search results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"search_summary_{timestamp}.json"
        
        summary = {
            'config': self.config,
            'total_trials': len(self.all_results),
            'results': self.all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nFull summary saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='ConSpec Hyperparameter Search')
    parser.add_argument('--config', type=str, default='configs/conspec_hyperparam_search.yaml',
                       help='Path to config file')
    parser.add_argument('--env', type=str, choices=['single', 'double', 'triple', 'all'],
                       default='all', help='Environment to search')
    parser.add_argument('--n-trials', type=int, help='Number of trials (overrides config)')
    parser.add_argument('--episodes', type=int, help='Episodes per trial (overrides config)')
    
    args = parser.parse_args()
    
    # Initialize searcher
    searcher = HyperparameterSearcher(args.config)
    
    # Override config if specified
    if args.n_trials:
        searcher.config['search']['n_trials'] = args.n_trials
    if args.episodes:
        searcher.config['training']['episodes'] = args.episodes
    
    # Run search
    env_type = None if args.env == 'all' else args.env
    searcher.run_search(env_type)
    
    # Save summary
    searcher.save_summary()
    
    print("\nHyperparameter search completed!")


if __name__ == "__main__":
    main()

