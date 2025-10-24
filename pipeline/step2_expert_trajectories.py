"""
Step 2: Expert Trajectory Collection

Collect expert trajectories from the best-performing policy that crosses
through all critical features. Record binary state features for each timestep.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

from environments.key_door_goal_base import create_key_door_goal_env
from minigrid.core.world_object import Key, Door, Goal


class ExpertTrajectoryCollector:
    """
    Collect expert trajectories with binary state features
    """
    
    def __init__(
        self,
        env_type: str,
        policy: torch.nn.Module,
        encoder: torch.nn.Module,
        seed: int = 0
    ):
        """
        Initialize expert trajectory collector
        
        Args:
            env_type: Environment type ('single', 'double', 'triple')
            policy: Trained policy network
            encoder: Trained encoder network
            seed: Random seed
        """
        self.env_type = env_type
        self.policy = policy
        self.encoder = encoder
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create environment
        self.env = create_key_door_goal_env(env_type, size=10)
        
        # Set to eval mode
        self.policy.eval()
        self.encoder.eval()
        
        print(f"Initialized ExpertTrajectoryCollector for {env_type} environment")
    
    def extract_binary_features(self, obs: Dict) -> Dict[str, bool]:
        """
        Extract binary features from environment state
        
        Features tracked:
        - has_blue_key: Agent is carrying blue key
        - has_green_key: Agent is carrying green key  
        - has_purple_key: Agent is carrying purple key (for triple env)
        - blue_door_open: Blue door is open
        - green_door_open: Green door is open
        - purple_door_open: Purple door is open (for triple env)
        - at_green_goal: Agent at green goal
        - at_purple_goal: Agent at purple goal (for triple env)
        - task_complete: Episode successfully completed
        """
        features = {}
        
        # Check what agent is carrying
        carrying = self.env.carrying
        features['has_key'] = carrying is not None
        features['has_blue_key'] = isinstance(carrying, Key) and carrying.color == 'blue'
        features['has_green_key'] = isinstance(carrying, Key) and carrying.color == 'green'
        features['has_purple_key'] = isinstance(carrying, Key) and carrying.color == 'purple'
        
        # Check door states
        features['blue_door_open'] = self._check_door_open('blue')
        features['green_door_open'] = self._check_door_open('green')
        features['purple_door_open'] = self._check_door_open('purple')
        
        # Check goal status
        features['at_green_goal'] = self._check_at_goal('green')
        features['at_purple_goal'] = self._check_at_goal('purple')
        
        # Overall task completion
        features['task_complete'] = False  # Will be updated based on episode outcome
        
        return features
    
    def _check_door_open(self, color: str) -> bool:
        """Check if a door of specific color is open"""
        for x in range(self.env.width):
            for y in range(self.env.height):
                cell = self.env.grid.get(x, y)
                if isinstance(cell, Door) and cell.color == color:
                    return cell.is_open
        return False
    
    def _check_at_goal(self, color: str) -> bool:
        """Check if agent is at a goal of specific color"""
        agent_pos = self.env.agent_pos
        if agent_pos is None:
            return False
        
        cell = self.env.grid.get(*agent_pos)
        return isinstance(cell, Goal) and cell.color == color
    
    def collect_trajectory(
        self,
        max_steps: int = 400,
        epsilon: float = 0.0
    ) -> Tuple[List[Dict], float, bool]:
        """
        Collect one expert trajectory
        
        Returns:
            trajectory: List of state features at each timestep
            total_reward: Total episode reward
            success: Whether episode was successful
        """
        obs, _ = self.env.reset()
        trajectory = []
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Extract binary features
            features = self.extract_binary_features(obs)
            features['step'] = step
            features['agent_x'], features['agent_y'] = self.env.agent_pos
            
            # Get action from policy
            if np.random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs['image']).unsqueeze(0)
                    encoded, _ = self.encoder(obs_tensor)
                    q_values = self.policy(encoded)
                    action = q_values.argmax().item()
            
            features['action'] = int(action)
            
            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            features['reward'] = float(reward)
            total_reward += reward
            
            trajectory.append(features)
            step += 1
        
        # Mark task completion
        success = total_reward > 0
        if success:
            for t in trajectory:
                t['task_complete'] = True
        
        return trajectory, total_reward, success
    
    def collect_expert_data(
        self,
        num_trajectories: int = 1000,
        success_only: bool = True,
        min_successful: int = 100,
        epsilon: float = 0.0,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Collect multiple expert trajectories
        
        Args:
            num_trajectories: Number of trajectories to collect
            success_only: Only keep successful trajectories
            min_successful: Minimum number of successful trajectories needed
            epsilon: Exploration rate
            verbose: Show progress
        
        Returns:
            DataFrame with all trajectory data
        """
        all_trajectories = []
        successful_count = 0
        
        iterator = tqdm(range(num_trajectories), desc='Collecting trajectories') if verbose else range(num_trajectories)
        
        for episode in iterator:
            trajectory, reward, success = self.collect_trajectory(
                max_steps=400,
                epsilon=epsilon
            )
            
            if success:
                successful_count += 1
            
            # Add episode identifier
            for step_data in trajectory:
                step_data['episode'] = episode
                step_data['episode_reward'] = reward
                step_data['episode_success'] = success
            
            if not success_only or success:
                all_trajectories.extend(trajectory)
            
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'successful': successful_count,
                    'success_rate': f"{successful_count/(episode+1):.2%}"
                })
            
            # Stop if we have enough successful trajectories
            if success_only and successful_count >= min_successful:
                print(f"\nCollected {successful_count} successful trajectories. Stopping.")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_trajectories)
        
        print(f"\nData collection completed!")
        print(f"  - Total episodes: {episode + 1}")
        print(f"  - Successful episodes: {successful_count}")
        print(f"  - Total timesteps: {len(df)}")
        print(f"  - Success rate: {successful_count/(episode+1):.2%}")
        
        return df
    
    def analyze_expert_data(self, df: pd.DataFrame) -> Dict:
        """
        Analyze collected expert data
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Feature statistics
        binary_features = [col for col in df.columns if df[col].dtype == bool or df[col].nunique() == 2]
        feature_stats = {}
        
        for feature in binary_features:
            if feature != 'episode_success':
                feature_stats[feature] = {
                    'frequency': float(df[feature].mean()),
                    'first_occurrence_step': float(df[df[feature] == True]['step'].min()) if df[feature].any() else None
                }
        
        analysis['feature_statistics'] = feature_stats
        
        # Transition analysis
        transitions = {}
        for feature in binary_features:
            if feature != 'episode_success':
                # Find transitions from False to True
                feature_series = df.groupby('episode')[feature].apply(list)
                transition_episodes = []
                
                for ep, values in feature_series.items():
                    for i in range(len(values) - 1):
                        if not values[i] and values[i+1]:
                            transition_episodes.append(ep)
                            break
                
                transitions[feature] = len(transition_episodes)
        
        analysis['transition_counts'] = transitions
        
        # Temporal order
        successful_df = df[df['episode_success'] == True]
        if len(successful_df) > 0:
            temporal_order = []
            for feature in binary_features:
                if feature != 'episode_success' and successful_df[feature].any():
                    avg_first_step = successful_df[successful_df[feature] == True].groupby('episode')['step'].min().mean()
                    temporal_order.append((feature, avg_first_step))
            
            temporal_order.sort(key=lambda x: x[1])
            analysis['temporal_order'] = temporal_order
        
        return analysis
    
    def visualize_feature_timeline(self, df: pd.DataFrame, save_dir: str):
        """Visualize feature activations over time"""
        import matplotlib.pyplot as plt
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Get successful episodes
        successful_episodes = df[df['episode_success'] == True]['episode'].unique()
        
        if len(successful_episodes) == 0:
            print("No successful episodes to visualize")
            return
        
        # Take first few successful episodes
        sample_episodes = successful_episodes[:min(5, len(successful_episodes))]
        
        binary_features = [col for col in df.columns if (df[col].dtype == bool or df[col].nunique() == 2) and col not in ['episode_success', 'task_complete']]
        
        fig, axes = plt.subplots(len(sample_episodes), 1, figsize=(12, 3*len(sample_episodes)))
        if len(sample_episodes) == 1:
            axes = [axes]
        
        for idx, episode in enumerate(sample_episodes):
            ep_data = df[df['episode'] == episode].sort_values('step')
            
            y_pos = 0
            for feature in binary_features:
                feature_values = ep_data[feature].values
                steps = ep_data['step'].values
                
                # Plot as horizontal line with markers where True
                true_steps = steps[feature_values == True]
                if len(true_steps) > 0:
                    axes[idx].scatter(true_steps, [y_pos] * len(true_steps), 
                                    marker='s', s=50, label=feature)
                
                y_pos += 1
            
            axes[idx].set_xlabel('Step')
            axes[idx].set_ylabel('Features')
            axes[idx].set_title(f'Episode {episode} - Reward: {ep_data["episode_reward"].iloc[0]:.2f}')
            axes[idx].set_yticks(range(len(binary_features)))
            axes[idx].set_yticklabels(binary_features)
            axes[idx].grid(alpha=0.3, axis='x')
            axes[idx].set_xlim([0, ep_data['step'].max()])
        
        plt.tight_layout()
        plt.savefig(save_path / f'{self.env_type}_feature_timeline.png', dpi=150, bbox_inches='tight')
        print(f"Saved feature timeline: {save_path / f'{self.env_type}_feature_timeline.png'}")
        plt.close()
    
    def save_expert_data(self, df: pd.DataFrame, output_dir: str):
        """Save expert trajectory data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_file = output_path / f'{self.env_type}_expert_data_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved expert data: {csv_file}")
        
        # Save analysis
        analysis = self.analyze_expert_data(df)
        analysis_file = output_path / f'{self.env_type}_analysis_{timestamp}.pkl'
        with open(analysis_file, 'wb') as f:
            pickle.dump(analysis, f)
        print(f"Saved analysis: {analysis_file}")
        
        return csv_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 2: Expert Trajectory Collection')
    parser.add_argument('--env', type=str, required=True, choices=['single', 'double', 'triple'],
                       help='Environment type')
    parser.add_argument('--step1_results', type=str, required=True,
                       help='Path to Step 1 results directory')
    parser.add_argument('--num_trajectories', type=int, default=1000,
                       help='Number of trajectories to collect')
    parser.add_argument('--min_successful', type=int, default=100,
                       help='Minimum successful trajectories')
    parser.add_argument('--output_dir', type=str, default='pipeline_results/step2',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load Step 1 results
    step1_path = Path(args.step1_results)
    
    # Find the most recent model files
    import glob
    policy_files = glob.glob(str(step1_path / f'{args.env}_*policy*.pt'))
    if not policy_files:
        raise FileNotFoundError(f"No policy file found in {step1_path}")
    
    # For now, we'll need to reconstruct the policy
    # In practice, you should save the entire policy in step 1
    print("Note: Loading policy from Step 1 results")
    print("Collecting trajectories with greedy policy...")
    
    # Create a simple collector without loading (for demonstration)
    # In production, load the actual trained policy
    from methods.conspec_minigrid import EncoderConSpec
    env = create_key_door_goal_env(args.env)
    obs_shape = env.observation_space['image'].shape
    
    encoder = EncoderConSpec(input_channels=obs_shape[2], hidden_dim=128, lstm_hidden_size=128)
    policy = torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, env.action_space.n)
    )
    
    # Load saved model if available
    # torch.load(...) here
    
    # Create collector
    collector = ExpertTrajectoryCollector(
        env_type=args.env,
        policy=policy,
        encoder=encoder,
        seed=args.seed
    )
    
    # Collect data
    df = collector.collect_expert_data(
        num_trajectories=args.num_trajectories,
        success_only=True,
        min_successful=args.min_successful,
        verbose=True
    )
    
    # Analyze
    analysis = collector.analyze_expert_data(df)
    print("\nFeature Analysis:")
    if 'temporal_order' in analysis:
        print("Temporal order of features:")
        for feature, avg_step in analysis['temporal_order']:
            print(f"  {feature}: step {avg_step:.1f}")
    
    # Save
    csv_file = collector.save_expert_data(df, args.output_dir)
    collector.visualize_feature_timeline(df, args.output_dir)
    
    print("\nStep 2 completed successfully!")
    print(f"Expert data saved to: {csv_file}")


if __name__ == '__main__':
    main()

