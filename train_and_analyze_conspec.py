"""
Train ConSpec on the single environment and analyze critical features
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from key_door_goal_base import create_key_door_goal_env
from conspec_minigrid import ConSpec
from key_door_goal_visualizer import KeyDoorGoalVisualizer


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


def train_conspec_single_environment(num_episodes=1000, device='cpu', save_models=True):
    """
    Train ConSpec on the single environment with detailed logging
    """
    print("Training ConSpec on Single Environment (Chain Graph)")
    print("=" * 60)
    
    # Create environment
    env = create_key_door_goal_env('single', size=10)
    obs_shape = (env.height, env.width, 3)
    num_actions = env.action_space.n
    
    # Create policy and ConSpec
    policy = SimplePolicy(obs_shape, num_actions).to(device)
    conspec = ConSpec(obs_shape, num_actions, device, 
                     intrinsic_reward_scale=0.1, learning_rate=1e-3)
    
    # Model saving variables
    best_success_rate = 0.0
    best_policy_state = None
    best_conspec_state = None
    convergence_episodes = 0
    convergence_threshold = 0.95  # Consider converged when success rate > 95%
    patience = 50  # Episodes to wait after convergence before saving
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    intrinsic_rewards_history = []
    conspec_losses = []
    
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
        
        while not done and episode_length < 200:  # Max 200 steps
            # Get action from policy (with some exploration)
            with torch.no_grad():
                logits, value = policy(obs['image'])
                action_probs = torch.softmax(logits, dim=-1)
                
                # Add exploration noise
                if episode < num_episodes // 2:  # More exploration in first half
                    action = torch.multinomial(action_probs, 1).item()
                else:
                    action = torch.argmax(action_probs, dim=-1).item()
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store episode data
            episode_observations.append(obs['image'])
            episode_actions.append(action)
            episode_rewards_list.append(reward)
            
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
        
        # Check for convergence and save best models
        if len(success_rate) >= 100:
            recent_success_rate = np.mean(success_rate[-100:])
            
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
            else:
                convergence_episodes += 1
            
            # Check if converged and save models
            if (recent_success_rate >= convergence_threshold and 
                convergence_episodes >= patience and 
                save_models and 
                best_policy_state is not None):
                
                print(f"\n🎉 POLICY CONVERGED! Saving optimal models...")
                print(f"Success rate: {recent_success_rate:.3f} (threshold: {convergence_threshold})")
                
                # Create models directory
                os.makedirs('models', exist_ok=True)
                
                # Save policy model
                torch.save({
                    'model_state_dict': best_policy_state,
                    'obs_shape': obs_shape,
                    'num_actions': num_actions,
                    'success_rate': best_success_rate,
                    'episode': episode
                }, 'models/optimal_policy_single_env.pth')
                
                # Save ConSpec model
                torch.save({
                    'encoder_state_dict': best_conspec_state['encoder'],
                    'prototype_net_state_dict': best_conspec_state['prototype_net'],
                    'obs_shape': obs_shape,
                    'num_actions': num_actions,
                    'intrinsic_reward_scale': conspec.intrinsic_reward_scale,
                    'num_prototypes': conspec.num_prototypes,
                    'success_rate': best_success_rate,
                    'episode': episode
                }, 'models/optimal_conspec_single_env.pth')
                
                # Save memory separately (as pickle since it contains complex objects)
                import pickle
                with open('models/conspec_memory_single_env.pkl', 'wb') as f:
                    pickle.dump(best_conspec_state['memory'], f)
                
                print(f"✅ Models saved to models/ directory:")
                print(f"   - optimal_policy_single_env.pth")
                print(f"   - optimal_conspec_single_env.pth") 
                print(f"   - conspec_memory_single_env.pkl")
                print(f"   Best success rate: {best_success_rate:.3f}")
                
                # Stop training early if converged
                break
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            avg_success = np.mean(success_rate[-100:]) if len(success_rate) >= 100 else np.mean(success_rate)
            avg_intrinsic = np.mean(intrinsic_rewards_history[-1000:]) if len(intrinsic_rewards_history) >= 1000 else np.mean(intrinsic_rewards_history)
            
            print(f"Episode {episode}: "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.1f}, "
                  f"Success Rate: {avg_success:.2f}, "
                  f"Avg Intrinsic: {avg_intrinsic:.4f}")
            print(f"  Memory - Success: {len(conspec.memory.success_memory)}, "
                  f"Failure: {len(conspec.memory.failure_memory)}")
            print(f"  Best Success Rate: {best_success_rate:.3f}")
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_rate,
        'intrinsic_rewards': intrinsic_rewards_history,
        'successful_trajectories': successful_trajectories,
        'failed_trajectories': failed_trajectories,
        'conspec': conspec
    }


def analyze_conspec_features(conspec, successful_trajectories, failed_trajectories):
    """
    Analyze what features ConSpec has learned to distinguish success from failure
    """
    print("\nAnalyzing ConSpec Features")
    print("=" * 40)
    
    device = conspec.device
    
    # Sample some trajectories for analysis
    num_samples = min(10, len(successful_trajectories), len(failed_trajectories))
    success_samples = successful_trajectories[:num_samples]
    failure_samples = failed_trajectories[:num_samples]
    
    print(f"Analyzing {num_samples} successful and {num_samples} failed trajectories")
    
    # Extract features from successful trajectories
    success_features = []
    for traj in success_samples:
        # Get the last observation (end state)
        last_obs = traj['observations'][-1]
        obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(device)
        encoded, _ = conspec.encoder(obs_tensor)
        last_feature = encoded[:, -1, :].detach().cpu().numpy()
        success_features.append(last_feature)
    
    # Extract features from failed trajectories
    failure_features = []
    for traj in failure_samples:
        # Get the last observation (end state)
        last_obs = traj['observations'][-1]
        obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(device)
        encoded, _ = conspec.encoder(obs_tensor)
        last_feature = encoded[:, -1, :].detach().cpu().numpy()
        failure_features.append(last_feature)
    
    # Convert to numpy arrays
    success_features = np.vstack(success_features)
    failure_features = np.vstack(failure_features)
    
    # Compute similarities to prototypes
    success_tensor = torch.FloatTensor(success_features).to(device)
    failure_tensor = torch.FloatTensor(failure_features).to(device)
    
    with torch.no_grad():
        _, success_similarities = conspec.prototype_net(success_tensor)
        _, failure_similarities = conspec.prototype_net(failure_tensor)
    
    success_similarities = success_similarities.cpu().numpy()
    failure_similarities = failure_similarities.cpu().numpy()
    
    # Analyze prototype similarities
    print(f"Success trajectories - Max similarity to prototypes: {np.max(success_similarities, axis=1).mean():.4f}")
    print(f"Failure trajectories - Max similarity to prototypes: {np.max(failure_similarities, axis=1).mean():.4f}")
    
    # Find most discriminative prototypes
    success_max_sim = np.max(success_similarities, axis=1)
    failure_max_sim = np.max(failure_similarities, axis=1)
    
    print(f"Success trajectories - Mean max similarity: {success_max_sim.mean():.4f} ± {success_max_sim.std():.4f}")
    print(f"Failure trajectories - Mean max similarity: {failure_max_sim.mean():.4f} ± {failure_max_sim.std():.4f}")
    
    return {
        'success_features': success_features,
        'failure_features': failure_features,
        'success_similarities': success_similarities,
        'failure_similarities': failure_similarities,
        'success_max_sim': success_max_sim,
        'failure_max_sim': failure_max_sim
    }


def visualize_training_results(results, save_dir='figures'):
    """
    Create comprehensive visualizations of training results
    """
    print("\nCreating visualizations...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ConSpec Training Results on Single Environment', fontsize=16, fontweight='bold')
    
    # Episode rewards
    axes[0, 0].plot(results['episode_rewards'], alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(results['episode_lengths'], alpha=0.7)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Success rate (moving average)
    window = 50
    if len(results['success_rate']) >= window:
        success_ma = np.convolve(results['success_rate'], np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(results['success_rate'])), success_ma)
    else:
        axes[1, 0].plot(results['success_rate'])
    axes[1, 0].set_title('Success Rate (Moving Average)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Intrinsic rewards (moving average)
    window = 100
    if len(results['intrinsic_rewards']) >= window:
        intrinsic_ma = np.convolve(results['intrinsic_rewards'], np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window-1, len(results['intrinsic_rewards'])), intrinsic_ma)
    else:
        axes[1, 1].plot(results['intrinsic_rewards'])
    axes[1, 1].set_title('Intrinsic Rewards (Moving Average)')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Intrinsic Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/conspec_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Trajectory analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('ConSpec Trajectory Analysis', fontsize=16, fontweight='bold')
    
    # Success vs failure trajectory lengths
    success_lengths = [t['length'] for t in results['successful_trajectories']]
    failure_lengths = [t['length'] for t in results['failed_trajectories']]
    
    axes[0].hist([success_lengths, failure_lengths], bins=20, alpha=0.7, 
                label=['Success', 'Failure'], color=['green', 'red'])
    axes[0].set_title('Trajectory Length Distribution')
    axes[0].set_xlabel('Episode Length')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Success vs failure intrinsic rewards
    success_intrinsic = []
    failure_intrinsic = []
    
    for traj in results['successful_trajectories'][:50]:  # Sample for visualization
        success_intrinsic.extend(traj['intrinsic_rewards'])
    
    for traj in results['failed_trajectories'][:50]:  # Sample for visualization
        failure_intrinsic.extend(traj['intrinsic_rewards'])
    
    axes[1].hist([success_intrinsic, failure_intrinsic], bins=30, alpha=0.7,
                label=['Success', 'Failure'], color=['green', 'red'])
    axes[1].set_title('Intrinsic Reward Distribution')
    axes[1].set_xlabel('Intrinsic Reward')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Success rate over time
    window = 100
    if len(results['success_rate']) >= window:
        success_ma = np.convolve(results['success_rate'], np.ones(window)/window, mode='valid')
        axes[2].plot(range(window-1, len(results['success_rate'])), success_ma, linewidth=2)
    else:
        axes[2].plot(results['success_rate'], linewidth=2)
    axes[2].set_title('Success Rate Over Time')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/conspec_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training visualizations saved to {save_dir}/")


def visualize_conspec_features(analysis_results, save_dir='figures'):
    """
    Visualize what features ConSpec has learned
    """
    print("Creating ConSpec feature analysis visualizations...")
    
    # 1. Feature space visualization (PCA)
    from sklearn.decomposition import PCA
    
    # Combine success and failure features
    all_features = np.vstack([analysis_results['success_features'], 
                             analysis_results['failure_features']])
    
    # Apply PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    
    # Split back into success and failure
    n_success = len(analysis_results['success_features'])
    success_2d = features_2d[:n_success]
    failure_2d = features_2d[n_success:]
    
    # Plot feature space
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('ConSpec Feature Analysis', fontsize=16, fontweight='bold')
    
    # Feature space scatter plot
    axes[0].scatter(success_2d[:, 0], success_2d[:, 1], c='green', alpha=0.7, 
                   label='Success', s=50)
    axes[0].scatter(failure_2d[:, 0], failure_2d[:, 1], c='red', alpha=0.7, 
                   label='Failure', s=50)
    axes[0].set_title('Learned Feature Space (PCA)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Prototype similarities
    axes[1].hist([analysis_results['success_max_sim'], 
                 analysis_results['failure_max_sim']], 
                bins=20, alpha=0.7, label=['Success', 'Failure'], 
                color=['green', 'red'])
    axes[1].set_title('Prototype Similarity Distribution')
    axes[1].set_xlabel('Max Similarity to Prototypes')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/conspec_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature analysis visualizations saved to {save_dir}/")


def visualize_sample_trajectories(successful_trajectories, failed_trajectories, save_dir='figures'):
    """
    Visualize sample successful and failed trajectories
    """
    print("Creating trajectory visualizations...")
    
    # Create visualizer
    visualizer = KeyDoorGoalVisualizer()
    
    # Sample a few trajectories
    num_samples = min(3, len(successful_trajectories), len(failed_trajectories))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(5*num_samples, 10))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Sample Trajectories: Success vs Failure', fontsize=16, fontweight='bold')
    
    # Visualize successful trajectories
    for i in range(num_samples):
        traj = successful_trajectories[i]
        
        # Create a temporary environment to visualize the final state
        env = create_key_door_goal_env('single', size=10)
        obs, _ = env.reset()
        
        # Replay the trajectory to get to the final state
        for action in traj['actions']:
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        
        # Visualize the final state
        visualizer._draw_grid(axes[0, i], env, env.width, env.height)
        visualizer._add_legend(axes[0, i], env)
        axes[0, i].set_title(f'Success Trajectory {i+1}\nLength: {traj["length"]}, Reward: {traj["total_reward"]}')
        axes[0, i].set_aspect('equal')
        axes[0, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        env.close()
    
    # Visualize failed trajectories
    for i in range(num_samples):
        traj = failed_trajectories[i]
        
        # Create a temporary environment to visualize the final state
        env = create_key_door_goal_env('single', size=10)
        obs, _ = env.reset()
        
        # Replay the trajectory to get to the final state
        for action in traj['actions']:
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        
        # Visualize the final state
        visualizer._draw_grid(axes[1, i], env, env.width, env.height)
        visualizer._add_legend(axes[1, i], env)
        axes[1, i].set_title(f'Failure Trajectory {i+1}\nLength: {traj["length"]}, Reward: {traj["total_reward"]}')
        axes[1, i].set_aspect('equal')
        axes[1, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        env.close()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/conspec_sample_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory visualizations saved to {save_dir}/")


def load_and_test_optimal_policy(device='cpu'):
    """
    Load and test the saved optimal policy
    """
    print("\n" + "=" * 60)
    print("TESTING OPTIMAL POLICY")
    print("=" * 60)
    
    # Check if models exist
    if not os.path.exists('models/optimal_policy_single_env.pth'):
        print("❌ No saved policy model found. Please train first.")
        return None
    
    # Load policy model
    policy_checkpoint = torch.load('models/optimal_policy_single_env.pth', map_location=device)
    obs_shape = policy_checkpoint['obs_shape']
    num_actions = policy_checkpoint['num_actions']
    success_rate = policy_checkpoint['success_rate']
    episode = policy_checkpoint['episode']
    
    print(f"Loading policy trained for {episode} episodes with {success_rate:.3f} success rate")
    
    # Create and load policy
    policy = SimplePolicy(obs_shape, num_actions).to(device)
    policy.load_state_dict(policy_checkpoint['model_state_dict'])
    policy.eval()
    
    # Test the policy
    env = create_key_door_goal_env('single', size=10)
    test_episodes = 10
    test_successes = 0
    test_rewards = []
    test_lengths = []
    
    print(f"Testing policy on {test_episodes} episodes...")
    
    for test_ep in range(test_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done and episode_length < 200:
            with torch.no_grad():
                logits, _ = policy(obs['image'])
                action_probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
        if episode_reward > 0:
            test_successes += 1
        
        print(f"  Test episode {test_ep + 1}: Reward={episode_reward}, Length={episode_length}")
    
    env.close()
    
    test_success_rate = test_successes / test_episodes
    avg_reward = np.mean(test_rewards)
    avg_length = np.mean(test_lengths)
    
    print(f"\n📊 OPTIMAL POLICY TEST RESULTS:")
    print(f"   Success rate: {test_success_rate:.2f} ({test_successes}/{test_episodes})")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Average length: {avg_length:.1f}")
    
    if test_success_rate >= 0.8:
        print("✅ Policy is performing well!")
    else:
        print("⚠️  Policy may need more training")
    
    return {
        'policy': policy,
        'test_success_rate': test_success_rate,
        'test_rewards': test_rewards,
        'test_lengths': test_lengths
    }


def main():
    """Main training and analysis function"""
    print("ConSpec Training and Analysis on Single Environment")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train ConSpec
    results = train_conspec_single_environment(num_episodes=1000, device=device, save_models=True)
    
    # Analyze features
    analysis_results = analyze_conspec_features(
        results['conspec'], 
        results['successful_trajectories'], 
        results['failed_trajectories']
    )
    
    # Create visualizations
    visualize_training_results(results)
    visualize_conspec_features(analysis_results)
    visualize_sample_trajectories(
        results['successful_trajectories'], 
        results['failed_trajectories']
    )
    
    # Test optimal policy if it was saved
    if os.path.exists('models/optimal_policy_single_env.pth'):
        test_results = load_and_test_optimal_policy(device)
    else:
        test_results = None
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {len(results['episode_rewards'])}")
    print(f"Final success rate: {np.mean(results['success_rate'][-100:]):.2f}")
    print(f"Average episode length: {np.mean(results['episode_lengths']):.1f}")
    print(f"Average intrinsic reward: {np.mean(results['intrinsic_rewards']):.4f}")
    print(f"Successful trajectories: {len(results['successful_trajectories'])}")
    print(f"Failed trajectories: {len(results['failed_trajectories'])}")
    print(f"ConSpec memory - Success: {len(results['conspec'].memory.success_memory)}")
    print(f"ConSpec memory - Failure: {len(results['conspec'].memory.failure_memory)}")
    
    print("\nConSpec has learned to distinguish between:")
    print(f"- Success trajectories (avg similarity: {analysis_results['success_max_sim'].mean():.4f})")
    print(f"- Failure trajectories (avg similarity: {analysis_results['failure_max_sim'].mean():.4f})")
    
    if test_results:
        print(f"\nOptimal policy test success rate: {test_results['test_success_rate']:.2f}")
    
    print("\nAll visualizations saved to figures/ directory!")
    if os.path.exists('models/'):
        print("Optimal models saved to models/ directory!")
    
    return results, analysis_results, test_results


if __name__ == "__main__":
    results, analysis = main()
