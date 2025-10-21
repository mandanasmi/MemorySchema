"""
Train ConSpec on all three environments and save best model checkpoints
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


def train_conspec_on_environment(env_type, num_episodes=1000, device='cpu', save_checkpoints=True):
    """
    Train ConSpec on a specific environment
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
                     intrinsic_reward_scale=0.1, learning_rate=1e-3)
    
    # Create trainer
    trainer = PPOTrainer(policy, device)
    
    # Model saving variables
    best_success_rate = 0.0
    best_policy_state = None
    best_conspec_state = None
    convergence_episodes = 0
    convergence_threshold = 0.7  # Consider converged when success rate > 70%
    patience = 20  # Episodes to wait after convergence before saving
    
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
        
        while not done and episode_length < 200:  # Max 200 steps
            # Get action from policy (with exploration)
            with torch.no_grad():
                logits, value = policy(obs['image'])
                action_probs = torch.softmax(logits, dim=-1)
                
                # Add exploration noise - more exploration throughout training
                if episode < num_episodes * 0.8:  # More exploration for 80% of training
                    # Add some noise to action probabilities for exploration
                    noise = torch.rand_like(action_probs) * 0.1
                    noisy_probs = action_probs + noise
                    noisy_probs = torch.softmax(noisy_probs, dim=-1)
                    action = torch.multinomial(noisy_probs, 1).item()
                else:
                    action = torch.argmax(action_probs, dim=-1).item()
                
                log_prob = torch.log(action_probs[0, action]).item()
            
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
            trainer.update(episode_observations, episode_actions, total_rewards, 
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
        
        # Check for convergence and save best models
        if len(success_rate) >= 50:
            recent_success_rate = np.mean(success_rate[-50:])
            
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
                save_checkpoints and 
                best_policy_state is not None):
                
                print(f"\n POLICY CONVERGED! Saving optimal models...")
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
                
                print(f" Models saved to checkpoints/ directory:")
                print(f"   - optimal_policy_{env_type}_env.pth")
                print(f"   - optimal_conspec_{env_type}_env.pth") 
                print(f"   - conspec_memory_{env_type}_env.pkl")
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


def analyze_critical_features(conspec, successful_trajectories, failed_trajectories, env_type):
    """
    Analyze what critical features ConSpec has learned to identify
    """
    print(f"\nAnalyzing Critical Features for {env_type.upper()} Environment")
    print("=" * 50)
    
    device = conspec.device
    
    # Sample trajectories for analysis
    num_samples = min(20, len(successful_trajectories), len(failed_trajectories))
    if num_samples == 0:
        print("No trajectories to analyze")
        return None
        
    success_samples = successful_trajectories[:num_samples]
    failure_samples = failed_trajectories[:num_samples]
    
    print(f"Analyzing {num_samples} successful and {num_samples} failed trajectories")
    
    # Extract features from different states in trajectories
    success_key_states = []      # States when agent is near keys
    success_door_states = []     # States when agent is near doors
    success_goal_states = []     # States when agent reaches goal
    failure_stuck_states = []    # States where agent gets stuck
    
    for traj in success_samples:
        # Get states at different points in successful trajectories
        if len(traj['observations']) > 10:
            # Early state (likely near key)
            success_key_states.append(traj['observations'][5])
            # Middle state (likely near door)
            success_door_states.append(traj['observations'][len(traj['observations'])//2])
            # Final state (goal reached)
            success_goal_states.append(traj['observations'][-1])
    
    for traj in failure_samples:
        # Get final states from failed trajectories (where agent got stuck)
        if len(traj['observations']) > 0:
            failure_stuck_states.append(traj['observations'][-1])
    
    # Encode all states
    def encode_states(states):
        if not states:
            return None
        encoded_features = []
        for state in states:
            obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            encoded, _ = conspec.encoder(obs_tensor)
            last_feature = encoded[:, -1, :].detach().cpu().numpy()
            encoded_features.append(last_feature)
        return np.vstack(encoded_features)
    
    success_key_features = encode_states(success_key_states)
    success_door_features = encode_states(success_door_states)
    success_goal_features = encode_states(success_goal_states)
    failure_stuck_features = encode_states(failure_stuck_states)
    
    # Compute similarities to prototypes
    def compute_similarities(features):
        if features is None:
            return None
        features_tensor = torch.FloatTensor(features).to(device)
        with torch.no_grad():
            _, similarities = conspec.prototype_net(features_tensor)
        return similarities.cpu().numpy()
    
    success_key_sim = compute_similarities(success_key_features)
    success_door_sim = compute_similarities(success_door_features)
    success_goal_sim = compute_similarities(success_goal_features)
    failure_stuck_sim = compute_similarities(failure_stuck_features)
    
    # Analyze what ConSpec has learned
    print(f"\nConSpec Feature Analysis for {env_type.upper()}:")
    
    if success_key_sim is not None:
        key_max_sim = np.max(success_key_sim, axis=1)
        print(f"  Key states - Mean max similarity: {key_max_sim.mean():.4f} Â {key_max_sim.std():.4f}")
    
    if success_door_sim is not None:
        door_max_sim = np.max(success_door_sim, axis=1)
        print(f"  Door states - Mean max similarity: {door_max_sim.mean():.4f} Â {door_max_sim.std():.4f}")
    
    if success_goal_sim is not None:
        goal_max_sim = np.max(success_goal_sim, axis=1)
        print(f"  Goal states - Mean max similarity: {goal_max_sim.mean():.4f} Â {goal_max_sim.std():.4f}")
    
    if failure_stuck_sim is not None:
        stuck_max_sim = np.max(failure_stuck_sim, axis=1)
        print(f"  Stuck states - Mean max similarity: {stuck_max_sim.mean():.4f} Â {stuck_max_sim.std():.4f}")
    
    return {
        'success_key_features': success_key_features,
        'success_door_features': success_door_features,
        'success_goal_features': success_goal_features,
        'failure_stuck_features': failure_stuck_features,
        'success_key_sim': success_key_sim,
        'success_door_sim': success_door_sim,
        'success_goal_sim': success_goal_sim,
        'failure_stuck_sim': failure_stuck_sim,
        'success_key_states': success_key_states,
        'success_door_states': success_door_states,
        'success_goal_states': success_goal_states,
        'failure_stuck_states': failure_stuck_states
    }


def visualize_critical_features(analysis_results, env_type, save_dir='figures'):
    """
    Visualize the critical features that ConSpec has learned to identify
    """
    print(f"Creating critical feature visualizations for {env_type.upper()}...")
    
    if analysis_results is None:
        print("No analysis results to visualize")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = KeyDoorGoalVisualizer()
    
    # Create figure with subplots for different critical states
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'ConSpec Critical Features - {env_type.upper()} Environment', fontsize=16, fontweight='bold')
    
    # Plot 1: Key states from successful trajectories
    if analysis_results['success_key_states']:
        # Create environment to visualize key state
        env = create_key_door_goal_env(env_type, size=10)
        try:
            visualizer._draw_grid(axes[0, 0], env, env.width, env.height)
            visualizer._add_legend(axes[0, 0], env)
            axes[0, 0].set_title('Key States (Success)')
            axes[0, 0].set_aspect('equal')
            axes[0, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f'Key States\n({len(analysis_results["success_key_states"])} samples)', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Key States (Success)')
        finally:
            env.close()
    
    # Plot 2: Door states from successful trajectories
    if analysis_results['success_door_states']:
        env = create_key_door_goal_env(env_type, size=10)
        try:
            visualizer._draw_grid(axes[0, 1], env, env.width, env.height)
            visualizer._add_legend(axes[0, 1], env)
            axes[0, 1].set_title('Door States (Success)')
            axes[0, 1].set_aspect('equal')
            axes[0, 1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Door States\n({len(analysis_results["success_door_states"])} samples)', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Door States (Success)')
        finally:
            env.close()
    
    # Plot 3: Goal states from successful trajectories
    if analysis_results['success_goal_states']:
        env = create_key_door_goal_env(env_type, size=10)
        try:
            visualizer._draw_grid(axes[0, 2], env, env.width, env.height)
            visualizer._add_legend(axes[0, 2], env)
            axes[0, 2].set_title('Goal States (Success)')
            axes[0, 2].set_aspect('equal')
            axes[0, 2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        except Exception as e:
            axes[0, 2].text(0.5, 0.5, f'Goal States\n({len(analysis_results["success_goal_states"])} samples)', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Goal States (Success)')
        finally:
            env.close()
    
    # Plot 4: Feature space visualization (PCA)
    from sklearn.decomposition import PCA
    
    all_features = []
    feature_labels = []
    
    if analysis_results['success_key_features'] is not None:
        all_features.append(analysis_results['success_key_features'])
        feature_labels.extend(['Key'] * len(analysis_results['success_key_features']))
    
    if analysis_results['success_door_features'] is not None:
        all_features.append(analysis_results['success_door_features'])
        feature_labels.extend(['Door'] * len(analysis_results['success_door_features']))
    
    if analysis_results['success_goal_features'] is not None:
        all_features.append(analysis_results['success_goal_features'])
        feature_labels.extend(['Goal'] * len(analysis_results['success_goal_features']))
    
    if analysis_results['failure_stuck_features'] is not None:
        all_features.append(analysis_results['failure_stuck_features'])
        feature_labels.extend(['Stuck'] * len(analysis_results['failure_stuck_features']))
    
    if all_features:
        all_features = np.vstack(all_features)
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_features)
        
        # Plot feature space
        colors = {'Key': 'blue', 'Door': 'orange', 'Goal': 'green', 'Stuck': 'red'}
        for label in set(feature_labels):
            mask = np.array(feature_labels) == label
            axes[1, 0].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                              c=colors[label], label=label, alpha=0.7, s=50)
        
        axes[1, 0].set_title('Learned Feature Space (PCA)')
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Prototype similarities
    similarities_data = []
    similarity_labels = []
    
    if analysis_results['success_key_sim'] is not None:
        similarities_data.append(np.max(analysis_results['success_key_sim'], axis=1))
        similarity_labels.extend(['Key'] * len(analysis_results['success_key_sim']))
    
    if analysis_results['success_door_sim'] is not None:
        similarities_data.append(np.max(analysis_results['success_door_sim'], axis=1))
        similarity_labels.extend(['Door'] * len(analysis_results['success_door_sim']))
    
    if analysis_results['success_goal_sim'] is not None:
        similarities_data.append(np.max(analysis_results['success_goal_sim'], axis=1))
        similarity_labels.extend(['Goal'] * len(analysis_results['success_goal_sim']))
    
    if analysis_results['failure_stuck_sim'] is not None:
        similarities_data.append(np.max(analysis_results['failure_stuck_sim'], axis=1))
        similarity_labels.extend(['Stuck'] * len(analysis_results['failure_stuck_sim']))
    
    if similarities_data:
        all_similarities = np.concatenate(similarities_data)
        for label in set(similarity_labels):
            mask = np.array(similarity_labels) == label
            axes[1, 1].hist(all_similarities[mask], bins=15, alpha=0.7, 
                           label=label, color=colors[label])
        
        axes[1, 1].set_title('Prototype Similarity Distribution')
        axes[1, 1].set_xlabel('Max Similarity to Prototypes')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Training summary
    axes[1, 2].text(0.1, 0.8, f'Environment: {env_type.upper()}', fontsize=14, fontweight='bold')
    axes[1, 2].text(0.1, 0.7, f'Key States: {len(analysis_results["success_key_states"])}', fontsize=12)
    axes[1, 2].text(0.1, 0.6, f'Door States: {len(analysis_results["success_door_states"])}', fontsize=12)
    axes[1, 2].text(0.1, 0.5, f'Goal States: {len(analysis_results["success_goal_states"])}', fontsize=12)
    axes[1, 2].text(0.1, 0.4, f'Stuck States: {len(analysis_results["failure_stuck_states"])}', fontsize=12)
    axes[1, 2].set_title('Analysis Summary')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/conspec_critical_features_{env_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Critical feature visualizations saved to {save_dir}/conspec_critical_features_{env_type}.png")


def create_performance_summary(all_results, save_dir='figures'):
    """
    Create a summary visualization of performance across all environments
    """
    print("Creating performance summary...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ConSpec Performance Across All Environments', fontsize=16, fontweight='bold')
    
    env_types = ['single', 'double', 'triple']
    colors = {'single': 'blue', 'double': 'green', 'triple': 'red'}
    
    # Plot 1: Success rates over time
    for env_type in env_types:
        if env_type in all_results:
            results = all_results[env_type]
            success_rate = results['success_rate']
            window = 50
            if len(success_rate) >= window:
                success_ma = np.convolve(success_rate, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(success_rate)), success_ma, 
                               label=f'{env_type.upper()}', color=colors[env_type], linewidth=2)
            else:
                axes[0, 0].plot(success_rate, label=f'{env_type.upper()}', color=colors[env_type], linewidth=2)
    
    axes[0, 0].set_title('Success Rate Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Final performance comparison
    final_success_rates = []
    env_names = []
    for env_type in env_types:
        if env_type in all_results:
            results = all_results[env_type]
            final_success = np.mean(results['success_rate'][-100:]) if len(results['success_rate']) >= 100 else np.mean(results['success_rate'])
            final_success_rates.append(final_success)
            env_names.append(env_type.upper())
    
    bars = axes[0, 1].bar(env_names, final_success_rates, color=[colors[env.lower()] for env in env_names])
    axes[0, 1].set_title('Final Success Rate')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, final_success_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{rate:.2f}', ha='center', va='bottom')
    
    # Plot 3: Episode lengths
    for env_type in env_types:
        if env_type in all_results:
            results = all_results[env_type]
            episode_lengths = results['episode_lengths']
            window = 50
            if len(episode_lengths) >= window:
                length_ma = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
                axes[1, 0].plot(range(window-1, len(episode_lengths)), length_ma, 
                               label=f'{env_type.upper()}', color=colors[env_type], linewidth=2)
            else:
                axes[1, 0].plot(episode_lengths, label=f'{env_type.upper()}', color=colors[env_type], linewidth=2)
    
    axes[1, 0].set_title('Episode Length Over Time')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Episode Length')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training summary table
    summary_data = []
    for env_type in env_types:
        if env_type in all_results:
            results = all_results[env_type]
            final_success = np.mean(results['success_rate'][-100:]) if len(results['success_rate']) >= 100 else np.mean(results['success_rate'])
            avg_length = np.mean(results['episode_lengths'])
            total_episodes = len(results['episode_rewards'])
            converged = results.get('converged', False)
            
            summary_data.append([
                env_type.upper(),
                f'{final_success:.2f}',
                f'{avg_length:.1f}',
                f'{total_episodes}',
                '' if converged else ''
            ])
    
    table = axes[1, 1].table(cellText=summary_data,
                            colLabels=['Environment', 'Success Rate', 'Avg Length', 'Episodes', 'Converged'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/conspec_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance summary saved to {save_dir}/conspec_performance_summary.png")


def main():
    """Main training function for all environments"""
    print("ConSpec Training on All Environments")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train on all three environments
    env_types = ['single', 'double', 'triple']
    all_results = {}
    all_analyses = {}
    
    for env_type in env_types:
        # Train ConSpec
        results = train_conspec_on_environment(
            env_type=env_type,
            num_episodes=800,  # Reduced for faster training
            device=device,
            save_checkpoints=True
        )
        
        all_results[env_type] = results
        
        # Analyze critical features
        analysis = analyze_critical_features(
            results['conspec'],
            results['successful_trajectories'],
            results['failed_trajectories'],
            env_type
        )
        
        all_analyses[env_type] = analysis
        
        # Visualize critical features
        visualize_critical_features(analysis, env_type)
    
    # Create performance summary
    create_performance_summary(all_results)
    
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
            print(f"  Total episodes: {len(results['episode_rewards'])}")
            print(f"  Converged: {'' if converged else ''}")
            print(f"  Successful trajectories: {len(results['successful_trajectories'])}")
            print(f"  Failed trajectories: {len(results['failed_trajectories'])}")
    
    print(f"\n All checkpoints saved to checkpoints/ directory!")
    print(f" All visualizations saved to figures/ directory!")
    
    return all_results, all_analyses


if __name__ == "__main__":
    results, analyses = main()
