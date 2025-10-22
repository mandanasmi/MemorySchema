"""
Load and use the saved optimal policy for the single environment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from key_door_goal_base import create_key_door_goal_env
from key_door_goal_visualizer import KeyDoorGoalVisualizer


class SimplePolicy(torch.nn.Module):
    """Simple policy network for the agent"""
    
    def __init__(self, obs_shape, num_actions, hidden_size=64):
        super(SimplePolicy, self).__init__()
        
        # CNN encoder
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(obs_shape[2], 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate conv output size
        conv_output_size = 64 * 4 * 4
        
        # Policy and value heads
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(conv_output_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_actions)
        )
        
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(conv_output_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
        
    def forward(self, obs):
        """Forward pass"""
        # Convert to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
            
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


def load_optimal_policy(device='cpu'):
    """
    Load the saved optimal policy
    """
    import os
    
    if not os.path.exists('models/optimal_policy_single_env.pth'):
        raise FileNotFoundError("No saved policy model found. Please train first using train_and_analyze_conspec.py")
    
    # Load policy model
    policy_checkpoint = torch.load('models/optimal_policy_single_env.pth', map_location=device)
    obs_shape = policy_checkpoint['obs_shape']
    num_actions = policy_checkpoint['num_actions']
    success_rate = policy_checkpoint['success_rate']
    episode = policy_checkpoint['episode']
    
    print(f"Loading optimal policy:")
    print(f"  Trained for {episode} episodes")
    print(f"  Success rate: {success_rate:.3f}")
    print(f"  Observation shape: {obs_shape}")
    print(f"  Number of actions: {num_actions}")
    
    # Create and load policy
    policy = SimplePolicy(obs_shape, num_actions).to(device)
    policy.load_state_dict(policy_checkpoint['model_state_dict'])
    policy.eval()
    
    return policy, obs_shape, num_actions


def test_optimal_policy(num_episodes=20, device='cpu', visualize=True):
    """
    Test the optimal policy and optionally visualize episodes
    """
    print(f"\nTesting optimal policy on {num_episodes} episodes...")
    
    # Load policy
    policy, obs_shape, num_actions = load_optimal_policy(device)
    
    # Create environment
    env = create_key_door_goal_env('single', size=10)
    
    # Test results
    test_successes = 0
    test_rewards = []
    test_lengths = []
    successful_episodes = []
    failed_episodes = []
    
    for test_ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_observations = []
        episode_actions = []
        
        # Store initial state
        episode_observations.append(obs['image'].copy())
        
        while not done and episode_length < 200:
            with torch.no_grad():
                logits, _ = policy(obs['image'])
                action_probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1).item()
            
            episode_actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            # Store state after action
            episode_observations.append(obs['image'].copy())
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
        
        if episode_reward > 0:
            test_successes += 1
            successful_episodes.append({
                'observations': episode_observations,
                'actions': episode_actions,
                'reward': episode_reward,
                'length': episode_length
            })
        else:
            failed_episodes.append({
                'observations': episode_observations,
                'actions': episode_actions,
                'reward': episode_reward,
                'length': episode_length
            })
        
        print(f"  Episode {test_ep + 1}: Reward={episode_reward}, Length={episode_length}")
    
    env.close()
    
    # Calculate statistics
    test_success_rate = test_successes / num_episodes
    avg_reward = np.mean(test_rewards)
    avg_length = np.mean(test_lengths)
    
    print(f"\n OPTIMAL POLICY TEST RESULTS:")
    print(f"   Success rate: {test_success_rate:.2f} ({test_successes}/{num_episodes})")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Average length: {avg_length:.1f}")
    print(f"   Successful episodes: {len(successful_episodes)}")
    print(f"   Failed episodes: {len(failed_episodes)}")
    
    if test_success_rate >= 0.8:
        print(" Policy is performing excellently!")
    elif test_success_rate >= 0.6:
        print(" Policy is performing well!")
    else:
        print("  Policy may need more training")
    
    # Visualize episodes if requested
    if visualize and (successful_episodes or failed_episodes):
        visualize_episodes(successful_episodes, failed_episodes)
    
    return {
        'policy': policy,
        'test_success_rate': test_success_rate,
        'test_rewards': test_rewards,
        'test_lengths': test_lengths,
        'successful_episodes': successful_episodes,
        'failed_episodes': failed_episodes
    }


def visualize_episodes(successful_episodes, failed_episodes, max_episodes=6):
    """
    Visualize sample successful and failed episodes
    """
    print(f"\nCreating episode visualizations...")
    
    # Create visualizer
    visualizer = KeyDoorGoalVisualizer()
    
    # Sample episodes to visualize
    num_success = min(3, len(successful_episodes))
    num_failure = min(3, len(failed_episodes))
    
    if num_success == 0 and num_failure == 0:
        print("No episodes to visualize")
        return
    
    fig, axes = plt.subplots(2, max(num_success, num_failure), figsize=(5*max(num_success, num_failure), 10))
    if max(num_success, num_failure) == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Optimal Policy Episode Visualizations', fontsize=16, fontweight='bold')
    
    # Visualize successful episodes
    for i in range(num_success):
        episode = successful_episodes[i]
        
        # Create environment and replay to final state
        env = create_key_door_goal_env('single', size=10)
        obs, _ = env.reset()
        
        for action in episode['actions']:
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        
        # Visualize final state
        visualizer._draw_grid(axes[0, i], env, env.width, env.height)
        visualizer._add_legend(axes[0, i], env)
        axes[0, i].set_title(f'Success Episode {i+1}\nLength: {episode["length"]}, Reward: {episode["reward"]}')
        axes[0, i].set_aspect('equal')
        axes[0, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        env.close()
    
    # Visualize failed episodes
    for i in range(num_failure):
        episode = failed_episodes[i]
        
        # Create environment and replay to final state
        env = create_key_door_goal_env('single', size=10)
        obs, _ = env.reset()
        
        for action in episode['actions']:
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        
        # Visualize final state
        visualizer._draw_grid(axes[1, i], env, env.width, env.height)
        visualizer._add_legend(axes[1, i], env)
        axes[1, i].set_title(f'Failure Episode {i+1}\nLength: {episode["length"]}, Reward: {episode["reward"]}')
        axes[1, i].set_aspect('equal')
        axes[1, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        env.close()
    
    # Hide unused subplots
    for i in range(max(num_success, num_failure), axes.shape[1]):
        axes[0, i].set_visible(False)
        axes[1, i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/optimal_policy_episodes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Episode visualizations saved to figures/optimal_policy_episodes.png")


def demonstrate_optimal_policy(device='cpu'):
    """
    Demonstrate the optimal policy step by step
    """
    print(f"\nDemonstrating optimal policy step by step...")
    
    # Load policy
    policy, obs_shape, num_actions = load_optimal_policy(device)
    
    # Create environment
    env = create_key_door_goal_env('single', size=10)
    obs, _ = env.reset()
    
    print(f"Mission: {env.mission}")
    print("Starting demonstration...")
    
    step = 0
    done = False
    
    while not done and step < 50:
        with torch.no_grad():
            logits, value = policy(obs['image'])
            action_probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(action_probs, dim=-1).item()
            confidence = torch.max(action_probs).item()
        
        action_names = ['Right', 'Down', 'Left', 'Up', 'Pickup', 'Drop', 'Done']
        print(f"Step {step + 1}: Action={action_names[action]} (confidence: {confidence:.3f})")
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
        
        if reward > 0:
            print(f" SUCCESS! Reward: {reward}")
            break
        elif done:
            print(f"Episode ended. Final reward: {reward}")
    
    env.close()
    
    if reward > 0:
        print(" Optimal policy successfully completed the task!")
    else:
        print(" Optimal policy failed to complete the task")


def main():
    """Main function to test the optimal policy"""
    print("Optimal Policy Testing and Demonstration")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Test the policy
        test_results = test_optimal_policy(num_episodes=20, device=device, visualize=True)
        
        # Demonstrate step by step
        demonstrate_optimal_policy(device)
        
        print(f"\n SUMMARY:")
        print(f"   Policy success rate: {test_results['test_success_rate']:.2f}")
        print(f"   Average episode length: {np.mean(test_results['test_lengths']):.1f}")
        print(f"   Successful episodes: {len(test_results['successful_episodes'])}")
        print(f"   Failed episodes: {len(test_results['failed_episodes'])}")
        
    except FileNotFoundError as e:
        print(f" Error: {e}")
        print("Please run train_and_analyze_conspec.py first to train and save the optimal policy.")


if __name__ == "__main__":
    main()
