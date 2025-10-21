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
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, policy, trainer, conspec, device):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    policy.load_state_dict(checkpoint['policy_state_dict'])
    trainer.load_state_dict(checkpoint['trainer_state_dict'])
    conspec.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    conspec.prototype_net.load_state_dict(checkpoint['prototype_net_state_dict'])
    
    # Restore memory
    conspec.memory.success_memory.clear()
    conspec.memory.failure_memory.clear()
    for traj in checkpoint['memory']['success_memory']:
        conspec.memory.success_memory.append(traj)
    for traj in checkpoint['memory']['failure_memory']:
        conspec.memory.failure_memory.append(traj)
    
    print(f"✅ Checkpoint loaded from episode {checkpoint['episode']}")
    print(f"   Best success rate: {checkpoint['best_success_rate']:.3f}")
    print(f"   Memory - Success: {len(conspec.memory.success_memory)}, "
          f"Failure: {len(conspec.memory.failure_memory)}")
    
    return checkpoint


def train_conspec_on_environment(env_type, num_prototypes, num_episodes=3000, device='cpu',
                                 checkpoint_dir='checkpoints', resume=True, use_wandb=True,
                                 save_interval=500):
    """
    Train ConSpec on a specific environment with checkpoint support
    """
    print(f"\n{'='*70}")
    print(f"Training ConSpec on {env_type.upper()} Environment")
    print(f"Number of Prototypes: {num_prototypes}")
    print(f"{'='*70}")
    
    # Create environment
    env = create_key_door_goal_env(env_type, size=10)
    obs_shape = (env.height, env.width, 3)
    num_actions = env.action_space.n
    
    # Create policy and ConSpec
    policy = SimplePolicy(obs_shape, num_actions).to(device)
    conspec = ConSpec(obs_shape, num_actions, device, 
                     intrinsic_reward_scale=0.5, 
                     learning_rate=1e-3,
                     num_prototypes=num_prototypes)
    
    # Create trainer
    trainer = PPOTrainer(policy, device, lr=3e-4, clip_ratio=0.2, 
                        value_coef=0.5, entropy_coef=0.05)
    
    # Training state
    start_episode = 0
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_rate': [],
        'intrinsic_rewards': []
    }
    best_success_rate = 0.0
    
    # Try to resume from checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, 
                                    f'checkpoint_{env_type}_proto{num_prototypes}_latest.pth')
    
    if resume and os.path.exists(latest_checkpoint):
        checkpoint = load_checkpoint(latest_checkpoint, policy, trainer, conspec, device)
        if checkpoint:
            start_episode = checkpoint['episode'] + 1
            training_stats = checkpoint['training_stats']
            best_success_rate = checkpoint['best_success_rate']
            print(f"🔄 Resuming training from episode {start_episode}")
    else:
        print(f"🆕 Starting new training from scratch")
    
    print(f"Environment: {env.__class__.__name__}")
    print(f"Grid size: {env.width}x{env.height}")
    print(f"Mission: {env.mission}")
    print(f"Training episodes: {start_episode} → {num_episodes}")
    print()
    
    # Initialize wandb run
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project='schema-learning',
            entity='samieima',
            name=f'{env_type}_proto{num_prototypes}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            config={
                'env_type': env_type,
                'num_prototypes': num_prototypes,
                'num_episodes': num_episodes,
                'intrinsic_reward_scale': 0.5,
                'learning_rate': 3e-4,
                'resume': resume and os.path.exists(latest_checkpoint)
            },
            resume='allow' if resume else False
        )
    
    # Training loop
    for episode in range(start_episode, num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        episode_observations = []
        episode_actions = []
        episode_rewards_list = []
        episode_intrinsic_rewards = []
        episode_log_probs = []
        
        # Epsilon-greedy exploration
        epsilon = max(0.1, 1.0 - episode / (num_episodes * 0.7))
        
        while not done and episode_length < 200:
            with torch.no_grad():
                logits, value = policy(obs['image'])
                action_probs = torch.softmax(logits, dim=-1)
                
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = torch.multinomial(action_probs, 1).item()
                
                log_prob = torch.log(action_probs[0, action] + 1e-8).item()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_observations.append(obs['image'])
            episode_actions.append(action)
            episode_rewards_list.append(reward)
            episode_log_probs.append(log_prob)
            
            intrinsic_reward = conspec.compute_intrinsic_reward(
                [obs], [action], [reward], [done]
            )
            episode_intrinsic_rewards.append(intrinsic_reward.item())
            
            conspec.train_step([obs], [action], [reward], [done])
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        # Update policy
        if len(episode_observations) > 0:
            total_rewards = np.array(episode_rewards_list) + np.array(episode_intrinsic_rewards)
            losses = trainer.update(episode_observations, episode_actions, total_rewards,
                                   [done] * len(episode_actions), episode_log_probs)
        
        # Store stats
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_length)
        training_stats['success_rate'].append(1 if episode_reward > 0 else 0)
        training_stats['intrinsic_rewards'].extend(episode_intrinsic_rewards)
        
        # Calculate metrics
        recent_success_rate = np.mean(training_stats['success_rate'][-50:]) if len(training_stats['success_rate']) >= 50 else np.mean(training_stats['success_rate'])
        recent_reward = np.mean(training_stats['episode_rewards'][-50:]) if len(training_stats['episode_rewards']) >= 50 else np.mean(training_stats['episode_rewards'])
        
        # Update best success rate
        if recent_success_rate > best_success_rate:
            best_success_rate = recent_success_rate
            
            # Save best model
            best_model_path = os.path.join(checkpoint_dir, 
                                          f'best_model_{env_type}_proto{num_prototypes}.pth')
            torch.save({
                'episode': episode,
                'env_type': env_type,
                'num_prototypes': num_prototypes,
                'policy_state_dict': policy.state_dict(),
                'encoder_state_dict': conspec.encoder.state_dict(),
                'prototype_net_state_dict': conspec.prototype_net.state_dict(),
                'best_success_rate': best_success_rate,
                'obs_shape': obs_shape,
                'num_actions': num_actions
            }, best_model_path)
        
        # Log to wandb
        if wandb_run:
            wandb.log({
                'episode': episode,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'success_rate': recent_success_rate,
                'best_success_rate': best_success_rate,
                'avg_intrinsic_reward': np.mean(episode_intrinsic_rewards),
                'epsilon': epsilon,
                'memory_success': len(conspec.memory.success_memory),
                'memory_failure': len(conspec.memory.failure_memory),
            })
        
        # Save checkpoint periodically
        if (episode + 1) % save_interval == 0:
            save_checkpoint(checkpoint_dir, env_type, num_prototypes, episode,
                          policy, trainer, conspec, training_stats, best_success_rate)
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}: "
                  f"Success Rate: {recent_success_rate:.3f}, "
                  f"Best: {best_success_rate:.3f}, "
                  f"Reward: {recent_reward:.2f}, "
                  f"Epsilon: {epsilon:.3f}")
    
    # Save final checkpoint
    save_checkpoint(checkpoint_dir, env_type, num_prototypes, num_episodes - 1,
                   policy, trainer, conspec, training_stats, best_success_rate)
    
    env.close()
    
    if wandb_run:
        wandb.finish()
    
    return {
        'env_type': env_type,
        'num_prototypes': num_prototypes,
        'best_success_rate': best_success_rate,
        'training_stats': training_stats
    }


def main():
    """Main training function with hyperparameter search"""
    parser = argparse.ArgumentParser(description='Train ConSpec on Mila cluster')
    parser.add_argument('--env', type=str, default='all', choices=['all', 'single', 'double', 'triple'],
                       help='Which environment to train on')
    parser.add_argument('--episodes', type=int, default=3000, help='Number of training episodes')
    parser.add_argument('--min-prototypes', type=int, default=3, help='Minimum number of prototypes')
    parser.add_argument('--max-prototypes', type=int, default=7, help='Maximum number of prototypes')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--no-resume', action='store_true', help='Do not resume from checkpoint')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--save-interval', type=int, default=500, help='Save checkpoint every N episodes')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ConSpec Training on Mila Cluster")
    print("Hyperparameter Search: Number of Prototypes")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Resume from checkpoint: {not args.no_resume}")
    print(f"Wandb logging: {not args.no_wandb and WANDB_AVAILABLE}")
    print()
    
    # Determine environments
    if args.env == 'all':
        env_types = ['single', 'double', 'triple']
    else:
        env_types = [args.env]
    
    # Prototype range
    prototype_range = range(args.min_prototypes, args.max_prototypes + 1)
    
    print(f"Environments: {env_types}")
    print(f"Prototypes to try: {list(prototype_range)}")
    print(f"Episodes per configuration: {args.episodes}")
    print()
    
    # Track all results
    all_results = []
    
    # Train on all combinations
    for env_type in env_types:
        for num_prototypes in prototype_range:
            print(f"\n{'#'*70}")
            print(f"Configuration: {env_type.upper()} with {num_prototypes} prototypes")
            print(f"{'#'*70}")
            
            result = train_conspec_on_environment(
                env_type=env_type,
                num_prototypes=num_prototypes,
                num_episodes=args.episodes,
                device=device,
                checkpoint_dir=args.checkpoint_dir,
                resume=not args.no_resume,
                use_wandb=not args.no_wandb and WANDB_AVAILABLE,
                save_interval=args.save_interval
            )
            
            all_results.append(result)
    
    # Save summary of all results
    summary_path = os.path.join(args.checkpoint_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'results': [
                {
                    'env_type': r['env_type'],
                    'num_prototypes': r['num_prototypes'],
                    'best_success_rate': r['best_success_rate'],
                    'final_success_rate': np.mean(r['training_stats']['success_rate'][-100:]) if len(r['training_stats']['success_rate']) >= 100 else np.mean(r['training_stats']['success_rate'])
                }
                for r in all_results
            ],
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    for result in all_results:
        final_success = np.mean(result['training_stats']['success_rate'][-100:]) if len(result['training_stats']['success_rate']) >= 100 else np.mean(result['training_stats']['success_rate'])
        print(f"\n{result['env_type'].upper()} - {result['num_prototypes']} prototypes:")
        print(f"  Best success rate: {result['best_success_rate']:.3f}")
        print(f"  Final success rate: {final_success:.3f}")
    
    print(f"\n📁 All checkpoints saved to {args.checkpoint_dir}/")
    print(f"📊 Summary saved to {summary_path}")
    
    return all_results


if __name__ == "__main__":
    results = main()
