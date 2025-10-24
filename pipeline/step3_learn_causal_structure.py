"""
Step 3: Learn Causal Structure with DAG-GFlowNet

Train DAG-GFlowNet on expert binary trajectory data to learn posterior
distributions over causal graphs. Compare uniform vs temporal priors.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rl_dag_gflownet'))

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import networkx as nx
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.scores import BDeScore, BGeScore, priors
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.metrics import expected_edges
from dag_gflownet.utils.jraph_utils import to_graphs_tuple
from dag_gflownet.utils import io


class TemporalPrior(priors.BasePrior):
    """
    Temporal prior that favors edges from earlier features to later features
    based on the temporal order observed in expert trajectories
    """
    
    def __init__(self, temporal_order: List[str], strength: float = 1.0):
        """
        Initialize temporal prior
        
        Args:
            temporal_order: List of feature names in temporal order
            strength: Strength of temporal bias (higher = more temporal bias)
        """
        self.temporal_order = temporal_order
        self.strength = strength
        self.num_variables = len(temporal_order)
        
        # Create temporal bias matrix
        self.temporal_bias = np.zeros((self.num_variables, self.num_variables))
        
        for i, from_var in enumerate(temporal_order):
            for j, to_var in enumerate(temporal_order):
                if i < j:  # Earlier feature -> Later feature (favored)
                    self.temporal_bias[i, j] = strength
                elif i > j:  # Later feature -> Earlier feature (discouraged)
                    self.temporal_bias[i, j] = -strength * 0.5
                # i == j: self-loops (neutral)
    
    def log_prior(self, adjacency: np.ndarray) -> float:
        """Compute log prior probability"""
        # Base uniform prior
        num_edges = adjacency.sum()
        uniform_log_prior = -num_edges * np.log(2)  # Each edge has 0.5 probability
        
        # Add temporal bias
        temporal_log_prior = np.sum(adjacency * self.temporal_bias)
        
        return uniform_log_prior + temporal_log_prior


class CausalStructureLearner:
    """
    Learn causal structure from expert binary trajectory data using DAG-GFlowNet
    """
    
    def __init__(
        self,
        expert_data: pd.DataFrame,
        feature_columns: List[str],
        env_type: str,
        seed: int = 0
    ):
        """
        Initialize causal structure learner
        
        Args:
            expert_data: DataFrame with binary features
            feature_columns: List of binary feature column names
            env_type: Environment type for naming
            seed: Random seed
        """
        self.expert_data = expert_data
        self.feature_columns = feature_columns
        self.env_type = env_type
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        jax.random.PRNGKey(seed)
        
        # Prepare data for DAG-GFlowNet
        self.binary_data = expert_data[feature_columns].astype(int)
        self.num_variables = len(feature_columns)
        
        print(f"Initialized CausalStructureLearner for {env_type}")
        print(f"  - Variables: {self.num_variables}")
        print(f"  - Data points: {len(self.binary_data)}")
        print(f"  - Features: {feature_columns}")
    
    def create_temporal_prior(self, temporal_order: List[str], strength: float = 1.0):
        """Create temporal prior based on observed temporal order"""
        return TemporalPrior(temporal_order, strength)
    
    def train_with_prior(
        self,
        prior_type: str = 'uniform',
        temporal_order: Optional[List[str]] = None,
        temporal_strength: float = 1.0,
        num_iterations: int = 50000,
        batch_size: int = 32,
        lr: float = 1e-5,
        num_envs: int = 8,
        num_samples_posterior: int = 1000,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Train DAG-GFlowNet with specified prior
        
        Args:
            prior_type: 'uniform' or 'temporal'
            temporal_order: Order of features for temporal prior
            temporal_strength: Strength of temporal bias
            num_iterations: Training iterations
            batch_size: Batch size
            lr: Learning rate
            num_envs: Number of parallel environments
            num_samples_posterior: Samples for posterior estimate
            verbose: Show progress
        
        Returns:
            posterior: Samples from posterior
            metrics: Training metrics
        """
        # Create prior
        if prior_type == 'uniform':
            prior = priors.UniformPrior()
        elif prior_type == 'temporal':
            if temporal_order is None:
                # Infer temporal order from data
                temporal_order = self._infer_temporal_order()
            prior = self.create_temporal_prior(temporal_order, temporal_strength)
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        
        # Create scorer (BDe for binary data)
        scorer = BDeScore(self.binary_data, prior)
        
        # Create environment
        env = GFlowNetDAGEnv(
            num_envs=num_envs,
            scorer=scorer,
            num_workers=4,
            context='spawn'
        )
        
        # Create replay buffer
        replay = ReplayBuffer(
            capacity=100000,
            num_variables=env.num_variables
        )
        
        # Create GFlowNet
        gflownet = DAGGFlowNet(delta=1.0)
        optimizer = optax.adam(lr)
        
        # Initialize parameters
        key = jax.random.PRNGKey(self.seed)
        key, subkey = jax.random.split(key)
        params, state = gflownet.init(
            subkey,
            optimizer,
            replay.dummy['graph'],
            replay.dummy['mask']
        )
        
        # Exploration schedule
        exploration_schedule = jax.jit(optax.linear_schedule(
            init_value=jnp.array(0.),
            end_value=jnp.array(0.9),
            transition_steps=num_iterations // 2,
            transition_begin=1000,
        ))
        
        # Training metrics
        metrics = {
            'losses': [],
            'errors': [],
            'epsilons': [],
            'replay_sizes': []
        }
        
        # Training loop
        indices = None
        observations = env.reset()
        
        iterator = tqdm(range(1000 + num_iterations), desc=f'Training {prior_type} prior') if verbose else range(1000 + num_iterations)
        
        for iteration in iterator:
            # Sample actions and execute
            epsilon = exploration_schedule(iteration)
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            actions, key, logs = gflownet.act(params, key, observations, epsilon)
            next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
            
            # Add to replay buffer
            indices = replay.add(
                observations,
                actions,
                logs['is_exploration'],
                next_observations,
                delta_scores,
                dones,
                prev_indices=indices
            )
            observations = next_observations
            
            # Update parameters
            if iteration >= 1000:
                samples = replay.sample(batch_size=batch_size, rng=np.random.default_rng(self.seed))
                params, state, logs = gflownet.step(params, state, samples)
                
                train_steps = iteration - 1000
                
                # Log metrics
                if (train_steps + 1) % 100 == 0:
                    metrics['losses'].append(float(logs['loss']))
                    metrics['errors'].append(float(jnp.abs(logs['error']).mean()))
                    metrics['epsilons'].append(float(epsilon))
                    metrics['replay_sizes'].append(len(replay))
                    
                    if verbose and hasattr(iterator, 'set_postfix'):
                        iterator.set_postfix(
                            loss=f"{logs['loss']:.2f}",
                            epsilon=f"{epsilon:.2f}",
                            replay=len(replay)
                        )
        
        # Sample from posterior
        if verbose:
            print(f"\nSampling from {prior_type} posterior...")
        
        posterior, _ = posterior_estimate(
            gflownet,
            params,
            env,
            key,
            num_samples=num_samples_posterior,
            desc=f'Sampling {prior_type} posterior' if verbose else None
        )
        
        # Compute final metrics
        metrics['expected_edges'] = float(expected_edges(posterior))
        metrics['num_unique_graphs'] = len(np.unique(posterior.reshape(len(posterior), -1), axis=0))
        
        if verbose:
            print(f"\n{prior_type.capitalize()} prior training complete!")
            print(f"  - Final loss: {metrics['losses'][-1]:.4f}")
            print(f"  - Expected edges: {metrics['expected_edges']:.2f}")
            print(f"  - Unique graphs: {metrics['num_unique_graphs']}")
        
        return posterior, metrics
    
    def _infer_temporal_order(self) -> List[str]:
        """Infer temporal order from expert data"""
        # Find average first occurrence of each feature
        first_occurrences = {}
        
        for feature in self.feature_columns:
            # For each episode, find first step where feature is True
            episode_firsts = []
            for episode in self.expert_data['episode'].unique():
                ep_data = self.expert_data[self.expert_data['episode'] == episode]
                feature_true = ep_data[ep_data[feature] == 1]
                if len(feature_true) > 0:
                    episode_firsts.append(feature_true['step'].min())
            
            if episode_firsts:
                first_occurrences[feature] = np.mean(episode_firsts)
            else:
                first_occurrences[feature] = float('inf')
        
        # Sort by average first occurrence
        temporal_order = sorted(first_occurrences.keys(), key=lambda x: first_occurrences[x])
        
        print(f"Inferred temporal order: {temporal_order}")
        return temporal_order
    
    def compare_priors(
        self,
        temporal_strength: float = 1.0,
        num_iterations: int = 50000,
        verbose: bool = True
    ) -> Dict:
        """
        Compare uniform vs temporal priors
        
        Returns:
            Dictionary with results from both priors
        """
        results = {}
        
        # Train with uniform prior
        if verbose:
            print("="*60)
            print("Training with UNIFORM prior")
            print("="*60)
        
        uniform_posterior, uniform_metrics = self.train_with_prior(
            prior_type='uniform',
            num_iterations=num_iterations,
            verbose=verbose
        )
        
        results['uniform'] = {
            'posterior': uniform_posterior,
            'metrics': uniform_metrics
        }
        
        # Train with temporal prior
        if verbose:
            print("\n" + "="*60)
            print("Training with TEMPORAL prior")
            print("="*60)
        
        temporal_posterior, temporal_metrics = self.train_with_prior(
            prior_type='temporal',
            temporal_strength=temporal_strength,
            num_iterations=num_iterations,
            verbose=verbose
        )
        
        results['temporal'] = {
            'posterior': temporal_posterior,
            'metrics': temporal_metrics
        }
        
        return results
    
    def analyze_posteriors(self, results: Dict, threshold: float = 0.5) -> Dict:
        """Analyze and compare posteriors"""
        analysis = {}
        
        for prior_type, result in results.items():
            posterior = result['posterior']
            
            # Compute edge probabilities
            edge_probs = posterior.mean(axis=0)
            
            # Get most likely graph
            most_likely = (edge_probs > threshold).astype(int)
            
            # Count unique graphs
            posterior_flat = posterior.reshape(len(posterior), -1)
            unique_graphs, counts = np.unique(posterior_flat, axis=0, return_counts=True)
            
            # Get top graphs
            top_indices = np.argsort(counts)[::-1][:5]
            top_graphs = unique_graphs[top_indices]
            top_probs = counts[top_indices] / len(posterior)
            
            analysis[prior_type] = {
                'edge_probabilities': edge_probs,
                'most_likely_graph': most_likely,
                'num_unique_graphs': len(unique_graphs),
                'top_graphs': top_graphs,
                'top_probabilities': top_probs,
                'expected_edges': float(expected_edges(posterior))
            }
        
        return analysis
    
    def visualize_comparison(self, results: Dict, analysis: Dict, save_dir: str):
        """Visualize comparison between priors"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot edge probabilities for both priors
        for idx, prior_type in enumerate(['uniform', 'temporal']):
            edge_probs = analysis[prior_type]['edge_probabilities']
            
            im = axes[0, idx].imshow(edge_probs, cmap='Blues', vmin=0, vmax=1)
            axes[0, idx].set_title(f'{prior_type.capitalize()} Prior - Edge Probabilities')
            axes[0, idx].set_xlabel('To Variable')
            axes[0, idx].set_ylabel('From Variable')
            
            # Add variable names
            axes[0, idx].set_xticks(range(len(self.feature_columns)))
            axes[0, idx].set_yticks(range(len(self.feature_columns)))
            axes[0, idx].set_xticklabels(self.feature_columns, rotation=45, ha='right')
            axes[0, idx].set_yticklabels(self.feature_columns)
            
            plt.colorbar(im, ax=axes[0, idx], label='Probability')
            
            # Add probability values
            for i in range(edge_probs.shape[0]):
                for j in range(edge_probs.shape[1]):
                    text = axes[0, idx].text(j, i, f'{edge_probs[i, j]:.2f}',
                                           ha="center", va="center", color="black", fontsize=8)
        
        # Plot most likely graphs
        for idx, prior_type in enumerate(['uniform', 'temporal']):
            most_likely = analysis[prior_type]['most_likely_graph']
            
            # Create networkx graph
            G = nx.DiGraph()
            G.add_nodes_from(range(len(self.feature_columns)))
            
            for i in range(most_likely.shape[0]):
                for j in range(most_likely.shape[1]):
                    if most_likely[i, j] == 1:
                        G.add_edge(i, j)
            
            pos = nx.spring_layout(G, seed=42)
            node_labels = {i: self.feature_columns[i] for i in range(len(self.feature_columns))}
            
            nx.draw(G, pos, ax=axes[1, idx], with_labels=True, labels=node_labels,
                   node_color='lightblue', node_size=1500, font_size=10,
                   font_weight='bold', arrows=True, arrowsize=20,
                   edge_color='gray', width=2)
            
            axes[1, idx].set_title(f'{prior_type.capitalize()} Prior - Most Likely Graph\n{G.number_of_edges()} edges')
        
        # Comparison plot
        uniform_edges = analysis['uniform']['expected_edges']
        temporal_edges = analysis['temporal']['expected_edges']
        
        axes[0, 2].bar(['Uniform', 'Temporal'], [uniform_edges, temporal_edges], 
                      color=['lightblue', 'lightcoral'], alpha=0.7)
        axes[0, 2].set_ylabel('Expected Number of Edges')
        axes[0, 2].set_title('Expected Edges Comparison')
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate([uniform_edges, temporal_edges]):
            axes[0, 2].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
        
        # Top graph probabilities
        uniform_top = analysis['uniform']['top_probabilities'][:3]
        temporal_top = analysis['temporal']['top_probabilities'][:3]
        
        x = np.arange(3)
        width = 0.35
        
        axes[1, 2].bar(x - width/2, uniform_top, width, label='Uniform', alpha=0.7)
        axes[1, 2].bar(x + width/2, temporal_top, width, label='Temporal', alpha=0.7)
        
        axes[1, 2].set_xlabel('Graph Rank')
        axes[1, 2].set_ylabel('Probability')
        axes[1, 2].set_title('Top 3 Graph Probabilities')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(['1st', '2nd', '3rd'])
        axes[1, 2].legend()
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / f'{self.env_type}_prior_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved comparison: {save_path / f'{self.env_type}_prior_comparison.png'}")
        plt.close()
    
    def save_results(self, results: Dict, analysis: Dict, output_dir: str):
        """Save all results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save posteriors
        for prior_type, result in results.items():
            posterior_file = output_path / f'{self.env_type}_{prior_type}_posterior_{timestamp}.npy'
            np.save(posterior_file, result['posterior'])
            print(f"Saved {prior_type} posterior: {posterior_file}")
        
        # Save analysis
        analysis_file = output_path / f'{self.env_type}_analysis_{timestamp}.pkl'
        with open(analysis_file, 'wb') as f:
            pickle.dump(analysis, f)
        print(f"Saved analysis: {analysis_file}")
        
        # Save summary
        summary = {
            'env_type': self.env_type,
            'feature_columns': self.feature_columns,
            'num_variables': self.num_variables,
            'timestamp': timestamp,
            'results': {
                prior_type: {
                    'expected_edges': analysis[prior_type]['expected_edges'],
                    'num_unique_graphs': analysis[prior_type]['num_unique_graphs'],
                    'top_graph_probabilities': analysis[prior_type]['top_probabilities'].tolist()
                }
                for prior_type in ['uniform', 'temporal']
            }
        }
        
        summary_file = output_path / f'{self.env_type}_summary_{timestamp}.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {summary_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 3: Learn Causal Structure with DAG-GFlowNet')
    parser.add_argument('--env', type=str, required=True, choices=['single', 'double', 'triple'],
                       help='Environment type')
    parser.add_argument('--expert_data', type=str, required=True,
                       help='Path to expert data CSV from Step 2')
    parser.add_argument('--num_iterations', type=int, default=50000,
                       help='Number of training iterations')
    parser.add_argument('--temporal_strength', type=float, default=1.0,
                       help='Strength of temporal prior')
    parser.add_argument('--output_dir', type=str, default='pipeline_results/step3',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load expert data
    print(f"Loading expert data from: {args.expert_data}")
    expert_data = pd.read_csv(args.expert_data)
    print(f"Data shape: {expert_data.shape}")
    
    # Define binary features to use
    binary_features = [
        'has_blue_key', 'has_green_key', 'has_purple_key',
        'blue_door_open', 'green_door_open', 'purple_door_open',
        'at_green_goal', 'at_purple_goal'
    ]
    
    # Filter to only features that exist in the data
    available_features = [f for f in binary_features if f in expert_data.columns]
    print(f"Using features: {available_features}")
    
    # Create learner
    learner = CausalStructureLearner(
        expert_data=expert_data,
        feature_columns=available_features,
        env_type=args.env,
        seed=args.seed
    )
    
    # Compare priors
    results = learner.compare_priors(
        temporal_strength=args.temporal_strength,
        num_iterations=args.num_iterations,
        verbose=True
    )
    
    # Analyze results
    analysis = learner.analyze_posteriors(results)
    
    # Print comparison
    print("\n" + "="*60)
    print("PRIOR COMPARISON RESULTS")
    print("="*60)
    
    for prior_type in ['uniform', 'temporal']:
        print(f"\n{prior_type.upper()} PRIOR:")
        print(f"  Expected edges: {analysis[prior_type]['expected_edges']:.2f}")
        print(f"  Unique graphs: {analysis[prior_type]['num_unique_graphs']}")
        print(f"  Top graph probability: {analysis[prior_type]['top_probabilities'][0]:.4f}")
    
    # Visualize
    learner.visualize_comparison(results, analysis, args.output_dir)
    
    # Save results
    learner.save_results(results, analysis, args.output_dir)
    
    print("\nStep 3 completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

