"""
Learn Posterior Distribution Over Graphs from Data

This script uses DAG-GFlowNet to learn a posterior distribution over causal
graphs from observational data. It supports custom data, various priors, and
different scoring functions.

Usage:
    python scripts/learn_posterior_from_data.py \
        --data trajectory_data.csv \
        --prior uniform \
        --score_type bge \
        --num_iterations 10000 \
        --output_dir results/

Author: Mandana Samiei
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rl_dag_gflownet'))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import networkx as nx
import pickle
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import trange
from typing import Optional, Dict, Tuple

from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.scores import BDeScore, BGeScore, priors
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.metrics import expected_edges
from dag_gflownet.utils.jraph_utils import to_graphs_tuple
from dag_gflownet.utils import io


class PosteriorLearner:
    """
    Learn posterior distribution over DAGs from data using GFlowNet
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        prior_type: str = 'uniform',
        score_type: str = 'bge',
        prior_kwargs: Optional[Dict] = None,
        scorer_kwargs: Optional[Dict] = None,
        seed: int = 0
    ):
        """
        Initialize the posterior learner
        
        Args:
            data: DataFrame with columns as variables and rows as samples
            prior_type: Type of prior ('uniform', 'erdos_renyi', 'edge', 'fair')
            score_type: Type of scoring function ('bge' for continuous, 'bde' for discrete)
            prior_kwargs: Additional arguments for prior
            scorer_kwargs: Additional arguments for scorer
            seed: Random seed
        """
        self.data = data
        self.num_variables = len(data.columns)
        self.seed = seed
        
        # Initialize random number generator
        self.rng = np.random.default_rng(seed)
        self.key = jax.random.PRNGKey(seed)
        
        # Get prior
        prior_kwargs = prior_kwargs or {}
        self.prior = self._get_prior(prior_type, **prior_kwargs)
        
        # Get scorer
        scorer_kwargs = scorer_kwargs or {}
        self.scorer = self._get_scorer(score_type, **scorer_kwargs)
        
        print(f"Initialized PosteriorLearner:")
        print(f"  - Data shape: {data.shape}")
        print(f"  - Number of variables: {self.num_variables}")
        print(f"  - Prior: {prior_type}")
        print(f"  - Score function: {score_type}")
        print(f"  - Seed: {seed}")
    
    def _get_prior(self, prior_type: str, **kwargs):
        """Get prior distribution over graphs"""
        priors_dict = {
            'uniform': priors.UniformPrior,
            'erdos_renyi': priors.ErdosRenyiPrior,
            'edge': priors.EdgePrior,
            'fair': priors.FairPrior
        }
        
        if prior_type not in priors_dict:
            raise ValueError(f"Unknown prior type: {prior_type}")
        
        # Set default kwargs for erdos_renyi
        if prior_type == 'erdos_renyi' and 'num_edges_per_node' not in kwargs:
            kwargs['num_edges_per_node'] = 1
        
        return priors_dict[prior_type](**kwargs)
    
    def _get_scorer(self, score_type: str, **kwargs):
        """Get scoring function"""
        scores_dict = {
            'bge': BGeScore,  # Bayesian Gaussian Equivalent (continuous data)
            'bde': BDeScore   # Bayesian Dirichlet Equivalent (discrete data)
        }
        
        if score_type not in scores_dict:
            raise ValueError(f"Unknown score type: {score_type}")
        
        return scores_dict[score_type](self.data, self.prior, **kwargs)
    
    def train(
        self,
        num_iterations: int = 100000,
        batch_size: int = 32,
        lr: float = 1e-5,
        delta: float = 1.0,
        num_envs: int = 8,
        replay_capacity: int = 100000,
        prefill: int = 1000,
        min_exploration: float = 0.1,
        num_workers: int = 4,
        log_every: int = 100,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Train GFlowNet to learn posterior distribution
        
        Args:
            num_iterations: Number of training iterations
            batch_size: Batch size for training
            lr: Learning rate
            delta: Delta for Huber loss
            num_envs: Number of parallel environments
            replay_capacity: Capacity of replay buffer
            prefill: Number of iterations to prefill buffer
            min_exploration: Minimum epsilon for exploration
            num_workers: Number of worker processes
            log_every: Logging frequency
            verbose: Whether to print progress
        
        Returns:
            posterior: Samples from posterior distribution
            metrics: Dictionary of training metrics
        """
        # Create environment
        env = GFlowNetDAGEnv(
            num_envs=num_envs,
            scorer=self.scorer,
            num_workers=num_workers,
            context='spawn'
        )
        
        # Create replay buffer
        replay = ReplayBuffer(
            replay_capacity,
            num_variables=env.num_variables
        )
        
        # Create GFlowNet
        gflownet = DAGGFlowNet(delta=delta)
        optimizer = optax.adam(lr)
        
        # Initialize parameters
        self.key, subkey = jax.random.split(self.key)
        params, state = gflownet.init(
            subkey,
            optimizer,
            replay.dummy['graph'],
            replay.dummy['mask']
        )
        
        # Exploration schedule
        exploration_schedule = jax.jit(optax.linear_schedule(
            init_value=jnp.array(0.),
            end_value=jnp.array(1. - min_exploration),
            transition_steps=num_iterations // 2,
            transition_begin=prefill,
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
        
        iterator = trange(prefill + num_iterations, desc='Training DAG-GFlowNet') if verbose else range(prefill + num_iterations)
        
        for iteration in iterator:
            # Sample actions and execute
            epsilon = exploration_schedule(iteration)
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            actions, self.key, logs = gflownet.act(params, self.key, observations, epsilon)
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
            if iteration >= prefill:
                samples = replay.sample(batch_size=batch_size, rng=self.rng)
                params, state, logs = gflownet.step(params, state, samples)
                
                train_steps = iteration - prefill
                
                # Log metrics
                if (train_steps + 1) % log_every == 0:
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
            print("\nSampling from learned posterior...")
        
        posterior, _ = posterior_estimate(
            gflownet,
            params,
            env,
            self.key,
            num_samples=1000,
            desc='Sampling posterior' if verbose else None
        )
        
        # Store for later use
        self.params = params
        self.gflownet = gflownet
        self.env = env
        self.replay = replay
        self.posterior = posterior
        
        # Compute final metrics
        metrics['expected_edges'] = float(expected_edges(posterior))
        metrics['num_unique_graphs'] = len(np.unique(posterior.reshape(len(posterior), -1), axis=0))
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"  - Final loss: {metrics['losses'][-1]:.4f}")
            print(f"  - Expected edges: {metrics['expected_edges']:.2f}")
            print(f"  - Unique graphs in posterior: {metrics['num_unique_graphs']}")
        
        return posterior, metrics
    
    def save_results(self, output_dir: str, prefix: str = ''):
        """
        Save learned posterior and model
        
        Args:
            output_dir: Directory to save results
            prefix: Prefix for output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{prefix}_" if prefix else ""
        
        # Save posterior
        posterior_file = output_path / f"{prefix}posterior_{timestamp}.npy"
        np.save(posterior_file, self.posterior)
        print(f"Saved posterior: {posterior_file}")
        
        # Save model parameters
        model_file = output_path / f"{prefix}model_{timestamp}.npz"
        io.save(str(model_file), params=self.params)
        print(f"Saved model: {model_file}")
        
        # Save data
        data_file = output_path / f"{prefix}data_{timestamp}.csv"
        self.data.to_csv(data_file, index=False)
        print(f"Saved data: {data_file}")
        
        # Save replay buffer
        replay_file = output_path / f"{prefix}replay_{timestamp}.npz"
        self.replay.save(str(replay_file))
        print(f"Saved replay buffer: {replay_file}")
        
        # Save configuration
        config = {
            'num_variables': self.num_variables,
            'data_shape': self.data.shape,
            'num_samples_posterior': len(self.posterior),
            'timestamp': timestamp
        }
        
        config_file = output_path / f"{prefix}config_{timestamp}.pkl"
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        print(f"Saved config: {config_file}")
    
    def analyze_posterior(self, threshold: float = 0.5) -> Dict:
        """
        Analyze the learned posterior distribution
        
        Args:
            threshold: Probability threshold for edge inclusion
        
        Returns:
            Dictionary with analysis results
        """
        # Compute edge probabilities
        edge_probs = self.posterior.mean(axis=0)
        
        # Get most likely graph (thresholded)
        most_likely_graph = (edge_probs > threshold).astype(int)
        
        # Count unique graphs
        posterior_flat = self.posterior.reshape(len(self.posterior), -1)
        unique_graphs, counts = np.unique(posterior_flat, axis=0, return_counts=True)
        
        # Get top graphs
        top_indices = np.argsort(counts)[::-1][:10]
        top_graphs = unique_graphs[top_indices]
        top_probs = counts[top_indices] / len(self.posterior)
        
        analysis = {
            'edge_probabilities': edge_probs,
            'most_likely_graph': most_likely_graph,
            'num_unique_graphs': len(unique_graphs),
            'top_graphs': top_graphs,
            'top_probabilities': top_probs,
            'expected_edges': float(expected_edges(self.posterior))
        }
        
        return analysis
    
    def visualize_posterior(self, output_file: Optional[str] = None):
        """
        Visualize the learned posterior
        
        Args:
            output_file: Path to save visualization (optional)
        """
        import matplotlib.pyplot as plt
        
        analysis = self.analyze_posterior()
        edge_probs = analysis['edge_probabilities']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot edge probabilities
        im = axes[0].imshow(edge_probs, cmap='Blues', vmin=0, vmax=1)
        axes[0].set_title('Edge Probabilities')
        axes[0].set_xlabel('To Variable')
        axes[0].set_ylabel('From Variable')
        
        # Add variable names if available
        if hasattr(self.data, 'columns'):
            axes[0].set_xticks(range(len(self.data.columns)))
            axes[0].set_yticks(range(len(self.data.columns)))
            axes[0].set_xticklabels(self.data.columns, rotation=45, ha='right')
            axes[0].set_yticklabels(self.data.columns)
        
        plt.colorbar(im, ax=axes[0], label='Probability')
        
        # Add probability values
        for i in range(edge_probs.shape[0]):
            for j in range(edge_probs.shape[1]):
                text = axes[0].text(j, i, f'{edge_probs[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        # Plot most likely graph
        most_likely = analysis['most_likely_graph']
        G = nx.DiGraph()
        
        if hasattr(self.data, 'columns'):
            node_labels = {i: col for i, col in enumerate(self.data.columns)}
            G.add_nodes_from(node_labels.keys())
        else:
            node_labels = {i: f'X{i}' for i in range(self.num_variables)}
            G.add_nodes_from(range(self.num_variables))
        
        for i in range(most_likely.shape[0]):
            for j in range(most_likely.shape[1]):
                if most_likely[i, j] == 1:
                    G.add_edge(i, j)
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=axes[1], with_labels=True, labels=node_labels,
                node_color='lightblue', node_size=1500, font_size=10,
                font_weight='bold', arrows=True, arrowsize=20,
                edge_color='gray', width=2)
        axes[1].set_title(f'Most Likely Graph (threshold=0.5)\n{G.number_of_edges()} edges')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {output_file}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Learn posterior distribution over causal graphs from data'
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data', type=str, required=True,
                           help='Path to CSV file with data')
    data_group.add_argument('--skip_rows', type=int, default=0,
                           help='Number of rows to skip')
    
    # Prior arguments
    prior_group = parser.add_argument_group('Prior')
    prior_group.add_argument('--prior', type=str, default='uniform',
                            choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
                            help='Prior over graphs')
    prior_group.add_argument('--prior_edges_per_node', type=float, default=1.0,
                            help='Expected edges per node (for erdos_renyi prior)')
    
    # Scoring arguments
    score_group = parser.add_argument_group('Scoring')
    score_group.add_argument('--score_type', type=str, default='bge',
                            choices=['bge', 'bde'],
                            help='Scoring function (bge for continuous, bde for discrete)')
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--num_iterations', type=int, default=100000,
                            help='Number of training iterations')
    train_group.add_argument('--batch_size', type=int, default=32,
                            help='Batch size')
    train_group.add_argument('--lr', type=float, default=1e-5,
                            help='Learning rate')
    train_group.add_argument('--num_envs', type=int, default=8,
                            help='Number of parallel environments')
    train_group.add_argument('--prefill', type=int, default=1000,
                            help='Prefill iterations for replay buffer')
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_dir', type=str, default='posterior_results',
                             help='Directory to save results')
    output_group.add_argument('--prefix', type=str, default='',
                             help='Prefix for output files')
    output_group.add_argument('--visualize', action='store_true',
                             help='Generate visualizations')
    
    # Miscellaneous
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--seed', type=int, default=0,
                           help='Random seed')
    misc_group.add_argument('--verbose', action='store_true',
                           help='Verbose output')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.data}")
    data = pd.read_csv(args.data, skiprows=args.skip_rows)
    print(f"Data shape: {data.shape}")
    print(f"Variables: {list(data.columns)}")
    
    # Setup prior kwargs
    prior_kwargs = {}
    if args.prior == 'erdos_renyi':
        prior_kwargs['num_edges_per_node'] = args.prior_edges_per_node
    
    # Create learner
    learner = PosteriorLearner(
        data=data,
        prior_type=args.prior,
        score_type=args.score_type,
        prior_kwargs=prior_kwargs,
        seed=args.seed
    )
    
    # Train
    posterior, metrics = learner.train(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        num_envs=args.num_envs,
        prefill=args.prefill,
        verbose=args.verbose
    )
    
    # Analyze
    print("\nAnalyzing posterior...")
    analysis = learner.analyze_posterior()
    
    print(f"\nPosterior Analysis:")
    print(f"  - Number of unique graphs: {analysis['num_unique_graphs']}")
    print(f"  - Expected number of edges: {analysis['expected_edges']:.2f}")
    print(f"\nTop 5 most probable graphs:")
    for i, (graph, prob) in enumerate(zip(analysis['top_graphs'][:5], analysis['top_probabilities'][:5])):
        num_edges = graph.sum()
        print(f"  {i+1}. Probability: {prob:.4f}, Edges: {num_edges}")
    
    print(f"\nEdge Probabilities:")
    print(analysis['edge_probabilities'])
    
    # Save results
    learner.save_results(args.output_dir, args.prefix)
    
    # Visualize
    if args.visualize:
        viz_file = Path(args.output_dir) / f"{args.prefix}_posterior_viz.png"
        learner.visualize_posterior(str(viz_file))
    
    print("\nDone!")


if __name__ == '__main__':
    main()

