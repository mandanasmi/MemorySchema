"""
Complete Pipeline: ConSpec → Expert Trajectories → DAG-GFlowNet

This script runs the complete pipeline:
1. Train ConSpec and analyze prototypes
2. Collect expert trajectories with binary features
3. Learn causal structure with DAG-GFlowNet (uniform vs temporal priors)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import json

from step1_conspec_analysis import ConSpecAnalyzer
from step2_expert_trajectories import ExpertTrajectoryCollector
from step3_learn_causal_structure import CausalStructureLearner


class FullPipeline:
    """
    Complete pipeline for learning causal structure from RL trajectories
    """
    
    def __init__(
        self,
        env_type: str,
        output_dir: str = 'pipeline_results',
        seed: int = 0
    ):
        """
        Initialize full pipeline
        
        Args:
            env_type: Environment type ('single', 'double', 'triple')
            output_dir: Base output directory
            seed: Random seed
        """
        self.env_type = env_type
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"{env_type}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized FullPipeline for {env_type}")
        print(f"  - Run directory: {self.run_dir}")
        print(f"  - Seed: {seed}")
    
    def run_step1(
        self,
        num_episodes: int = 5000,
        num_prototypes: int = 5,
        **kwargs
    ) -> str:
        """
        Run Step 1: ConSpec Training and Analysis
        
        Returns:
            Path to Step 1 results
        """
        print("\n" + "="*60)
        print("STEP 1: CONSPEC TRAINING AND ANALYSIS")
        print("="*60)
        
        step1_dir = self.run_dir / "step1"
        
        # Create analyzer
        analyzer = ConSpecAnalyzer(
            env_type=self.env_type,
            num_prototypes=num_prototypes,
            seed=self.seed
        )
        
        # Train
        results = analyzer.train(
            num_episodes=num_episodes,
            verbose=True,
            **kwargs
        )
        
        # Save results
        analyzer.save_results(str(step1_dir))
        analyzer.visualize_metrics(str(step1_dir))
        
        print(f"\nStep 1 completed! Results saved to: {step1_dir}")
        return str(step1_dir)
    
    def run_step2(
        self,
        step1_dir: str,
        num_trajectories: int = 1000,
        min_successful: int = 100,
        **kwargs
    ) -> str:
        """
        Run Step 2: Expert Trajectory Collection
        
        Args:
            step1_dir: Path to Step 1 results
            num_trajectories: Number of trajectories to collect
            min_successful: Minimum successful trajectories
        
        Returns:
            Path to expert data CSV
        """
        print("\n" + "="*60)
        print("STEP 2: EXPERT TRAJECTORY COLLECTION")
        print("="*60)
        
        step2_dir = self.run_dir / "step2"
        
        # For now, we'll create a mock collector since we need to load the actual policy
        # In production, load the trained policy from step1_dir
        print("Note: Loading policy from Step 1 results...")
        
        # Create mock policy and encoder (in production, load from step1_dir)
        import torch
        from methods.conspec_minigrid import EncoderConSpec
        from environments.key_door_goal_base import create_key_door_goal_env
        
        env = create_key_door_goal_env(self.env_type)
        obs_shape = env.observation_space['image'].shape
        
        encoder = EncoderConSpec(input_channels=obs_shape[2], hidden_dim=128, lstm_hidden_size=128)
        policy = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, env.action_space.n)
        )
        
        # Create collector
        collector = ExpertTrajectoryCollector(
            env_type=self.env_type,
            policy=policy,
            encoder=encoder,
            seed=self.seed
        )
        
        # Collect data
        df = collector.collect_expert_data(
            num_trajectories=num_trajectories,
            success_only=True,
            min_successful=min_successful,
            verbose=True,
            **kwargs
        )
        
        # Save data
        csv_file = collector.save_expert_data(df, str(step2_dir))
        collector.visualize_feature_timeline(df, str(step2_dir))
        
        print(f"\nStep 2 completed! Expert data saved to: {csv_file}")
        return str(csv_file)
    
    def run_step3(
        self,
        expert_data_path: str,
        num_iterations: int = 50000,
        temporal_strength: float = 1.0,
        **kwargs
    ) -> str:
        """
        Run Step 3: Learn Causal Structure with DAG-GFlowNet
        
        Args:
            expert_data_path: Path to expert data CSV
            num_iterations: Training iterations
            temporal_strength: Strength of temporal prior
        
        Returns:
            Path to Step 3 results
        """
        print("\n" + "="*60)
        print("STEP 3: LEARN CAUSAL STRUCTURE WITH DAG-GFLOWNET")
        print("="*60)
        
        step3_dir = self.run_dir / "step3"
        
        # Load expert data
        import pandas as pd
        expert_data = pd.read_csv(expert_data_path)
        
        # Define binary features
        binary_features = [
            'has_blue_key', 'has_green_key', 'has_purple_key',
            'blue_door_open', 'green_door_open', 'purple_door_open',
            'at_green_goal', 'at_purple_goal'
        ]
        
        # Filter to available features
        available_features = [f for f in binary_features if f in expert_data.columns]
        
        # Create learner
        learner = CausalStructureLearner(
            expert_data=expert_data,
            feature_columns=available_features,
            env_type=self.env_type,
            seed=self.seed
        )
        
        # Compare priors
        results = learner.compare_priors(
            temporal_strength=temporal_strength,
            num_iterations=num_iterations,
            verbose=True,
            **kwargs
        )
        
        # Analyze results
        analysis = learner.analyze_posteriors(results)
        
        # Visualize
        learner.visualize_comparison(results, analysis, str(step3_dir))
        
        # Save results
        learner.save_results(results, analysis, str(step3_dir))
        
        print(f"\nStep 3 completed! Results saved to: {step3_dir}")
        return str(step3_dir)
    
    def run_full_pipeline(
        self,
        step1_kwargs: dict = None,
        step2_kwargs: dict = None,
        step3_kwargs: dict = None
    ) -> Dict[str, str]:
        """
        Run the complete pipeline
        
        Returns:
            Dictionary with paths to results from each step
        """
        step1_kwargs = step1_kwargs or {}
        step2_kwargs = step2_kwargs or {}
        step3_kwargs = step3_kwargs or {}
        
        print(f"\n{'='*80}")
        print(f"STARTING FULL PIPELINE FOR {self.env_type.upper()} ENVIRONMENT")
        print(f"{'='*80}")
        
        results = {}
        
        # Step 1: ConSpec Training
        step1_dir = self.run_step1(**step1_kwargs)
        results['step1'] = step1_dir
        
        # Step 2: Expert Trajectory Collection
        expert_data_path = self.run_step2(step1_dir, **step2_kwargs)
        results['step2'] = expert_data_path
        
        # Step 3: Causal Structure Learning
        step3_dir = self.run_step3(expert_data_path, **step3_kwargs)
        results['step3'] = step3_dir
        
        # Save pipeline summary
        summary = {
            'env_type': self.env_type,
            'seed': self.seed,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        summary_file = self.run_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Results directory: {self.run_dir}")
        print(f"Summary: {summary_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Run complete ConSpec → DAG-GFlowNet pipeline')
    
    # Pipeline arguments
    parser.add_argument('--env', type=str, required=True, choices=['single', 'double', 'triple'],
                       help='Environment type')
    parser.add_argument('--output_dir', type=str, default='pipeline_results',
                       help='Base output directory')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    # Step 1 arguments
    parser.add_argument('--step1_episodes', type=int, default=5000,
                       help='Number of ConSpec training episodes')
    parser.add_argument('--step1_prototypes', type=int, default=5,
                       help='Number of ConSpec prototypes')
    
    # Step 2 arguments
    parser.add_argument('--step2_trajectories', type=int, default=1000,
                       help='Number of expert trajectories to collect')
    parser.add_argument('--step2_min_successful', type=int, default=100,
                       help='Minimum successful trajectories')
    
    # Step 3 arguments
    parser.add_argument('--step3_iterations', type=int, default=50000,
                       help='Number of DAG-GFlowNet training iterations')
    parser.add_argument('--step3_temporal_strength', type=float, default=1.0,
                       help='Strength of temporal prior')
    
    # Control arguments
    parser.add_argument('--skip_step1', action='store_true',
                       help='Skip Step 1 (use existing results)')
    parser.add_argument('--skip_step2', action='store_true',
                       help='Skip Step 2 (use existing data)')
    parser.add_argument('--step1_dir', type=str,
                       help='Path to existing Step 1 results')
    parser.add_argument('--expert_data', type=str,
                       help='Path to existing expert data')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = FullPipeline(
        env_type=args.env,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Prepare step arguments
    step1_kwargs = {
        'num_episodes': args.step1_episodes,
        'num_prototypes': args.step1_prototypes
    }
    
    step2_kwargs = {
        'num_trajectories': args.step2_trajectories,
        'min_successful': args.step2_min_successful
    }
    
    step3_kwargs = {
        'num_iterations': args.step3_iterations,
        'temporal_strength': args.step3_temporal_strength
    }
    
    # Run pipeline
    if args.skip_step1 and args.skip_step2:
        # Only run Step 3
        if not args.expert_data:
            raise ValueError("Must provide --expert_data when skipping steps 1 and 2")
        
        results = {}
        results['step3'] = pipeline.run_step3(args.expert_data, **step3_kwargs)
        
    elif args.skip_step1:
        # Run Steps 2 and 3
        if not args.step1_dir:
            raise ValueError("Must provide --step1_dir when skipping step 1")
        
        results = {}
        results['step1'] = args.step1_dir
        expert_data_path = pipeline.run_step2(args.step1_dir, **step2_kwargs)
        results['step2'] = expert_data_path
        results['step3'] = pipeline.run_step3(expert_data_path, **step3_kwargs)
        
    else:
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            step1_kwargs=step1_kwargs,
            step2_kwargs=step2_kwargs,
            step3_kwargs=step3_kwargs
        )
    
    print(f"\nPipeline completed! All results in: {pipeline.run_dir}")


if __name__ == '__main__':
    main()

