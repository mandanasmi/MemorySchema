"""
Analyze hyperparameter search results and generate reports
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import argparse

class HyperparameterAnalyzer:
    """Analyze hyperparameter search results"""
    
    def __init__(self, results_dir: str = "hyperparam_search_results"):
        self.results_dir = Path(results_dir)
        self.results = []
        
    def load_results(self, pattern: str = "search_*.json"):
        """Load all result files"""
        result_files = list(self.results_dir.glob(pattern))
        
        print(f"Found {len(result_files)} result files")
        
        for file_path in result_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.results.extend(data)
                elif isinstance(data, dict) and 'results' in data:
                    self.results.extend(data['results'])
        
        print(f"Loaded {len(self.results)} trials")
        
    def create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        rows = []
        
        for trial in self.results:
            row = {
                'environment': trial['environment'],
                'trial': trial['trial'],
                **trial['hyperparameters'],
                **trial['results']
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def analyze_by_environment(self, df: pd.DataFrame):
        """Analyze results grouped by environment"""
        print("\n" + "="*70)
        print("Analysis by Environment")
        print("="*70 + "\n")
        
        for env in df['environment'].unique():
            env_df = df[df['environment'] == env]
            
            print(f"\n{env.upper()} Environment:")
            print(f"  Trials: {len(env_df)}")
            print(f"  Mean Reward: {env_df['mean_reward'].mean():.3f} ± {env_df['mean_reward'].std():.3f}")
            print(f"  Best Reward: {env_df['mean_reward'].max():.3f}")
            print(f"  Success Rate: {env_df['success_rate'].mean():.2%} ± {env_df['success_rate'].std():.2%}")
            
            # Best configuration
            best_idx = env_df['mean_reward'].idxmax()
            best_trial = env_df.loc[best_idx]
            
            print(f"\n  Best Configuration:")
            hyperparam_cols = ['num_prototypes', 'learning_rate', 'intrinsic_reward_scale', 
                              'gamma', 'epsilon_decay', 'hidden_dim', 'lstm_hidden_size', 
                              'batch_size', 'memory_size']
            for col in hyperparam_cols:
                if col in best_trial:
                    print(f"    {col}: {best_trial[col]}")
    
    def analyze_hyperparameter_importance(self, df: pd.DataFrame):
        """Analyze which hyperparameters matter most"""
        print("\n" + "="*70)
        print("Hyperparameter Importance Analysis")
        print("="*70 + "\n")
        
        hyperparam_cols = ['num_prototypes', 'learning_rate', 'intrinsic_reward_scale', 
                          'gamma', 'epsilon_decay', 'hidden_dim', 'lstm_hidden_size', 
                          'batch_size', 'memory_size']
        
        for env in df['environment'].unique():
            env_df = df[df['environment'] == env]
            
            print(f"\n{env.upper()} Environment:")
            
            correlations = {}
            for col in hyperparam_cols:
                if col in env_df.columns:
                    corr = env_df[col].corr(env_df['mean_reward'])
                    correlations[col] = abs(corr)
            
            # Sort by absolute correlation
            sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            print("\n  Correlation with Mean Reward:")
            for param, corr in sorted_correlations:
                print(f"    {param:25s}: {corr:.3f}")
    
    def plot_results(self, df: pd.DataFrame, save_dir: str = "figures"):
        """Generate visualization plots"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Reward distribution by environment
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, env in enumerate(['single', 'double', 'triple']):
            if env in df['environment'].values:
                env_df = df[df['environment'] == env]
                axes[idx].hist(env_df['mean_reward'], bins=20, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{env.capitalize()} Environment')
                axes[idx].set_xlabel('Mean Reward')
                axes[idx].set_ylabel('Frequency')
                axes[idx].axvline(env_df['mean_reward'].mean(), color='red', 
                                 linestyle='--', label='Mean')
                axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(save_path / 'reward_distributions.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path / 'reward_distributions.png'}")
        plt.close()
        
        # 2. Hyperparameter impact
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        key_params = ['num_prototypes', 'learning_rate', 'intrinsic_reward_scale', 'gamma']
        
        for idx, param in enumerate(key_params):
            if param in df.columns:
                for env in df['environment'].unique():
                    env_df = df[df['environment'] == env]
                    axes[idx].scatter(env_df[param], env_df['mean_reward'], 
                                     alpha=0.6, label=env)
                
                axes[idx].set_xlabel(param)
                axes[idx].set_ylabel('Mean Reward')
                axes[idx].set_title(f'Impact of {param}')
                axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(save_path / 'hyperparameter_impact.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path / 'hyperparameter_impact.png'}")
        plt.close()
        
        # 3. Top configurations comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top_configs = []
        for env in df['environment'].unique():
            env_df = df[df['environment'] == env]
            best_idx = env_df['mean_reward'].idxmax()
            best_trial = env_df.loc[best_idx]
            top_configs.append({
                'Environment': env.capitalize(),
                'Mean Reward': best_trial['mean_reward'],
                'Success Rate': best_trial['success_rate']
            })
        
        top_df = pd.DataFrame(top_configs)
        x = np.arange(len(top_df))
        width = 0.35
        
        ax.bar(x - width/2, top_df['Mean Reward'], width, label='Mean Reward', alpha=0.8)
        ax.bar(x + width/2, top_df['Success Rate'], width, label='Success Rate', alpha=0.8)
        
        ax.set_xlabel('Environment')
        ax.set_ylabel('Score')
        ax.set_title('Best Configuration Performance by Environment')
        ax.set_xticks(x)
        ax.set_xticklabels(top_df['Environment'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'best_configs_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path / 'best_configs_comparison.png'}")
        plt.close()
    
    def generate_report(self, output_file: str = "hyperparam_search_report.md"):
        """Generate markdown report"""
        df = self.create_dataframe()
        
        with open(output_file, 'w') as f:
            f.write("# ConSpec Hyperparameter Search Report\n\n")
            f.write(f"Total Trials: {len(df)}\n\n")
            
            f.write("## Best Configurations by Environment\n\n")
            
            for env in df['environment'].unique():
                env_df = df[df['environment'] == env]
                best_idx = env_df['mean_reward'].idxmax()
                best_trial = env_df.loc[best_idx]
                
                f.write(f"### {env.capitalize()} Environment\n\n")
                f.write(f"- **Mean Reward**: {best_trial['mean_reward']:.3f}\n")
                f.write(f"- **Success Rate**: {best_trial['success_rate']:.2%}\n")
                f.write(f"- **Max Reward**: {best_trial['max_reward']:.3f}\n\n")
                
                f.write("**Hyperparameters:**\n\n")
                hyperparam_cols = ['num_prototypes', 'learning_rate', 'intrinsic_reward_scale', 
                                  'gamma', 'epsilon_decay', 'hidden_dim', 'lstm_hidden_size', 
                                  'batch_size', 'memory_size']
                
                for col in hyperparam_cols:
                    if col in best_trial:
                        f.write(f"- `{col}`: {best_trial[col]}\n")
                f.write("\n")
            
            f.write("## Hyperparameter Correlations\n\n")
            f.write("(See analysis output above)\n\n")
            
            f.write("## Visualizations\n\n")
            f.write("- `figures/reward_distributions.png`\n")
            f.write("- `figures/hyperparameter_impact.png`\n")
            f.write("- `figures/best_configs_comparison.png`\n")
        
        print(f"\nReport saved: {output_file}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        self.load_results()
        
        if len(self.results) == 0:
            print("No results found!")
            return
        
        df = self.create_dataframe()
        
        # Print basic stats
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Environments: {df['environment'].unique()}")
        
        # Run analyses
        self.analyze_by_environment(df)
        self.analyze_hyperparameter_importance(df)
        self.plot_results(df)
        self.generate_report()
        
        print("\n" + "="*70)
        print("Analysis Complete!")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Analyze ConSpec hyperparameter search results')
    parser.add_argument('--results-dir', type=str, default='hyperparam_search_results',
                       help='Directory containing results')
    parser.add_argument('--output', type=str, default='hyperparam_search_report.md',
                       help='Output report file')
    
    args = parser.parse_args()
    
    analyzer = HyperparameterAnalyzer(args.results_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

