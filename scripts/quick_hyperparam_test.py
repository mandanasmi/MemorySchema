"""
Quick hyperparameter search test with fewer trials
Use this for local testing before running full search on Mila
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics_experiments.hyperparameter_search_conspec import HyperparameterSearcher

def main():
    print("Quick Hyperparameter Search Test")
    print("="*70)
    
    # Create searcher with reduced trials
    searcher = HyperparameterSearcher()
    
    # Override for quick test
    searcher.config['search']['n_trials'] = 3  # Just 3 trials
    searcher.config['training']['episodes'] = 500  # Fewer episodes
    searcher.config['logging']['log_frequency'] = 50
    
    # Test on single environment only
    print("\nRunning quick test on 'single' environment...")
    print("Trials: 3")
    print("Episodes per trial: 500")
    print()
    
    searcher.run_search(env_type='single')
    searcher.save_summary()
    
    print("\nQuick test completed!")
    print("Check hyperparam_search_results/ for results")

if __name__ == "__main__":
    main()

