"""Example script demonstrating how to use the comparison module programmatically."""

import numpy as np
from src.env.env import make_blackjack_env
from src.comparison import (
    run_comparison,
    print_comparison_summary,
    save_comparison_report,
    generate_all_visualizations,
)


def main() -> None:
    """Run a simple comparison example."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create environment
    env = make_blackjack_env(natural=False, sab=True)
    
    print("\n" + "="*80)
    print(" " * 20 + "AGENT COMPARISON EXAMPLE")
    print("="*80 + "\n")
    
    # Run comparison with custom parameters
    results, agents_dict = run_comparison(
        env=env,
        n_train_episodes=1_000_000,  # Smaller for quick example
        n_eval_episodes=10_000,
        pretrained_dir="src/pretrained",
        force_retrain=False,  # Load pre-trained if available
        verbose=True
    )
    
    # Print summary
    print_comparison_summary(results)
    
    # Save detailed report
    save_comparison_report(
        results,
        save_path="results/comparison/example_report.txt"
    )
    
    # Generate all visualizations
    generate_all_visualizations(
        results,
        agents_dict,
        save_dir="results/comparison"
    )
    
    # Access specific results programmatically
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80 + "\n")
    
    for agent_name, data in results.items():
        print(f"\n{agent_name}:")
        print(f"  Evaluation Win Rate: {data['eval']['win_rate']:.4f}")
        print(f"  Evaluation Avg Reward: {data['eval']['avg_reward']:.4f}")
        
        if data['agent_stats']:
            stats = data['agent_stats']
            print(f"  Unique States Explored: {stats['unique_states']}")
            print(f"  Avg Epsilon: {stats['avg_current_epsilon']:.4f}")
    
    print("\n" + "="*80)
    print("✅ Example complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

