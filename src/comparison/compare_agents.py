"""Main entry point for running agent comparisons."""

import argparse
from typing import Optional
import numpy as np

from ..env.env import make_blackjack_env
from .runner import run_comparison, print_comparison_summary, save_comparison_report
from .visualizations import generate_all_visualizations


def main(
    n_train_episodes: int = 250_000,
    n_eval_episodes: int = 10_000,
    pretrained_dir: str = "src/pretrained",
    output_dir: str = "results/comparison",
    force_retrain: bool = False,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> None:
    """Run complete agent comparison with visualizations.
    
    Args:
        n_train_episodes: Number of training episodes (default: 250,000)
        n_eval_episodes: Number of evaluation episodes (default: 10,000)
        pretrained_dir: Directory containing pre-trained agents
        output_dir: Directory to save results and visualizations
        force_retrain: Whether to force retraining even if pretrained agents exist
        seed: Random seed for reproducibility (None for no seed)
        verbose: Whether to show detailed progress and output
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        print(f"🎲 Random seed set to {seed}\n")
    
    # Create environment
    env = make_blackjack_env(natural=False, sab=True)
    
    # Run comparison
    results, agents_dict = run_comparison(
        env=env,
        n_train_episodes=n_train_episodes,
        n_eval_episodes=n_eval_episodes,
        pretrained_dir=pretrained_dir,
        force_retrain=force_retrain,
        verbose=verbose
    )
    
    # Print summary
    print_comparison_summary(results)
    
    # Save report
    save_comparison_report(results, save_path=f"{output_dir}/comparison_report.txt")
    
    # Generate visualizations
    generate_all_visualizations(results, agents_dict, save_dir=output_dir)
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print(f"Results saved to: {output_dir}/")
    print("="*80 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Compare different RL agents on Blackjack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=250_000,
        help="Number of training episodes per agent"
    )
    
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10_000,
        help="Number of evaluation episodes per agent"
    )
    
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default="src/pretrained",
        help="Directory containing pre-trained agent files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Directory to save results and visualizations"
    )
    
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if pre-trained agents exist"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (use -1 for no seed)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Handle seed
    seed = None if args.seed == -1 else args.seed
    
    main(
        n_train_episodes=args.train_episodes,
        n_eval_episodes=args.eval_episodes,
        pretrained_dir=args.pretrained_dir,
        output_dir=args.output_dir,
        force_retrain=args.force_retrain,
        seed=seed,
        verbose=not args.quiet
    )

