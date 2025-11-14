"""Example script demonstrating Monte Carlo Control agent usage."""

from ..env.env import make_blackjack_env
from ..agents.MonteCarloAgent import MonteCarloAgent
from ..training.train import train_agent
from ..training.evaluate import evaluate_agent


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_stats(stats: dict, title: str = "Results") -> None:
    """Print training/evaluation statistics."""
    print_section(title)
    print(f"Total Episodes: {stats['total_episodes']:,}")
    print(f"Wins: {stats['wins']:,} ({stats['win_rate']:.1%})")
    print(f"Losses: {stats['losses']:,} ({stats['loss_rate']:.1%})")
    print(f"Draws: {stats['draws']:,} ({stats['draw_rate']:.1%})")
    print(f"Average Reward: {stats['avg_reward']:.4f}")
    if 'std_reward' in stats:
        print(f"Reward Std Dev: {stats['std_reward']:.4f}")
    print("=" * 60)


def main() -> None:
    """Train and evaluate a Monte Carlo Control agent on Blackjack."""
    
    # Training hyperparameters
    n_episodes = 500_000        # Number of games to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.01        # Keep minimal exploration
    discount_factor = 1.0       # Value future rewards (use 1.0 for episodic tasks)
    use_first_visit = True      # Use first-visit MC (True) or every-visit MC (False)
    
    # Create environment
    env = make_blackjack_env(natural=False, sab=True, render_mode=None)
    
    # Create Monte Carlo agent
    agent = MonteCarloAgent(
        env=env,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
        use_first_visit=use_first_visit,
    )

    print_section("Blackjack Monte Carlo Control Training")
    print(f"Episodes: {n_episodes:,}")
    print(f"Discount Factor: {discount_factor}")
    print(f"Epsilon: {start_epsilon} → {final_epsilon}")
    print(f"Method: {'First-visit' if use_first_visit else 'Every-visit'} MC")
    print("=" * 60)
    
    # Train the agent
    print("\n🎓 Training Phase...")
    train_stats = train_agent(agent, env, n_episodes, verbose=True)
    print_stats(train_stats, "Training Results")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    
    # Evaluate the agent
    print("\n📊 Evaluation Phase...")
    eval_stats = evaluate_agent(agent, env, n_episodes=10_000, verbose=True)
    print_stats(eval_stats, "Evaluation Results")
    
    # Print Q-table statistics
    print_section("Q-Table Statistics")
    print(f"Total states visited: {len(agent.q_values):,}")
    print(f"Q-table size: {len(agent.q_values) * env.action_space.n:,} entries")
    
    # Show some example Q-values and visit counts for interesting states
    print("\nExample Q-values for interesting states:")
    example_states = [
        (20, 10, 0),  # Player has 20, dealer shows 10, no usable ace
        (16, 10, 0),  # Player has 16, dealer shows 10, no usable ace
        (11, 5, 0),   # Player has 11, dealer shows 5, no usable ace
        (21, 10, 0),  # Player has 21, dealer shows 10, no usable ace
    ]
    
    for state in example_states:
        if state in agent.q_values:
            q_vals = agent.q_values[state]
            visit_counts = agent.returns_count[state]
            action = "Stick" if q_vals[0] > q_vals[1] else "Hit"
            print(f"State {state}:")
            print(f"  Q(Stick)={q_vals[0]:.3f} (visits: {int(visit_counts[0])})")
            print(f"  Q(Hit)={q_vals[1]:.3f} (visits: {int(visit_counts[1])}) → {action}")
    
    print("\n✅ Training complete!")
    
    # Compare with theory
    print_section("Monte Carlo vs Q-Learning Comparison")
    print("Monte Carlo Control:")
    print("  ✓ Learns from complete episodes (offline)")
    print("  ✓ Uses actual returns (no bootstrapping)")
    print("  ✓ Requires episodic tasks")
    print("  ✓ Unbiased estimates of value functions")
    print("\nQ-Learning:")
    print("  ✓ Learns from each step (online)")
    print("  ✓ Uses bootstrapped estimates")
    print("  ✓ Works with continuing tasks")
    print("  ✓ Often faster convergence in practice")
    print("=" * 60)


if __name__ == "__main__":
    main()

