"""Main training script for Blackjack Q-Learning agent."""

from .env.env import make_blackjack_env
from .agents.SimpleAgent import SimpleAgent
from .training.train import train_agent
from .training.evaluate import evaluate_agent


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


def demo_game(env, agent) -> None:
    """Run a demo game with visualization."""
    print_section("Demo Game")
    
    obs, _ = env.reset()
    print(f"Initial state: Player sum={obs[0]}, Dealer showing={obs[1]}, Usable ace={obs[2]}")
    
    done = False
    step = 0
    
    while not done:
        step += 1
        action = agent.get_action(obs)
        action_name = "Hit" if action == 1 else "Stick"
        
        print(f"\nStep {step}: {action_name} (action={action})")
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        if not terminated:
            print(f"State: Player sum={next_obs[0]}, Dealer showing={next_obs[1]}, Usable ace={next_obs[2]}")
        
        done = terminated or truncated
        obs = next_obs
    
    # Print final result
    if reward > 0:
        result = "WIN!"
    elif reward < 0:
        result = "LOSS"
    else:
        result = "DRAW"
    
    print(f"\nGame Over! {result}")
    print(f"Final reward: {reward}")
    print(f"Final state: Player sum={obs[0]}, Dealer showing={obs[1]}, Usable ace={obs[2]}")
    print("=" * 60)


def main() -> None:
    """Main training loop for Blackjack Q-Learning agent."""
    
    # Training hyperparameters
    learning_rate = 0.01        # How fast to learn (lower for Blackjack due to stochasticity)
    n_episodes = 500_000        # Number of games to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.01        # Keep minimal exploration
    discount_factor = 1.0       # Value future rewards (use 1.0 for episodic tasks)
    
    # Create environment
    env = make_blackjack_env(natural=False, sab=True, render_mode=None)
    
    # Create agent
    agent = SimpleAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    print_section("Blackjack Q-Learning Training")
    print(f"Episodes: {n_episodes:,}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Discount Factor: {discount_factor}")
    print(f"Epsilon: {start_epsilon} → {final_epsilon}")
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
    
    # Run demo games
    demo_env = make_blackjack_env(natural=False, sab=True, render_mode=None)
    
    print("\n🎮 Demo Games (Agent plays with learned policy):")
    for i in range(3):
        print(f"\n{'='*20} Demo Game {i+1} {'='*20}")
        agent.epsilon = 0.0  # Pure exploitation for demo
        demo_game(demo_env, agent)
    
    # Print Q-table statistics
    print_section("Q-Table Statistics")
    print(f"Total states visited: {len(agent.q_values):,}")
    print(f"Q-table size: {len(agent.q_values) * env.action_space.n:,} entries")
    
    # Show some example Q-values for interesting states
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
            action = "Stick" if q_vals[0] > q_vals[1] else "Hit"
            print(f"State {state}: Q(Stick)={q_vals[0]:.3f}, Q(Hit)={q_vals[1]:.3f} → {action}")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
