"""Training loop for AdaptiveEpsilonAgent with detailed insights and analysis.

This script trains an AdaptiveEpsilonAgent and provides real-time insights into:
- State visit frequency distribution
- Epsilon value evolution across states
- Exploration vs exploitation balance
- Learning progress and convergence
"""

from typing import Dict, List
import numpy as np
from tqdm import tqdm
import gymnasium as gym

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.AdaptiveEpsilonAgent import AdaptiveEpsilonAgent
from src.env.env import make_blackjack_env


def train_adaptive_agent_with_insights(
    agent: AdaptiveEpsilonAgent,
    env: gym.Env,
    n_episodes: int,
    insight_interval: int = 10000,
    verbose: bool = True
) -> Dict:
    """Train AdaptiveEpsilonAgent with periodic insights.
    
    Args:
        agent: The adaptive epsilon agent
        env: Gymnasium environment
        n_episodes: Number of episodes to train
        insight_interval: How often to print insights (in episodes)
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with training statistics and insights
    """
    total_rewards = []
    wins = 0
    losses = 0
    draws = 0
    
    # Track exploration vs exploitation
    exploration_actions = 0
    exploitation_actions = 0
    
    # Track epsilon statistics over time
    epsilon_snapshots = []
    state_coverage_snapshots = []
    
    iterator = tqdm(range(n_episodes), desc="Training") if verbose else range(n_episodes)
    
    for episode in iterator:
        obs, info = env.reset()
        episode_reward = 0
        done = False
        episode_explorations = 0
        episode_exploitations = 0
        
        while not done:
            state = agent._obs_to_state(obs)
            current_epsilon = agent._compute_state_epsilon(state)
            
            # Track whether we explored or exploited
            if np.random.random() < current_epsilon:
                episode_explorations += 1
            else:
                episode_exploitations += 1
            
            # Get action from agent
            action = agent.get_action(obs)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update agent
            agent.update(obs, action, reward, terminated, next_obs)
            
            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
        
        # Track statistics
        total_rewards.append(episode_reward)
        exploration_actions += episode_explorations
        exploitation_actions += episode_exploitations
        
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
        
        # Decay exploration rate (base epsilon for some strategies)
        agent.decay_epsilon()
        
        # Periodic insights
        if (episode + 1) % insight_interval == 0:
            print_training_insights(
                agent, episode + 1, total_rewards[-insight_interval:],
                wins, losses, draws, n_episodes,
                exploration_actions, exploitation_actions
            )
            
            # Snapshot statistics
            stats = agent.get_state_statistics()
            epsilon_snapshots.append({
                'episode': episode + 1,
                'avg_epsilon': stats['avg_current_epsilon'],
                'min_epsilon': stats['min_current_epsilon'],
                'max_epsilon': stats['max_current_epsilon'],
            })
            state_coverage_snapshots.append({
                'episode': episode + 1,
                'unique_states': stats['unique_states'],
                'total_visits': stats['total_visits'],
            })
        
        # Update progress bar
        if verbose and episode % 100 == 0:
            recent_rewards = total_rewards[-100:] if len(total_rewards) >= 100 else total_rewards
            avg_reward = np.mean(recent_rewards)
            explore_rate = exploration_actions / max(1, exploration_actions + exploitation_actions)
            iterator.set_postfix({
                'avg_reward': f'{avg_reward:.3f}',
                'explore%': f'{explore_rate*100:.1f}',
                'states': agent.unique_states_visited
            })
    
    # Final comprehensive analysis
    print("\n" + "="*80)
    print(" " * 30 + "FINAL ANALYSIS")
    print("="*80)
    
    final_stats = agent.get_state_statistics()
    
    return {
        "total_episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "avg_reward": np.mean(total_rewards),
        "total_rewards": total_rewards,
        "exploration_rate": exploration_actions / (exploration_actions + exploitation_actions),
        "epsilon_snapshots": epsilon_snapshots,
        "state_coverage_snapshots": state_coverage_snapshots,
        "final_stats": final_stats,
    }


def print_training_insights(
    agent: AdaptiveEpsilonAgent,
    episode: int,
    recent_rewards: List[float],
    wins: int,
    losses: int,
    draws: int,
    total_episodes: int,
    exploration_actions: int,
    exploitation_actions: int
) -> None:
    """Print detailed insights about training progress.
    
    Args:
        agent: The adaptive epsilon agent
        episode: Current episode number
        recent_rewards: Recent episode rewards
        wins, losses, draws: Win/loss/draw counts
        total_episodes: Total episodes for training
        exploration_actions: Count of exploratory actions
        exploitation_actions: Count of exploitative actions
    """
    print("\n" + "="*80)
    print(f"📊 INSIGHTS AT EPISODE {episode:,} / {total_episodes:,} ({episode/total_episodes*100:.1f}%)")
    print("="*80)
    
    # Performance metrics
    avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0
    current_win_rate = wins / episode if episode > 0 else 0
    
    print("\n🎯 Performance:")
    print(f"   Recent Avg Reward: {avg_recent_reward:.3f}")
    print(f"   Overall Win Rate:  {current_win_rate:.1%} ({wins:,} wins, {losses:,} losses, {draws:,} draws)")
    
    # Exploration statistics
    total_actions = exploration_actions + exploitation_actions
    explore_rate = exploration_actions / total_actions if total_actions > 0 else 0
    
    print("\n🔍 Exploration vs Exploitation:")
    print(f"   Exploration Rate:  {explore_rate:.1%} ({exploration_actions:,} exploratory actions)")
    print(f"   Exploitation Rate: {1-explore_rate:.1%} ({exploitation_actions:,} exploitative actions)")
    
    # State space coverage
    stats = agent.get_state_statistics()
    
    print("\n🗺️  State Space Coverage:")
    print(f"   Unique States:     {stats['unique_states']:,}")
    print(f"   Total Visits:      {stats['total_visits']:,}")
    print(f"   Avg Visits/State:  {stats['avg_visits_per_state']:.1f}")
    print(f"   Median Visits:     {stats['median_visits']:.0f}")
    print(f"   Visit Range:       {stats['min_visits']} - {stats['max_visits']:,}")
    
    # Epsilon distribution
    print("\n📈 Epsilon Distribution:")
    print(f"   Strategy:          {agent.strategy}")
    print(f"   Average ε:         {stats['avg_current_epsilon']:.3f}")
    print(f"   ε Range:           {stats['min_current_epsilon']:.3f} - {stats['max_current_epsilon']:.3f}")
    
    # State visit distribution
    visit_dist = stats['visit_distribution']
    rare_states = sum(count for visits, count in visit_dist.items() if visits < 10)
    common_states = sum(count for visits, count in visit_dist.items() if visits >= 100)
    
    print("\n📊 State Visit Distribution:")
    print(f"   Rare States (<10 visits):    {rare_states:,} states")
    print(f"   Common States (≥100 visits): {common_states:,} states")
    
    # Top visited states with their epsilon values
    print("\n🔝 Top 5 Most Visited States:")
    for i, (state, count) in enumerate(stats['states_by_visits'][:5], 1):
        player_sum, dealer_card, usable_ace = state
        epsilon = agent.get_epsilon_for_state(state)
        ace_str = "A" if usable_ace else " "
        print(f"   {i}. P={player_sum:2d}, D={dealer_card:2d} [{ace_str}] | "
              f"Visits: {count:6,} | ε={epsilon:.3f}")
    
    # Rare states with high epsilon
    rare_states_dict = agent.get_rare_states(threshold=5)
    if rare_states_dict:
        print("\n🆕 Sample Rare States (High Exploration):")
        sample_rare = list(rare_states_dict.items())[:3]
        for state, count in sample_rare:
            player_sum, dealer_card, usable_ace = state
            epsilon = agent.get_epsilon_for_state(state)
            ace_str = "A" if usable_ace else " "
            print(f"   P={player_sum:2d}, D={dealer_card:2d} [{ace_str}] | "
                  f"Visits: {count} | ε={epsilon:.3f} (High exploration)")
    
    print("="*80)


def analyze_final_results(results: Dict, agent: AdaptiveEpsilonAgent) -> None:
    """Print comprehensive final analysis.
    
    Args:
        results: Training results dictionary
        agent: Trained agent
    """
    print("\n" + "="*80)
    print(" " * 25 + "COMPREHENSIVE FINAL ANALYSIS")
    print("="*80)
    
    print("\n📊 Overall Performance:")
    print(f"   Total Episodes:    {results['total_episodes']:,}")
    print(f"   Final Win Rate:    {results['win_rate']:.1%}")
    print(f"   Final Loss Rate:   {results['loss_rate']:.1%}")
    print(f"   Final Draw Rate:   {results['draw_rate']:.1%}")
    print(f"   Average Reward:    {results['avg_reward']:.3f}")
    
    print("\n🎯 Exploration Statistics:")
    print(f"   Overall Exploration Rate: {results['exploration_rate']:.1%}")
    print("   (This shows how often we explored vs exploited)")
    
    final_stats = results['final_stats']
    
    print("\n🗺️  Final State Space Analysis:")
    print(f"   Unique States Visited: {final_stats['unique_states']}")
    print(f"   Total State Visits:    {final_stats['total_visits']:,}")
    print(f"   Avg Visits per State:  {final_stats['avg_visits_per_state']:.1f}")
    print(f"   Std Dev of Visits:     {final_stats['std_visits']:.1f}")
    
    print("\n📈 Final Epsilon Values:")
    print(f"   Average ε across states: {final_stats['avg_current_epsilon']:.3f}")
    print(f"   Min ε (most exploited):  {final_stats['min_current_epsilon']:.3f}")
    print(f"   Max ε (most explored):   {final_stats['max_current_epsilon']:.3f}")
    
    # Analyze state visit distribution
    visit_dist = final_stats['visit_distribution']
    print("\n📊 State Visit Distribution:")
    
    # Group by visit ranges
    ranges = [
        (0, 10, "Very Rare"),
        (10, 50, "Rare"),
        (50, 100, "Moderate"),
        (100, 500, "Common"),
        (500, float('inf'), "Very Common"),
    ]
    
    for low, high, label in ranges:
        count = sum(cnt for visits, cnt in visit_dist.items() 
                   if low <= visits < high)
        if count > 0:
            print(f"   {label:12s} ({low:3d}-{high if high != float('inf') else '∞':>3s} visits): {count:4d} states")
    
    print("\n🔝 Top 10 Most Visited States (with Q-values):")
    print(f"   {'Rank':<6} {'State':<20} {'Visits':<10} {'ε':<8} {'Q(hit)':<10} {'Q(stick)':<10}")
    print(f"   {'-'*78}")
    
    for i, (state, count) in enumerate(final_stats['states_by_visits'][:10], 1):
        player_sum, dealer_card, usable_ace = state
        epsilon = agent.get_epsilon_for_state(state)
        q_values = agent.get_policy(state)
        ace_str = "A" if usable_ace else " "
        state_str = f"P={player_sum:2d}, D={dealer_card:2d} [{ace_str}]"
        print(f"   {i:<6} {state_str:<20} {count:<10,} {epsilon:<8.3f} "
              f"{q_values[1]:<10.3f} {q_values[0]:<10.3f}")
    
    # Key insights about strategy
    print("\n💡 KEY INSIGHTS:")
    print("="*80)
    
    print(f"\n1. ADAPTIVE EXPLORATION STRATEGY: {agent.strategy.upper()}")
    if agent.strategy == "count_based":
        print(f"   • Uses formula: ε(s) = {agent.k_param}/(k + N(s))")
        print(f"   • Higher k = {agent.k_param} means more initial exploration")
        print("   • Epsilon decays hyperbolically with visit count")
    elif agent.strategy == "ucb":
        print(f"   • Uses formula: ε(s) = {agent.base_epsilon} + {agent.c_param}/sqrt(N(s)+1)")
        print("   • Adds exploration bonus inversely proportional to sqrt(visits)")
    elif agent.strategy == "threshold":
        print(f"   • Uses piecewise thresholds: {agent.thresholds}")
        print(f"   • Epsilon levels: {agent.epsilon_levels}")
    
    print("\n2. EXPLORATION EFFICIENCY:")
    rare_count = len(agent.get_rare_states(threshold=10))
    common_count = len(agent.get_common_states(threshold=100))
    print(f"   • Rare states (<10 visits): {rare_count} - these get HIGH epsilon")
    print(f"   • Common states (≥100 visits): {common_count} - these get LOW epsilon")
    print("   • This focuses exploration where uncertainty is highest!")
    
    print("\n3. LEARNING CONVERGENCE:")
    final_avg_reward = np.mean(results['total_rewards'][-1000:])
    early_avg_reward = np.mean(results['total_rewards'][:1000])
    improvement = final_avg_reward - early_avg_reward
    print(f"   • Early avg reward (first 1k):  {early_avg_reward:.3f}")
    print(f"   • Final avg reward (last 1k):   {final_avg_reward:.3f}")
    print(f"   • Improvement:                   {improvement:+.3f}")
    
    print("\n4. STATE SPACE COVERAGE:")
    # Blackjack has roughly 280 possible states
    theoretical_states = 280  # Approximate
    coverage = final_stats['unique_states'] / theoretical_states * 100
    print(f"   • Visited {final_stats['unique_states']} out of ~{theoretical_states} possible states")
    print(f"   • State space coverage: ~{coverage:.1f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main training script with insights."""
    print("\n" + "="*80)
    print(" " * 15 + "ADAPTIVE EPSILON AGENT TRAINING WITH INSIGHTS")
    print("="*80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configuration
    n_episodes = 100_000
    insight_interval = 20_000
    
    # Create environment
    env = make_blackjack_env(natural=False, sab=True)
    
    # Create adaptive epsilon agent
    print("\nCreating AdaptiveEpsilonAgent...")
    print("  Strategy: count_based")
    print("  k parameter: 10.0")
    print(f"  Episodes: {n_episodes:,}")
    
    agent = AdaptiveEpsilonAgent(
        env=env,
        learning_rate=0.01,
        initial_epsilon=1.0,
        epsilon_decay=0,  # No global decay, purely state-based
        final_epsilon=0.01,
        discount_factor=1.0,
        strategy="count_based",
        k_param=10.0,
    )
    
    print("\nStarting training with periodic insights...")
    print(f"Insights will be printed every {insight_interval:,} episodes\n")
    
    # Train with insights
    results = train_adaptive_agent_with_insights(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        insight_interval=insight_interval,
        verbose=True
    )
    
    # Final analysis
    analyze_final_results(results, agent)
    
    # Demonstrate learned policy on key states
    print("\n" + "="*80)
    print(" " * 20 + "LEARNED POLICY ON KEY STATES")
    print("="*80)
    
    key_states = [
        ((20, 10, False), "Strong hand vs dealer 10"),
        ((16, 10, False), "Weak hand vs dealer 10"),
        ((12, 6, False), "Marginal hand vs dealer 6"),
        ((11, 5, False), "Low hand vs dealer 5"),
        ((18, 9, True), "Soft 18 vs dealer 9"),
    ]
    
    print("\nPolicy: 0 = STICK, 1 = HIT")
    print(f"{'State':<25} {'Description':<30} {'Visits':<10} {'Best Action':<15} {'Q(hit)':<10} {'Q(stick)':<10}")
    print("-" * 100)
    
    for state, description in key_states:
        visits = agent.state_counts.get(state, 0)
        q_values = agent.get_policy(state)
        best_action = np.argmax(q_values)
        action_str = "HIT" if best_action == 1 else "STICK"
        player, dealer, ace = state
        ace_str = "A" if ace else " "
        state_str = f"P={player:2d}, D={dealer:2d} [{ace_str}]"
        
        print(f"{state_str:<25} {description:<30} {visits:<10,} {action_str:<15} "
              f"{q_values[1]:<10.3f} {q_values[0]:<10.3f}")
    
    print("\n" + "="*80)
    print("Training complete! The agent has learned to adapt its exploration")
    print("based on state familiarity, leading to more efficient learning.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

