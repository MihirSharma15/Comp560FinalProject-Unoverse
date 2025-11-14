"""Orchestration module for running agent comparisons."""

from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
import gymnasium as gym

from ..agents.BaseAgent import BaseAgent
from ..agents.AdaptiveEpsilonAgent import AdaptiveEpsilonAgent
from .agent_loader import load_or_train_agents


def evaluate_agent(
    agent: BaseAgent,
    env: gym.Env,
    n_episodes: int = 10_000,
    verbose: bool = True
) -> Dict[str, any]:
    """Evaluate agent's performance without learning.
    
    Args:
        agent: The trained agent to evaluate
        env: The evaluation environment
        n_episodes: Number of evaluation episodes
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary containing evaluation statistics
    """
    # Save and disable exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    total_rewards = []
    wins = 0
    losses = 0
    draws = 0
    
    iterator = tqdm(range(n_episodes), desc="Evaluating") if verbose else range(n_episodes)
    
    for _ in iterator:
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return {
        "total_episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "total_rewards": total_rewards,
    }


def get_agent_statistics(agent: BaseAgent) -> Optional[Dict]:
    """Extract statistics from an agent (if available).
    
    Args:
        agent: The agent to get statistics from
        
    Returns:
        Dictionary of agent statistics, or None if not applicable
    """
    if isinstance(agent, AdaptiveEpsilonAgent):
        return agent.get_state_statistics()
    return None


def run_comparison(
    env: gym.Env,
    n_train_episodes: int = 250_000,
    n_eval_episodes: int = 10_000,
    pretrained_dir: str = "src/pretrained",
    force_retrain: bool = False,
    verbose: bool = True
) -> Dict[str, Dict]:
    """Run complete comparison of all agents.
    
    Args:
        env: The environment for training and evaluation
        n_train_episodes: Number of training episodes
        n_eval_episodes: Number of evaluation episodes
        pretrained_dir: Directory containing pre-trained agents
        force_retrain: Whether to force retraining even if pretrained agents exist
        verbose: Whether to show progress and detailed output
        
    Returns:
        Dictionary mapping agent names to their results
    """
    if verbose:
        print("\n" + "="*80)
        print(" " * 25 + "AGENT COMPARISON")
        print("="*80)
        print("\nConfiguration:")
        print(f"  Training episodes: {n_train_episodes:,}")
        print(f"  Evaluation episodes: {n_eval_episodes:,}")
        print(f"  Pre-trained directory: {pretrained_dir}")
        print(f"  Force retrain: {force_retrain}")
        print("\n" + "="*80 + "\n")
    
    # Load or train agents
    if verbose:
        print("📚 Loading/Training Agents...")
        print("-" * 80)
    
    agents_data = load_or_train_agents(
        env=env,
        n_train_episodes=n_train_episodes,
        pretrained_dir=pretrained_dir,
        force_retrain=force_retrain,
        verbose=verbose
    )
    
    # Prepare results dictionary
    results = {}
    agents_dict = {}
    
    # Evaluate each agent
    if verbose:
        print("\n" + "="*80)
        print("📊 Evaluating Agents...")
        print("="*80 + "\n")
    
    for agent_name, (agent, training_stats) in agents_data.items():
        if verbose:
            print(f"\n{'='*80}")
            print(f"Evaluating: {agent_name}")
            print(f"{'='*80}")
        
        # Evaluate agent
        eval_stats = evaluate_agent(agent, env, n_eval_episodes, verbose=verbose)
        
        # Get agent-specific statistics
        agent_stats = get_agent_statistics(agent)
        
        # Store results
        results[agent_name] = {
            'agent': agent,
            'train': training_stats,
            'eval': eval_stats,
            'agent_stats': agent_stats,
        }
        
        agents_dict[agent_name] = agent
        
        if verbose:
            print("\n📈 Evaluation Results:")
            print(f"  Win Rate: {eval_stats['win_rate']:.4f}")
            print(f"  Loss Rate: {eval_stats['loss_rate']:.4f}")
            print(f"  Draw Rate: {eval_stats['draw_rate']:.4f}")
            print(f"  Avg Reward: {eval_stats['avg_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
            
            if agent_stats:
                print("\n🔍 Exploration Statistics:")
                print(f"  Unique States: {agent_stats['unique_states']}")
                print(f"  Total Visits: {agent_stats['total_visits']:,}")
                print(f"  Avg Visits/State: {agent_stats['avg_visits_per_state']:.2f}")
                print(f"  Median Visits: {agent_stats['median_visits']:.2f}")
    
    if verbose:
        print("\n" + "="*80)
        print("✅ All agents evaluated successfully!")
        print("="*80 + "\n")
    
    return results, agents_dict


def print_comparison_summary(results: Dict[str, Dict]) -> None:
    """Print a formatted summary of comparison results.
    
    Args:
        results: Dictionary mapping agent names to their results
    """
    print("\n" + "="*80)
    print(" " * 28 + "COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    # Sort agents by evaluation performance
    sorted_agents = sorted(
        results.items(),
        key=lambda x: x[1]['eval']['avg_reward'],
        reverse=True
    )
    
    print(f"{'Rank':<6} {'Agent':<30} {'Win Rate':<12} {'Avg Reward':<15} {'Std Dev':<10}")
    print("-" * 80)
    
    for rank, (agent_name, data) in enumerate(sorted_agents, 1):
        eval_stats = data['eval']
        print(f"{rank:<6} {agent_name:<30} "
              f"{eval_stats['win_rate']:.4f}      "
              f"{eval_stats['avg_reward']:>7.4f}       "
              f"{eval_stats['std_reward']:>6.4f}")
    
    print("\n" + "="*80)
    
    # Best agent details
    best_agent_name, best_data = sorted_agents[0]
    print(f"\n🏆 Best Performer: {best_agent_name}")
    print(f"   Win Rate: {best_data['eval']['win_rate']:.4f}")
    print(f"   Avg Reward: {best_data['eval']['avg_reward']:.4f} ± {best_data['eval']['std_reward']:.4f}")
    
    if best_data['agent_stats']:
        stats = best_data['agent_stats']
        print(f"   Unique States Explored: {stats['unique_states']}")
        print(f"   Average Epsilon: {stats['avg_current_epsilon']:.4f}")
    
    print("\n" + "="*80 + "\n")


def save_comparison_report(
    results: Dict[str, Dict],
    save_path: str = "results/comparison/comparison_report.txt"
) -> None:
    """Save detailed comparison report to text file.
    
    Args:
        results: Dictionary mapping agent names to their results
        save_path: Path to save the report
    """
    import os
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" " * 25 + "AGENT COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Sort by performance
        sorted_agents = sorted(
            results.items(),
            key=lambda x: x[1]['eval']['avg_reward'],
            reverse=True
        )
        
        for rank, (agent_name, data) in enumerate(sorted_agents, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"Rank {rank}: {agent_name}\n")
            f.write(f"{'='*80}\n\n")
            
            # Evaluation results
            eval_stats = data['eval']
            f.write("EVALUATION RESULTS:\n")
            f.write(f"  Total Episodes: {eval_stats['total_episodes']:,}\n")
            f.write(f"  Win Rate:  {eval_stats['win_rate']:.4f} ({eval_stats['wins']:,} wins)\n")
            f.write(f"  Loss Rate: {eval_stats['loss_rate']:.4f} ({eval_stats['losses']:,} losses)\n")
            f.write(f"  Draw Rate: {eval_stats['draw_rate']:.4f} ({eval_stats['draws']:,} draws)\n")
            f.write(f"  Average Reward: {eval_stats['avg_reward']:.4f} ± {eval_stats['std_reward']:.4f}\n")
            
            # Training results (if available)
            if data['train'] is not None:
                train_stats = data['train']
                f.write("\nTRAINING RESULTS:\n")
                f.write(f"  Total Episodes: {train_stats['total_episodes']:,}\n")
                f.write(f"  Win Rate:  {train_stats['win_rate']:.4f}\n")
                f.write(f"  Loss Rate: {train_stats['loss_rate']:.4f}\n")
                f.write(f"  Draw Rate: {train_stats['draw_rate']:.4f}\n")
                f.write(f"  Average Reward: {train_stats['avg_reward']:.4f}\n")
            
            # Agent statistics (if available)
            if data['agent_stats'] is not None:
                stats = data['agent_stats']
                f.write("\nEXPLORATION STATISTICS:\n")
                f.write(f"  Unique States Visited: {stats['unique_states']}\n")
                f.write(f"  Total State Visits: {stats['total_visits']:,}\n")
                f.write(f"  Average Visits per State: {stats['avg_visits_per_state']:.2f}\n")
                f.write(f"  Median Visits: {stats['median_visits']:.2f}\n")
                f.write(f"  Std Dev Visits: {stats['std_visits']:.2f}\n")
                f.write(f"  Min/Max Visits: {stats['min_visits']} / {stats['max_visits']:,}\n")
                f.write(f"  Average Current Epsilon: {stats['avg_current_epsilon']:.4f}\n")
                f.write(f"  Epsilon Range: [{stats['min_current_epsilon']:.4f}, {stats['max_current_epsilon']:.4f}]\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")
    
    print(f"📄 Comparison report saved to {save_path}")

