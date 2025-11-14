"""Visualization functions for comparing RL agents."""

import os
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from ..agents.BaseAgent import BaseAgent
from ..agents.SimpleAgent import SimpleAgent
from ..agents.AdaptiveEpsilonAgent import AdaptiveEpsilonAgent


# Set seaborn style for cleaner plots
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_q_values_3d(
    agent: BaseAgent,
    agent_name: str,
    action: int = 1,
    save_path: Optional[str] = None
) -> None:
    """Create 3D surface plot of Q-values across state space.
    
    Args:
        agent: Trained agent to visualize
        agent_name: Name of the agent for title
        action: Which action to visualize (0=stick, 1=hit)
        save_path: Optional path to save the figure
    """
    # Define state space ranges
    player_sums = np.arange(4, 22)  # 4-21
    dealer_cards = np.arange(1, 11)  # 1-10 (Ace-10)
    
    # Create meshgrid
    X, Y = np.meshgrid(dealer_cards, player_sums)
    
    # Create figure with two subplots (usable ace vs no usable ace)
    fig = plt.figure(figsize=(16, 6))
    
    for idx, usable_ace in enumerate([False, True]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        
        # Compute Q-values for each state
        Z = np.zeros_like(X, dtype=float)
        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                state = (player_sum, dealer_card, usable_ace)
                if isinstance(agent, SimpleAgent):
                    Z[i, j] = agent.get_q_value(state, action)
                else:
                    Z[i, j] = 0.0
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Dealer Showing', fontsize=10)
        ax.set_ylabel('Player Sum', fontsize=10)
        ax.set_zlabel(f'Q-value ({"Hit" if action == 1 else "Stick"})', fontsize=10)
        ax.set_title(f'{"Usable" if usable_ace else "No"} Ace', fontsize=12, fontweight='bold')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
    
    fig.suptitle(f'{agent_name}: Q-Values for {"Hit" if action == 1 else "Stick"} Action', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Q-value 3D plot to {save_path}")
    
    plt.close()


def plot_epsilon_values_3d(
    agent: AdaptiveEpsilonAgent,
    agent_name: str,
    save_path: Optional[str] = None
) -> None:
    """Create 3D surface plot of epsilon values across state space.
    
    Args:
        agent: Adaptive epsilon agent to visualize
        agent_name: Name of the agent for title
        save_path: Optional path to save the figure
    """
    if not isinstance(agent, AdaptiveEpsilonAgent):
        print(f"Skipping epsilon 3D plot for {agent_name} (not an AdaptiveEpsilonAgent)")
        return
    
    # Define state space ranges
    player_sums = np.arange(4, 22)
    dealer_cards = np.arange(1, 11)
    
    # Create meshgrid
    X, Y = np.meshgrid(dealer_cards, player_sums)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 6))
    
    for idx, usable_ace in enumerate([False, True]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        
        # Compute epsilon values for each state
        Z = np.zeros_like(X, dtype=float)
        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                state = (player_sum, dealer_card, usable_ace)
                Z[i, j] = agent.get_epsilon_for_state(state)
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Dealer Showing', fontsize=10)
        ax.set_ylabel('Player Sum', fontsize=10)
        ax.set_zlabel('Epsilon Value', fontsize=10)
        ax.set_title(f'{"Usable" if usable_ace else "No"} Ace', fontsize=12, fontweight='bold')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
    
    fig.suptitle(f'{agent_name}: Epsilon Values Across State Space', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved epsilon 3D plot to {save_path}")
    
    plt.close()


def plot_win_rate_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """Create bar chart comparing win/loss/draw rates across agents.
    
    Args:
        results: Dictionary mapping agent names to their evaluation results
        save_path: Optional path to save the figure
    """
    agents = list(results.keys())
    win_rates = [results[agent]['eval']['win_rate'] for agent in agents]
    loss_rates = [results[agent]['eval']['loss_rate'] for agent in agents]
    draw_rates = [results[agent]['eval']['draw_rate'] for agent in agents]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(agents))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, win_rates, width, label='Wins', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, loss_rates, width, label='Losses', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, draw_rates, width, label='Draws', color='#95a5a6', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Agent Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
    ax.set_title('Win/Loss/Draw Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved win rate comparison to {save_path}")
    
    plt.close()


def plot_training_performance(
    results: Dict[str, Dict],
    window_size: int = 1000,
    save_path: Optional[str] = None
) -> None:
    """Create line plot showing training reward progression over episodes.
    
    Args:
        results: Dictionary mapping agent names to their results
        window_size: Window size for moving average smoothing
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for agent_name, data in results.items():
        if data['train'] is None:
            continue
        
        rewards = data['train']['total_rewards']
        
        # Compute moving average
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            x = np.arange(window_size-1, len(rewards))
            ax.plot(x, moving_avg, label=agent_name, linewidth=2, alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Training Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Average Reward (window={window_size})', fontsize=12, fontweight='bold')
    ax.set_title('Training Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training performance plot to {save_path}")
    
    plt.close()


def plot_exploration_statistics(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """Create visualization of exploration statistics across agents.
    
    Args:
        results: Dictionary mapping agent names to their results
        save_path: Optional path to save the figure
    """
    agents = []
    unique_states = []
    avg_visits = []
    
    for agent_name, data in results.items():
        if data['agent_stats'] is not None:
            stats = data['agent_stats']
            agents.append(agent_name)
            unique_states.append(stats['unique_states'])
            avg_visits.append(stats['avg_visits_per_state'])
    
    if not agents:
        print("No exploration statistics available to plot")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Unique states visited
    colors = sns.color_palette("husl", len(agents))
    bars1 = ax1.bar(agents, unique_states, color=colors, alpha=0.8)
    ax1.set_xlabel('Agent Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Unique States Visited', fontsize=12, fontweight='bold')
    ax1.set_title('State Space Coverage', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Average visits per state
    bars2 = ax2.bar(agents, avg_visits, color=colors, alpha=0.8)
    ax2.set_xlabel('Agent Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Visits per State', fontsize=12, fontweight='bold')
    ax2.set_title('State Visit Distribution', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved exploration statistics to {save_path}")
    
    plt.close()


def plot_policy_heatmap(
    agent: BaseAgent,
    agent_name: str,
    save_path: Optional[str] = None
) -> None:
    """Create 2D heatmap showing learned policy (best action per state).
    
    Args:
        agent: Trained agent to visualize
        agent_name: Name of the agent for title
        save_path: Optional path to save the figure
    """
    # Define state space
    player_sums = np.arange(12, 22)  # Standard blackjack range
    dealer_cards = np.arange(1, 11)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, usable_ace in zip([ax1, ax2], [False, True]):
        # Create policy matrix (0=stick, 1=hit)
        policy = np.zeros((len(player_sums), len(dealer_cards)))
        
        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                state = (player_sum, dealer_card, usable_ace)
                if isinstance(agent, SimpleAgent):
                    q_values = agent.get_policy(state)
                    policy[i, j] = np.argmax(q_values)
                else:
                    policy[i, j] = 0
        
        # Create heatmap
        sns.heatmap(policy, ax=ax, cmap='RdYlGn', cbar_kws={'label': 'Action'},
                   xticklabels=dealer_cards, yticklabels=player_sums,
                   linewidths=0.5, linecolor='gray', alpha=0.8)
        
        ax.set_xlabel('Dealer Showing', fontsize=11, fontweight='bold')
        ax.set_ylabel('Player Sum', fontsize=11, fontweight='bold')
        ax.set_title(f'{"Usable" if usable_ace else "No"} Ace', fontsize=12, fontweight='bold')
        
        # Customize colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Stick', 'Hit'])
    
    fig.suptitle(f'{agent_name}: Learned Policy', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved policy heatmap to {save_path}")
    
    plt.close()


def plot_average_reward_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """Create bar chart comparing average rewards during evaluation.
    
    Args:
        results: Dictionary mapping agent names to their evaluation results
        save_path: Optional path to save the figure
    """
    agents = list(results.keys())
    avg_rewards = [results[agent]['eval']['avg_reward'] for agent in agents]
    std_rewards = [results[agent]['eval']['std_reward'] for agent in agents]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(agents))
    colors = sns.color_palette("coolwarm", len(agents))
    
    # Create bars with error bars
    bars = ax.bar(x, avg_rewards, yerr=std_rewards, capsize=5, 
                   color=colors, alpha=0.8, ecolor='black', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Agent Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('Evaluation Performance: Average Reward ± Std Dev', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha='right')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.4)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, avg, std) in enumerate(zip(bars, avg_rewards, std_rewards)):
        ax.text(bar.get_x() + bar.get_width()/2., avg,
               f'{avg:.3f}\n±{std:.3f}',
               ha='center', va='bottom' if avg >= 0 else 'top', 
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved average reward comparison to {save_path}")
    
    plt.close()


def plot_policy_difference_heatmap(
    agents_dict: Dict[str, BaseAgent],
    baseline_name: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """Create heatmap showing where agents disagree with baseline policy.
    
    Args:
        agents_dict: Dictionary mapping agent names to agent instances
        baseline_name: Name of baseline agent (default: first agent)
        save_path: Optional path to save the figure
    """
    if baseline_name is None:
        baseline_name = list(agents_dict.keys())[0]
    
    baseline_agent = agents_dict[baseline_name]
    other_agents = [(name, agent) for name, agent in agents_dict.items() if name != baseline_name]
    
    if not other_agents:
        print("Need at least 2 agents for policy difference plot")
        return
    
    # Define state space
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    
    # Create figure
    fig, axes = plt.subplots(2, len(other_agents), figsize=(6*len(other_agents), 10))
    if len(other_agents) == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, (agent_name, agent) in enumerate(other_agents):
        for row_idx, usable_ace in enumerate([False, True]):
            ax = axes[row_idx, col_idx]
            
            # Compute policy disagreement
            disagreement = np.zeros((len(player_sums), len(dealer_cards)))
            
            for i, player_sum in enumerate(player_sums):
                for j, dealer_card in enumerate(dealer_cards):
                    state = (player_sum, dealer_card, usable_ace)
                    
                    if isinstance(baseline_agent, SimpleAgent) and isinstance(agent, SimpleAgent):
                        baseline_action = np.argmax(baseline_agent.get_policy(state))
                        agent_action = np.argmax(agent.get_policy(state))
                        disagreement[i, j] = 1 if baseline_action != agent_action else 0
            
            # Create heatmap
            disagreement_pct = disagreement.mean() * 100
            sns.heatmap(disagreement, ax=ax, cmap='RdYlGn_r', cbar_kws={'label': 'Disagree'},
                       xticklabels=dealer_cards, yticklabels=player_sums,
                       linewidths=0.5, linecolor='gray', alpha=0.8, vmin=0, vmax=1)
            
            ax.set_xlabel('Dealer Showing', fontsize=10, fontweight='bold')
            ax.set_ylabel('Player Sum', fontsize=10, fontweight='bold')
            ax.set_title(f'{agent_name} vs {baseline_name}\n'
                        f'{"Usable" if usable_ace else "No"} Ace '
                        f'({disagreement_pct:.1f}% disagree)', 
                        fontsize=11, fontweight='bold')
    
    fig.suptitle('Policy Disagreement Heatmaps', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved policy difference heatmap to {save_path}")
    
    plt.close()


def plot_action_value_gap_heatmap(
    agent: BaseAgent,
    agent_name: str,
    save_path: Optional[str] = None
) -> None:
    """Create heatmap showing Q(s,hit) - Q(s,stick) for decision confidence.
    
    Args:
        agent: Trained agent to visualize
        agent_name: Name of the agent for title
        save_path: Optional path to save the figure
    """
    # Define state space
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, usable_ace in zip([ax1, ax2], [False, True]):
        # Compute action value gap
        value_gap = np.zeros((len(player_sums), len(dealer_cards)))
        
        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                state = (player_sum, dealer_card, usable_ace)
                if isinstance(agent, SimpleAgent):
                    q_values = agent.get_policy(state)
                    value_gap[i, j] = q_values[1] - q_values[0]  # Hit - Stick
        
        # Create heatmap with diverging colormap
        # Positive = prefer Hit, Negative = prefer Stick
        max_abs = max(abs(value_gap.min()), abs(value_gap.max()))
        sns.heatmap(value_gap, ax=ax, cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Q(Hit) - Q(Stick)'},
                   xticklabels=dealer_cards, yticklabels=player_sums,
                   linewidths=0.5, linecolor='gray', alpha=0.8,
                   vmin=-max_abs, vmax=max_abs)
        
        ax.set_xlabel('Dealer Showing', fontsize=11, fontweight='bold')
        ax.set_ylabel('Player Sum', fontsize=11, fontweight='bold')
        ax.set_title(f'{"Usable" if usable_ace else "No"} Ace', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'{agent_name}: Action Value Gap\n(Red=Prefer Hit, Blue=Prefer Stick)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved action value gap heatmap to {save_path}")
    
    plt.close()


def plot_visit_count_heatmap(
    agent: BaseAgent,
    agent_name: str,
    save_path: Optional[str] = None
) -> None:
    """Create heatmap showing state visit counts during training.
    
    Args:
        agent: Trained agent to visualize (must have state_counts attribute)
        agent_name: Name of the agent for title
        save_path: Optional path to save the figure
    """
    if not isinstance(agent, AdaptiveEpsilonAgent):
        print(f"Skipping visit count heatmap for {agent_name} (no state_counts available)")
        return
    
    # Define state space
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, usable_ace in zip([ax1, ax2], [False, True]):
        # Get visit counts
        visit_counts = np.zeros((len(player_sums), len(dealer_cards)))
        
        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                state = (player_sum, dealer_card, usable_ace)
                visit_counts[i, j] = agent.state_counts.get(state, 0)
        
        # Use log scale for better visualization
        log_counts = np.log1p(visit_counts)  # log(1 + x) to handle zeros
        
        # Create heatmap
        sns.heatmap(log_counts, ax=ax, cmap='YlOrRd',
                   cbar_kws={'label': 'log(1 + visits)'},
                   xticklabels=dealer_cards, yticklabels=player_sums,
                   linewidths=0.5, linecolor='gray', alpha=0.8)
        
        ax.set_xlabel('Dealer Showing', fontsize=11, fontweight='bold')
        ax.set_ylabel('Player Sum', fontsize=11, fontweight='bold')
        ax.set_title(f'{"Usable" if usable_ace else "No"} Ace', fontsize=12, fontweight='bold')
        
        # Add text annotations for actual counts (if not too cluttered)
        if len(player_sums) * len(dealer_cards) <= 100:
            for i in range(len(player_sums)):
                for j in range(len(dealer_cards)):
                    count = int(visit_counts[i, j])
                    if count > 0:
                        ax.text(j + 0.5, i + 0.5, f'{count}',
                               ha='center', va='center', fontsize=7, color='black')
    
    fig.suptitle(f'{agent_name}: State Visit Counts', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visit count heatmap to {save_path}")
    
    plt.close()


def plot_learning_convergence(
    results: Dict[str, Dict],
    window_size: int = 10_000,
    convergence_threshold: float = 0.001,
    save_path: Optional[str] = None
) -> None:
    """Create plot showing learning convergence with rolling average.
    
    Args:
        results: Dictionary mapping agent names to their results
        window_size: Window size for rolling average smoothing
        convergence_threshold: Threshold for marking convergence point (based on slope)
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = sns.color_palette("husl", len(results))
    
    for idx, (agent_name, data) in enumerate(results.items()):
        if data['train'] is None:
            continue
        
        rewards = np.array(data['train']['total_rewards'])
        
        if len(rewards) < window_size:
            print(f"Skipping {agent_name}: not enough training data")
            continue
        
        # Compute rolling mean
        rolling_mean = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        x = np.arange(window_size-1, len(rewards))
        
        # Plot mean line
        ax.plot(x, rolling_mean, label=agent_name, linewidth=2.5, 
                alpha=0.9, color=colors[idx])
        
        # Find convergence point (where slope is below threshold)
        if len(rolling_mean) > window_size:
            slopes = np.diff(rolling_mean)
            converged_idx = np.where(np.abs(slopes) < convergence_threshold)[0]
            if len(converged_idx) > 0:
                convergence_episode = x[converged_idx[0]]
                ax.axvline(x=convergence_episode, color=colors[idx], 
                          linestyle='--', alpha=0.5, linewidth=1)
                ax.text(convergence_episode, ax.get_ylim()[1] * 0.95, 
                       f'{convergence_episode:,}',
                       rotation=90, va='top', ha='right', fontsize=8,
                       color=colors[idx])
    
    # Customize plot
    ax.set_xlabel('Training Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Average Reward (window={window_size:,})', fontsize=12, fontweight='bold')
    ax.set_title('Learning Convergence Analysis', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved learning convergence plot to {save_path}")
    
    plt.close()


def generate_all_visualizations(
    results: Dict[str, Dict],
    agents_dict: Dict[str, BaseAgent],
    save_dir: str = "results/comparison"
) -> None:
    """Generate all comparison visualizations and save to directory.
    
    Args:
        results: Dictionary mapping agent names to their results
        agents_dict: Dictionary mapping agent names to agent instances
        save_dir: Directory to save all plots
    """
    print(f"\n{'='*80}")
    print("Generating Visualizations")
    print(f"{'='*80}\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Win rate comparison
    print("📊 Creating win rate comparison...")
    plot_win_rate_comparison(results, os.path.join(save_dir, "win_rate_comparison.png"))
    
    # 2. Average reward comparison
    print("📊 Creating average reward comparison...")
    plot_average_reward_comparison(results, os.path.join(save_dir, "avg_reward_comparison.png"))
    
    # 3. Training performance
    print("📈 Creating training performance plot...")
    plot_training_performance(results, window_size=1000, 
                             save_path=os.path.join(save_dir, "training_performance.png"))
    
    # 4. Exploration statistics
    print("🔍 Creating exploration statistics...")
    plot_exploration_statistics(results, os.path.join(save_dir, "exploration_stats.png"))
    
    # 5. Learning convergence analysis
    print("📉 Creating learning convergence plot...")
    plot_learning_convergence(results, window_size=10_000,
                              save_path=os.path.join(save_dir, "learning_convergence.png"))
    
    # 6. Policy difference heatmap
    print("🔀 Creating policy difference heatmap...")
    plot_policy_difference_heatmap(agents_dict, 
                                   save_path=os.path.join(save_dir, "policy_differences.png"))
    
    # 7. Individual agent visualizations
    for agent_name, agent in agents_dict.items():
        print(f"\n📈 Creating visualizations for {agent_name}...")
        
        # Q-value 3D plots
        safe_name = agent_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        plot_q_values_3d(agent, agent_name, action=1,
                        save_path=os.path.join(save_dir, f"q_values_3d_hit_{safe_name}.png"))
        plot_q_values_3d(agent, agent_name, action=0,
                        save_path=os.path.join(save_dir, f"q_values_3d_stick_{safe_name}.png"))
        
        # Epsilon 3D plot (for adaptive agents)
        if isinstance(agent, AdaptiveEpsilonAgent):
            plot_epsilon_values_3d(agent, agent_name,
                                  save_path=os.path.join(save_dir, f"epsilon_3d_{safe_name}.png"))
        
        # Policy heatmap
        plot_policy_heatmap(agent, agent_name,
                          save_path=os.path.join(save_dir, f"policy_heatmap_{safe_name}.png"))
        
        # Action value gap heatmap
        plot_action_value_gap_heatmap(agent, agent_name,
                                      save_path=os.path.join(save_dir, f"action_value_gap_{safe_name}.png"))
        
        # Visit count heatmap (for adaptive agents)
        if isinstance(agent, AdaptiveEpsilonAgent):
            plot_visit_count_heatmap(agent, agent_name,
                                    save_path=os.path.join(save_dir, f"visit_counts_{safe_name}.png"))
    
    print(f"\n{'='*80}")
    print(f"✅ All visualizations saved to {save_dir}/")
    print(f"{'='*80}\n")

