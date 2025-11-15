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
    save_path: Optional[str] = None,
    z_limits: Optional[tuple] = None
) -> None:
    """Create 3D surface plot of Q-values across state space.
    
    Args:
        agent: Trained agent to visualize
        agent_name: Name of the agent for title
        action: Which action to visualize (0=stick, 1=hit)
        save_path: Optional path to save the figure
        z_limits: Optional tuple of (min, max) for z-axis limits across all agents
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
        
        # Create surface plot with consistent color scale
        if z_limits:
            vmin, vmax = z_limits
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, 
                                  antialiased=True, alpha=0.8, vmin=vmin, vmax=vmax)
        else:
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, 
                                  antialiased=True, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Dealer Showing', fontsize=10)
        ax.set_ylabel('Player Sum', fontsize=10)
        ax.set_zlabel(f'Q-value ({"Hit" if action == 1 else "Stick"})', fontsize=10)
        ax.set_title(f'{"Usable" if usable_ace else "No"} Ace', fontsize=12, fontweight='bold')
        
        # Set consistent z-axis limits if provided
        if z_limits:
            ax.set_zlim(z_limits)
        
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
    save_path: Optional[str] = None,
    z_limits: Optional[tuple] = None
) -> None:
    """Create 3D surface plot of epsilon values across state space.
    
    Args:
        agent: Adaptive epsilon agent to visualize
        agent_name: Name of the agent for title
        save_path: Optional path to save the figure
        z_limits: Optional tuple of (min, max) for z-axis limits across all agents
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
        
        # Create surface plot with consistent color scale
        if z_limits:
            vmin, vmax = z_limits
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, 
                                  antialiased=True, alpha=0.8, vmin=vmin, vmax=vmax)
        else:
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, 
                                  antialiased=True, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Dealer Showing', fontsize=10)
        ax.set_ylabel('Player Sum', fontsize=10)
        ax.set_zlabel('Epsilon Value', fontsize=10)
        ax.set_title(f'{"Usable" if usable_ace else "No"} Ace', fontsize=12, fontweight='bold')
        
        # Set consistent z-axis limits if provided
        if z_limits:
            ax.set_zlim(z_limits)
        
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


def plot_convergence_envelope(
    results: Dict[str, Dict],
    window_size: int = 5000,
    envelope_percentiles: tuple = (10, 90),
    save_path: Optional[str] = None
) -> None:
    """Plot convergence detection via mean envelope drift analysis.
    
    Shows when the smoothed central tendency stops drifting upward/downward.
    Convergence is detected when the mean envelope stabilizes into a band.
    
    Args:
        results: Dictionary mapping agent names to their results
        window_size: Window size for computing rolling statistics
        envelope_percentiles: Tuple of (lower, upper) percentiles for envelope
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 5 * len(results)))
    if len(results) == 1:
        axes = [axes]
    
    colors = sns.color_palette("husl", len(results))
    
    for idx, (agent_name, data) in enumerate(results.items()):
        ax = axes[idx]
        
        if data['train'] is None:
            ax.text(0.5, 0.5, f'{agent_name}: No training data', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue
        
        rewards = np.array(data['train']['total_rewards'])
        
        if len(rewards) < window_size:
            ax.text(0.5, 0.5, f'{agent_name}: Insufficient data', 
                   ha='center', va='center', fontsize=12)
            continue
        
        # Compute rolling statistics
        rolling_mean = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Compute rolling percentiles for envelope
        lower_env = []
        upper_env = []
        for i in range(len(rewards) - window_size + 1):
            window = rewards[i:i+window_size]
            lower_env.append(np.percentile(window, envelope_percentiles[0]))
            upper_env.append(np.percentile(window, envelope_percentiles[1]))
        
        lower_env = np.array(lower_env)
        upper_env = np.array(upper_env)
        
        x = np.arange(window_size-1, len(rewards))
        
        # Plot envelope cloud
        ax.fill_between(x, lower_env, upper_env, alpha=0.2, color=colors[idx], 
                        label=f'{envelope_percentiles[0]}-{envelope_percentiles[1]}th percentile')
        
        # Plot mean line
        ax.plot(x, rolling_mean, linewidth=2.5, color=colors[idx], 
               label='Rolling Mean', alpha=0.9)
        
        # Detect convergence: when mean stops drifting
        # Use derivative of rolling mean
        mean_derivative = np.diff(rolling_mean)
        smoothed_derivative = np.convolve(mean_derivative, 
                                         np.ones(window_size//10)/(window_size//10), 
                                         mode='valid')
        
        # Find where derivative variance becomes small (stable)
        if len(smoothed_derivative) > window_size//5:
            # Split into chunks and compute variance
            chunk_size = max(1, len(smoothed_derivative) // 20)
            variances = []
            positions = []
            for i in range(0, len(smoothed_derivative) - chunk_size, chunk_size):
                chunk = smoothed_derivative[i:i+chunk_size]
                variances.append(np.var(chunk))
                positions.append(x[i + window_size//10])
            
            # Find first point where variance stays below threshold
            if variances:
                threshold = np.percentile(variances, 25)  # Lower quartile
                converged_idx = None
                for i, var in enumerate(variances):
                    if var < threshold:
                        # Check if it stays low
                        if i < len(variances) - 3:
                            if all(v < threshold * 1.5 for v in variances[i:i+3]):
                                converged_idx = i
                                break
                
                if converged_idx is not None:
                    convergence_episode = positions[converged_idx]
                    ax.axvline(x=convergence_episode, color='red', 
                             linestyle='--', alpha=0.7, linewidth=2,
                             label=f'Envelope Stabilizes ~{convergence_episode:,}')
        
        # Customize subplot
        ax.set_xlabel('Training Episode', fontsize=11, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'{agent_name}: Convergence via Envelope Drift', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    
    fig.suptitle('Convergence Detection: Mean Envelope Stabilization', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved convergence envelope plot to {save_path}")
    
    plt.close()


def plot_q_value_change_derivative(
    results: Dict[str, Dict],
    window_size: int = 1000,
    save_path: Optional[str] = None
) -> None:
    """Plot Q-value change derivative to detect convergence.
    
    Convergence is indicated when the derivative (slope) of Q-value changes
    approaches zero, typically after significant training episodes.
    
    Args:
        results: Dictionary mapping agent names to their results  
        window_size: Window size for smoothing the derivative
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = sns.color_palette("husl", len(results))
    
    for idx, (agent_name, data) in enumerate(results.items()):
        if data['train'] is None:
            continue
        
        # For Q-value changes, we can use the reward variance or 
        # approximate from episode-to-episode reward changes
        rewards = np.array(data['train']['total_rewards'])
        
        if len(rewards) < window_size * 2:
            print(f"Skipping {agent_name}: insufficient data for derivative analysis")
            continue
        
        # Compute episode-to-episode changes (proxy for Q-value updates)
        # The magnitude of reward changes correlates with Q-value updates
        episode_changes = np.abs(np.diff(rewards))
        
        # Smooth the changes
        smoothed_changes = np.convolve(episode_changes, 
                                      np.ones(window_size)/window_size, 
                                      mode='valid')
        
        x1 = np.arange(window_size, len(rewards))
        
        # Plot 1: Magnitude of changes
        ax1.plot(x1, smoothed_changes, linewidth=2, alpha=0.8, 
                color=colors[idx], label=agent_name)
        
        # Compute derivative of changes (rate of change of change)
        derivative = np.diff(smoothed_changes)
        # Smooth the derivative
        if len(derivative) >= window_size//2:
            smoothed_derivative = np.convolve(derivative, 
                                             np.ones(window_size//2)/(window_size//2),
                                             mode='valid')
            x2 = np.arange(window_size + window_size//2, len(rewards))
            
            # Plot 2: Derivative (slope)
            ax2.plot(x2, smoothed_derivative, linewidth=2, alpha=0.8,
                    color=colors[idx], label=agent_name)
            
            # Find convergence: where derivative approaches zero and stays there
            abs_derivative = np.abs(smoothed_derivative)
            threshold = np.percentile(abs_derivative, 10)  # Low threshold
            
            # Find sustained low derivative
            below_threshold = abs_derivative < threshold
            if np.any(below_threshold):
                # Find first sustained period
                min_sustain = max(1, len(below_threshold) // 20)
                for i in range(len(below_threshold) - min_sustain):
                    if np.all(below_threshold[i:i+min_sustain]):
                        convergence_ep = x2[i]
                        ax2.axvline(x=convergence_ep, color=colors[idx],
                                   linestyle='--', alpha=0.5, linewidth=1.5)
                        ax2.text(convergence_ep, ax2.get_ylim()[1] * 0.9,
                                f'{convergence_ep:,}',
                                rotation=90, va='top', ha='right', 
                                fontsize=8, color=colors[idx],
                                bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='white', alpha=0.7))
                        break
    
    # Customize plot 1
    ax1.set_xlabel('Training Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('|Reward Change| (smoothed)', fontsize=12, fontweight='bold')
    ax1.set_title('Magnitude of Episode-to-Episode Changes', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')  # Log scale to see convergence better
    
    # Customize plot 2  
    ax2.set_xlabel('Training Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('d(Change)/d(Episode) (smoothed)', fontsize=12, fontweight='bold')
    ax2.set_title('Derivative of Changes → Zero Indicates Convergence', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    fig.suptitle('Q-Value Change Derivative Analysis for Convergence Detection', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Q-value change derivative plot to {save_path}")
    
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
    
    # 6. Convergence envelope analysis (new)
    print("Creating convergence envelope plot...")
    plot_convergence_envelope(results, window_size=5000,
                             save_path=os.path.join(save_dir, "convergence_envelope.png"))
    
    # 7. Q-value change derivative analysis (new)
    print("📉 Creating Q-value change derivative plot...")
    plot_q_value_change_derivative(results, window_size=1000,
                                   save_path=os.path.join(save_dir, "q_value_derivative.png"))
    
    # 8. Policy difference heatmap
    print("🔀 Creating policy difference heatmap...")
    plot_policy_difference_heatmap(agents_dict, 
                                   save_path=os.path.join(save_dir, "policy_differences.png"))
    
    # 9. Compute global scales for consistent comparison
    print("\n📊 Computing global scales for consistent visualization...")
    
    # Define state space for scanning
    player_sums = np.arange(4, 22)
    dealer_cards = np.arange(1, 11)
    
    # Compute global Q-value ranges for each action
    q_mins = {0: float('inf'), 1: float('inf')}
    q_maxs = {0: float('-inf'), 1: float('-inf')}
    
    for agent in agents_dict.values():
        if isinstance(agent, SimpleAgent):
            for action in [0, 1]:
                for player_sum in player_sums:
                    for dealer_card in dealer_cards:
                        for usable_ace in [False, True]:
                            state = (player_sum, dealer_card, usable_ace)
                            q_val = agent.get_q_value(state, action)
                            q_mins[action] = min(q_mins[action], q_val)
                            q_maxs[action] = max(q_maxs[action], q_val)
    
    # Compute global epsilon ranges for adaptive agents
    epsilon_min = float('inf')
    epsilon_max = float('-inf')
    
    for agent in agents_dict.values():
        if isinstance(agent, AdaptiveEpsilonAgent):
            for player_sum in player_sums:
                for dealer_card in dealer_cards:
                    for usable_ace in [False, True]:
                        state = (player_sum, dealer_card, usable_ace)
                        eps_val = agent.get_epsilon_for_state(state)
                        epsilon_min = min(epsilon_min, eps_val)
                        epsilon_max = max(epsilon_max, eps_val)
    
    # Set limits with small padding
    q_limits_hit = (q_mins[1] * 1.05 if q_mins[1] < 0 else q_mins[1] * 0.95,
                    q_maxs[1] * 1.05 if q_maxs[1] > 0 else q_maxs[1] * 0.95)
    q_limits_stick = (q_mins[0] * 1.05 if q_mins[0] < 0 else q_mins[0] * 0.95,
                      q_maxs[0] * 1.05 if q_maxs[0] > 0 else q_maxs[0] * 0.95)
    
    if epsilon_min != float('inf'):
        epsilon_limits = (max(0, epsilon_min - 0.05), min(1, epsilon_max + 0.05))
    else:
        epsilon_limits = (0, 1)
    
    print(f"  Q-value range (Hit): [{q_limits_hit[0]:.3f}, {q_limits_hit[1]:.3f}]")
    print(f"  Q-value range (Stick): [{q_limits_stick[0]:.3f}, {q_limits_stick[1]:.3f}]")
    print(f"  Epsilon range: [{epsilon_limits[0]:.3f}, {epsilon_limits[1]:.3f}]")
    
    # 10. Individual agent visualizations with consistent scales
    for agent_name, agent in agents_dict.items():
        print(f"\n📈 Creating visualizations for {agent_name}...")
        
        # Q-value 3D plots with consistent scales
        safe_name = agent_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        plot_q_values_3d(agent, agent_name, action=1,
                        save_path=os.path.join(save_dir, f"q_values_3d_hit_{safe_name}.png"),
                        z_limits=q_limits_hit)
        plot_q_values_3d(agent, agent_name, action=0,
                        save_path=os.path.join(save_dir, f"q_values_3d_stick_{safe_name}.png"),
                        z_limits=q_limits_stick)
        
        # Epsilon 3D plot (for adaptive agents) with consistent scales
        if isinstance(agent, AdaptiveEpsilonAgent):
            plot_epsilon_values_3d(agent, agent_name,
                                  save_path=os.path.join(save_dir, f"epsilon_3d_{safe_name}.png"),
                                  z_limits=epsilon_limits)
        
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

