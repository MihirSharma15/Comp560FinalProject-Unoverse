"""Plot viewer for displaying matplotlib and plotly visualizations."""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import plotly.graph_objects as go
import tempfile
import webbrowser
import os

from ...agents.BaseAgent import BaseAgent
from ...agents.SimpleAgent import SimpleAgent
from ...agents.AdaptiveEpsilonAgent import AdaptiveEpsilonAgent
from .interactive_plots import (
    create_interactive_q_values_3d,
    create_interactive_epsilon_3d,
)


class PlotViewer(ttk.Frame):
    """Widget for displaying and managing plot visualizations."""
    
    def __init__(self, parent: tk.Widget) -> None:
        """Initialize the plot viewer.
        
        Args:
            parent: Parent tkinter widget
        """
        super().__init__(parent)
        
        self.current_canvas: Optional[FigureCanvasTkAgg] = None
        self.current_toolbar: Optional[NavigationToolbar2Tk] = None
        self.current_figure: Optional[Figure] = None
        self.plot_cache: Dict[str, Any] = {}
        
        # Create container for plot
        self.plot_container = ttk.Frame(self)
        self.plot_container.pack(fill='both', expand=True)
        
        # Placeholder label
        self.placeholder = ttk.Label(
            self.plot_container,
            text="Select a visualization to display",
            font=('Arial', 16)
        )
        self.placeholder.pack(expand=True)
    
    def clear(self) -> None:
        """Clear the current plot display."""
        # Destroy all widgets in plot_container
        for widget in self.plot_container.winfo_children():
            widget.destroy()
        
        if self.current_canvas:
            self.current_canvas = None
        
        if self.current_toolbar:
            self.current_toolbar = None
        
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None
        
        # Recreate placeholder
        self.placeholder = ttk.Label(
            self.plot_container,
            text="Select a visualization to display",
            font=('Arial', 16)
        )
        self.placeholder.pack(expand=True)
    
    def display_matplotlib_figure(self, fig: Figure) -> None:
        """Display a matplotlib figure in the viewer.
        
        Args:
            fig: Matplotlib Figure object
        """
        # Clear existing content first
        self.clear()
        
        # Create canvas
        self.current_figure = fig
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.current_canvas.draw()
        
        # Add toolbar in a dedicated frame
        toolbar_frame = ttk.Frame(self.plot_container)
        toolbar_frame.pack(side='top', fill='x')
        self.current_toolbar = NavigationToolbar2Tk(self.current_canvas, toolbar_frame)
        self.current_toolbar.update()
        
        # Pack canvas below toolbar
        self.current_canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
    
    def display_plotly_figure(self, fig: go.Figure) -> None:
        """Display a plotly figure by opening in browser.
        
        Args:
            fig: Plotly Figure object
        """
        # Clear existing content first
        self.clear()
        
        # Save to temporary HTML file and open in browser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            fig.write_html(f.name)
            temp_path = f.name
        
        # Open in default browser
        webbrowser.open('file://' + os.path.abspath(temp_path))
        
        # Update placeholder message (already created by clear())
        self.placeholder.config(text="Interactive 3D plot opened in browser\n\nClick and drag to rotate the plot")
    
    def create_comparison_plot(
        self,
        viz_name: str,
        results: Dict[str, Dict],
        agents_dict: Dict[str, BaseAgent],
        selected_agents: Optional[list] = None
    ) -> None:
        """Create and display a comparison plot.
        
        Args:
            viz_name: Name of the visualization
            results: Dictionary of results for all agents
            agents_dict: Dictionary of agent instances
            selected_agents: List of agent names to include (None for all)
        """
        # Filter results and agents if selection provided
        if selected_agents:
            results = {k: v for k, v in results.items() if k in selected_agents}
            agents_dict = {k: v for k, v in agents_dict.items() if k in selected_agents}
        
        # Check cache
        cache_key = f"{viz_name}_{','.join(sorted(results.keys()))}"
        if cache_key in self.plot_cache:
            fig = self.plot_cache[cache_key]
            self.display_matplotlib_figure(fig)
            return
        
        # Create figure based on visualization type
        fig = Figure(figsize=(12, 8))
        
        if viz_name == "Win Rate Comparison":
            self._plot_win_rate(fig, results)
        elif viz_name == "Average Reward Comparison":
            self._plot_avg_reward(fig, results)
        elif viz_name == "Training Performance":
            self._plot_training_performance(fig, results)
        elif viz_name == "Exploration Statistics":
            self._plot_exploration_stats(fig, results)
        elif viz_name == "Learning Convergence":
            self._plot_learning_convergence(fig, results)
        elif viz_name == "Policy Differences":
            self._plot_policy_differences(fig, agents_dict)
        else:
            # Unknown visualization
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Visualization '{viz_name}' not implemented",
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
        
        # Cache and display
        self.plot_cache[cache_key] = fig
        self.display_matplotlib_figure(fig)
    
    def create_agent_plot(
        self,
        viz_name: str,
        agent_name: str,
        agent: BaseAgent,
        use_plotly: bool = False
    ) -> None:
        """Create and display an agent-specific plot.
        
        Args:
            viz_name: Name of the visualization
            agent_name: Name of the agent
            agent: Agent instance
            use_plotly: Whether to use plotly for 3D plots
        """
        # Check cache
        cache_key = f"{viz_name}_{agent_name}"
        
        if "Q-Values" in viz_name and "3D" in viz_name:
            action = 1 if "Hit" in viz_name else 0
            
            if use_plotly:
                fig = create_interactive_q_values_3d(agent, agent_name, action)
                self.display_plotly_figure(fig)
            else:
                if cache_key not in self.plot_cache:
                    fig = Figure(figsize=(16, 6))
                    self._plot_q_values_3d_mpl(fig, agent, agent_name, action)
                    self.plot_cache[cache_key] = fig
                self.display_matplotlib_figure(self.plot_cache[cache_key])
        
        elif "Epsilon" in viz_name and "3D" in viz_name:
            if use_plotly:
                fig = create_interactive_epsilon_3d(agent, agent_name)
                self.display_plotly_figure(fig)
            else:
                if cache_key not in self.plot_cache:
                    fig = Figure(figsize=(16, 6))
                    self._plot_epsilon_3d_mpl(fig, agent, agent_name)
                    self.plot_cache[cache_key] = fig
                self.display_matplotlib_figure(self.plot_cache[cache_key])
        
        elif "Policy Heatmap" in viz_name:
            if cache_key not in self.plot_cache:
                fig = Figure(figsize=(14, 6))
                self._plot_policy_heatmap(fig, agent, agent_name)
                self.plot_cache[cache_key] = fig
            self.display_matplotlib_figure(self.plot_cache[cache_key])
        
        elif "Action Value Gap" in viz_name:
            if cache_key not in self.plot_cache:
                fig = Figure(figsize=(14, 6))
                self._plot_action_value_gap(fig, agent, agent_name)
                self.plot_cache[cache_key] = fig
            self.display_matplotlib_figure(self.plot_cache[cache_key])
        
        elif "Visit Counts" in viz_name:
            if cache_key not in self.plot_cache:
                fig = Figure(figsize=(14, 6))
                self._plot_visit_counts(fig, agent, agent_name)
                self.plot_cache[cache_key] = fig
            self.display_matplotlib_figure(self.plot_cache[cache_key])
        
        else:
            # Unknown visualization
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Visualization '{viz_name}' not implemented",
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.display_matplotlib_figure(fig)
    
    def _plot_win_rate(self, fig: Figure, results: Dict) -> None:
        """Plot win rate comparison."""
        ax = fig.add_subplot(111)
        
        agents = list(results.keys())
        win_rates = [results[agent]['eval']['win_rate'] for agent in agents]
        loss_rates = [results[agent]['eval']['loss_rate'] for agent in agents]
        draw_rates = [results[agent]['eval']['draw_rate'] for agent in agents]
        
        x = range(len(agents))
        width = 0.25
        
        ax.bar([i - width for i in x], win_rates, width, label='Wins', color='#2ecc71', alpha=0.8)
        ax.bar(x, loss_rates, width, label='Losses', color='#e74c3c', alpha=0.8)
        ax.bar([i + width for i in x], draw_rates, width, label='Draws', color='#95a5a6', alpha=0.8)
        
        ax.set_xlabel('Agent Type', fontweight='bold')
        ax.set_ylabel('Rate', fontweight='bold')
        ax.set_title('Win/Loss/Draw Rate Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
    
    def _plot_avg_reward(self, fig: Figure, results: Dict) -> None:
        """Plot average reward comparison."""
        import seaborn as sns
        ax = fig.add_subplot(111)
        
        agents = list(results.keys())
        avg_rewards = [results[agent]['eval']['avg_reward'] for agent in agents]
        std_rewards = [results[agent]['eval']['std_reward'] for agent in agents]
        
        colors = sns.color_palette("coolwarm", len(agents))
        ax.bar(range(len(agents)), avg_rewards, yerr=std_rewards, capsize=5,
               color=colors, alpha=0.8, ecolor='black')
        
        ax.set_xlabel('Agent Type', fontweight='bold')
        ax.set_ylabel('Average Reward', fontweight='bold')
        ax.set_title('Evaluation Performance: Average Reward ± Std Dev', 
                     fontweight='bold', fontsize=14)
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, rotation=15, ha='right')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.4)
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
    
    def _plot_training_performance(self, fig: Figure, results: Dict) -> None:
        """Plot training performance."""
        import numpy as np
        ax = fig.add_subplot(111)
        
        window_size = 1000
        for agent_name, data in results.items():
            if data['train'] is None:
                continue
            
            rewards = np.array(data['train']['total_rewards'])
            if len(rewards) >= window_size:
                # Compute rolling mean efficiently using cumsum
                cumsum = np.cumsum(np.insert(rewards, 0, 0))
                moving_avg = np.zeros(len(rewards))
                # For early episodes (< window_size), use expanding window
                for i in range(min(window_size, len(rewards))):
                    moving_avg[i] = cumsum[i+1] / (i+1)
                # For later episodes, use fixed window
                if len(rewards) > window_size:
                    moving_avg[window_size:] = (cumsum[window_size+1:] - cumsum[:-window_size]) / window_size
                x = np.arange(len(rewards))
                ax.plot(x, moving_avg, label=agent_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Training Episode', fontweight='bold')
        ax.set_ylabel(f'Average Reward (window={window_size})', fontweight='bold')
        ax.set_title('Training Performance Comparison', fontweight='bold', fontsize=14)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        
        fig.tight_layout()
    
    def _plot_exploration_stats(self, fig: Figure, results: Dict) -> None:
        """Plot exploration statistics."""
        import seaborn as sns
        
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
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No exploration statistics available\n(Only adaptive agents have this data)",
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            fig.tight_layout()
            return
        
        # Create two subplots
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        colors = sns.color_palette("husl", len(agents))
        
        # Plot 1: Unique states visited
        bars1 = ax1.bar(agents, unique_states, color=colors, alpha=0.8)
        ax1.set_xlabel('Agent Type', fontweight='bold')
        ax1.set_ylabel('Unique States Visited', fontweight='bold')
        ax1.set_title('State Space Coverage', fontweight='bold')
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Average visits per state
        bars2 = ax2.bar(agents, avg_visits, color=colors, alpha=0.8)
        ax2.set_xlabel('Agent Type', fontweight='bold')
        ax2.set_ylabel('Average Visits per State', fontweight='bold')
        ax2.set_title('State Visit Distribution', fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        fig.tight_layout()
    
    def _plot_learning_convergence(self, fig: Figure, results: Dict) -> None:
        """Plot learning convergence."""
        import numpy as np
        import seaborn as sns
        
        # Check if any agents have training data
        has_training_data = any(data['train'] is not None for data in results.values())
        
        if not has_training_data:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 
                   "Training data not available\n\n"
                   "Agents were loaded from pretrained files.\n"
                   "Click 'Refresh Data' and retrain to see learning curves,\n"
                   "or check 'Training Performance' for available data.",
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            fig.tight_layout()
            return
        
        ax = fig.add_subplot(111)
        colors = sns.color_palette("husl", len(results))
        
        window_size = 10_000
        convergence_threshold = 0.001
        plotted_any = False
        
        for idx, (agent_name, data) in enumerate(results.items()):
            if data['train'] is None:
                continue
            
            rewards = np.array(data['train']['total_rewards'])
            if len(rewards) < window_size:
                continue
            
            plotted_any = True
            
            # Compute rolling mean efficiently using cumsum
            cumsum = np.cumsum(np.insert(rewards, 0, 0))
            rolling_mean = np.zeros(len(rewards))
            # For early episodes (< window_size), use expanding window
            for i in range(min(window_size, len(rewards))):
                rolling_mean[i] = cumsum[i+1] / (i+1)
            # For later episodes, use fixed window
            if len(rewards) > window_size:
                rolling_mean[window_size:] = (cumsum[window_size+1:] - cumsum[:-window_size]) / window_size
            x = np.arange(len(rewards))
            
            # Plot mean line
            ax.plot(x, rolling_mean, label=agent_name, linewidth=2.5, 
                   alpha=0.9, color=colors[idx])
            
            # Find convergence point
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
        
        if not plotted_any:
            ax.text(0.5, 0.5, 
                   "Training episodes too short for convergence analysis\n"
                   f"(requires at least {window_size:,} episodes)",
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
        else:
            ax.set_xlabel('Training Episode', fontweight='bold')
            ax.set_ylabel(f'Average Reward (window={window_size:,})', fontweight='bold')
            ax.set_title('Learning Convergence Analysis', fontweight='bold', fontsize=14)
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        
        fig.tight_layout()
    
    def _plot_policy_differences(self, fig: Figure, agents_dict: Dict) -> None:
        """Plot policy differences."""
        import numpy as np
        
        if len(agents_dict) < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Need at least 2 agents for policy comparison",
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            fig.tight_layout()
            return
        
        baseline_name = list(agents_dict.keys())[0]
        baseline_agent = agents_dict[baseline_name]
        other_agents = [(name, agent) for name, agent in agents_dict.items() if name != baseline_name]
        
        player_sums = np.arange(12, 22)
        dealer_cards = np.arange(1, 11)
        
        # Create subplots
        n_agents = len(other_agents)
        fig.clear()
        
        for col_idx, (agent_name, agent) in enumerate(other_agents):
            for row_idx, usable_ace in enumerate([False, True]):
                ax = fig.add_subplot(2, n_agents, row_idx * n_agents + col_idx + 1)
                
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
                im = ax.imshow(disagreement, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
                
                ax.set_xlabel('Dealer Showing', fontsize=9, fontweight='bold')
                ax.set_ylabel('Player Sum', fontsize=9, fontweight='bold')
                ax.set_title(f'{agent_name} vs {baseline_name}\n'
                            f'{"Usable" if usable_ace else "No"} Ace '
                            f'({disagreement_pct:.1f}% disagree)', 
                            fontsize=10, fontweight='bold')
                
                ax.set_xticks(range(len(dealer_cards)))
                ax.set_xticklabels(dealer_cards)
                ax.set_yticks(range(len(player_sums)))
                ax.set_yticklabels(player_sums)
        
        fig.suptitle('Policy Disagreement Heatmaps', fontweight='bold', fontsize=12)
        fig.tight_layout()
    
    def _plot_q_values_3d_mpl(self, fig: Figure, agent: BaseAgent, agent_name: str, action: int) -> None:
        """Plot Q-values in 3D using matplotlib."""
        import numpy as np
        
        player_sums = np.arange(4, 22)
        dealer_cards = np.arange(1, 11)
        X, Y = np.meshgrid(dealer_cards, player_sums)
        
        for idx, usable_ace in enumerate([False, True]):
            ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
            
            Z = np.zeros_like(X, dtype=float)
            for i, player_sum in enumerate(player_sums):
                for j, dealer_card in enumerate(dealer_cards):
                    state = (player_sum, dealer_card, usable_ace)
                    if isinstance(agent, SimpleAgent):
                        Z[i, j] = agent.get_q_value(state, action)
            
            surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
            ax.set_xlabel('Dealer')
            ax.set_ylabel('Player')
            ax.set_zlabel('Q-value')
            ax.set_title(f'{"Usable" if usable_ace else "No"} Ace')
        
        fig.suptitle(f'{agent_name}: Q-Values', fontweight='bold')
        fig.tight_layout()
    
    def _plot_epsilon_3d_mpl(self, fig: Figure, agent: BaseAgent, agent_name: str) -> None:
        """Plot epsilon values in 3D using matplotlib."""
        import numpy as np
        
        if not isinstance(agent, AdaptiveEpsilonAgent):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Epsilon 3D only available for adaptive agents",
                   ha='center', va='center')
            ax.axis('off')
            fig.tight_layout()
            return
        
        player_sums = np.arange(4, 22)
        dealer_cards = np.arange(1, 11)
        X, Y = np.meshgrid(dealer_cards, player_sums)
        
        for idx, usable_ace in enumerate([False, True]):
            ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
            
            Z = np.zeros_like(X, dtype=float)
            for i, player_sum in enumerate(player_sums):
                for j, dealer_card in enumerate(dealer_cards):
                    state = (player_sum, dealer_card, usable_ace)
                    Z[i, j] = agent.get_epsilon_for_state(state)
            
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax.set_xlabel('Dealer')
            ax.set_ylabel('Player')
            ax.set_zlabel('Epsilon')
            ax.set_title(f'{"Usable" if usable_ace else "No"} Ace')
        
        fig.suptitle(f'{agent_name}: Epsilon Values', fontweight='bold')
        fig.tight_layout()
    
    def _plot_policy_heatmap(self, fig: Figure, agent: BaseAgent, agent_name: str) -> None:
        """Plot policy heatmap."""
        import numpy as np
        import seaborn as sns
        
        player_sums = np.arange(12, 22)
        dealer_cards = np.arange(1, 11)
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        for ax, usable_ace in zip([ax1, ax2], [False, True]):
            policy = np.zeros((len(player_sums), len(dealer_cards)))
            
            for i, player_sum in enumerate(player_sums):
                for j, dealer_card in enumerate(dealer_cards):
                    state = (player_sum, dealer_card, usable_ace)
                    if isinstance(agent, SimpleAgent):
                        q_values = agent.get_policy(state)
                        policy[i, j] = np.argmax(q_values)
            
            sns.heatmap(policy, ax=ax, cmap='RdYlGn', cbar_kws={'label': 'Action'},
                       xticklabels=dealer_cards, yticklabels=player_sums,
                       linewidths=0.5, linecolor='gray', alpha=0.8, vmin=0, vmax=1)
            
            ax.set_xlabel('Dealer Showing', fontweight='bold')
            ax.set_ylabel('Player Sum', fontweight='bold')
            ax.set_title(f'{"Usable" if usable_ace else "No"} Ace', fontweight='bold')
            
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0.25, 0.75])
            cbar.set_ticklabels(['Stick', 'Hit'])
        
        fig.suptitle(f'{agent_name}: Learned Policy', fontweight='bold', fontsize=14)
        fig.tight_layout()
    
    def _plot_action_value_gap(self, fig: Figure, agent: BaseAgent, agent_name: str) -> None:
        """Plot action value gap."""
        import numpy as np
        import seaborn as sns
        
        player_sums = np.arange(12, 22)
        dealer_cards = np.arange(1, 11)
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        for ax, usable_ace in zip([ax1, ax2], [False, True]):
            value_gap = np.zeros((len(player_sums), len(dealer_cards)))
            
            for i, player_sum in enumerate(player_sums):
                for j, dealer_card in enumerate(dealer_cards):
                    state = (player_sum, dealer_card, usable_ace)
                    if isinstance(agent, SimpleAgent):
                        q_values = agent.get_policy(state)
                        value_gap[i, j] = q_values[1] - q_values[0]  # Hit - Stick
            
            max_abs = max(abs(value_gap.min()), abs(value_gap.max())) if value_gap.size > 0 else 1
            sns.heatmap(value_gap, ax=ax, cmap='RdBu_r', center=0,
                       cbar_kws={'label': 'Q(Hit) - Q(Stick)'},
                       xticklabels=dealer_cards, yticklabels=player_sums,
                       linewidths=0.5, linecolor='gray', alpha=0.8,
                       vmin=-max_abs, vmax=max_abs)
            
            ax.set_xlabel('Dealer Showing', fontweight='bold')
            ax.set_ylabel('Player Sum', fontweight='bold')
            ax.set_title(f'{"Usable" if usable_ace else "No"} Ace', fontweight='bold')
        
        fig.suptitle(f'{agent_name}: Action Value Gap\n(Red=Prefer Hit, Blue=Prefer Stick)', 
                     fontweight='bold', fontsize=14)
        fig.tight_layout()
    
    def _plot_visit_counts(self, fig: Figure, agent: BaseAgent, agent_name: str) -> None:
        """Plot visit counts."""
        import numpy as np
        import seaborn as sns
        
        if not isinstance(agent, AdaptiveEpsilonAgent):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Visit counts only available for adaptive agents",
                   ha='center', va='center')
            ax.axis('off')
            fig.tight_layout()
            return
        
        player_sums = np.arange(12, 22)
        dealer_cards = np.arange(1, 11)
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        for ax, usable_ace in zip([ax1, ax2], [False, True]):
            visit_counts = np.zeros((len(player_sums), len(dealer_cards)))
            
            for i, player_sum in enumerate(player_sums):
                for j, dealer_card in enumerate(dealer_cards):
                    state = (player_sum, dealer_card, usable_ace)
                    visit_counts[i, j] = agent.state_counts.get(state, 0)
            
            log_counts = np.log1p(visit_counts)
            
            sns.heatmap(log_counts, ax=ax, cmap='YlOrRd',
                       cbar_kws={'label': 'log(1 + visits)'},
                       xticklabels=dealer_cards, yticklabels=player_sums,
                       linewidths=0.5, linecolor='gray', alpha=0.8)
            
            ax.set_xlabel('Dealer Showing', fontweight='bold')
            ax.set_ylabel('Player Sum', fontweight='bold')
            ax.set_title(f'{"Usable" if usable_ace else "No"} Ace', fontweight='bold')
        
        fig.suptitle(f'{agent_name}: State Visit Counts', fontweight='bold', fontsize=14)
        fig.tight_layout()
    
    def clear_cache(self) -> None:
        """Clear the plot cache to free memory."""
        for fig in self.plot_cache.values():
            if isinstance(fig, Figure):
                plt.close(fig)
        self.plot_cache.clear()
    
    def export_current_plot(self, filepath: str) -> None:
        """Export the currently displayed plot.
        
        Args:
            filepath: Path to save the plot
        """
        if self.current_figure:
            self.current_figure.savefig(filepath, dpi=150, bbox_inches='tight')

