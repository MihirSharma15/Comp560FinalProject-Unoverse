"""Main GUI application for interactive visualization of RL agent comparisons."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict
import os

from ...agents.BaseAgent import BaseAgent
from ..agent_loader import load_or_train_agents
from ...env.env import make_blackjack_env
from ..runner import evaluate_agent, get_agent_statistics

from .widgets import (
    AgentSelectorPanel,
    VisualizationSelectorPanel,
    ControlPanel,
    StatusBar
)
from .plot_viewer import PlotViewer


class VisualizationApp:
    """Main application class for the visualization GUI."""
    
    def __init__(self, root: tk.Tk) -> None:
        """Initialize the visualization application.
        
        Args:
            root: Root tkinter window
        """
        self.root = root
        self.root.title("RL Agent Comparison Visualizer")
        self.root.geometry("1400x900")
        
        # Data storage
        self.agents_dict: Dict[str, BaseAgent] = {}
        self.results: Dict[str, Dict] = {}
        self.use_plotly_3d = tk.BooleanVar(value=True)
        
        # Define available visualizations
        self.visualizations = {
            "Comparison Plots": [
                "Win Rate Comparison",
                "Average Reward Comparison",
                "Training Performance",
                "Exploration Statistics",
                "Learning Convergence",
                "Policy Differences"
            ],
            "3D Q-Values (Hit)": [],  # Will be populated with agent names
            "3D Q-Values (Stick)": [],
            "3D Epsilon Values": [],
            "2D Heatmaps - Policy": [],
            "2D Heatmaps - Value Gap": [],
            "2D Heatmaps - Visit Counts": []
        }
        
        # Setup GUI
        self._setup_ui()
        
        # Load data
        self.root.after(100, self._load_data)
    
    def _setup_ui(self) -> None:
        """Setup the user interface components."""
        # Create main container
        main_container = ttk.PanedWindow(self.root, orient='horizontal')
        main_container.pack(fill='both', expand=True)
        
        # Left sidebar
        sidebar = ttk.Frame(main_container, width=250)
        main_container.add(sidebar, weight=0)
        
        # Right plot area
        plot_area = ttk.Frame(main_container)
        main_container.add(plot_area, weight=1)
        
        # === Sidebar Components ===
        
        # Title
        title_label = ttk.Label(
            sidebar,
            text="Visualization Controls",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)
        
        # Agent selector (will be populated after data load)
        self.agent_selector = None
        self.agent_frame = ttk.Frame(sidebar)
        self.agent_frame.pack(fill='x', padx=5, pady=5)
        
        # Visualization selector (will be populated after data load)
        self.viz_selector = None
        self.viz_frame = ttk.Frame(sidebar)
        self.viz_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 3D Plot options
        options_frame = ttk.LabelFrame(sidebar, text="3D Plot Options", padding=10)
        options_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(
            options_frame,
            text="Use Interactive Plotly\n(opens in browser)",
            variable=self.use_plotly_3d
        ).pack(anchor='w')
        
        # Control panel
        self.control_panel = ControlPanel(
            sidebar,
            on_refresh=self._refresh_data,
            on_export=self._export_plot,
            on_clear_cache=self._clear_cache
        )
        self.control_panel.pack(fill='x', padx=5, pady=5)
        
        # === Plot Area ===
        
        # Plot viewer
        self.plot_viewer = PlotViewer(plot_area)
        self.plot_viewer.pack(fill='both', expand=True)
        
        # === Status Bar ===
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(side='bottom', fill='x')
    
    def _load_data(self) -> None:
        """Load agents and results data."""
        self.status_bar.set_status("Loading data...")
        self.root.update()
        
        try:
            # Create environment
            env = make_blackjack_env(natural=False, sab=True)
            
            # Check if pretrained agents exist
            pretrained_dir = "src/pretrained"
            if not os.path.exists(pretrained_dir):
                os.makedirs(pretrained_dir, exist_ok=True)
            
            # Try to load agents
            agents_data = load_or_train_agents(
                env=env,
                n_train_episodes=250_000,
                pretrained_dir=pretrained_dir,
                force_retrain=False,
                verbose=False
            )
            
            # Extract agents and build results
            for agent_name, (agent, training_stats) in agents_data.items():
                self.agents_dict[agent_name] = agent
                
                # Evaluate if no training stats (loaded from disk)
                if training_stats is None:
                    eval_stats = evaluate_agent(agent, env, n_episodes=10_000, verbose=False)
                else:
                    eval_stats = evaluate_agent(agent, env, n_episodes=10_000, verbose=False)
                
                agent_stats = get_agent_statistics(agent)
                
                self.results[agent_name] = {
                    'agent': agent,
                    'train': training_stats,
                    'eval': eval_stats,
                    'agent_stats': agent_stats
                }
            
            # Populate visualizations with agent-specific options
            agent_names = list(self.agents_dict.keys())
            self.visualizations["3D Q-Values (Hit)"] = [f"{name}" for name in agent_names]
            self.visualizations["3D Q-Values (Stick)"] = [f"{name}" for name in agent_names]
            self.visualizations["3D Epsilon Values"] = [
                name for name in agent_names 
                if any(x in name for x in ["Count", "UCB", "Threshold"])
            ]
            self.visualizations["2D Heatmaps - Policy"] = agent_names
            self.visualizations["2D Heatmaps - Value Gap"] = agent_names
            self.visualizations["2D Heatmaps - Visit Counts"] = [
                name for name in agent_names 
                if any(x in name for x in ["Count", "UCB", "Threshold"])
            ]
            
            # Setup controls now that we have data
            self._setup_controls()
            
            self.status_bar.set_status("Ready")
            self.control_panel.update_info(f"Loaded {len(self.agents_dict)} agents")
            
        except Exception as e:
            self.status_bar.set_status("Error loading data")
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
    
    def _setup_controls(self) -> None:
        """Setup control widgets after data is loaded."""
        # Agent selector
        self.agent_selector = AgentSelectorPanel(
            self.agent_frame,
            list(self.agents_dict.keys()),
            on_change=self._on_agent_selection_change
        )
        self.agent_selector.pack(fill='both', expand=True)
        
        # Visualization selector
        self.viz_selector = VisualizationSelectorPanel(
            self.viz_frame,
            self.visualizations,
            on_change=self._on_viz_selection_change
        )
        self.viz_selector.pack(fill='both', expand=True)
        
        # Update status bar
        self._on_agent_selection_change()
    
    def _on_agent_selection_change(self) -> None:
        """Handle agent selection change."""
        if self.agent_selector:
            selected = self.agent_selector.get_selected_agents()
            self.status_bar.set_agent_info(selected)
            self._update_visualization()
    
    def _on_viz_selection_change(self) -> None:
        """Handle visualization selection change."""
        self._update_visualization()
    
    def _update_visualization(self) -> None:
        """Update the displayed visualization based on current selections."""
        if not self.viz_selector or not self.agent_selector:
            return
        
        selection = self.viz_selector.get_selected_visualization()
        if not selection or selection[0] is None:
            return
        
        category, viz_name = selection
        selected_agents = self.agent_selector.get_selected_agents()
        
        if not selected_agents and category == "Comparison Plots":
            messagebox.showwarning("No Selection", "Please select at least one agent")
            return
        
        self.status_bar.set_status(f"Generating {viz_name}...")
        self.root.update()
        
        try:
            # Comparison plots
            if category == "Comparison Plots":
                filtered_results = {k: v for k, v in self.results.items() if k in selected_agents}
                filtered_agents = {k: v for k, v in self.agents_dict.items() if k in selected_agents}
                self.plot_viewer.create_comparison_plot(
                    viz_name, filtered_results, filtered_agents, selected_agents
                )
            
            # Agent-specific plots
            elif category.startswith("3D Q-Values"):
                action = 1 if "Hit" in category else 0
                agent_name = viz_name
                if agent_name in self.agents_dict:
                    self.plot_viewer.create_agent_plot(
                        f"Q-Values 3D {'Hit' if action == 1 else 'Stick'}",
                        agent_name,
                        self.agents_dict[agent_name],
                        use_plotly=self.use_plotly_3d.get()
                    )
            
            elif category == "3D Epsilon Values":
                agent_name = viz_name
                if agent_name in self.agents_dict:
                    self.plot_viewer.create_agent_plot(
                        "Epsilon 3D",
                        agent_name,
                        self.agents_dict[agent_name],
                        use_plotly=self.use_plotly_3d.get()
                    )
            
            elif category == "2D Heatmaps - Policy":
                agent_name = viz_name
                if agent_name in self.agents_dict:
                    self.plot_viewer.create_agent_plot(
                        "Policy Heatmap",
                        agent_name,
                        self.agents_dict[agent_name]
                    )
            
            elif category == "2D Heatmaps - Value Gap":
                agent_name = viz_name
                if agent_name in self.agents_dict:
                    self.plot_viewer.create_agent_plot(
                        "Action Value Gap",
                        agent_name,
                        self.agents_dict[agent_name]
                    )
            
            elif category == "2D Heatmaps - Visit Counts":
                agent_name = viz_name
                if agent_name in self.agents_dict:
                    self.plot_viewer.create_agent_plot(
                        "Visit Counts",
                        agent_name,
                        self.agents_dict[agent_name]
                    )
            
            self.status_bar.set_status("Ready")
            self.control_panel.update_info(f"Displaying: {viz_name}")
            
        except Exception as e:
            self.status_bar.set_status("Error generating plot")
            messagebox.showerror("Error", f"Failed to generate plot:\n{str(e)}")
    
    def _refresh_data(self) -> None:
        """Refresh data by reloading agents and results."""
        response = messagebox.askyesno(
            "Refresh Data",
            "This will reload all agents and results. Continue?"
        )
        if response:
            self.plot_viewer.clear()
            self.plot_viewer.clear_cache()
            self.agents_dict.clear()
            self.results.clear()
            self._load_data()
    
    def _export_plot(self) -> None:
        """Export the current plot to a file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.plot_viewer.export_current_plot(filepath)
                messagebox.showinfo("Success", f"Plot exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plot:\n{str(e)}")
    
    def _clear_cache(self) -> None:
        """Clear the plot cache."""
        self.plot_viewer.clear_cache()
        self.control_panel.update_info("Cache cleared")
        messagebox.showinfo("Cache Cleared", "Plot cache has been cleared")
    
    def run(self) -> None:
        """Start the GUI event loop."""
        self.root.mainloop()


def launch_gui() -> None:
    """Launch the visualization GUI application."""
    root = tk.Tk()
    app = VisualizationApp(root)
    app.run()


if __name__ == "__main__":
    launch_gui()

