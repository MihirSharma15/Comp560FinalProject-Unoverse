"""Comparison module for evaluating and visualizing RL agents."""

from .agent_loader import (
    get_agent_configs,
    create_agent,
    save_agent,
    load_agent,
    train_agent,
    load_or_train_agents,
)

from .runner import (
    evaluate_agent,
    get_agent_statistics,
    run_comparison,
    print_comparison_summary,
    save_comparison_report,
)

from .visualizations import (
    plot_q_values_3d,
    plot_epsilon_values_3d,
    plot_win_rate_comparison,
    plot_training_performance,
    plot_exploration_statistics,
    plot_policy_heatmap,
    plot_average_reward_comparison,
    plot_policy_difference_heatmap,
    plot_action_value_gap_heatmap,
    plot_visit_count_heatmap,
    plot_learning_convergence,
    generate_all_visualizations,
)

from .compare_agents import main as compare_agents_main

__all__ = [
    # Agent loader functions
    "get_agent_configs",
    "create_agent",
    "save_agent",
    "load_agent",
    "train_agent",
    "load_or_train_agents",
    # Runner functions
    "evaluate_agent",
    "get_agent_statistics",
    "run_comparison",
    "print_comparison_summary",
    "save_comparison_report",
    # Visualization functions
    "plot_q_values_3d",
    "plot_epsilon_values_3d",
    "plot_win_rate_comparison",
    "plot_training_performance",
    "plot_exploration_statistics",
    "plot_policy_heatmap",
    "plot_average_reward_comparison",
    "plot_policy_difference_heatmap",
    "plot_action_value_gap_heatmap",
    "plot_visit_count_heatmap",
    "plot_learning_convergence",
    "generate_all_visualizations",
    # Main entry point
    "compare_agents_main",
]

