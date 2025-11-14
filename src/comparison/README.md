# Agent Comparison Module

A comprehensive module for comparing and visualizing different reinforcement learning agents on the Blackjack environment.

## Overview

This module provides tools to:
- Train or load 4 different RL agents (Simple, Count-Based, UCB, Threshold)
- Evaluate their performance on Blackjack
- Generate clean, publication-ready visualizations
- Compare exploration strategies and learned policies

## Quick Start

### Option 1: Run from project root

```bash
python run_comparison.py
```

### Option 2: Run with custom parameters

```bash
python run_comparison.py \
    --train-episodes 500000 \
    --eval-episodes 20000 \
    --output-dir results/my_comparison \
    --force-retrain
```

### Option 3: Use as a Python module

```python
from src.comparison import compare_agents_main

compare_agents_main(
    n_train_episodes=250_000,
    n_eval_episodes=10_000,
    output_dir="results/comparison",
    force_retrain=False,
    seed=42
)
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train-episodes` | int | 250,000 | Number of training episodes per agent |
| `--eval-episodes` | int | 10,000 | Number of evaluation episodes per agent |
| `--pretrained-dir` | str | `src/pretrained` | Directory containing pre-trained agents |
| `--output-dir` | str | `results/comparison` | Directory to save results |
| `--force-retrain` | flag | False | Force retraining even if pretrained agents exist |
| `--seed` | int | 42 | Random seed (-1 for no seed) |
| `--quiet` | flag | False | Suppress verbose output |

## Agents Compared

### 1. Simple (Fixed Epsilon)
- **Strategy**: Fixed epsilon with linear decay
- **Configuration**: ε starts at 1.0, decays to 0.01 over training
- **Description**: Baseline agent with standard ε-greedy exploration

### 2. Count-Based (k=10)
- **Strategy**: Adaptive epsilon based on state visit counts
- **Formula**: ε(s) = k / (k + N(s))
- **Description**: Higher exploration in rarely-visited states

### 3. UCB-Style (c=1.0)
- **Strategy**: Upper Confidence Bound inspired exploration
- **Formula**: ε(s) = ε_base + c / √(N(s) + 1)
- **Description**: Exploration bonus that decreases with √visits

### 4. Threshold-Based
- **Strategy**: Piecewise constant epsilon based on visit thresholds
- **Thresholds**: (10, 100, 1000) visits
- **Epsilon Levels**: (0.9, 0.5, 0.2, 0.05)
- **Description**: Discrete exploration levels based on state familiarity

## Generated Visualizations

### Comparison Plots

1. **Win Rate Comparison** (`win_rate_comparison.png`)
   - Bar chart showing win/loss/draw rates for each agent
   - Useful for understanding overall performance

2. **Average Reward Comparison** (`avg_reward_comparison.png`)
   - Bar chart with error bars showing mean reward ± std dev
   - Best single metric for comparing agent quality

3. **Training Performance** (`training_performance.png`)
   - Line plot showing reward progression during training
   - Smoothed with moving average (window=1000)

4. **Exploration Statistics** (`exploration_stats.png`)
   - Two-panel plot showing:
     - Unique states visited (state space coverage)
     - Average visits per state (exploration efficiency)

### Individual Agent Plots

For each agent, the following visualizations are generated:

1. **Q-Value 3D Plots** (`q_values_3d_hit_*.png`, `q_values_3d_stick_*.png`)
   - 3D surface plots showing Q-values across (dealer card, player sum)
   - Separate plots for "Hit" and "Stick" actions
   - Two subplots: usable ace vs no usable ace

2. **Epsilon 3D Plots** (`epsilon_3d_*.png`) _(Adaptive agents only)_
   - 3D surface plots showing epsilon values across state space
   - Visualizes how exploration is allocated

3. **Policy Heatmaps** (`policy_heatmap_*.png`)
   - 2D heatmaps showing learned policy (best action per state)
   - Color-coded: Green=Hit, Red=Stick
   - Two subplots: usable ace vs no usable ace

## Module Structure

```
src/comparison/
├── __init__.py              # Module exports
├── agent_loader.py          # Agent creation, loading, saving
├── runner.py                # Training and evaluation orchestration
├── visualizations.py        # All visualization functions
├── compare_agents.py        # Main entry point with CLI
└── README.md                # This file
```

## File Descriptions

### `agent_loader.py`
- `get_agent_configs()`: Get default configurations for all agent types
- `create_agent()`: Create an agent instance with specified config
- `save_agent()` / `load_agent()`: Pickle-based agent persistence
- `train_agent()`: Train an agent with progress tracking
- `load_or_train_agents()`: Main function to load or train all agents

### `runner.py`
- `evaluate_agent()`: Evaluate agent performance (no learning)
- `get_agent_statistics()`: Extract exploration statistics from agents
- `run_comparison()`: Orchestrate complete comparison workflow
- `print_comparison_summary()`: Print formatted results table
- `save_comparison_report()`: Save detailed text report

### `visualizations.py`
- `plot_q_values_3d()`: 3D Q-value surface plots
- `plot_epsilon_values_3d()`: 3D epsilon surface plots
- `plot_win_rate_comparison()`: Win/loss/draw bar chart
- `plot_training_performance()`: Training reward progression
- `plot_exploration_statistics()`: Exploration metrics
- `plot_policy_heatmap()`: 2D policy visualization
- `plot_average_reward_comparison()`: Evaluation reward comparison
- `generate_all_visualizations()`: Create all plots at once

### `compare_agents.py`
- `main()`: Main entry point function
- `parse_args()`: Command-line argument parsing

## Output Files

After running the comparison, the output directory will contain:

```
results/comparison/
├── comparison_report.txt                    # Detailed text report
├── win_rate_comparison.png                  # Win/loss/draw rates
├── avg_reward_comparison.png                # Average rewards
├── training_performance.png                 # Training curves
├── exploration_stats.png                    # Exploration metrics
├── q_values_3d_hit_Simple_Fixed_Epsilon.png
├── q_values_3d_stick_Simple_Fixed_Epsilon.png
├── policy_heatmap_Simple_Fixed_Epsilon.png
├── epsilon_3d_Count-Based_k10.png
├── ...                                      # Similar files for each agent
```

## Pre-trained Agents

Agents can be saved/loaded from the `src/pretrained/` directory:

```
src/pretrained/
├── simple.pkl          # Simple agent
├── count_based.pkl     # Count-based agent
├── ucb.pkl             # UCB-style agent
└── threshold.pkl       # Threshold-based agent
```

To use pre-trained agents:
1. Place agent `.pkl` files in `src/pretrained/`
2. Run comparison without `--force-retrain`
3. Missing agents will be trained automatically

## Customization

### Custom Agent Parameters

Modify `agent_loader.py` → `get_agent_configs()` to change default parameters:

```python
"count_based": {
    "params": {
        "learning_rate": 0.01,
        "k_param": 20.0,  # Change exploration parameter
        # ...
    }
}
```

### Custom Visualizations

Add new visualization functions in `visualizations.py`:

```python
def plot_my_custom_visualization(results, save_path=None):
    # Your visualization code
    pass
```

Then call it in `generate_all_visualizations()`.

## Requirements

- Python 3.10+
- gymnasium
- numpy
- matplotlib
- seaborn
- tqdm

## Tips

1. **Fast Testing**: Use fewer episodes for quick tests:
   ```bash
   python run_comparison.py --train-episodes 50000 --eval-episodes 1000
   ```

2. **Reproducibility**: Always use the same seed:
   ```bash
   python run_comparison.py --seed 42
   ```

3. **Pre-training**: Train agents once, then load for quick comparisons:
   ```bash
   # First run (trains and saves)
   python run_comparison.py
   
   # Subsequent runs (loads from disk)
   python run_comparison.py  # Fast!
   ```

4. **Clean Retrain**: Force fresh training:
   ```bash
   python run_comparison.py --force-retrain
   ```

## Troubleshooting

### Issue: "No module named 'src'"
- Run from project root directory
- Or add project root to PYTHONPATH

### Issue: Plots look cluttered
- Increase figure DPI in `visualizations.py`
- Adjust seaborn style settings

### Issue: Training takes too long
- Reduce `--train-episodes`
- Use pre-trained agents (remove `--force-retrain`)

## License

Part of the Comp560FinalProject-Unoverse.

