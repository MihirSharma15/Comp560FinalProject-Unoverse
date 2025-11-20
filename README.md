# Comp560 Final Project - Unoverse (Blackjack RL Agent Comparison)

Slides Deck: https://docs.google.com/presentation/d/19labg7LYWOEc9VMf5BaQ8uGpuzO_Sr9nnQyQg39kyog/edit?usp=sharing
Final Paper: 
A comprehensive reinforcement learning project comparing different Q-Learning agents with various exploration strategies on the Blackjack environment. Features both command-line comparison tools and an interactive GUI for visualization.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Terminology](#key-terminology)
- [Setup & Installation](#setup--installation)
- [Quick Start](#quick-start)
- [Core Functionalities](#core-functionalities)
- [Agents Explained](#agents-explained)
- [Visualizations](#visualizations)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements and compares **4 different reinforcement learning agents** learning to play Blackjack using Q-Learning with different exploration strategies:

1. **Simple (Fixed Epsilon)** - Baseline agent with linear epsilon decay
2. **Count-Based** - Adaptive exploration based on state visit counts
3. **UCB-Style** - Upper Confidence Bound inspired exploration
4. **Threshold-Based** - Piecewise epsilon based on visit thresholds

### What This Project Does

- **Trains** RL agents to play Blackjack optimally using Q-Learning
- **Compares** different exploration strategies (how agents balance exploration vs exploitation)
- **Visualizes** agent performance, learned policies, and decision-making
- **Provides** an interactive GUI for exploring results with rotatable 3D plots
- **Generates** publication-ready comparison plots and heatmaps

### Key Features

✅ 4 different exploration strategies implemented  
✅ 11+ visualization types (3D Q-values, heatmaps, performance charts)  
✅ Interactive GUI with rotatable 3D plots  
✅ Comprehensive comparison framework  
✅ Pre-trained agent support  
✅ Extensible architecture for adding new agents  

---

## 📁 Project Structure

```
Comp560FinalProject-Unoverse/
├── README.md                    # This file
├── QUICKSTART.md               # 5-minute setup guide
├── requirements.txt            # pip dependencies
├── pyproject.toml              # Project dependencies (uv/pip)
├── uv.lock                     # Locked dependencies
│
├── src/                        # Source code
│   ├── agents/                 # RL agent implementations
│   │   ├── BaseAgent.py        # Abstract base class
│   │   ├── SimpleAgent.py      # Fixed epsilon Q-Learning
│   │   ├── AdaptiveEpsilonAgent.py  # Adaptive exploration agents
│   │   └── MonteCarloAgent.py  # Monte Carlo (alternative approach)
│   │
│   ├── env/                    # Environment wrappers
│   │   └── env.py              # Blackjack environment setup
│   │
│   ├── training/               # Training & evaluation utilities
│   │   ├── train.py            # Training functions
│   │   └── evaluate.py         # Evaluation functions
│   │
│   ├── comparison/             # 🎯 Main comparison module
│   │   ├── __init__.py         # Module exports
│   │   ├── agent_loader.py     # Agent creation & loading
│   │   ├── runner.py           # Comparison orchestration
│   │   ├── visualizations.py   # All plotting functions
│   │   ├── compare_agents.py   # CLI entry point
│   │   ├── README.md           # Comparison module docs
│   │   │
│   │   └── gui/                # 🖥️ Interactive GUI
│   │       ├── __init__.py
│   │       ├── app.py          # Main GUI application
│   │       ├── widgets.py      # Custom tkinter widgets
│   │       ├── plot_viewer.py  # Plot display logic
│   │       ├── interactive_plots.py  # Plotly 3D plots
│   │       └── README.md       # GUI documentation
│   │
│   ├── pretrained/             # Pre-trained agent storage
│   │   ├── simple.pkl
│   │   ├── count_based.pkl
│   │   ├── ucb.pkl
│   │   └── threshold.pkl
│   │
│   └── misc/                   # Miscellaneous utilities
│
├── results/                    # Generated visualizations
│   └── comparison/             # Comparison results
│       ├── *.png               # All generated plots
│       └── comparison_report.txt  # Detailed text report
│
├── run_comparison.py           # 🚀 CLI comparison launcher
├── view_comparison.py          # 🎨 GUI launcher
├── example_comparison.py       # Programmatic usage example
├── demo_adaptive_epsilon.py    # Legacy demo script
└── main_monte_carlo.py         # Monte Carlo experiments
```

---

## 📚 Key Terminology

### Reinforcement Learning Terms

- **Q-Learning**: A model-free RL algorithm that learns a Q-function Q(s,a) estimating expected reward
- **Q-Value / Q(s,a)**: Expected cumulative reward for taking action `a` in state `s`
- **Policy**: Strategy for choosing actions (π: S → A)
- **Epsilon (ε)**: Exploration rate - probability of taking a random action
- **Exploration vs Exploitation**: Balance between trying new actions (explore) vs using known good actions (exploit)
- **State**: Current situation (in Blackjack: player sum, dealer card, usable ace)
- **Action**: Decision to make (in Blackjack: Hit or Stick)
- **Reward**: Feedback from environment (+1 win, -1 loss, 0 draw)

### Project-Specific Terms

- **Adaptive Epsilon**: Exploration rate that changes based on state familiarity
- **State Counts**: How many times each state has been visited during training
- **Visit Count**: Number of times a specific state was encountered
- **Convergence**: When learning stabilizes (Q-values stop changing significantly)
- **Policy Heatmap**: 2D visualization showing best action for each state
- **Action Value Gap**: Difference between Q(Hit) and Q(Stick) showing decision confidence

### Blackjack Terms

- **Player Sum**: Total value of player's cards (goal: get close to 21)
- **Dealer Showing**: Dealer's visible card
- **Usable Ace**: An ace counting as 11 (vs 1) without busting
- **Hit**: Take another card
- **Stick**: Stop taking cards
- **Bust**: Go over 21 (automatic loss)

---

## 🛠️ Setup & Installation

### Prerequisites

- **Python 3.10+** (Python 3.11 recommended)
- **uv** package manager (recommended) or pip

### Step 1: Install UV Package Manager

UV is a fast Python package manager. Install it:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

Verify installation:
```bash
uv --version
```

### Step 2: Clone/Navigate to Project

```bash
cd "/path/to/Comp560FinalProject-Unoverse"
```

### Step 3: Install Dependencies

Using UV (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test import
python -c "from src.comparison import compare_agents_main; print('✅ Setup complete!')"
```

---

## 🚀 Quick Start

### Option 1: Run Comparison with Default Settings

```bash
python run_comparison.py
```

This will:
- Load or train 4 agents (250K episodes each)
- Evaluate each agent (10K episodes)
- Generate ~20-30 visualization plots
- Save results to `results/comparison/`
- Print comparison summary

**Time**: 5-30 minutes depending on whether agents need training

### Option 2: Launch Interactive GUI

```bash
python view_comparison.py
```

This will:
- Open a GUI window
- Load pre-trained agents (or train if needed)
- Allow browsing all visualizations interactively
- Support rotatable 3D plots in browser
- Enable exporting any visualization

**Time**: 10-60 seconds to launch (faster if agents pre-trained)

---

## 🎯 Core Functionalities

### 1. Command-Line Comparison (`run_comparison.py`)

**Basic Usage:**
```bash
python run_comparison.py
```

**With Custom Parameters:**
```bash
python run_comparison.py \
    --train-episodes 500000 \
    --eval-episodes 20000 \
    --output-dir results/my_comparison \
    --force-retrain \
    --seed 42
```

**Available Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train-episodes` | int | 250,000 | Training episodes per agent |
| `--eval-episodes` | int | 10,000 | Evaluation episodes per agent |
| `--pretrained-dir` | str | `src/pretrained` | Directory for saved agents |
| `--output-dir` | str | `results/comparison` | Output directory for plots |
| `--force-retrain` | flag | False | Force retraining even if agents exist |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--quiet` | flag | False | Suppress verbose output |

**Output:**
- `results/comparison/*.png` - All visualization plots
- `results/comparison/comparison_report.txt` - Detailed text report
- `src/pretrained/*.pkl` - Trained agent files (if trained)

### 2. Interactive GUI (`view_comparison.py`)

**Launch:**
```bash
python view_comparison.py
```

**Features:**
- **Agent Selection Panel**: Check/uncheck agents to compare
- **Visualization Browser**: Categorized list of all available plots
- **3D Plot Options**: Toggle between matplotlib (embedded) and plotly (browser)
- **Control Buttons**:
  - 🔄 Refresh Data - Reload agents
  - 💾 Export Plot - Save current visualization
  - 🗑️ Clear Cache - Free memory

**Navigation:**
1. Select agents using checkboxes (left sidebar)
2. Click any visualization from the list
3. For 3D plots: Check "Use Interactive Plotly" for browser rotation
4. Use matplotlib toolbar for zoom/pan on 2D plots
5. Export any plot with the Export button

### 3. Programmatic Usage

```python
from src.comparison import run_comparison, generate_all_visualizations
from src.env.env import make_blackjack_env

# Create environment
env = make_blackjack_env(natural=False, sab=True)

# Run comparison
results, agents_dict = run_comparison(
    env=env,
    n_train_episodes=250_000,
    n_eval_episodes=10_000,
    pretrained_dir="src/pretrained",
    force_retrain=False,
    verbose=True
)

# Generate all visualizations
generate_all_visualizations(
    results,
    agents_dict,
    save_dir="results/my_comparison"
)

# Access specific agent results
for agent_name, data in results.items():
    print(f"{agent_name}:")
    print(f"  Win Rate: {data['eval']['win_rate']:.4f}")
    print(f"  Avg Reward: {data['eval']['avg_reward']:.4f}")
```

See `example_comparison.py` for more examples.

---

## 🤖 Agents Explained

### 1. Simple (Fixed Epsilon)

**Strategy**: Standard ε-greedy with linear decay

**How it works:**
- Starts with ε = 1.0 (100% exploration)
- Linearly decays to ε = 0.01 over training
- Explores less as it learns more
- Same epsilon for all states

**Formula**: `ε(t) = max(ε_final, ε_initial - decay_rate × t)`

**Best for**: Baseline comparison, simple problems

### 2. Count-Based (k=10)

**Strategy**: State-dependent ε based on visit counts

**How it works:**
- Each state has its own ε based on how often it's been visited
- Rarely-visited states → High ε (explore more)
- Frequently-visited states → Low ε (exploit more)
- Automatically balances exploration

**Formula**: `ε(s) = k / (k + N(s))`
- k = exploration parameter (higher = more exploration)
- N(s) = visit count for state s

**Best for**: Large state spaces, uneven state distributions

### 3. UCB-Style (c=1.0)

**Strategy**: Upper Confidence Bound inspired exploration bonus

**How it works:**
- Base epsilon + exploration bonus
- Bonus decreases with √(visits)
- Slower decay than count-based
- Confidence-based exploration

**Formula**: `ε(s) = ε_base + c / √(N(s) + 1)`
- c = exploration bonus coefficient
- Higher c = more exploration

**Best for**: When you want sustained exploration

### 4. Threshold-Based

**Strategy**: Piecewise constant ε with discrete levels

**How it works:**
- Different ε for different visit count ranges
- Discrete jumps at thresholds
- Very high ε for new states (0.9)
- Very low ε for well-known states (0.05)

**Configuration:**
- Visits < 10: ε = 0.9 (high exploration)
- Visits 10-99: ε = 0.5 (medium)
- Visits 100-999: ε = 0.2 (low)
- Visits 1000+: ε = 0.05 (very low)

**Best for**: Clear exploration phases, interpretable behavior

---

## 📊 Visualizations

### Comparison Plots (Multi-Agent)

1. **Win Rate Comparison**
   - Bar chart: Win/Loss/Draw rates
   - Shows overall performance

2. **Average Reward Comparison**
   - Bar chart with error bars
   - Best metric for comparing agents

3. **Training Performance**
   - Line plot with 1K-episode window
   - Shows learning progression

4. **Exploration Statistics**
   - Bar charts: Unique states, avg visits
   - Shows exploration efficiency

5. **Learning Convergence**
   - Line plot with 10K-episode window
   - Shows when learning stabilizes
   - Marks convergence points

6. **Policy Differences**
   - Heatmap showing where agents disagree
   - Reveals strategic differences

### Agent-Specific Plots

#### 3D Visualizations

7. **Q-Values 3D (Hit)**
   - Surface plot: Dealer × Player Sum × Q-value
   - Shows learned action values for hitting
   - Two subplots: Usable ace vs No ace
   - **Rotatable** in browser (plotly mode)

8. **Q-Values 3D (Stick)**
   - Same as above for sticking action

9. **Epsilon Values 3D** (Adaptive agents only)
   - Surface plot showing exploration rates
   - Visualizes adaptive exploration strategy

#### 2D Heatmaps

10. **Policy Heatmap**
    - Color-coded: Best action per state
    - Green = Hit, Red = Stick
    - Shows learned strategy

11. **Action Value Gap**
    - Q(Hit) - Q(Stick) per state
    - Red = Prefer Hit, Blue = Prefer Stick
    - Shows decision confidence

12. **Visit Counts** (Adaptive agents only)
    - Log-scale heatmap of state visits
    - Shows which states were explored

---

## 🔧 Advanced Usage

### Training with Custom Hyperparameters

Edit `src/comparison/agent_loader.py` → `get_agent_configs()`:

```python
"count_based": {
    "params": {
        "learning_rate": 0.02,      # Change learning rate
        "k_param": 20.0,            # More aggressive exploration
        "final_epsilon": 0.001,     # Lower minimum epsilon
    }
}
```

### Adding a New Agent

1. Create agent class in `src/agents/MyAgent.py`:
```python
from .SimpleAgent import SimpleAgent

class MyAgent(SimpleAgent):
    def get_action(self, obs, valid_actions=None):
        # Your custom exploration strategy
        pass
```

2. Add to `agent_loader.py`:
```python
"my_agent": {
    "name": "My Custom Agent",
    "class": MyAgent,
    "params": {...}
}
```

3. Run comparison - it will automatically be included!

### Batch Experiments

```python
# Run multiple experiments with different seeds
for seed in [42, 123, 456]:
    results, agents = run_comparison(
        env=env,
        n_train_episodes=250_000,
        output_dir=f"results/seed_{seed}",
        seed=seed
    )
```

### Loading Pre-trained Agents

```python
from src.comparison.agent_loader import load_agent

agent = load_agent("src/pretrained/count_based.pkl")
# Use agent for evaluation or further training
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'src'"

**Cause**: Running from wrong directory

**Solution**:
```bash
cd "/path/to/Comp560FinalProject-Unoverse"
python run_comparison.py
```

### Issue: "ModuleNotFoundError: No module named 'gymnasium'"

**Cause**: Dependencies not installed

**Solution**:
```bash
uv sync
# Or
pip install -r requirements.txt
```

### Issue: GUI won't launch

**Cause**: Tkinter not available (usually installed with Python)

**Solution** (macOS):
```bash
brew install python-tk
```

**Solution** (Ubuntu/Debian):
```bash
sudo apt-get install python3-tk
```

### Issue: 3D plots don't open in browser

**Cause**: Browser not configured or popup blocker

**Solution**:
1. Check browser allows pop-ups for local files
2. Try unchecking "Use Interactive Plotly" in GUI (uses embedded matplotlib instead)
3. Manually open HTML files in temp directory

### Issue: Training takes too long

**Solution**: Use fewer episodes for testing:
```bash
python run_comparison.py --train-episodes 50000 --eval-episodes 1000
```

### Issue: "Training data not available" in Learning Convergence plot

**Cause**: Agents loaded from pretrained files don't have training history

**Solution**:
1. Use "Training Performance" plot instead (different window size)
2. Click "Refresh Data" in GUI and force retrain
3. Run with `--force-retrain` flag

### Issue: Out of memory

**Cause**: Too many cached plots in GUI

**Solution**: Click "Clear Cache" button in GUI

---

## 📖 Additional Resources

### Documentation Files

- `QUICKSTART.md` - 5-minute setup guide
- `src/comparison/README.md` - Comparison module documentation
- `src/comparison/gui/README.md` - GUI documentation
- `NEW_VISUALIZATIONS.md` - Details on new visualization types
- `GUI_IMPLEMENTATION_SUMMARY.md` - GUI implementation details
- `MONTE_CARLO_README.md` - Monte Carlo agent documentation

### Key Papers & References

- **Sutton & Barto** - *Reinforcement Learning: An Introduction*
- **Gymnasium Blackjack** - https://gymnasium.farama.org/environments/toy_text/blackjack/
- **Q-Learning** - Watkins & Dayan (1992)

### Project Info

- **Course**: COMP 560 (Fall 2025)
- **Institution**: UNC
- **Environment**: Blackjack-v1 (Gymnasium)
- **Algorithm**: Q-Learning with various exploration strategies
- **Implementation**: Python 3.10+, NumPy, Matplotlib, Seaborn, Plotly, Tkinter

---

## 🎓 Learning Outcomes

After using this project, you will understand:

✅ How Q-Learning works in practice  
✅ The exploration-exploitation tradeoff  
✅ Different exploration strategies and their tradeoffs  
✅ How to evaluate and compare RL agents  
✅ Visualization techniques for RL analysis  
✅ State-dependent vs global exploration rates  
✅ The impact of hyperparameters on learning  

---

## 🤝 For Your Team

### Quick Team Onboarding

1. **Install UV**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **Clone/Navigate**: `cd Comp560FinalProject-Unoverse`
3. **Install Dependencies**: `uv sync`
4. **Run Comparison**: `python run_comparison.py`
5. **Launch GUI**: `python view_comparison.py`
6. **Read the Output**: Check `results/comparison/` folder

### What Each Team Member Should Know

**For Analysis/Interpretation:**
- Focus on `results/comparison/` plots
- Read `comparison_report.txt` for numerical results
- Use GUI for interactive exploration

**For Code Understanding:**
- Start with `src/agents/` to understand each agent
- Review `src/comparison/visualizations.py` for plot generation
- Check `src/comparison/runner.py` for comparison logic

**For Extension:**
- Edit `src/comparison/agent_loader.py` for new agents
- Modify hyperparameters in `get_agent_configs()`
- Add visualizations in `visualizations.py`

---

## 📝 Notes

- **Training Time**: 250K episodes × 4 agents ≈ 10-30 minutes (first run)
- **Loading Time**: < 10 seconds if pre-trained agents exist
- **Output Size**: ~20-30 PNG files + 1 text report
- **Memory**: ~500MB-1GB during training, ~100MB for GUI
- **Best Practices**: Always use same seed for reproducibility
- **Tip**: Start with GUI for quick exploration, use CLI for batch experiments

---

## 📜 License

Part of UNC COMP 560 Final Project - Fall 2025

---

**Happy Learning! 🎓🎰**

*For questions about specific components, see the README files in respective directories.*

*Quick setup? See QUICKSTART.md for a 5-minute guide!*
