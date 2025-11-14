# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install UV (Fast Python Package Manager)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Step 2: Navigate to Project

```bash
cd "/Users/manu/Library/Mobile Documents/com~apple~CloudDocs/UNC Courses/Fall 2025/comp560/Comp560FinalProject-Unoverse"
```

## Step 3: Install Dependencies

```bash
uv sync
```

**Alternative** (if you prefer pip):
```bash
pip install -r requirements.txt
```

## Step 4: Run It!

### Option A: Interactive GUI (Recommended for First-Time)

```bash
python view_comparison.py
```

- Opens a GUI window
- Click through visualizations
- 3D plots open in browser (rotatable!)
- Takes 10-60 seconds to load

### Option B: Command-Line Comparison

```bash
python run_comparison.py
```

- Generates all plots automatically
- Saves to `results/comparison/`
- Takes 5-30 minutes (depending on if agents need training)
- Prints summary at the end

## Step 5: View Results

**GUI**: Visualizations appear in the window or browser

**CLI**: Check these folders:
- `results/comparison/` - All generated plots (PNG files)
- `results/comparison/comparison_report.txt` - Detailed text report

## 🎯 What You'll See

The project compares 4 different RL agents learning to play Blackjack:

1. **Simple** - Basic fixed exploration
2. **Count-Based** - Smart adaptive exploration
3. **UCB-Style** - Confidence-based exploration  
4. **Threshold** - Step-wise exploration

You'll get visualizations showing:
- Who wins more often
- How they learn over time
- What strategies they learned
- How much they explore
- Where they make different decisions

## 🐛 Troubleshooting

### "No module named 'src'"
Make sure you're in the project directory:
```bash
pwd  # Should show the project path
```

### "ModuleNotFoundError: gymnasium"
Install dependencies:
```bash
uv sync
```

### GUI won't open
Install tkinter (usually already installed):
```bash
# macOS
brew install python-tk

# Ubuntu/Debian  
sudo apt-get install python3-tk
```

### Too slow?
Use fewer training episodes:
```bash
python run_comparison.py --train-episodes 50000 --eval-episodes 1000
```

## 📖 Need More Info?

See the full **README.md** for:
- Detailed explanations
- Advanced usage
- All available options
- Terminology guide
- Project structure

## ✅ Success Checklist

- [ ] UV installed (`uv --version` works)
- [ ] Dependencies installed (`uv sync` completed)
- [ ] GUI launches (`python view_comparison.py`)
- [ ] Can see visualizations in `results/comparison/`

**You're all set! 🎉**

