# Interactive Visualization GUI

An interactive GUI application for browsing and exploring RL agent comparison visualizations.

## Features

### 🎮 Interactive Controls
- **Agent Selection**: Toggle agents on/off with checkboxes
- **Visualization Browser**: Browse all available visualizations in categorized lists
- **3D Plot Rotation**: Interactive 3D plots open in browser with full rotation controls
- **2D Plot Interaction**: Zoom, pan, and explore 2D plots with matplotlib toolbar

### 📊 Visualization Categories

1. **Comparison Plots**
   - Win Rate Comparison
   - Average Reward Comparison
   - Training Performance
   - Exploration Statistics
   - Learning Convergence
   - Policy Differences

2. **3D Q-Values** (per agent)
   - Hit action Q-values
   - Stick action Q-values
   - Fully rotatable 3D surfaces

3. **3D Epsilon Values** (adaptive agents only)
   - Exploration rate across state space
   - Interactive rotation

4. **2D Heatmaps**
   - Policy heatmaps
   - Action value gap heatmaps
   - Visit count heatmaps

### 🔧 Utility Features
- **Refresh Data**: Reload agents and results
- **Export Plot**: Save current visualization to PNG/PDF
- **Clear Cache**: Free memory by clearing cached plots
- **Status Bar**: Shows current selection and loading status

## Quick Start

### Launch the GUI

From project root:

```bash
python view_comparison.py
```

Or using the module directly:

```bash
python -m src.comparison.gui.app
```

Or programmatically:

```python
from src.comparison.gui import launch_gui

launch_gui()
```

## User Interface

```
┌──────────────────────────────────────────────────────────┐
│  RL Agent Comparison Visualizer                          │
├───────────────┬──────────────────────────────────────────┤
│               │                                          │
│ Agent         │                                          │
│ Selection     │                                          │
│ ☑ Simple      │          Main Plot Area                 │
│ ☑ Count-Based │     (Matplotlib or Plotly Display)      │
│ ☑ UCB         │                                          │
│ ☑ Threshold   │                                          │
│               │                                          │
│ Visualization │                                          │
│ Type          │                                          │
│ - Comparison  │                                          │
│ - 3D Q-Values │                                          │
│ - 3D Epsilon  │                                          │
│ - 2D Heatmaps │                                          │
│               │                                          │
│ Options       │                                          │
│ ☑ Interactive │                                          │
│   Plotly      │                                          │
│               │                                          │
│ [Refresh]     │                                          │
│ [Export]      │                                          │
│ [Clear Cache] │                                          │
├───────────────┴──────────────────────────────────────────┤
│ Status: Ready | Agents: Simple, Count-Based, UCB        │
└──────────────────────────────────────────────────────────┘
```

## Workflow

1. **Launch**: Start the GUI with `python view_comparison.py`
2. **Load**: Agents and data load automatically (or prompt to train if missing)
3. **Select Agents**: Use checkboxes to select which agents to compare
4. **Choose Visualization**: Click on any visualization from the list
5. **Interact**: 
   - For 2D plots: Use toolbar to zoom, pan, save
   - For 3D plots: Opens in browser with full rotation controls
6. **Export**: Click "Export Plot" to save current visualization
7. **Switch**: Click any other visualization to switch instantly

## 3D Plot Interaction

### Matplotlib Mode (unchecked "Use Interactive Plotly")
- Click and drag to rotate
- Scroll to zoom
- Right-click and drag to pan
- Embedded directly in GUI window

### Plotly Mode (checked "Use Interactive Plotly")
- Opens in default web browser
- Click and drag to rotate smoothly
- Scroll to zoom
- Hover to see values
- Double-click to reset view
- Better performance for complex surfaces

## Data Loading

### Automatic Loading
On startup, the GUI attempts to:
1. Load pre-trained agents from `src/pretrained/`
2. Evaluate agents if needed
3. Build comparison results

### If No Agents Found
If no pre-trained agents exist, they will be trained automatically:
- Training: 250,000 episodes per agent
- Evaluation: 10,000 episodes per agent
- Progress displayed in console

### Refresh Data
Click "Refresh Data" to:
- Reload agents from disk
- Re-run evaluation
- Clear all caches

## Performance

### Caching
- Plots are cached after first generation
- Switching between cached plots is instant
- Cache persists until cleared or app closed

### Memory Management
- Click "Clear Cache" to free memory
- Cache automatically cleared on refresh
- Each agent's plots cached independently

### Load Times
- Initial load: 5-30 seconds (depending on data)
- First plot generation: 1-3 seconds
- Cached plot display: < 0.1 seconds
- 3D Plotly plots: 2-5 seconds to open browser

## Keyboard Shortcuts

Currently, the GUI uses mouse-based interaction. Future versions may add:
- `Ctrl+R`: Refresh data
- `Ctrl+E`: Export plot
- `Ctrl+C`: Clear cache
- Arrow keys: Navigate visualizations

## Troubleshooting

### Issue: GUI won't launch
**Solution**: Check dependencies are installed:
```bash
uv add plotly kaleido
```

### Issue: 3D plots don't open
**Solution**: 
1. Check default browser is set
2. Try unchecking "Use Interactive Plotly"
3. Check browser allows local file access

### Issue: Plots look distorted
**Solution**:
1. Try resizing the window
2. Clear cache and regenerate
3. Export and view externally

### Issue: "No agents found"
**Solution**: 
- Let GUI train agents automatically, or
- Run `python run_comparison.py` first to generate pretrained agents

### Issue: GUI freezes
**Solution**:
- Generating plots can take a few seconds
- Check status bar for progress
- For very long freezes, restart GUI

## Technical Details

### Architecture
- **Framework**: tkinter (Python standard library)
- **Plotting**: matplotlib (2D, embedded 3D) + plotly (interactive 3D)
- **Backend**: matplotlib.backends.backend_tkagg
- **3D Engine**: plotly.graph_objects for browser-based interaction

### File Structure
```
src/comparison/gui/
├── __init__.py          # Module exports
├── app.py               # Main application logic
├── widgets.py           # Custom tkinter widgets
├── plot_viewer.py       # Plot display management
├── interactive_plots.py # Plotly 3D plot generators
└── README.md            # This file
```

### Dependencies
- `tkinter`: GUI framework (built-in)
- `matplotlib`: 2D and embedded 3D plots
- `plotly`: Interactive 3D plots
- `kaleido`: Plotly static image export
- `numpy`: Numerical computations
- `seaborn`: Statistical visualization styling

## Extending the GUI

### Adding New Visualizations

1. **Add to visualization dictionary** in `app.py`:
```python
self.visualizations = {
    "My Category": ["New Viz 1", "New Viz 2"]
}
```

2. **Implement plot function** in `plot_viewer.py`:
```python
def _plot_my_viz(self, fig: Figure, data: Any) -> None:
    # Your plotting code
    pass
```

3. **Handle in update_visualization** in `app.py`:
```python
elif category == "My Category":
    self.plot_viewer.create_my_plot(...)
```

### Customizing Appearance

Edit widget styling in `widgets.py`:
- Colors: Modify ttk styles
- Fonts: Change font parameters
- Layout: Adjust pack/grid parameters

## Known Limitations

1. **3D Plotly plots** open in browser (cannot embed in tkinter)
2. **Large datasets** may slow down initial rendering
3. **Multiple 3D plots** open multiple browser tabs
4. **Export** only works for currently displayed matplotlib plots

## Future Enhancements

Potential improvements:
- [ ] Keyboard shortcuts
- [ ] Side-by-side comparison view
- [ ] Plot history/breadcrumbs
- [ ] Custom plot configurations
- [ ] Theme support (dark mode)
- [ ] Embedded web view for Plotly (using tkinterweb)
- [ ] Animation of training progress
- [ ] Real-time training visualization

## License

Part of the Comp560FinalProject-Unoverse.

