"""Launcher script for the interactive visualization GUI."""

from src.comparison.gui import launch_gui

if __name__ == "__main__":
    print("="*80)
    print(" " * 20 + "RL Agent Comparison Visualizer")
    print("="*80)
    print("\nLaunching GUI...")
    print("\nFeatures:")
    print("  • Interactive 3D plot rotation (opens in browser)")
    print("  • Compare multiple agents side-by-side")
    print("  • Switch between visualizations with one click")
    print("  • Export plots to PNG/PDF")
    print("\nLoading agents and data...\n")
    
    launch_gui()

