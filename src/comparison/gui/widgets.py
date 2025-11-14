"""Custom tkinter widgets for the visualization GUI."""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Callable, Optional


class AgentSelectorPanel(ttk.LabelFrame):
    """Widget for selecting which agents to display."""
    
    def __init__(
        self,
        parent: tk.Widget,
        agent_names: List[str],
        on_change: Optional[Callable] = None
    ) -> None:
        """Initialize the agent selector panel.
        
        Args:
            parent: Parent tkinter widget
            agent_names: List of agent names to display
            on_change: Callback function when selection changes
        """
        super().__init__(parent, text="Agent Selection", padding=10)
        
        self.agent_names = agent_names
        self.on_change = on_change
        self.check_vars: Dict[str, tk.BooleanVar] = {}
        
        # Create checkboxes for each agent
        for agent_name in agent_names:
            var = tk.BooleanVar(value=True)
            self.check_vars[agent_name] = var
            
            cb = ttk.Checkbutton(
                self,
                text=agent_name,
                variable=var,
                command=self._on_checkbox_change
            )
            cb.pack(anchor='w', pady=2)
        
        # Add select all / deselect all buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="Select All",
            command=self.select_all
        ).pack(side='left', padx=2)
        
        ttk.Button(
            button_frame,
            text="Deselect All",
            command=self.deselect_all
        ).pack(side='left', padx=2)
    
    def _on_checkbox_change(self) -> None:
        """Handle checkbox state change."""
        if self.on_change:
            self.on_change()
    
    def select_all(self) -> None:
        """Select all agent checkboxes."""
        for var in self.check_vars.values():
            var.set(True)
        self._on_checkbox_change()
    
    def deselect_all(self) -> None:
        """Deselect all agent checkboxes."""
        for var in self.check_vars.values():
            var.set(False)
        self._on_checkbox_change()
    
    def get_selected_agents(self) -> List[str]:
        """Get list of currently selected agent names.
        
        Returns:
            List of selected agent names
        """
        return [
            name for name, var in self.check_vars.items()
            if var.get()
        ]


class VisualizationSelectorPanel(ttk.LabelFrame):
    """Widget for selecting which visualization to display."""
    
    def __init__(
        self,
        parent: tk.Widget,
        visualizations: Dict[str, List[str]],
        on_change: Optional[Callable] = None
    ) -> None:
        """Initialize the visualization selector panel.
        
        Args:
            parent: Parent tkinter widget
            visualizations: Dictionary mapping categories to visualization names
            on_change: Callback function when selection changes
        """
        super().__init__(parent, text="Visualization Type", padding=10)
        
        self.visualizations = visualizations
        self.on_change = on_change
        
        # Create listbox with scrollbar
        list_frame = ttk.Frame(self)
        list_frame.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode='single',
            height=20
        )
        self.listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        # Populate listbox with visualizations
        self.viz_list: List[tuple] = []
        for category, viz_names in visualizations.items():
            # Add category header
            self.listbox.insert(tk.END, f"─── {category} ───")
            self.listbox.itemconfig(tk.END, {'bg': '#e0e0e0'})
            self.viz_list.append((None, None))  # Placeholder for header
            
            # Add visualizations in this category
            for viz_name in viz_names:
                self.listbox.insert(tk.END, f"  {viz_name}")
                self.viz_list.append((category, viz_name))
        
        # Bind selection event
        self.listbox.bind('<<ListboxSelect>>', self._on_selection)
        
        # Select first visualization by default
        self.listbox.selection_set(1)  # Skip first header
    
    def _on_selection(self, event: tk.Event) -> None:
        """Handle listbox selection change."""
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            category, viz_name = self.viz_list[idx]
            
            # Don't trigger callback for headers
            if category is not None and self.on_change:
                self.on_change()
    
    def get_selected_visualization(self) -> Optional[tuple]:
        """Get currently selected visualization.
        
        Returns:
            Tuple of (category, visualization_name) or None if header selected
        """
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            return self.viz_list[idx]
        return None


class ControlPanel(ttk.Frame):
    """Widget for control buttons (refresh, export, etc.)."""
    
    def __init__(
        self,
        parent: tk.Widget,
        on_refresh: Optional[Callable] = None,
        on_export: Optional[Callable] = None,
        on_clear_cache: Optional[Callable] = None
    ) -> None:
        """Initialize the control panel.
        
        Args:
            parent: Parent tkinter widget
            on_refresh: Callback for refresh button
            on_export: Callback for export button
            on_clear_cache: Callback for clear cache button
        """
        super().__init__(parent, padding=10)
        
        # Refresh button
        if on_refresh:
            ttk.Button(
                self,
                text="🔄 Refresh Data",
                command=on_refresh
            ).pack(fill='x', pady=2)
        
        # Export button
        if on_export:
            ttk.Button(
                self,
                text="💾 Export Plot",
                command=on_export
            ).pack(fill='x', pady=2)
        
        # Clear cache button
        if on_clear_cache:
            ttk.Button(
                self,
                text="🗑️ Clear Cache",
                command=on_clear_cache
            ).pack(fill='x', pady=2)
        
        # Add separator
        ttk.Separator(self, orient='horizontal').pack(fill='x', pady=10)
        
        # Info label
        self.info_label = ttk.Label(
            self,
            text="Select a visualization to view",
            wraplength=200,
            justify='left'
        )
        self.info_label.pack(fill='x', pady=5)
    
    def update_info(self, text: str) -> None:
        """Update the information label text.
        
        Args:
            text: Text to display
        """
        self.info_label.config(text=text)


class StatusBar(ttk.Frame):
    """Status bar widget for displaying current state."""
    
    def __init__(self, parent: tk.Widget) -> None:
        """Initialize the status bar.
        
        Args:
            parent: Parent tkinter widget
        """
        super().__init__(parent, relief='sunken', padding=2)
        
        self.status_label = ttk.Label(self, text="Ready")
        self.status_label.pack(side='left')
        
        # Add separator
        ttk.Separator(self, orient='vertical').pack(side='left', fill='y', padx=5)
        
        # Agent info label
        self.agent_label = ttk.Label(self, text="")
        self.agent_label.pack(side='left')
    
    def set_status(self, text: str) -> None:
        """Set the status text.
        
        Args:
            text: Status text to display
        """
        self.status_label.config(text=text)
    
    def set_agent_info(self, agents: List[str]) -> None:
        """Set the agent information text.
        
        Args:
            agents: List of selected agent names
        """
        if agents:
            text = f"Agents: {', '.join(agents)}"
        else:
            text = "No agents selected"
        self.agent_label.config(text=text)

