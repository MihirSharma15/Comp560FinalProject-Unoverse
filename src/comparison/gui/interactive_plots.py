"""Interactive 3D plots using Plotly for better rotation and interaction."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...agents.BaseAgent import BaseAgent
from ...agents.SimpleAgent import SimpleAgent
from ...agents.AdaptiveEpsilonAgent import AdaptiveEpsilonAgent


def create_interactive_q_values_3d(
    agent: BaseAgent,
    agent_name: str,
    action: int = 1
) -> go.Figure:
    """Create interactive 3D surface plot of Q-values using Plotly.
    
    Args:
        agent: Trained agent to visualize
        agent_name: Name of the agent for title
        action: Which action to visualize (0=stick, 1=hit)
        
    Returns:
        Plotly Figure object with interactive 3D plots
    """
    # Define state space ranges
    player_sums = np.arange(4, 22)  # 4-21
    dealer_cards = np.arange(1, 11)  # 1-10
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('No Usable Ace', 'Usable Ace'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        horizontal_spacing=0.1
    )
    
    for idx, usable_ace in enumerate([False, True]):
        # Compute Q-values for each state
        Z = np.zeros((len(player_sums), len(dealer_cards)))
        
        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                state = (player_sum, dealer_card, usable_ace)
                if isinstance(agent, SimpleAgent):
                    Z[i, j] = agent.get_q_value(state, action)
        
        # Create surface plot
        surface = go.Surface(
            x=dealer_cards,
            y=player_sums,
            z=Z,
            colorscale='RdBu',
            showscale=(idx == 1),  # Show colorbar only for second plot
            name=f'{"Usable" if usable_ace else "No"} Ace'
        )
        
        fig.add_trace(surface, row=1, col=idx+1)
        
        # Update axes
        fig.update_scenes(
            xaxis_title='Dealer Showing',
            yaxis_title='Player Sum',
            zaxis_title=f'Q-value ({"Hit" if action == 1 else "Stick"})',
            row=1, col=idx+1
        )
    
    # Update layout
    action_name = "Hit" if action == 1 else "Stick"
    fig.update_layout(
        title=f'{agent_name}: Q-Values for {action_name} Action<br><sub>Click and drag to rotate</sub>',
        height=600,
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig


def create_interactive_epsilon_3d(
    agent: AdaptiveEpsilonAgent,
    agent_name: str
) -> go.Figure:
    """Create interactive 3D surface plot of epsilon values using Plotly.
    
    Args:
        agent: Adaptive epsilon agent to visualize
        agent_name: Name of the agent for title
        
    Returns:
        Plotly Figure object with interactive 3D plots
    """
    if not isinstance(agent, AdaptiveEpsilonAgent):
        raise TypeError("Agent must be an AdaptiveEpsilonAgent")
    
    # Define state space ranges
    player_sums = np.arange(4, 22)
    dealer_cards = np.arange(1, 11)
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('No Usable Ace', 'Usable Ace'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        horizontal_spacing=0.1
    )
    
    for idx, usable_ace in enumerate([False, True]):
        # Compute epsilon values for each state
        Z = np.zeros((len(player_sums), len(dealer_cards)))
        
        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                state = (player_sum, dealer_card, usable_ace)
                Z[i, j] = agent.get_epsilon_for_state(state)
        
        # Create surface plot
        surface = go.Surface(
            x=dealer_cards,
            y=player_sums,
            z=Z,
            colorscale='Viridis',
            showscale=(idx == 1),
            name=f'{"Usable" if usable_ace else "No"} Ace'
        )
        
        fig.add_trace(surface, row=1, col=idx+1)
        
        # Update axes
        fig.update_scenes(
            xaxis_title='Dealer Showing',
            yaxis_title='Player Sum',
            zaxis_title='Epsilon Value',
            row=1, col=idx+1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{agent_name}: Epsilon Values Across State Space<br><sub>Click and drag to rotate</sub>',
        height=600,
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig


def create_interactive_policy_3d(
    agent: BaseAgent,
    agent_name: str
) -> go.Figure:
    """Create interactive 3D visualization of learned policy.
    
    Args:
        agent: Trained agent to visualize
        agent_name: Name of the agent for title
        
    Returns:
        Plotly Figure object showing policy as 3D surface
    """
    # Define state space
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('No Usable Ace', 'Usable Ace'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        horizontal_spacing=0.1
    )
    
    for idx, usable_ace in enumerate([False, True]):
        # Create policy matrix (0=stick, 1=hit)
        Z = np.zeros((len(player_sums), len(dealer_cards)))
        
        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                state = (player_sum, dealer_card, usable_ace)
                if isinstance(agent, SimpleAgent):
                    q_values = agent.get_policy(state)
                    Z[i, j] = np.argmax(q_values)
        
        # Create surface plot
        surface = go.Surface(
            x=dealer_cards,
            y=player_sums,
            z=Z,
            colorscale=[[0, 'red'], [1, 'green']],
            showscale=(idx == 1),
            cmin=0,
            cmax=1,
            name=f'{"Usable" if usable_ace else "No"} Ace',
            colorbar=dict(
                tickvals=[0.25, 0.75],
                ticktext=['Stick', 'Hit']
            ) if idx == 1 else None
        )
        
        fig.add_trace(surface, row=1, col=idx+1)
        
        # Update axes
        fig.update_scenes(
            xaxis_title='Dealer Showing',
            yaxis_title='Player Sum',
            zaxis_title='Action',
            row=1, col=idx+1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{agent_name}: Learned Policy<br><sub>Click and drag to rotate</sub>',
        height=600,
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig

