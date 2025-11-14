"""Environment wrapper for Blackjack using Gymnasium."""

import gymnasium as gym


def make_blackjack_env(
    natural: bool = False,
    sab: bool = True,
    render_mode: str | None = None
) -> gym.Env:
    """Create a Blackjack environment.
    
    Blackjack is a card game where the goal is to beat the dealer by obtaining 
    cards that sum to closer to 21 (without going over 21) than the dealer's cards.
    
    Observation Space: Tuple(Discrete(32), Discrete(11), Discrete(2))
        - Player's current sum (0-31)
        - Dealer's showing card value (1-10, where 1 is ace)
        - Whether player holds a usable ace (0 or 1)
    
    Action Space: Discrete(2)
        - 0: Stick (stop taking cards)
        - 1: Hit (take another card)
    
    Rewards:
        - +1 for winning
        - -1 for losing
        - 0 for draw
        - +1.5 for winning with natural blackjack (if natural=True)
    
    Args:
        natural: Whether to give additional reward for natural blackjack
        sab: Whether to follow exact Sutton & Barto book rules
        render_mode: How to render ("human", "rgb_array", or None)
        
    Returns:
        Blackjack environment
        
    Reference:
        https://gymnasium.farama.org/environments/toy_text/blackjack/
    """
    return gym.make(
        "Blackjack-v1",
        natural=natural,
        sab=sab,
        render_mode=render_mode
    )
