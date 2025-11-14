"""Abstract base class for all agents."""

from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym


class BaseAgent(ABC):
    """Abstract base class for all agents.
    
    All agents should inherit from this class and implement the required methods.
    """
    
    def __init__(self, env: gym.Env) -> None:
        """Initialize the agent.
        
        Args:
            env: The training environment
        """
        self.env = env
    
    @abstractmethod
    def get_action(self, obs: np.ndarray | tuple, valid_actions: np.ndarray | None = None) -> int:
        """Select an action given the current observation.
        
        Args:
            obs: Current observation from the environment
            valid_actions: Optional array of valid action indices
            
        Returns:
            action: Integer representing the action to take
        """
        pass
    
    @abstractmethod
    def update(
        self,
        obs: np.ndarray | tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray | tuple,
        valid_next_actions: np.ndarray | None = None,
    ) -> None:
        """Update agent's internal state based on experience.
        
        Args:
            obs: Previous observation
            action: Action taken
            reward: Reward received
            terminated: Whether the episode ended
            next_obs: New observation after taking action
            valid_next_actions: Optional array of valid actions in next state
        """
        pass
    
    @abstractmethod
    def decay_epsilon(self) -> None:
        """Decay exploration rate (if applicable).
        
        Should be called after each episode.
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state at the start of an episode (optional).
        
        Override this if the agent needs to reset internal state.
        """
        pass