"""Simple Q-Learning agent for reinforcement learning."""

from collections import defaultdict
from typing import List
import numpy as np
import gymnasium as gym
from .BaseAgent import BaseAgent


class SimpleAgent(BaseAgent):
    """Q-Learning agent using epsilon-greedy exploration.
    
    This agent learns to play by maintaining a Q-table that maps
    (state, action) pairs to expected rewards.
    """
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ) -> None:
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        super().__init__(env)

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error: List[float] = []

    def _obs_to_state(self, obs: np.ndarray | tuple) -> tuple:
        """Convert observation to hashable tuple for Q-table lookup.
        
        Args:
            obs: Observation from environment (numpy array or tuple)
            
        Returns:
            Hashable tuple representation of the state
        """
        # Handle tuple observations (e.g., Blackjack)
        if isinstance(obs, tuple):
            return obs
        # Handle array observations (e.g., board games)
        return tuple(obs.flatten())

    def get_action(self, obs: np.ndarray | tuple, valid_actions: np.ndarray | None = None) -> int:
        """Choose an action using epsilon-greedy strategy.
        
        Args:
            obs: Current observation from the environment
            valid_actions: Optional array of valid action indices
            
        Returns:
            action: Integer representing the chosen action
        """
        state = self._obs_to_state(obs)
        
        # Get valid actions if not provided
        if valid_actions is None:
            if hasattr(self.env, 'get_valid_actions'):
                valid_actions = self.env.get_valid_actions()
            else:
                valid_actions = np.arange(self.env.action_space.n)
        
        # If no valid actions available, return random action
        if len(valid_actions) == 0:
            return self.env.action_space.sample()
        
        # With probability epsilon: explore (random valid action)
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        # With probability (1-epsilon): exploit (best known valid action)
        else:
            q_values_for_state = self.q_values[state]
            # Mask invalid actions with very negative value
            masked_q_values = np.full(len(q_values_for_state), -np.inf)
            masked_q_values[valid_actions] = q_values_for_state[valid_actions]
            return int(np.argmax(masked_q_values))

    def update(
        self,
        obs: np.ndarray | tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray | tuple,
        valid_next_actions: np.ndarray | None = None,
    ) -> None:
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        
        Args:
            obs: Previous observation
            action: Action taken
            reward: Reward received
            terminated: Whether the episode ended
            next_obs: New observation after taking action
            valid_next_actions: Optional array of valid actions in next state
        """
        state = self._obs_to_state(obs)
        next_state = self._obs_to_state(next_obs)
        
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        if not terminated:
            # Only consider valid actions when computing max Q-value
            # This prevents overestimation from invalid actions
            if valid_next_actions is not None and len(valid_next_actions) > 0:
                future_q_value = np.max(self.q_values[next_state][valid_next_actions])
            else:
                # Fall back to max over all actions if valid_next_actions not provided
                future_q_value = np.max(self.q_values[next_state])
        else:
            future_q_value = 0.0

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[state][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[state][action] = (
            self.q_values[state][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self) -> None:
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def get_q_value(self, obs: np.ndarray | tuple, action: int) -> float:
        """Get the Q-value for a specific state-action pair.
        
        Args:
            obs: Observation from environment
            action: Action index
            
        Returns:
            Q-value for the state-action pair
        """
        state = self._obs_to_state(obs)
        return self.q_values[state][action]
    
    def get_policy(self, obs: np.ndarray | tuple) -> np.ndarray:
        """Get the current policy (Q-values) for a given state.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Array of Q-values for all actions in this state
        """
        state = self._obs_to_state(obs)
        return self.q_values[state].copy()
    
    def __getstate__(self) -> dict:
        """Prepare agent for pickling by converting defaultdict to regular dict.
        
        Returns:
            Dictionary containing agent state without lambda functions
        """
        state = self.__dict__.copy()
        # Convert defaultdict to regular dict (removes unpicklable lambda)
        state['q_values'] = dict(state['q_values'])
        return state
    
    def __setstate__(self, state: dict) -> None:
        """Restore agent from pickle by recreating defaultdict.
        
        Args:
            state: Dictionary containing agent state
        """
        self.__dict__.update(state)
        # Recreate defaultdict with lambda
        q_dict = state['q_values']
        action_size = len(next(iter(q_dict.values()))) if q_dict else self.env.action_space.n
        self.q_values = defaultdict(lambda: np.zeros(action_size))
        # Restore all Q-values
        self.q_values.update(q_dict)