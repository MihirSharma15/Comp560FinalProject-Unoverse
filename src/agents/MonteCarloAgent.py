"""Monte Carlo Control agent for reinforcement learning."""

from collections import defaultdict
from typing import List, Tuple
import numpy as np
import gymnasium as gym
from .BaseAgent import BaseAgent


class MonteCarloAgent(BaseAgent):
    """Monte Carlo Control agent using epsilon-greedy exploration.
    
    This agent learns action-values from complete episodes using Monte Carlo methods.
    Unlike Q-learning, it waits until episode completion to update Q-values based on
    actual returns rather than bootstrapping from estimated values.
    """
    
    def __init__(
        self,
        env: gym.Env,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 1.0,
        use_first_visit: bool = True,
    ) -> None:
        """Initialize a Monte Carlo Control agent.

        Args:
            env: The training environment
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
            use_first_visit: If True, use first-visit MC; else every-visit MC
        """
        super().__init__(env)

        # Q-table: maps (state, action) to expected return
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # Returns tracker: stores all returns for each (state, action) pair
        # Used to compute average return (incremental mean)
        self.returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns_count = defaultdict(lambda: np.zeros(env.action_space.n))

        self.discount_factor = discount_factor
        self.use_first_visit = use_first_visit

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Episode buffer: stores (state, action, reward) tuples for current episode
        self.episode: List[Tuple[tuple, int, float]] = []
        
        # Track learning progress
        self.episode_returns: List[float] = []

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
        """Store experience tuple for current episode.

        In Monte Carlo methods, we don't update Q-values immediately.
        Instead, we store the experience and update at episode end.
        
        Args:
            obs: Previous observation
            action: Action taken
            reward: Reward received
            terminated: Whether the episode ended
            next_obs: New observation after taking action
            valid_next_actions: Optional array of valid actions in next state (unused in MC)
        """
        state = self._obs_to_state(obs)
        
        # Store the (state, action, reward) tuple
        self.episode.append((state, action, reward))
        
        # If episode terminated, update Q-values based on returns
        if terminated:
            self._update_from_episode()

    def _update_from_episode(self) -> None:
        """Update Q-values based on the complete episode.
        
        Uses Monte Carlo prediction to calculate returns for each state-action pair
        and updates Q-values as the average of observed returns.
        """
        # Track which (state, action) pairs we've seen in this episode (for first-visit)
        visited_state_actions = set()
        
        # Calculate returns for each step in the episode (backward from end)
        G = 0.0  # Return (cumulative discounted reward)
        
        # Iterate through episode in reverse
        for t in range(len(self.episode) - 1, -1, -1):
            state, action, reward = self.episode[t]
            
            # Update return (accumulate discounted rewards going backward)
            G = reward + self.discount_factor * G
            
            state_action = (state, action)
            
            # First-visit MC: only update on first occurrence of (state, action)
            if self.use_first_visit:
                if state_action not in visited_state_actions:
                    visited_state_actions.add(state_action)
                    self._update_q_value(state, action, G)
            # Every-visit MC: update on every occurrence
            else:
                self._update_q_value(state, action, G)
        
        # Track total return for the episode (for monitoring)
        if len(self.episode) > 0:
            # Total return is the return from the first state
            _, _, first_reward = self.episode[0]
            total_return = sum(
                reward * (self.discount_factor ** i) 
                for i, (_, _, reward) in enumerate(self.episode)
            )
            self.episode_returns.append(total_return)
        
        # Clear episode buffer for next episode
        self.episode = []

    def _update_q_value(self, state: tuple, action: int, G: float) -> None:
        """Update Q-value for a state-action pair using incremental mean.
        
        Args:
            state: The state (as tuple)
            action: The action taken
            G: The return observed
        """
        # Incremental update: Q(s,a) = average of all returns observed for (s,a)
        self.returns_sum[state][action] += G
        self.returns_count[state][action] += 1
        
        # Update Q-value as the mean of all observed returns
        self.q_values[state][action] = (
            self.returns_sum[state][action] / self.returns_count[state][action]
        )

    def decay_epsilon(self) -> None:
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def reset(self) -> None:
        """Reset episode buffer at the start of a new episode."""
        self.episode = []
    
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
    
    def get_visit_counts(self, obs: np.ndarray | tuple) -> np.ndarray:
        """Get visit counts for all actions in a given state.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Array of visit counts for all actions in this state
        """
        state = self._obs_to_state(obs)
        return self.returns_count[state].copy()
    
    def __getstate__(self) -> dict:
        """Prepare agent for pickling by converting defaultdicts to regular dicts.
        
        Returns:
            Dictionary containing agent state without lambda functions
        """
        state = self.__dict__.copy()
        # Convert all defaultdicts to regular dicts (removes unpicklable lambdas)
        state['q_values'] = dict(state['q_values'])
        state['returns_sum'] = dict(state['returns_sum'])
        state['returns_count'] = dict(state['returns_count'])
        return state
    
    def __setstate__(self, state: dict) -> None:
        """Restore agent from pickle by recreating defaultdicts.
        
        Args:
            state: Dictionary containing agent state
        """
        self.__dict__.update(state)
        # Recreate defaultdicts with lambdas
        q_dict = state['q_values']
        action_size = len(next(iter(q_dict.values()))) if q_dict else self.env.action_space.n
        
        self.q_values = defaultdict(lambda: np.zeros(action_size))
        self.returns_sum = defaultdict(lambda: np.zeros(action_size))
        self.returns_count = defaultdict(lambda: np.zeros(action_size))
        
        # Restore all values
        self.q_values.update(q_dict)
        self.returns_sum.update(state['returns_sum'])
        self.returns_count.update(state['returns_count'])

