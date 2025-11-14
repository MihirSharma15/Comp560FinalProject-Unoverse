"""Q-Learning agent with state-dependent adaptive epsilon exploration.

This agent adjusts its exploration rate based on how frequently each state has been visited.
Common states get low epsilon (exploit more), rare states get high epsilon (explore more).
"""

from collections import defaultdict
from typing import Literal, Dict, Tuple
import numpy as np
import gymnasium as gym
from .SimpleAgent import SimpleAgent


class AdaptiveEpsilonAgent(SimpleAgent):
    """Q-Learning agent with adaptive epsilon based on state visit counts.
    
    This agent maintains visit counts for each state and adjusts epsilon dynamically:
    - Rare states (low visit count): High epsilon → explore more
    - Common states (high visit count): Low epsilon → exploit more
    
    Multiple strategies are available for computing epsilon from visit counts.
    """
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        strategy: Literal["count_based", "ucb", "threshold"] = "count_based",
        # Strategy-specific hyperparameters
        k_param: float = 10.0,  # For count_based: ε = k/(k+N)
        c_param: float = 1.0,   # For UCB: ε = ε_base + c/sqrt(N+1)
        thresholds: Tuple[int, int, int] = (10, 100, 1000),  # For threshold
        epsilon_levels: Tuple[float, float, float, float] = (0.9, 0.5, 0.2, 0.05),  # For threshold
    ) -> None:
        """Initialize an Adaptive Epsilon Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (base epsilon for some strategies)
            epsilon_decay: How much to reduce global epsilon each episode
            final_epsilon: Minimum exploration rate
            discount_factor: How much to value future rewards (0-1)
            strategy: Which epsilon computation strategy to use
                - "count_based": ε(s) = k/(k + N(s))
                - "ucb": ε(s) = ε_base + c/sqrt(N(s)+1)
                - "threshold": Piecewise function based on thresholds
            k_param: Hyperparameter for count_based strategy (higher = more exploration)
            c_param: Exploration bonus coefficient for UCB strategy
            thresholds: (low, medium, high) visit count thresholds for threshold strategy
            epsilon_levels: (very_high, high, medium, low) epsilon values for threshold strategy
        """
        super().__init__(
            env=env,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
        # State visit counting
        self.state_counts: Dict[tuple, int] = defaultdict(int)
        
        # Strategy selection
        self.strategy = strategy
        
        # Store base epsilon for strategies that need it
        self.base_epsilon = initial_epsilon
        
        # Strategy hyperparameters
        self.k_param = k_param
        self.c_param = c_param
        self.thresholds = sorted(thresholds)  # Ensure sorted order
        self.epsilon_levels = epsilon_levels
        
        # Statistics tracking
        self.epsilon_history: Dict[tuple, list] = defaultdict(list)
        self.unique_states_visited: int = 0
        
    def _compute_state_epsilon(self, state: tuple) -> float:
        """Compute epsilon for a specific state based on visit count.
        
        Args:
            state: The state tuple
            
        Returns:
            Epsilon value for this state
        """
        visit_count = self.state_counts[state]
        
        if self.strategy == "count_based":
            # ε(s) = k / (k + N(s))
            # High k = more exploration, starts at ε=1.0 and decays hyperbolically
            epsilon = self.k_param / (self.k_param + visit_count)
            
        elif self.strategy == "ucb":
            # ε(s) = ε_base + c / sqrt(N(s) + 1)
            # Adds exploration bonus that decreases with sqrt(visits)
            exploration_bonus = self.c_param / np.sqrt(visit_count + 1)
            epsilon = self.base_epsilon + exploration_bonus
            
        elif self.strategy == "threshold":
            # Piecewise constant function based on visit thresholds
            low, medium, high = self.thresholds
            very_high_eps, high_eps, medium_eps, low_eps = self.epsilon_levels
            
            if visit_count < low:
                epsilon = very_high_eps
            elif visit_count < medium:
                epsilon = high_eps
            elif visit_count < high:
                epsilon = medium_eps
            else:
                epsilon = low_eps
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Clamp between final_epsilon and 1.0
        epsilon = max(self.final_epsilon, min(1.0, epsilon))
        
        return epsilon
    
    def get_action(self, obs: np.ndarray | tuple, valid_actions: np.ndarray | None = None) -> int:
        """Choose an action using state-adaptive epsilon-greedy strategy.
        
        Args:
            obs: Current observation from the environment
            valid_actions: Optional array of valid action indices
            
        Returns:
            action: Integer representing the chosen action
        """
        state = self._obs_to_state(obs)
        
        # Compute state-specific epsilon BEFORE incrementing count
        # This ensures epsilon reflects prior experience, not current visit
        state_epsilon = self._compute_state_epsilon(state)
        
        # Increment visit count for this state (after computing epsilon)
        if self.state_counts[state] == 0:
            self.unique_states_visited += 1
        self.state_counts[state] += 1
        
        # Track epsilon history for this state
        self.epsilon_history[state].append(state_epsilon)
        
        # Get valid actions if not provided
        if valid_actions is None:
            if hasattr(self.env, 'get_valid_actions'):
                valid_actions = self.env.get_valid_actions()
            else:
                valid_actions = np.arange(self.env.action_space.n)
                
        # If no valid actions available, return random action
        if len(valid_actions) == 0:
            return self.env.action_space.sample()
        
        # With probability state_epsilon: explore (random valid action)
        if np.random.random() < state_epsilon:
            return np.random.choice(valid_actions)
        
        # With probability (1-state_epsilon): exploit (best known valid action)
        else:
            q_values_for_state = self.q_values[state]
            # Mask invalid actions with very negative value
            masked_q_values = np.full(len(q_values_for_state), -np.inf)
            masked_q_values[valid_actions] = q_values_for_state[valid_actions]
            return int(np.argmax(masked_q_values))
    
    def decay_epsilon(self) -> None:
        """Reduce base exploration rate after each episode.
        
        For UCB strategy, this reduces the base epsilon.
        For count_based and threshold, this has minimal effect since
        epsilon is primarily determined by state visit counts.
        """
        self.base_epsilon = max(self.final_epsilon, self.base_epsilon - self.epsilon_decay)
        # Also update self.epsilon for compatibility with parent class
        self.epsilon = self.base_epsilon
    
    def get_state_statistics(self) -> Dict:
        """Get comprehensive statistics about state visits and epsilon values.
        
        Returns:
            Dictionary with visit count statistics, epsilon distributions, etc.
        """
        if not self.state_counts:
            return {
                "unique_states": 0,
                "total_visits": 0,
                "avg_visits_per_state": 0.0,
                "min_visits": 0,
                "max_visits": 0,
                "visit_distribution": {},
            }
        
        visit_counts = list(self.state_counts.values())
        total_visits = sum(visit_counts)
        
        # Compute epsilon for each visited state
        current_epsilons = {
            state: self._compute_state_epsilon(state)
            for state in self.state_counts.keys()
        }
        
        # Visit distribution: how many states have 1, 2, 3, ... visits
        from collections import Counter
        visit_distribution = Counter(visit_counts)
        
        return {
            "unique_states": len(self.state_counts),
            "total_visits": total_visits,
            "avg_visits_per_state": np.mean(visit_counts),
            "median_visits": np.median(visit_counts),
            "std_visits": np.std(visit_counts),
            "min_visits": min(visit_counts),
            "max_visits": max(visit_counts),
            "visit_distribution": dict(sorted(visit_distribution.items())),
            "avg_current_epsilon": np.mean(list(current_epsilons.values())),
            "min_current_epsilon": min(current_epsilons.values()),
            "max_current_epsilon": max(current_epsilons.values()),
            "states_by_visits": sorted(
                [(state, count) for state, count in self.state_counts.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20],  # Top 20 most visited states
        }
    
    def get_epsilon_for_state(self, obs: np.ndarray | tuple) -> float:
        """Get the current epsilon value that would be used for a given state.
        
        Useful for analysis and debugging.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Current epsilon value for this state
        """
        state = self._obs_to_state(obs)
        return self._compute_state_epsilon(state)
    
    def get_rare_states(self, threshold: int = 5) -> Dict[tuple, int]:
        """Get states that have been visited fewer than threshold times.
        
        Args:
            threshold: Maximum visit count to be considered "rare"
            
        Returns:
            Dictionary of {state: visit_count} for rare states
        """
        return {
            state: count
            for state, count in self.state_counts.items()
            if count < threshold
        }
    
    def get_common_states(self, threshold: int = 100) -> Dict[tuple, int]:
        """Get states that have been visited more than threshold times.
        
        Args:
            threshold: Minimum visit count to be considered "common"
            
        Returns:
            Dictionary of {state: visit_count} for common states
        """
        return {
            state: count
            for state, count in self.state_counts.items()
            if count >= threshold
        }
    
    def __getstate__(self) -> dict:
        """Prepare agent for pickling by converting defaultdicts to regular dicts.
        
        Returns:
            Dictionary containing agent state without lambda functions
        """
        # Get parent class state (handles q_values defaultdict)
        state = super().__getstate__()
        
        # Convert additional defaultdicts to regular dicts
        state['state_counts'] = dict(state['state_counts'])
        state['epsilon_history'] = dict(state['epsilon_history'])
        
        return state
    
    def __setstate__(self, state: dict) -> None:
        """Restore agent from pickle by recreating defaultdicts.
        
        Args:
            state: Dictionary containing agent state
        """
        # Restore parent class state (handles q_values defaultdict)
        super().__setstate__(state)
        
        # Recreate additional defaultdicts
        state_counts_dict = state['state_counts']
        epsilon_history_dict = state['epsilon_history']
        
        self.state_counts = defaultdict(int)
        self.state_counts.update(state_counts_dict)
        
        self.epsilon_history = defaultdict(list)
        self.epsilon_history.update(epsilon_history_dict)

