"""Agent loading, saving, and training utilities for comparison module."""

import os
import pickle
from typing import Dict, Tuple, Optional
import gymnasium as gym
from tqdm import tqdm

from ..agents.SimpleAgent import SimpleAgent
from ..agents.AdaptiveEpsilonAgent import AdaptiveEpsilonAgent
from ..agents.BaseAgent import BaseAgent


def get_agent_configs() -> Dict[str, Dict]:
    """Get default configurations for all agent types.
    
    Returns:
        Dictionary mapping agent names to their configuration parameters
    """
    return {
        "simple": {
            "name": "Simple (Fixed Epsilon)",
            "class": SimpleAgent,
            "params": {
                "learning_rate": 0.01,
                "initial_epsilon": 1.0,
                "epsilon_decay": None,  # Will be computed based on n_episodes
                "final_epsilon": 0.01,
                "discount_factor": 1.0,
            },
        },
        "count_based": {
            "name": "Count-Based (k=10)",
            "class": AdaptiveEpsilonAgent,
            "params": {
                "learning_rate": 0.01,
                "initial_epsilon": 1.0,
                "epsilon_decay": 0,
                "final_epsilon": 0.01,
                "discount_factor": 1.0,
                "strategy": "count_based",
                "k_param": 10.0,
            },
        },
        "count_based_high_k": {
            "name": "Count-Based (k=100)",
            "class": AdaptiveEpsilonAgent,
            "params": {
                "learning_rate": 0.01,
                "initial_epsilon": 1.0,
                "epsilon_decay": 0,
                "final_epsilon": 0.01,
                "discount_factor": 1.0,
                "strategy": "count_based",
                "k_param": 100.0,
            },
        },
        "count_based_low_k": {
            "name": "Count-Based (k=2)",
            "class": AdaptiveEpsilonAgent,
            "params": {
                "learning_rate": 0.01,
                "initial_epsilon": 1.0,
                "epsilon_decay": 0,
                "final_epsilon": 0.01,
                "discount_factor": 1.0,
                "strategy": "count_based",
                "k_param": 2.0,
            },
        },
        "count_based_ultra_high_k": {
            "name": "Count-Based (k=1000)",
            "class": AdaptiveEpsilonAgent,
            "params": {
                "learning_rate": 0.01,
                "initial_epsilon": 1.0,
                "epsilon_decay": 0,
                "final_epsilon": 0.01,
                "discount_factor": 1.0,
                "strategy": "count_based",
                "k_param": 1000.0,
            },
        },
        "ucb": {
            "name": "UCB-Style (c=1.0)",
            "class": AdaptiveEpsilonAgent,
            "params": {
                "learning_rate": 0.01,
                "initial_epsilon": 0.1,
                "epsilon_decay": 0,
                "final_epsilon": 0.01,
                "discount_factor": 1.0,
                "strategy": "ucb",
                "c_param": 1.0,
            },
        },
        "ucb_high_c": {
            "name": "UCB-Style (c=10.0)",
            "class": AdaptiveEpsilonAgent,
            "params": {
                "learning_rate": 0.01,
                "initial_epsilon": 0.1,
                "epsilon_decay": 0,
                "final_epsilon": 0.01,
                "discount_factor": 1.0,
                "strategy": "ucb",
                "c_param": 10.0,
            },
        },
        "threshold": {
            "name": "Threshold-Based",
            "class": AdaptiveEpsilonAgent,
            "params": {
                "learning_rate": 0.01,
                "initial_epsilon": 1.0,
                "epsilon_decay": 0,
                "final_epsilon": 0.01,
                "discount_factor": 1.0,
                "strategy": "threshold",
                "thresholds": (10, 100, 1000),
                "epsilon_levels": (0.9, 0.5, 0.2, 0.05),
            },
        },
    }


def create_agent(
    agent_type: str,
    env: gym.Env,
    n_train_episodes: int,
    custom_params: Optional[Dict] = None
) -> Tuple[str, BaseAgent]:
    """Create an agent instance with the specified configuration.
    
    Args:
        agent_type: Type of agent ("simple", "count_based", "ucb", "threshold")
        env: The training environment
        n_train_episodes: Number of training episodes (used for epsilon decay)
        custom_params: Optional custom parameters to override defaults
        
    Returns:
        Tuple of (agent_name, agent_instance)
    """
    configs = get_agent_configs()
    
    if agent_type not in configs:
        raise ValueError(f"Unknown agent type: {agent_type}. Must be one of {list(configs.keys())}")
    
    config = configs[agent_type]
    params = config["params"].copy()
    
    # Compute epsilon decay for simple agent
    if agent_type == "simple" and params["epsilon_decay"] is None:
        params["epsilon_decay"] = (params["initial_epsilon"] - params["final_epsilon"]) / n_train_episodes
    
    # Override with custom parameters if provided
    if custom_params:
        params.update(custom_params)
    
    # Create agent instance
    agent = config["class"](env=env, **params)
    
    return config["name"], agent


def save_agent(agent: BaseAgent, filepath: str) -> None:
    """Save a trained agent to disk.
    
    Args:
        agent: The agent to save
        filepath: Path to save the agent (will add .pkl extension if not present)
    """
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(agent, f)
    
    print(f"Agent saved to {filepath}")


def load_agent(filepath: str) -> Optional[BaseAgent]:
    """Load a trained agent from disk.
    
    Args:
        filepath: Path to the saved agent file
        
    Returns:
        Loaded agent instance, or None if file doesn't exist
    """
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        agent = pickle.load(f)
    
    return agent


def save_agent_with_stats(agent: BaseAgent, training_stats: Dict, filepath: str) -> None:
    """Save a trained agent and its training statistics to disk.
    
    Args:
        agent: The agent to save
        training_stats: Training statistics dictionary (including total_rewards history)
        filepath: Path to save the agent (will add .pkl extension if not present)
    """
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save agent
    with open(filepath, 'wb') as f:
        pickle.dump(agent, f)
    
    # Save training stats separately
    stats_filepath = filepath.replace('.pkl', '_stats.pkl')
    with open(stats_filepath, 'wb') as f:
        pickle.dump(training_stats, f)
    
    print(f"Agent saved to {filepath}")
    print(f"Training stats saved to {stats_filepath}")


def load_agent_with_stats(filepath: str) -> Tuple[Optional[BaseAgent], Optional[Dict]]:
    """Load a trained agent and its training statistics from disk.
    
    Args:
        filepath: Path to the saved agent file
        
    Returns:
        Tuple of (agent, training_stats), or (None, None) if files don't exist
    """
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    
    if not os.path.exists(filepath):
        return None, None
    
    # Load agent
    with open(filepath, 'rb') as f:
        agent = pickle.load(f)
    
    # Load training stats if they exist
    stats_filepath = filepath.replace('.pkl', '_stats.pkl')
    training_stats = None
    if os.path.exists(stats_filepath):
        with open(stats_filepath, 'rb') as f:
            training_stats = pickle.load(f)
    
    return agent, training_stats


def train_agent(
    agent: BaseAgent,
    env: gym.Env,
    n_episodes: int,
    verbose: bool = True
) -> Dict[str, any]:
    """Train an agent on the environment.
    
    Args:
        agent: The agent to train
        env: The training environment
        n_episodes: Number of training episodes
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary containing training statistics
    """
    import numpy as np
    
    total_rewards = []
    wins = 0
    losses = 0
    draws = 0
    
    iterator = tqdm(range(n_episodes), desc="Training") if verbose else range(n_episodes)
    
    for episode in iterator:
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            
            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
        
        agent.decay_epsilon()
        
        if verbose and episode % 1000 == 0:
            recent_rewards = total_rewards[-1000:] if len(total_rewards) >= 1000 else total_rewards
            avg_reward = np.mean(recent_rewards)
            iterator.set_postfix({
                'avg_reward': f'{avg_reward:.3f}',
                'epsilon': f'{agent.epsilon:.3f}'
            })
    
    return {
        "total_episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "avg_reward": np.mean(total_rewards),
        "total_rewards": total_rewards,
    }


def load_or_train_agents(
    env: gym.Env,
    n_train_episodes: int = 250_000,
    pretrained_dir: str = "src/pretrained",
    force_retrain: bool = False,
    verbose: bool = True
) -> Dict[str, Tuple[BaseAgent, Dict]]:
    """Load pre-trained agents or train them from scratch.
    
    Args:
        env: The training environment
        n_train_episodes: Number of training episodes if training from scratch
        pretrained_dir: Directory containing pre-trained agent files
        force_retrain: If True, train from scratch even if pretrained agents exist
        verbose: Whether to show progress during training
        
    Returns:
        Dictionary mapping agent names to (agent, training_stats) tuples
    """
    configs = get_agent_configs()
    results = {}
    
    for agent_type in configs.keys():
        agent_name, agent = create_agent(agent_type, env, n_train_episodes)
        filepath = os.path.join(pretrained_dir, f"{agent_type}.pkl")
        
        if not force_retrain:
            loaded_agent, training_stats = load_agent_with_stats(filepath)
            if loaded_agent is not None:
                if verbose:
                    stats_msg = " (with training history)" if training_stats else " (no training history)"
                    print(f"✓ Loaded pre-trained {agent_name} from {filepath}{stats_msg}")
                results[agent_name] = (loaded_agent, training_stats)
                continue
        
        # Train from scratch
        if verbose:
            print(f"\nTraining {agent_name}...")
        
        training_stats = train_agent(agent, env, n_train_episodes, verbose=verbose)
        
        # Save the trained agent with stats
        save_agent_with_stats(agent, training_stats, filepath)
        
        results[agent_name] = (agent, training_stats)
    
    return results

