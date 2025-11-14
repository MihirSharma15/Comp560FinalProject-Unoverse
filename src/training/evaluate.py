"""Evaluation functions for RL agents."""

from typing import Dict
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from ..agents.BaseAgent import BaseAgent


def evaluate_agent(
    agent: BaseAgent,
    env: gym.Env,
    n_episodes: int = 1000,
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate agent's performance (no learning).
    
    Args:
        agent: The trained agent to evaluate
        env: Gymnasium environment
        n_episodes: Number of evaluation episodes
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with evaluation statistics
    """
    # Save current epsilon and set to 0 for pure exploitation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    total_rewards = []
    wins = 0
    losses = 0
    draws = 0
    
    iterator = tqdm(range(n_episodes), desc="Evaluating") if verbose else range(n_episodes)
    
    for _ in iterator:
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from agent (exploitation only)
            action = agent.get_action(obs)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
        
        # Track statistics
        total_rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return {
        "total_episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "total_rewards": total_rewards,
    }

