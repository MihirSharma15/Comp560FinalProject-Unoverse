"""Training functions for RL agents."""

from typing import Dict
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from ..agents.BaseAgent import BaseAgent


def train_agent(
    agent: BaseAgent,
    env: gym.Env,
    n_episodes: int,
    verbose: bool = True
) -> Dict[str, float]:
    """Train agent on the environment.
    
    Args:
        agent: The learning agent
        env: Gymnasium environment
        n_episodes: Number of episodes to train
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with training statistics
    """
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
            # Get action from agent
            action = agent.get_action(obs)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update agent
            agent.update(obs, action, reward, terminated, next_obs)
            
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
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Update progress bar with running statistics
        if verbose and episode % 100 == 0:
            recent_rewards = total_rewards[-100:] if len(total_rewards) >= 100 else total_rewards
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

