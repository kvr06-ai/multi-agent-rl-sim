"""
Base agent interface for Prisoner's Dilemma agents.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class BaseAgent(ABC):
    """
    Abstract base class for all Prisoner's Dilemma agents.
    
    This class defines the interface that all agent implementations must follow.
    """
    
    def __init__(self, name: str = "BaseAgent"):
        """
        Initialize the agent.
        
        Args:
            name: The name of the agent
        """
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """Reset the agent's state between episodes."""
        self.opponent_history = []
        self.action_history = []
        self.reward_history = []
        self.cumulative_reward = 0.0
    
    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """
        Choose an action based on the observation.
        
        Args:
            observation: The current observation from the environment
            
        Returns:
            The chosen action (0 for cooperate, 1 for defect)
        """
        pass
    
    def update(self, observation: np.ndarray, action: int, reward: float, 
               next_observation: np.ndarray, done: bool) -> None:
        """
        Update the agent after taking an action.
        
        Args:
            observation: The observation before taking the action
            action: The action taken
            reward: The reward received
            next_observation: The next observation after taking the action
            done: Whether the episode is done
        """
        # Record the action and reward
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.cumulative_reward += reward
        
        # Record opponent's action if available in the observation
        if len(next_observation) > 0 and next_observation[-1] >= 0:
            self.opponent_history.append(int(next_observation[-1]))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's performance.
        
        Returns:
            Dictionary with statistics
        """
        # Calculate cooperation rate
        if len(self.action_history) > 0:
            cooperation_rate = sum(1 for a in self.action_history if a == 0) / len(self.action_history)
        else:
            cooperation_rate = 0.0
        
        # Count reciprocated actions (mutual cooperation or mutual defection)
        mutual_actions = 0
        for i in range(min(len(self.action_history), len(self.opponent_history))):
            if self.action_history[i] == self.opponent_history[i]:
                mutual_actions += 1
        
        if len(self.opponent_history) > 0:
            reciprocation_rate = mutual_actions / len(self.opponent_history)
        else:
            reciprocation_rate = 0.0
        
        return {
            "name": self.name,
            "total_steps": len(self.action_history),
            "cumulative_reward": self.cumulative_reward,
            "cooperation_rate": cooperation_rate,
            "reciprocation_rate": reciprocation_rate,
            "avg_reward": self.cumulative_reward / max(1, len(self.reward_history))
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name}"
    
    def save(self, path: str) -> None:
        """
        Save the agent to a file.
        
        Args:
            path: Path to save the agent to
        """
        pass  # Default implementation does nothing
    
    def load(self, path: str) -> None:
        """
        Load the agent from a file.
        
        Args:
            path: Path to load the agent from
        """
        pass  # Default implementation does nothing 