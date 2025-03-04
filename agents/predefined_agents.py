"""
Predefined strategy agents for the Prisoner's Dilemma.
"""
import numpy as np
import random
from typing import List, Optional
from .base_agent import BaseAgent

class AlwaysCooperateAgent(BaseAgent):
    """Agent that always cooperates."""
    
    def __init__(self, name: str = "Always Cooperate"):
        """Initialize the agent."""
        super().__init__(name)
    
    def act(self, observation: np.ndarray) -> int:
        """Always return cooperate (0)."""
        return 0

class AlwaysDefectAgent(BaseAgent):
    """Agent that always defects."""
    
    def __init__(self, name: str = "Always Defect"):
        """Initialize the agent."""
        super().__init__(name)
    
    def act(self, observation: np.ndarray) -> int:
        """Always return defect (1)."""
        return 1

class RandomAgent(BaseAgent):
    """Agent that acts randomly."""
    
    def __init__(self, name: str = "Random", p_cooperate: float = 0.5):
        """
        Initialize the agent.
        
        Args:
            name: The name of the agent
            p_cooperate: Probability of cooperation
        """
        super().__init__(name)
        self.p_cooperate = p_cooperate
    
    def act(self, observation: np.ndarray) -> int:
        """Return cooperate with probability p_cooperate, defect otherwise."""
        return 0 if random.random() < self.p_cooperate else 1

class TitForTatAgent(BaseAgent):
    """
    Agent that implements the Tit-for-Tat strategy.
    
    This agent cooperates on the first move, then copies the opponent's previous move.
    """
    
    def __init__(self, name: str = "Tit-for-Tat"):
        """Initialize the agent."""
        super().__init__(name)
    
    def act(self, observation: np.ndarray) -> int:
        """
        Cooperate on first move, then copy opponent's last move.
        
        Args:
            observation: Array containing opponent's last actions
            
        Returns:
            Action to take (0=cooperate, 1=defect)
        """
        # If no valid opponent action observed yet, cooperate
        if len(observation) == 0 or observation[-1] < 0:
            return 0
        
        # Otherwise copy the opponent's last action
        return int(observation[-1])

class GradualAgent(BaseAgent):
    """
    Agent that implements the Gradual strategy.
    
    This agent cooperates until the opponent defects. After that, it defects n times,
    where n is the number of times the opponent has defected so far, then cooperates twice.
    """
    
    def __init__(self, name: str = "Gradual"):
        """Initialize the agent."""
        super().__init__(name)
        self.defect_count = 0
        self.retaliation_count = 0
        self.forgiveness_count = 0
    
    def reset(self) -> None:
        """Reset the agent's state."""
        super().reset()
        self.defect_count = 0
        self.retaliation_count = 0
        self.forgiveness_count = 0
    
    def act(self, observation: np.ndarray) -> int:
        """
        Implements the Gradual strategy.
        
        Args:
            observation: Array containing opponent's last actions
            
        Returns:
            Action to take (0=cooperate, 1=defect)
        """
        # If we're in forgiveness mode, cooperate and decrement counter
        if self.forgiveness_count > 0:
            self.forgiveness_count -= 1
            return 0
        
        # If we're in retaliation mode, defect and decrement counter
        if self.retaliation_count > 0:
            self.retaliation_count -= 1
            # If this was the last retaliation, switch to forgiveness mode
            if self.retaliation_count == 0:
                self.forgiveness_count = 2
            return 1
        
        # Check opponent's last action
        if len(observation) > 0 and observation[-1] == 1:  # Opponent defected
            self.defect_count += 1
            self.retaliation_count = self.defect_count
            return 1
        
        # Default: cooperate
        return 0

class FirmButFairAgent(BaseAgent):
    """
    Agent that implements the Firm-But-Fair strategy.
    
    This agent cooperates unless the opponent defected while it cooperated in the previous round.
    """
    
    def __init__(self, name: str = "Firm-But-Fair"):
        """Initialize the agent."""
        super().__init__(name)
        self.last_action = None
    
    def reset(self) -> None:
        """Reset the agent's state."""
        super().reset()
        self.last_action = None
    
    def act(self, observation: np.ndarray) -> int:
        """
        Cooperate unless betrayed in the previous round.
        
        Args:
            observation: Array containing opponent's last actions
            
        Returns:
            Action to take (0=cooperate, 1=defect)
        """
        # First move or no valid observation
        if self.last_action is None or len(observation) == 0 or observation[-1] < 0:
            self.last_action = 0
            return 0
        
        # If I cooperated and opponent defected, defect
        if self.last_action == 0 and observation[-1] == 1:
            self.last_action = 1
            return 1
        
        # Otherwise cooperate
        self.last_action = 0
        return 0
    
    def update(self, observation: np.ndarray, action: int, reward: float, 
               next_observation: np.ndarray, done: bool) -> None:
        """Update the agent after taking an action."""
        super().update(observation, action, reward, next_observation, done)
        self.last_action = action

class WinStayLoseShiftAgent(BaseAgent):
    """
    Agent that implements the Win-Stay, Lose-Shift strategy (also known as Pavlov).
    
    This agent repeats its previous action if it received a good reward, 
    and changes its action if it received a bad reward.
    """
    
    def __init__(self, name: str = "Win-Stay-Lose-Shift", good_reward_threshold: float = 2.0):
        """
        Initialize the agent.
        
        Args:
            name: The name of the agent
            good_reward_threshold: Threshold above which a reward is considered "good"
        """
        super().__init__(name)
        self.good_reward_threshold = good_reward_threshold
        self.last_action = None
        self.last_reward = None
    
    def reset(self) -> None:
        """Reset the agent's state."""
        super().reset()
        self.last_action = None
        self.last_reward = None
    
    def act(self, observation: np.ndarray) -> int:
        """
        Implements Win-Stay, Lose-Shift strategy.
        
        Args:
            observation: Array containing opponent's last actions
            
        Returns:
            Action to take (0=cooperate, 1=defect)
        """
        # First move
        if self.last_action is None:
            return 0
        
        # If last outcome was good, repeat last action
        if self.last_reward >= self.good_reward_threshold:
            return self.last_action
        
        # If last outcome was bad, change action
        return 1 - self.last_action
    
    def update(self, observation: np.ndarray, action: int, reward: float, 
               next_observation: np.ndarray, done: bool) -> None:
        """Update the agent after taking an action."""
        super().update(observation, action, reward, next_observation, done)
        self.last_action = action
        self.last_reward = reward

def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """
    Factory function to create an agent of the specified type.
    
    Args:
        agent_type: The type of agent to create
        **kwargs: Additional arguments to pass to the agent constructor
        
    Returns:
        An instance of the specified agent type
        
    Raises:
        ValueError: If the agent type is not recognized
    """
    agent_map = {
        "always_cooperate": AlwaysCooperateAgent,
        "always_defect": AlwaysDefectAgent,
        "random": RandomAgent,
        "tit_for_tat": TitForTatAgent,
        "gradual": GradualAgent,
        "firm_but_fair": FirmButFairAgent,
        "win_stay_lose_shift": WinStayLoseShiftAgent
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_map[agent_type](**kwargs) 