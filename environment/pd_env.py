"""
Prisoner's Dilemma environment implementation.
Uses Gymnasium framework for reinforcement learning compatibility.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Union, Any

# Define the payoff matrix for the standard Prisoner's Dilemma
# (row player's payoff, column player's payoff)
# Actions: 0 = Cooperate, 1 = Defect
DEFAULT_PAYOFF_MATRIX = np.array([
    [(3, 3), (0, 5)],  # Row player cooperates
    [(5, 0), (1, 1)]   # Row player defects
])

class PrisonersDilemmaEnv(gym.Env):
    """
    Prisoner's Dilemma environment following the Gymnasium API.
    
    This environment simulates the classic Prisoner's Dilemma game where:
    - Two agents interact simultaneously
    - Each agent can either cooperate (0) or defect (1)
    - Rewards are determined by a payoff matrix
    
    The observation space includes:
    - The opponent's most recent action
    - Optional: history of n previous actions
    
    The action space is discrete:
    - 0: Cooperate
    - 1: Defect
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        payoff_matrix: np.ndarray = DEFAULT_PAYOFF_MATRIX,
        memory_size: int = 1,
        max_steps: int = 100,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the Prisoner's Dilemma environment.
        
        Args:
            payoff_matrix: The payoff matrix for the game
            memory_size: Number of previous actions to include in the observation
            max_steps: Maximum number of steps per episode
            render_mode: The render mode to use
        """
        super().__init__()
        
        self.payoff_matrix = payoff_matrix
        self.memory_size = memory_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Action space: cooperate (0) or defect (1)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: opponent's last n actions
        # -1 represents no previous action (start of game)
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=(memory_size,), 
            dtype=np.int8
        )
        
        # Initialize game state
        self.reset()
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the initial state.
        
        Args:
            seed: Random seed
            options: Additional options for environment reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Initialize memory with -1 (no previous actions)
        self.memory = np.full(self.memory_size, -1, dtype=np.int8)
        
        # Initialize step counter
        self.step_count = 0
        
        # Initialize cumulative rewards
        self.cumulative_rewards = [0, 0]
        
        # Game history for analysis
        self.history = {
            "actions": [],  # List of (agent action, opponent action) pairs
            "rewards": []   # List of (agent reward, opponent reward) pairs
        }
        
        return self._get_observation(), self._get_info()
    
    def step(
        self, 
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The agent's action (0: cooperate, 1: defect)
            
        Returns:
            observation: The new observation
            reward: The reward for the agent
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Opponent action (to be replaced with actual agent decision)
        # For now, we'll use a simple random policy
        opponent_action = self.np_random.integers(0, 2)
        
        # Get rewards based on payoff matrix
        agent_reward, opponent_reward = self.payoff_matrix[action, opponent_action]
        
        # Update memory with opponent's action
        self.memory = np.roll(self.memory, -1)
        self.memory[-1] = opponent_action
        
        # Update cumulative rewards
        self.cumulative_rewards[0] += agent_reward
        self.cumulative_rewards[1] += opponent_reward
        
        # Update history
        self.history["actions"].append((action, opponent_action))
        self.history["rewards"].append((agent_reward, opponent_reward))
        
        # Increment step counter
        self.step_count += 1
        
        # Check if episode is done
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Handle rendering
        if self.render_mode == "human":
            self.render()
        
        return observation, agent_reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            Current observation as a numpy array
        """
        return self.memory.copy()
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current state.
        
        Returns:
            Dictionary with additional information
        """
        return {
            "step_count": self.step_count,
            "cumulative_rewards": self.cumulative_rewards.copy(),
            "memory": self.memory.copy()
        }
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            Either None or an RGB array if mode is 'rgb_array'
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self) -> Optional[np.ndarray]:
        """
        Render a frame of the environment.
        
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        # Implement visualization logic here
        # For now, we'll just print the current state
        if self.step_count > 0:
            last_actions = self.history["actions"][-1]
            last_rewards = self.history["rewards"][-1]
            
            print(f"Step {self.step_count}:")
            print(f"  Actions: Agent={last_actions[0]}, Opponent={last_actions[1]}")
            print(f"  Rewards: Agent={last_rewards[0]}, Opponent={last_rewards[1]}")
            print(f"  Cumulative: Agent={self.cumulative_rewards[0]}, Opponent={self.cumulative_rewards[1]}")
        
        # Return an empty array for rgb_array mode
        if self.render_mode == "rgb_array":
            return np.zeros((300, 400, 3), dtype=np.uint8)
        
        return None

class MultiAgentPrisonersDilemmaEnv:
    """
    A wrapper for PrisonersDilemmaEnv that allows for interaction between
    two learning agents rather than an agent and a fixed opponent.
    """
    
    def __init__(
        self,
        payoff_matrix: np.ndarray = DEFAULT_PAYOFF_MATRIX,
        memory_size: int = 1,
        max_steps: int = 100,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the multi-agent environment.
        
        Args:
            payoff_matrix: The payoff matrix for the game
            memory_size: Number of previous actions to include in the observation
            max_steps: Maximum number of steps per episode
            render_mode: The render mode to use
        """
        self.env1 = PrisonersDilemmaEnv(payoff_matrix, memory_size, max_steps, render_mode)
        self.env2 = PrisonersDilemmaEnv(payoff_matrix, memory_size, max_steps, render_mode)
        
        self.max_steps = max_steps
        self.step_count = 0
        
        # Game history
        self.history = {
            "actions": [],  # List of (agent1 action, agent2 action) pairs
            "rewards": []   # List of (agent1 reward, agent2 reward) pairs
        }
    
    def reset(
        self, 
        seed: Optional[int] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Reset both environments.
        
        Args:
            seed: Random seed
            
        Returns:
            observations: Initial observations for both agents
            info: Additional information
        """
        obs1, info1 = self.env1.reset(seed=seed)
        obs2, info2 = self.env2.reset(seed=seed)
        
        self.step_count = 0
        self.history = {
            "actions": [],
            "rewards": []
        }
        
        return [obs1, obs2], {"agent1": info1, "agent2": info2}
    
    def step(
        self, 
        actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict[str, Any]]:
        """
        Step both environments forward with the actions from both agents.
        
        Args:
            actions: List of actions for both agents [agent1_action, agent2_action]
            
        Returns:
            observations: New observations for both agents
            rewards: Rewards for both agents
            terminated: Whether the episode is terminated for both agents
            truncated: Whether the episode is truncated for both agents
            info: Additional information for both agents
        """
        # Unpack actions
        action1, action2 = actions
        
        # Update environments with opponent actions
        self.env1.memory = np.roll(self.env1.memory, -1)
        self.env1.memory[-1] = action2
        
        self.env2.memory = np.roll(self.env2.memory, -1)
        self.env2.memory[-1] = action1
        
        # Get rewards based on payoff matrix
        reward1, reward2 = self.env1.payoff_matrix[action1, action2]
        
        # Update histories
        self.env1.step_count += 1
        self.env2.step_count += 1
        
        self.env1.cumulative_rewards[0] += reward1
        self.env2.cumulative_rewards[0] += reward2
        
        self.env1.history["actions"].append((action1, action2))
        self.env1.history["rewards"].append((reward1, reward2))
        
        self.env2.history["actions"].append((action2, action1))
        self.env2.history["rewards"].append((reward2, reward1))
        
        # Update own history
        self.history["actions"].append((action1, action2))
        self.history["rewards"].append((reward1, reward2))
        
        # Increment step counter
        self.step_count += 1
        
        # Check if episode is done
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        # Get new observations and info
        obs1 = self.env1._get_observation()
        obs2 = self.env2._get_observation()
        
        info1 = self.env1._get_info()
        info2 = self.env2._get_info()
        
        return [obs1, obs2], [reward1, reward2], terminated, truncated, {"agent1": info1, "agent2": info2}
    
    def render(self, mode: str = 'human'):
        """
        Render the environment.
        
        Args:
            mode: The render mode
        """
        # For simplicity, we'll just use env1's render method
        return self.env1.render()
    
    def get_payoff_matrix(self) -> np.ndarray:
        """
        Get the payoff matrix.
        
        Returns:
            The payoff matrix
        """
        return self.env1.payoff_matrix 