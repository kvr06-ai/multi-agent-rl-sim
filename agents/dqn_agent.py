"""
DQN agent implementation for the Prisoner's Dilemma.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Any, Optional

from .base_agent import BaseAgent

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Replay buffer for storing experiences."""
    
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of experiences
        """
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return len(self.memory)

class QNetwork(nn.Module):
    """Neural network for Q-function approximation."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        """
        Initialize the Q-network.
        
        Args:
            input_dim: Dimension of the input state
            output_dim: Dimension of the output (number of actions)
            hidden_dim: Dimension of the hidden layer
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(BaseAgent):
    """Deep Q-Network agent for the Prisoner's Dilemma."""
    
    def __init__(
        self,
        state_dim: int,
        name: str = "DQN Agent",
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = 'auto'
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of the state space
            name: Name of the agent
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency for updating the target network
            device: Device to use for training ('cpu', 'cuda', or 'auto')
        """
        super().__init__(name)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # DQN parameters
        self.state_dim = state_dim
        self.action_dim = 2  # Cooperate or defect
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Initialize networks
        self.policy_net = QNetwork(state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.steps_done = 0
        self.training_enabled = True
        
        # Track metrics for visualization
        self.epsilon_history = []
        self.q_values_history = []  # Track Q-values over time
        self.loss_history = []      # Track loss over time
        self.latest_q_values = None
    
    def preprocess_state(self, observation: np.ndarray) -> torch.Tensor:
        """
        Preprocess the observation for input to the neural network.
        
        Args:
            observation: Raw observation from the environment
            
        Returns:
            Preprocessed state tensor
        """
        # Debug the observation shape
        observation = np.array(observation).flatten()  # Ensure it's a flat array
        
        # Create a fixed size feature vector
        # For the Prisoner's Dilemma, we'll create a one-hot encoding for each action
        # where -1 = no previous action, 0 = cooperate, 1 = defect
        features = []
        
        # Handle memory size of 1 (default)
        # Typically observation is [-1] at the start of an episode
        # or the opponent's previous move [0] or [1] during the episode
        for action in observation:
            if action == -1:  # No previous action
                features.extend([0.5, 0.5])  # No information, equal probability
            elif action == 0:  # Cooperate
                features.extend([1.0, 0.0])
            elif action == 1:  # Defect
                features.extend([0.0, 1.0])
        
        # If we need more features to match self.state_dim, pad with zeros
        while len(features) < self.state_dim:
            features.append(0.0)
        
        # If we have too many features, truncate
        features = features[:self.state_dim]
        
        # Convert to tensor
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)
    
    def act(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current state observation
        
        Returns:
            int: Selected action (0 for cooperate, 1 for defect)
        """
        state = self.preprocess_state(observation)
        
        # Epsilon-greedy action selection
        if self.training_enabled and random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            # Get Q-values and select best action
            with torch.no_grad():
                q_values = self.policy_net(state)
                
                # Store Q-values for visualization
                self.latest_q_values = q_values.numpy()[0]
                self.q_values_history.append(self.latest_q_values.copy())
                
                action = q_values.max(1)[1].item()
        
        return action
    
    def update(self, observation: np.ndarray, action: int, reward: float, 
             next_observation: np.ndarray, done: bool) -> None:
        """
        Update the agent's knowledge based on the received experience.
        
        Args:
            observation: Current state observation
            action: Action taken
            reward: Reward received
            next_observation: Next state observation
            done: Whether the episode is done
        """
        if not self.training_enabled:
            return
        
        # Store the transition in the replay buffer
        self.replay_buffer.add(observation, action, reward, next_observation, done)
        
        # Increment step counter
        self.steps_done += 1
        
        # Only update if we have enough samples in the replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch from the replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Separate experiences into batches
        states = torch.cat([self.preprocess_state(exp.state) for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], device=self.device).unsqueeze(1)
        rewards = torch.tensor([exp.reward for exp in experiences], device=self.device)
        next_states = torch.cat([self.preprocess_state(exp.next_state) for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32, device=self.device)
        
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Compute target Q values (using double DQN approach)
        with torch.no_grad():
            # Get actions from policy network
            policy_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Get Q-values from target network for those actions
            next_q = self.target_net(next_states).gather(1, policy_actions).squeeze(1)
            # Compute target Q-values
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Save loss for visualization
        self.loss_history.append(loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update epsilon for exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Track exploration rate over time
        self.epsilon_history.append(self.epsilon)
    
    def enable_training(self, enabled: bool = True) -> None:
        """
        Enable or disable training.
        
        Args:
            enabled: Whether to enable training
        """
        self.training_enabled = enabled
    
    def reset(self) -> None:
        """Reset the agent for a new episode."""
        super().reset()
        # Reset tracking for a new episode
        self.epsilon_history = []
        self.q_values_history = []
        self.loss_history = []
        self.latest_q_values = None
    
    def save(self, path: str) -> None:
        """
        Save the agent's models to a file.
        
        Args:
            path: Path to save the models to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the agent's models from a file.
        
        Args:
            path: Path to load the models from
        """
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's performance.
        
        Returns:
            Dictionary with statistics
        """
        stats = super().get_stats()
        stats.update({
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'buffer_size': len(self.replay_buffer),
            'epsilon_history': self.epsilon_history,
            'q_values_history': self.q_values_history,
            'loss_history': self.loss_history,
            'latest_q_values': self.latest_q_values,
            'coop_probability': 1 - self.epsilon if self.q_values_history and self.q_values_history[-1][0] > self.q_values_history[-1][1] else self.epsilon
        })
        return stats 