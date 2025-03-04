"""
Test cases for the Prisoner's Dilemma environment.
"""

import unittest
import numpy as np
from environment.pd_env import PrisonersDilemmaEnv, MultiAgentPrisonersDilemmaEnv

class TestPrisonersDilemmaEnv(unittest.TestCase):
    """Test cases for the PrisonersDilemmaEnv class."""
    
    def setUp(self):
        """Set up the test case."""
        self.env = PrisonersDilemmaEnv(memory_size=3, max_steps=10)
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.memory_size, 3)
        self.assertEqual(self.env.max_steps, 10)
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(len(self.env.memory), 3)
        self.assertTrue(np.all(self.env.memory == -1))
    
    def test_step(self):
        """Test environment step."""
        # Take a step with action 0 (cooperate)
        obs, reward, terminated, truncated, info = self.env.step(0)
        
        # Check that observation is the right shape
        self.assertEqual(obs.shape, (3,))
        
        # Check that the step counter incremented
        self.assertEqual(self.env.step_count, 1)
        
        # Check that the history was updated
        self.assertEqual(len(self.env.history["actions"]), 1)
        self.assertEqual(len(self.env.history["rewards"]), 1)
    
    def test_reset(self):
        """Test environment reset."""
        # Take a few steps
        self.env.step(0)
        self.env.step(1)
        
        # Reset the environment
        obs, info = self.env.reset()
        
        # Check that everything was reset
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(len(self.env.history["actions"]), 0)
        self.assertEqual(len(self.env.history["rewards"]), 0)
        self.assertTrue(np.all(self.env.memory == -1))

class TestMultiAgentPrisonersDilemmaEnv(unittest.TestCase):
    """Test cases for the MultiAgentPrisonersDilemmaEnv class."""
    
    def setUp(self):
        """Set up the test case."""
        self.env = MultiAgentPrisonersDilemmaEnv(memory_size=2, max_steps=5)
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.max_steps, 5)
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(len(self.env.history["actions"]), 0)
        self.assertEqual(len(self.env.history["rewards"]), 0)
    
    def test_step(self):
        """Test environment step."""
        # Reset the environment
        obs, info = self.env.reset()
        
        # Check that observations are the right shape
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0].shape, (2,))
        self.assertEqual(obs[1].shape, (2,))
        
        # Take a step with both agents cooperating
        new_obs, rewards, terminated, truncated, info = self.env.step([0, 0])
        
        # Check that observations are updated
        self.assertEqual(len(new_obs), 2)
        
        # Check that rewards are correct for mutual cooperation
        self.assertEqual(rewards[0], 3)
        self.assertEqual(rewards[1], 3)
        
        # Check that the step counter incremented
        self.assertEqual(self.env.step_count, 1)
        
        # Check that the history was updated
        self.assertEqual(len(self.env.history["actions"]), 1)
        self.assertEqual(len(self.env.history["rewards"]), 1)
        
        # Test defection scenario
        new_obs, rewards, terminated, truncated, info = self.env.step([1, 0])
        
        # Check that rewards are correct when agent1 defects and agent2 cooperates
        self.assertEqual(rewards[0], 5)
        self.assertEqual(rewards[1], 0)

if __name__ == "__main__":
    unittest.main() 