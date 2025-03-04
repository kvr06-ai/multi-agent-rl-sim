#!/usr/bin/env python3
"""
Example script demonstrating the Prisoner's Dilemma environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import PrisonersDilemmaEnv, MultiAgentPrisonersDilemmaEnv

def run_random_agent_example(episodes=5, steps=10):
    """
    Run an example with a random agent playing the Prisoner's Dilemma.
    
    Args:
        episodes: Number of episodes to run
        steps: Number of steps per episode
    """
    print("\n=== Random Agent Example ===")
    
    # Create environment
    env = PrisonersDilemmaEnv(memory_size=1, max_steps=steps)
    
    total_rewards = []
    
    # Run multiple episodes
    for episode in range(episodes):
        # Reset environment
        observation, info = env.reset()
        
        episode_reward = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        # Run steps
        for step in range(steps):
            # Choose random action
            action = np.random.choice([0, 1])
            
            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Update total reward
            episode_reward += reward
            
            # Print step information
            print(f"  Step {step + 1}: Action={action}, Reward={reward}")
            
            if terminated or truncated:
                break
        
        # Print episode information
        print(f"  Episode reward: {episode_reward}")
        total_rewards.append(episode_reward)
    
    print(f"\nAverage reward over {episodes} episodes: {np.mean(total_rewards):.2f}")

def run_multi_agent_example(episodes=3, steps=5):
    """
    Run an example with two agents playing the Prisoner's Dilemma.
    
    Args:
        episodes: Number of episodes to run
        steps: Number of steps per episode
    """
    print("\n=== Multi-Agent Example ===")
    
    # Create environment
    env = MultiAgentPrisonersDilemmaEnv(memory_size=1, max_steps=steps)
    
    # Define simple agent strategies
    strategies = {
        "always_cooperate": lambda obs: 0,
        "always_defect": lambda obs: 1,
        "tit_for_tat": lambda obs: obs[0] if obs[0] >= 0 else 0,
        "random": lambda obs: np.random.choice([0, 1])
    }
    
    # Get all combinations of agent pairs
    agent_pairs = []
    for agent1 in strategies.keys():
        for agent2 in strategies.keys():
            agent_pairs.append((agent1, agent2))
    
    # Run all combinations
    results = {}
    
    for agent1_name, agent2_name in agent_pairs:
        print(f"\nRunning: {agent1_name} vs {agent2_name}")
        
        agent1_strategy = strategies[agent1_name]
        agent2_strategy = strategies[agent2_name]
        
        agent1_total_reward = 0
        agent2_total_reward = 0
        
        # Run multiple episodes
        for episode in range(episodes):
            # Reset environment
            observations, info = env.reset()
            
            # Run steps
            for step in range(steps):
                # Get agent actions
                agent1_action = agent1_strategy(observations[0])
                agent2_action = agent2_strategy(observations[1])
                
                # Take step in environment
                observations, rewards, terminated, truncated, info = env.step([agent1_action, agent2_action])
                
                # Update total rewards
                agent1_total_reward += rewards[0]
                agent2_total_reward += rewards[1]
                
                # Print step information
                print(f"  Step {step + 1}: Actions=({agent1_action}, {agent2_action}), Rewards=({rewards[0]}, {rewards[1]})")
                
                if terminated or truncated:
                    break
        
        # Store average results
        avg_agent1_reward = agent1_total_reward / (episodes * steps)
        avg_agent2_reward = agent2_total_reward / (episodes * steps)
        
        results[(agent1_name, agent2_name)] = (avg_agent1_reward, avg_agent2_reward)
        
        print(f"  Average rewards: {agent1_name}={avg_agent1_reward:.2f}, {agent2_name}={avg_agent2_reward:.2f}")
    
    # Return results for analysis
    return results

def visualize_results(results):
    """
    Visualize the results of multi-agent simulations.
    
    Args:
        results: Dictionary of results from run_multi_agent_example
    """
    # Prepare data
    agent_names = sorted(set([name for pair in results.keys() for name in pair]))
    
    # Create matrix of average rewards
    matrix = np.zeros((len(agent_names), len(agent_names)))
    
    for i, agent1 in enumerate(agent_names):
        for j, agent2 in enumerate(agent_names):
            if (agent1, agent2) in results:
                matrix[i, j] = results[(agent1, agent2)][0]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation='nearest', cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Average Reward')
    
    # Add labels
    plt.xticks(range(len(agent_names)), agent_names, rotation=45)
    plt.yticks(range(len(agent_names)), agent_names)
    
    plt.xlabel('Agent 2 Strategy')
    plt.ylabel('Agent 1 Strategy')
    plt.title('Average Rewards for Agent 1 in Prisoner\'s Dilemma')
    
    plt.tight_layout()
    plt.savefig('prisoner_dilemma_results.png')
    plt.close()

def main():
    """Main function."""
    # Run examples
    run_random_agent_example()
    results = run_multi_agent_example()
    
    # Visualize results
    visualize_results(results)
    
    print("\nResults visualization saved to 'prisoner_dilemma_results.png'")

if __name__ == "__main__":
    main() 