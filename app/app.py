#!/usr/bin/env python3
"""
Streamlit application for the Prisoner's Dilemma DRL Interactive Demo.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from the environment module
from environment import PrisonersDilemmaEnv, MultiAgentPrisonersDilemmaEnv

# Set page config
st.set_page_config(
    page_title="Prisoner's Dilemma DRL Demo",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_payoff_matrix(reward, temptation, sucker, punishment):
    """
    Create a payoff matrix for the Prisoner's Dilemma.
    
    Args:
        reward: Reward for mutual cooperation
        temptation: Temptation to defect
        sucker: Sucker's payoff
        punishment: Punishment for mutual defection
        
    Returns:
        Numpy array with payoff matrix
    """
    return np.array([
        [(reward, reward), (sucker, temptation)],  # Row player cooperates
        [(temptation, sucker), (punishment, punishment)]  # Row player defects
    ])

def get_agent_strategy(agent_name):
    """
    Get a function that implements the specified agent strategy.
    
    Args:
        agent_name: Name of the agent strategy
        
    Returns:
        Function that takes an observation and returns an action
    """
    if agent_name == "Always Cooperate":
        return lambda obs: 0
    elif agent_name == "Always Defect":
        return lambda obs: 1
    elif agent_name == "Tit-for-Tat":
        return lambda obs: obs[0] if obs[0] >= 0 else 0
    elif agent_name == "Random":
        return lambda obs: np.random.choice([0, 1])
    elif agent_name == "DQN Agent":
        # Placeholder for future DQN implementation
        # For now, use a slightly smarter random strategy
        return lambda obs: 0 if np.random.random() < 0.7 else 1
    else:
        return lambda obs: 0  # Default to always cooperate

def run_simulation(env, agent1_strategy, agent2_strategy, rounds):
    """
    Run a simulation of the Prisoner's Dilemma.
    
    Args:
        env: The environment to use
        agent1_strategy: Strategy function for agent 1
        agent2_strategy: Strategy function for agent 2
        rounds: Number of rounds to run
        
    Returns:
        Dictionary with simulation results
    """
    # Initialize results
    results = {
        "actions": [],  # List of (agent1_action, agent2_action) pairs
        "rewards": [],  # List of (agent1_reward, agent2_reward) pairs
        "cumulative_rewards": [(0, 0)],  # Cumulative rewards for each agent
    }
    
    # Reset environment
    observations, _ = env.reset()
    
    # Run rounds
    for _ in range(rounds):
        # Get agent actions
        agent1_action = agent1_strategy(observations[0])
        agent2_action = agent2_strategy(observations[1])
        
        # Take step in environment
        observations, rewards, terminated, truncated, _ = env.step([agent1_action, agent2_action])
        
        # Record results
        results["actions"].append((agent1_action, agent2_action))
        results["rewards"].append((rewards[0], rewards[1]))
        
        # Update cumulative rewards
        last_cum_reward = results["cumulative_rewards"][-1]
        new_cum_reward = (last_cum_reward[0] + rewards[0], last_cum_reward[1] + rewards[1])
        results["cumulative_rewards"].append(new_cum_reward)
        
        # Check if episode is done
        if terminated or truncated:
            break
    
    return results

def plot_decisions(results):
    """
    Plot the decisions made by the agents over time.
    
    Args:
        results: Simulation results
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    rounds = range(1, len(results["actions"]) + 1)
    agent1_actions = [action[0] for action in results["actions"]]
    agent2_actions = [action[1] for action in results["actions"]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot actions (0 = cooperate, 1 = defect)
    ax.plot(rounds, agent1_actions, 'b-', label="Agent 1")
    ax.plot(rounds, agent2_actions, 'r-', label="Agent 2")
    
    # Add legend and labels
    ax.legend()
    ax.set_xlabel("Round")
    ax.set_ylabel("Action (0=Cooperate, 1=Defect)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Cooperate", "Defect"])
    ax.set_title("Agent Decisions Over Time")
    ax.grid(True)
    
    return fig

def plot_rewards(results):
    """
    Plot the cumulative rewards for the agents.
    
    Args:
        results: Simulation results
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    rounds = range(len(results["cumulative_rewards"]))
    agent1_cum_rewards = [reward[0] for reward in results["cumulative_rewards"]]
    agent2_cum_rewards = [reward[1] for reward in results["cumulative_rewards"]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot cumulative rewards
    ax.plot(rounds, agent1_cum_rewards, 'b-', label="Agent 1")
    ax.plot(rounds, agent2_cum_rewards, 'r-', label="Agent 2")
    
    # Add legend and labels
    ax.legend()
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Rewards Over Time")
    ax.grid(True)
    
    return fig

def main():
    """Main application function."""
    # Title and introduction
    st.title("Prisoner's Dilemma Interactive Demo")
    
    st.markdown("""
    ## 🎮 Explore Game Theory with Reinforcement Learning
    
    This interactive demo allows you to explore the famous **Prisoner's Dilemma** and 
    observe how different strategies perform, including those learned through 
    deep reinforcement learning.
    
    ### What is the Prisoner's Dilemma?
    
    The Prisoner's Dilemma is a classic game theory scenario that demonstrates why 
    two completely rational individuals might not cooperate, even when it appears 
    in their best interests to do so.
    """)
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Game Parameters")
        
        # Payoff matrix values
        st.subheader("Payoff Matrix")
        col1, col2 = st.columns(2)
        with col1:
            temptation = st.slider("Temptation (T)", 0, 10, 5, 
                                  help="Payoff for defecting while other cooperates")
            punishment = st.slider("Punishment (P)", 0, 10, 1, 
                                  help="Payoff when both defect")
        with col2:
            reward = st.slider("Reward (R)", 0, 10, 3, 
                              help="Payoff when both cooperate")
            sucker = st.slider("Sucker's payoff (S)", 0, 10, 0, 
                              help="Payoff for cooperating while other defects")
        
        # Check if it's a valid Prisoner's Dilemma
        is_valid_pd = (temptation > reward > punishment > sucker) and (2*reward > temptation + sucker)
        if not is_valid_pd:
            st.warning("⚠️ These values don't represent a valid Prisoner's Dilemma. " +
                      "For a valid PD: T > R > P > S and 2R > T + S")
        
        # Number of rounds
        rounds = st.slider("Number of rounds", 1, 200, 50)
        
        # Memory size
        memory_size = st.slider("Agent memory size", 1, 10, 1,
                              help="Number of previous actions agents can remember")
        
        # Agent selection
        st.subheader("Agents")
        agent1 = st.selectbox("Agent 1", 
                             ["Always Cooperate", "Always Defect", "Tit-for-Tat", "Random", "DQN Agent"], 
                             index=2)
        agent2 = st.selectbox("Agent 2", 
                             ["Always Cooperate", "Always Defect", "Tit-for-Tat", "Random", "DQN Agent"], 
                             index=4)
        
        # Run button
        start_button = st.button("Start Simulation")
    
    # Main content area
    # Payoff matrix
    st.subheader("Payoff Matrix")
    
    # Create payoff matrix for display
    matrix_data = [
        [f"({reward}, {reward})", f"({sucker}, {temptation})"],
        [f"({temptation}, {sucker})", f"({punishment}, {punishment})"]
    ]
    
    df = pd.DataFrame(
        matrix_data,
        columns=["Agent 2: Cooperate", "Agent 2: Defect"],
        index=["Agent 1: Cooperate", "Agent 1: Defect"]
    )
    
    st.table(df)
    
    # Run simulation when button is clicked
    if start_button:
        # Create environment
        payoff_matrix = create_payoff_matrix(reward, temptation, sucker, punishment)
        env = MultiAgentPrisonersDilemmaEnv(
            payoff_matrix=payoff_matrix,
            memory_size=memory_size,
            max_steps=rounds
        )
        
        # Get agent strategies
        agent1_strategy = get_agent_strategy(agent1)
        agent2_strategy = get_agent_strategy(agent2)
        
        # Run simulation
        with st.spinner("Running simulation..."):
            results = run_simulation(env, agent1_strategy, agent2_strategy, rounds)
        
        # Show simulation statistics
        st.subheader("Simulation Results")
        
        # Calculate statistics
        agent1_coop = sum(1 for action in results["actions"] if action[0] == 0)
        agent2_coop = sum(1 for action in results["actions"] if action[1] == 0)
        
        agent1_defect = sum(1 for action in results["actions"] if action[0] == 1)
        agent2_defect = sum(1 for action in results["actions"] if action[1] == 1)
        
        mutual_coop = sum(1 for action in results["actions"] if action[0] == 0 and action[1] == 0)
        mutual_defect = sum(1 for action in results["actions"] if action[0] == 1 and action[1] == 1)
        
        agent1_final_reward = results["cumulative_rewards"][-1][0]
        agent2_final_reward = results["cumulative_rewards"][-1][1]
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{agent1} (Agent 1)", 
                     f"{agent1_final_reward} points",
                     f"{agent1_coop} cooperations")
        
        with col2:
            st.metric(f"{agent2} (Agent 2)", 
                     f"{agent2_final_reward} points",
                     f"{agent2_coop} cooperations")
        
        with col3:
            st.metric("Game Statistics", 
                     f"{rounds} rounds",
                     f"{mutual_coop} mutual cooperations")
        
        # Plot results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Decisions Over Time")
            decisions_fig = plot_decisions(results)
            st.pyplot(decisions_fig)
        
        with col2:
            st.markdown("### Cumulative Rewards")
            rewards_fig = plot_rewards(results)
            st.pyplot(rewards_fig)
        
        # Display action history
        with st.expander("View detailed action history"):
            # Create a dataframe for the action history
            action_history = pd.DataFrame({
                "Round": range(1, len(results["actions"]) + 1),
                "Agent 1 Action": ["Cooperate" if a[0] == 0 else "Defect" for a in results["actions"]],
                "Agent 2 Action": ["Cooperate" if a[1] == 0 else "Defect" for a in results["actions"]],
                "Agent 1 Reward": [r[0] for r in results["rewards"]],
                "Agent 2 Reward": [r[1] for r in results["rewards"]],
            })
            
            st.dataframe(action_history)

if __name__ == "__main__":
    main() 