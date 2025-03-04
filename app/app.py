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
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from the environment module
from environment import PrisonersDilemmaEnv, MultiAgentPrisonersDilemmaEnv
from agents.predefined_agents import AlwaysCooperateAgent, AlwaysDefectAgent, TitForTatAgent, RandomAgent
from agents.dqn_agent import DQNAgent

# Set page config
st.set_page_config(
    page_title="Prisoner's Dilemma DRL Demo",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a singleton to store the DQN agents between reruns
if "dqn_agent1" not in st.session_state:
    # For memory_size=1, we get observations with 1 previous action
    # Each action gets encoded as 2 features (one-hot), so state_dim = 2
    st.session_state.dqn_agent1 = DQNAgent(state_dim=2, name="DQN Agent 1", 
                                         epsilon_start=0.3, epsilon_end=0.05)
    
if "dqn_agent2" not in st.session_state:
    st.session_state.dqn_agent2 = DQNAgent(state_dim=2, name="DQN Agent 2", 
                                         epsilon_start=0.3, epsilon_end=0.05)

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

def get_agent(agent_name):
    """
    Get an agent instance based on the agent name.
    
    Args:
        agent_name: Name of the agent type
        
    Returns:
        Agent instance implementing the BaseAgent interface
    """
    if agent_name == "Always Cooperate":
        return AlwaysCooperateAgent()
    elif agent_name == "Always Defect":
        return AlwaysDefectAgent()
    elif agent_name == "Tit-for-Tat":
        return TitForTatAgent()
    elif agent_name == "Random":
        return RandomAgent()
    elif agent_name == "DQN Agent 1":
        return st.session_state.dqn_agent1
    elif agent_name == "DQN Agent 2":
        return st.session_state.dqn_agent2
    else:
        # Default to Always Cooperate
        return AlwaysCooperateAgent()

def run_simulation(env, agent1, agent2, rounds):
    """
    Run a simulation of the Prisoner's Dilemma.
    
    Args:
        env: The environment to use
        agent1: Agent 1 instance
        agent2: Agent 2 instance
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
    
    # Track agent-specific metrics like exploration rate
    results["agent1_metrics"] = {"type": type(agent1).__name__}
    results["agent2_metrics"] = {"type": type(agent2).__name__}
    
    # If agents are DQN agents, initialize epsilon tracking
    if hasattr(agent1, 'epsilon'):
        results["agent1_metrics"]["epsilon_history"] = []
    if hasattr(agent2, 'epsilon'):
        results["agent2_metrics"]["epsilon_history"] = []
    
    # Reset environment and agents
    observations, _ = env.reset()
    agent1.reset()
    agent2.reset()
    
    # Run rounds
    for _ in range(rounds):
        # Record current epsilon if agents are DQN agents
        if hasattr(agent1, 'epsilon'):
            results["agent1_metrics"]["epsilon_history"].append(agent1.epsilon)
        if hasattr(agent2, 'epsilon'):
            results["agent2_metrics"]["epsilon_history"].append(agent2.epsilon)
        
        # Get agent actions
        agent1_action = agent1.act(observations[0])
        agent2_action = agent2.act(observations[1])
        
        # Take step in environment
        next_observations, rewards, terminated, truncated, _ = env.step([agent1_action, agent2_action])
        
        # Update agents
        agent1.update(observations[0], agent1_action, rewards[0], next_observations[0], terminated or truncated)
        agent2.update(observations[1], agent2_action, rewards[1], next_observations[1], terminated or truncated)
        
        # Store results
        results["actions"].append((agent1_action, agent2_action))
        results["rewards"].append((rewards[0], rewards[1]))
        cum_rewards = results["cumulative_rewards"][-1]
        results["cumulative_rewards"].append((cum_rewards[0] + rewards[0], cum_rewards[1] + rewards[1]))
        
        # Update observations
        observations = next_observations
        
        # Check if done
        if terminated or truncated:
            break
    
    # Final recording of agent metrics
    if hasattr(agent1, 'epsilon'):
        results["agent1_metrics"]["final_epsilon"] = agent1.epsilon
    if hasattr(agent2, 'epsilon'):
        results["agent2_metrics"]["final_epsilon"] = agent2.epsilon
    
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

def plot_exploration_rates(results):
    """
    Plot the exploration rates over time for DQN agents.
    
    Args:
        results: Dictionary with simulation results
        
    Returns:
        Matplotlib figure with exploration rate plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot agent 1 epsilon history if available
    if "agent1_metrics" in results and "epsilon_history" in results["agent1_metrics"]:
        epsilon_history = results["agent1_metrics"]["epsilon_history"]
        if epsilon_history:
            ax.plot(range(len(epsilon_history)), epsilon_history, label="Agent 1 Exploration Rate", color="#1f77b4", marker="o")
    
    # Plot agent 2 epsilon history if available
    if "agent2_metrics" in results and "epsilon_history" in results["agent2_metrics"]:
        epsilon_history = results["agent2_metrics"]["epsilon_history"]
        if epsilon_history:
            ax.plot(range(len(epsilon_history)), epsilon_history, label="Agent 2 Exploration Rate", color="#ff7f0e", marker="x")
    
    # Only add legend if we plotted exploration rates
    if ax.has_data():
        ax.legend()
        ax.set_xlabel("Round")
        ax.set_ylabel("Exploration Rate (ε)")
        ax.set_title("Agent Exploration Rates Over Time")
        ax.grid(True)
        ax.set_ylim(bottom=0)  # Start y-axis at 0
    else:
        fig.text(0.5, 0.5, "No exploration rate data available", ha="center", va="center", fontsize=12)
    
    return fig

def plot_q_values(results):
    """
    Plot the Q-values over time for DQN agents.
    
    Args:
        results: Dictionary with simulation results
        
    Returns:
        Matplotlib figure with Q-values plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot agent 1 Q-values if available
    if "agent1_metrics" in results and "q_values_history" in results["agent1_metrics"]:
        q_values = results["agent1_metrics"]["q_values_history"]
        if q_values:
            rounds = range(len(q_values))
            q_coop = [q[0] for q in q_values]  # Q-value for cooperation
            q_defect = [q[1] for q in q_values]  # Q-value for defection
            
            ax.plot(rounds, q_coop, label="Agent 1 Q(Cooperate)", color="#1f77b4", linestyle="-")
            ax.plot(rounds, q_defect, label="Agent 1 Q(Defect)", color="#1f77b4", linestyle="--")
    
    # Plot agent 2 Q-values if available
    if "agent2_metrics" in results and "q_values_history" in results["agent2_metrics"]:
        q_values = results["agent2_metrics"]["q_values_history"]
        if q_values:
            rounds = range(len(q_values))
            q_coop = [q[0] for q in q_values]  # Q-value for cooperation
            q_defect = [q[1] for q in q_values]  # Q-value for defection
            
            ax.plot(rounds, q_coop, label="Agent 2 Q(Cooperate)", color="#ff7f0e", linestyle="-")
            ax.plot(rounds, q_defect, label="Agent 2 Q(Defect)", color="#ff7f0e", linestyle="--")
    
    # Only add legend if we plotted Q-values
    if ax.has_data():
        ax.legend()
        ax.set_xlabel("Round")
        ax.set_ylabel("Q-Value")
        ax.set_title("Q-Values Over Time")
        ax.grid(True)
    else:
        fig.text(0.5, 0.5, "No Q-value data available", ha="center", va="center", fontsize=12)
    
    return fig

def plot_policy_distribution(results):
    """
    Visualize the agents' policy distribution (probability of actions) over time.
    
    Args:
        results: Dictionary with simulation results
        
    Returns:
        Matplotlib figure with policy distribution plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Analyze actions
    if "actions" in results and results["actions"]:
        actions = np.array(results["actions"])
        rounds = range(1, len(actions) + 1)
        
        # Use a rolling window to calculate cooperation probability
        window_size = min(10, len(actions))
        agent1_coop_prob = []
        agent2_coop_prob = []
        
        for i in range(len(actions)):
            start_idx = max(0, i - window_size + 1)
            window = actions[start_idx:i+1]
            agent1_coop_prob.append(np.mean(window[:, 0] == 0))
            agent2_coop_prob.append(np.mean(window[:, 1] == 0))
        
        ax.plot(rounds, agent1_coop_prob, label="Agent 1 P(Cooperate)", color="#1f77b4")
        ax.plot(rounds, agent2_coop_prob, label="Agent 2 P(Cooperate)", color="#ff7f0e")
        
        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Round")
        ax.set_ylabel("Probability of Cooperation")
        ax.set_title("Policy Distribution Over Time")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
    else:
        fig.text(0.5, 0.5, "No action data available", ha="center", va="center", fontsize=12)
    
    return fig

def plot_learning_curve(agent_type, session_key, title, color="#1f77b4"):
    """
    Plot the learning curve across multiple simulations for a DQN agent.
    
    Args:
        agent_type: The agent type to plot for
        session_key: The session state key for tracking cumulative rewards
        title: Title for the plot
        color: Color for the plot line
        
    Returns:
        Matplotlib figure with learning curve
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get simulation history from session state
    if session_key in st.session_state:
        history = st.session_state[session_key]
        simulations = range(1, len(history) + 1)
        
        ax.plot(simulations, history, marker='o', color=color)
        ax.set_xlabel("Simulation")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(title)
        ax.grid(True)
    else:
        fig.text(0.5, 0.5, "No simulation history available", ha="center", va="center", fontsize=12)
    
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
        agent1_type = st.selectbox("Agent 1", 
                             ["Always Cooperate", "Always Defect", "Tit-for-Tat", "Random", "DQN Agent 1"], 
                             index=2)
        agent2_type = st.selectbox("Agent 2", 
                             ["Always Cooperate", "Always Defect", "Tit-for-Tat", "Random", "DQN Agent 2"], 
                             index=4)
        
        # DQN training control
        if agent1_type == "DQN Agent 1" or agent2_type == "DQN Agent 2":
            st.subheader("DQN Settings")
            enable_training = st.checkbox("Enable DQN Training", value=True,
                                         help="When enabled, DQN agents will learn from experiences")
            
            if enable_training:
                # Apply training settings
                if agent1_type == "DQN Agent 1":
                    st.session_state.dqn_agent1.enable_training(True)
                if agent2_type == "DQN Agent 2":
                    st.session_state.dqn_agent2.enable_training(True)
                
                # Show exploration rate (epsilon)
                if agent1_type == "DQN Agent 1":
                    st.text(f"DQN Agent 1 epsilon: {st.session_state.dqn_agent1.epsilon:.3f}")
                if agent2_type == "DQN Agent 2":
                    st.text(f"DQN Agent 2 epsilon: {st.session_state.dqn_agent2.epsilon:.3f}")
            else:
                # Disable training
                if agent1_type == "DQN Agent 1":
                    st.session_state.dqn_agent1.enable_training(False)
                if agent2_type == "DQN Agent 2":
                    st.session_state.dqn_agent2.enable_training(False)
        
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
    
    # Initialize session state for tracking rewards across simulations
    if "dqn_agent1_rewards" not in st.session_state:
        st.session_state.dqn_agent1_rewards = []
    
    if "dqn_agent2_rewards" not in st.session_state:
        st.session_state.dqn_agent2_rewards = []
    
    # Run simulation when button is clicked
    if start_button:
        # Create environment
        payoff_matrix = create_payoff_matrix(reward, temptation, sucker, punishment)
        env = MultiAgentPrisonersDilemmaEnv(
            payoff_matrix=payoff_matrix,
            memory_size=memory_size,
            max_steps=rounds
        )
        
        # Get agent instances
        agent1 = get_agent(agent1_type)
        agent2 = get_agent(agent2_type)
        
        # Run simulation
        with st.spinner("Running simulation..."):
            results = run_simulation(env, agent1, agent2, rounds)
        
        # Store cumulative rewards for learning curve plotting
        final_cum_rewards = results["cumulative_rewards"][-1]
        if agent1_type == "DQN Agent 1":
            st.session_state.dqn_agent1_rewards.append(final_cum_rewards[0])
        
        if agent2_type == "DQN Agent 2":
            st.session_state.dqn_agent2_rewards.append(final_cum_rewards[1])
        
        # Display results
        st.subheader("Simulation Results")
        
        # Agent decisions over time
        st.write("#### Agent Decisions Over Time")
        decisions_fig = plot_decisions(results)
        st.pyplot(decisions_fig)
        
        # Policy distribution over time
        st.write("#### Policy Distribution Over Time")
        policy_fig = plot_policy_distribution(results)
        st.pyplot(policy_fig)
        
        # Rewards over time
        st.write("#### Rewards Over Time")
        rewards_fig = plot_rewards(results)
        st.pyplot(rewards_fig)
        
        # Exploration rates over time (only show if at least one DQN agent is used)
        if (agent1_type.startswith("DQN") or agent2_type.startswith("DQN")):
            st.write("#### Exploration Rates Over Time")
            exploration_fig = plot_exploration_rates(results)
            st.pyplot(exploration_fig)
            
            # Q-values over time
            st.write("#### Q-Values Over Time")
            q_values_fig = plot_q_values(results)
            st.pyplot(q_values_fig)
            
            # Learning curves across simulations
            st.write("#### Learning Curves (Across Simulations)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if agent1_type == "DQN Agent 1" and len(st.session_state.dqn_agent1_rewards) > 0:
                    learning_curve1 = plot_learning_curve(
                        "DQN Agent 1", 
                        "dqn_agent1_rewards", 
                        "Agent 1 Learning Curve", 
                        "#1f77b4"
                    )
                    st.pyplot(learning_curve1)
            
            with col2:
                if agent2_type == "DQN Agent 2" and len(st.session_state.dqn_agent2_rewards) > 0:
                    learning_curve2 = plot_learning_curve(
                        "DQN Agent 2", 
                        "dqn_agent2_rewards", 
                        "Agent 2 Learning Curve", 
                        "#ff7f0e"
                    )
                    st.pyplot(learning_curve2)
        
        # Final statistics
        st.write("#### Final Statistics")
        final_cum_rewards = results["cumulative_rewards"][-1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Agent 1 Total Reward", f"{final_cum_rewards[0]:.1f}")
            
            # If DQN, show additional stats
            if agent1_type == "DQN Agent 1":
                agent1_stats = st.session_state.dqn_agent1.get_stats()
                st.write(f"Exploration rate (ε): {agent1_stats['epsilon']:.3f}")
                st.write(f"Training steps: {agent1_stats['steps_done']}")
                st.write(f"Buffer size: {agent1_stats['buffer_size']}")
                
                if 'latest_q_values' in agent1_stats:
                    q_vals = agent1_stats['latest_q_values']
                    st.write(f"Latest Q-values: Cooperate={q_vals[0]:.3f}, Defect={q_vals[1]:.3f}")
        
        with col2:
            st.metric("Agent 2 Total Reward", f"{final_cum_rewards[1]:.1f}")
            
            # If DQN, show additional stats
            if agent2_type == "DQN Agent 2":
                agent2_stats = st.session_state.dqn_agent2.get_stats()
                st.write(f"Exploration rate (ε): {agent2_stats['epsilon']:.3f}")
                st.write(f"Training steps: {agent2_stats['steps_done']}")
                st.write(f"Buffer size: {agent2_stats['buffer_size']}")
                
                if 'latest_q_values' in agent2_stats:
                    q_vals = agent2_stats['latest_q_values']
                    st.write(f"Latest Q-values: Cooperate={q_vals[0]:.3f}, Defect={q_vals[1]:.3f}")
        
        # Action frequencies
        actions_arr = np.array(results["actions"])
        if len(actions_arr) > 0:
            st.write("#### Action Frequencies")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Agent 1:")
                a1_coop_rate = np.mean(actions_arr[:, 0] == 0) * 100
                a1_defect_rate = 100 - a1_coop_rate
                st.write(f"- Cooperate: {a1_coop_rate:.1f}%")
                st.write(f"- Defect: {a1_defect_rate:.1f}%")
            
            with col2:
                st.write("Agent 2:")
                a2_coop_rate = np.mean(actions_arr[:, 1] == 0) * 100
                a2_defect_rate = 100 - a2_coop_rate
                st.write(f"- Cooperate: {a2_coop_rate:.1f}%")
                st.write(f"- Defect: {a2_defect_rate:.1f}%")

if __name__ == "__main__":
    main() 