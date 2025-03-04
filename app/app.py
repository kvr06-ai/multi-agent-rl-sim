#!/usr/bin/env python3
"""
Streamlit application for the Prisoner's Dilemma DRL Interactive Demo.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Prisoner's Dilemma DRL Demo",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    # Placeholder for future implementation
    if start_button:
        st.info("Simulation functionality will be implemented in the next phase.")
        
        # For now, show a sample visualization of the payoff matrix
        st.subheader("Payoff Matrix")
        
        # Create sample payoff matrix
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
        
        # Placeholder for future game results
        st.subheader("Game Results (Coming Soon)")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("### Decisions Over Time")
            st.text("Visualization of agent decisions will appear here.")
        
        with cols[1]:
            st.markdown("### Cumulative Rewards")
            st.text("Reward tracking will appear here.")

if __name__ == "__main__":
    main() 