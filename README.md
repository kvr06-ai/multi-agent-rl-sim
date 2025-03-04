# Prisoner's Dilemma DRL Interactive Demo

An interactive demonstration of the Prisoner's Dilemma game using Deep Reinforcement Learning (DRL) techniques.

## Overview

This project implements an interactive simulation of the classic Prisoner's Dilemma, a fundamental game theory scenario. Users can explore strategic interactions between agents, observe the emergence of cooperation or defection strategies, and understand key game-theoretic concepts through a visually engaging interface.

## Features

- Interactive web-based UI built with Streamlit
- Configurable game parameters (payoff values, number of rounds)
- Multiple agent types:
  - Fixed-strategy agents (Always Cooperate, Always Defect, Tit-for-Tat, etc.)
  - Learning agents powered by deep reinforcement learning algorithms
- Real-time visualization of game outcomes and agent learning
- Educational explanations of game theory concepts

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   streamlit run app/app.py
   ```

## Project Structure

- `app/`: Streamlit application and UI components
- `agents/`: Implementation of various agent types
- `environment/`: Prisoner's Dilemma game environment
- `visualizations/`: Visualization tools and components
- `models/`: Saved model weights for pre-trained agents

## License

MIT 