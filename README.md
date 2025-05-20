# Reinforcement Learning - GridWorld with DQN Agent

This project implements a simple reinforcement learning environment called GridWorld. A Deep Q-Network (DQN) agent learns to navigate this grid to reach a specified goal while avoiding obstacles.

The primary purpose of this project is to provide a foundational example for understanding:
- Basic reinforcement learning concepts (states, actions, rewards, episodes).
- The structure of an RL agent.
- How a simple neural network (as part of DQN) can be used to learn a policy.
- The Deep Q-Learning algorithm with experience replay.

## Environment
- **GridWorld**: A customizable grid (default 20x20) where the agent operates.
- **Agent (A)**: Starts at a defined position.
- **Goal (G)**: A target position the agent tries to reach.
- **Obstacles (X)**: Locations the agent must avoid.

## Agent
- **DQN Agent**: Uses a neural network to approximate Q-values for state-action pairs.
- **Learning**: Employs epsilon-greedy strategy for exploration/exploitation and experience replay for training efficiency.

## Running
(Instructions to be added once `main.py` is implemented)

## Dependencies
- Python 3.x
- NumPy
- TensorFlow
