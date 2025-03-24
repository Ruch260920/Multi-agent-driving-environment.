# 🚗 Multi-Agent Driving Environment

This project provides a simple **multi-agent driving simulation environment** built using **Gymnasium** and **Pygame**. It is designed for reinforcement learning experiments involving multiple autonomous agents navigating a 2D space.

## 📦 Features

- Multi-agent support with customizable number of agents
- 2D simulation of agents with position, velocity, and orientation
- Continuous updates of motion dynamics
- Visual rendering using Pygame
- Gym-compatible for easy integration with RL algorithms

## 📁 Structure

- `MultiAgentDrivingEnv`: A Gym-compatible environment
- Actions: Each agent can **Throttle/Brake** and **Steer**
- Observations: Position, velocity, and angle (normalized)
- Reward: Encourages movement, penalizes boundary collisions

## 📷 Demo

🧾 Requirements
Python 3.8+

Gymnasium

Pygame

NumPy

See requirements.txt for exact versions.

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/multi-agent-driving-env.git
cd multi-agent-driving-env

2. Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt

 ## 🧠 Usage
python
import gymnasium as gym
from multi_agent_env import MultiAgentDrivingEnv

env = MultiAgentDrivingEnv(num_agents=5)
obs, _ = env.reset()

for _ in range(200):
    actions = env.action_space.sample()
    obs, reward, done, _, _ = env.step(actions)
    env.render()
    if done:
        break

env.close()
