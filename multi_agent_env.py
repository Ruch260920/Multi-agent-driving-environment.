{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "''' Multi-Agent Driving Environment This module defines a 2D multi-agent driving environment for reinforcement learning. Agents navigate within a bounded space, and their movements are determined by throttle and steering actions. Key Features: - Multi-agent support with vectorized operations - Simple physics-based movement with angle and velocity adjustments - Pygame-based visualization for debugging - Gymnasium-compatible environment '''"
      ],
      "metadata": {
        "id": "ctDHocY7-4Ve"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "oD6oJVqS-y-2"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import gymnasium as gym\n",
        "from gymnasium import spaces\n",
        "import pygame\n",
        "\n",
        "class MultiAgentDrivingEnv(gym.Env):\n",
        "    '''\n",
        "    A Gymnasium-compatible multi-agent driving environment.\n",
        "    '''\n",
        "    def _init_(self, num_agents=5, map_size=(500, 500)):\n",
        "        super(MultiAgentDrivingEnv, self)._init_()\n",
        "        self.num_agents = num_agents\n",
        "        self.map_size = np.array(map_size)\n",
        "\n",
        "        # Define action and observation space\n",
        "        self.action_space = spaces.MultiDiscrete([3, 3] * num_agents)  # [Throttle/Brake, Steering] per agent\n",
        "        self.observation_space = spaces.Box(low=0, high=1, shape=(num_agents, 6), dtype=np.float32)  # Position, velocity, angle\n",
        "\n",
        "        self.reset()\n",
        "\n",
        "        # Pygame rendering setup\n",
        "        self.screen = None\n",
        "        self.clock = None\n",
        "\n",
        "    def reset(self, seed=None, options=None):\n",
        "        '''\n",
        "        Resets the environment to its initial state.\n",
        "        '''\n",
        "        super().reset(seed=seed)\n",
        "        self.agent_positions = np.random.rand(self.num_agents, 2) * self.map_size\n",
        "        self.agent_velocities = np.zeros((self.num_agents, 2))\n",
        "        self.agent_angles = np.zeros(self.num_agents)  # Orientation angles for movement\n",
        "        self.steps = 0\n",
        "        return self._get_observation(), {}\n",
        "\n",
        "    def step(self, actions):\n",
        "        '''\n",
        "        Executes a step in the environment based on the given actions.\n",
        "        '''\n",
        "        self.steps += 1\n",
        "        actions = np.reshape(actions, (self.num_agents, 2))\n",
        "        self._update_agents(actions)\n",
        "\n",
        "        # Compute reward\n",
        "        rewards = self._compute_rewards()\n",
        "        done = self.steps > 200  # Episode ends after 200 steps\n",
        "        return self._get_observation(), rewards, done, False, {}\n",
        "\n",
        "    def _update_agents(self, actions):\n",
        "        '''\n",
        "        Updates agent positions, velocities, and angles based on actions.\n",
        "        '''\n",
        "        for i, (throttle, steering) in enumerate(actions):\n",
        "            self.agent_angles[i] += np.clip((steering - 1) * 0.1, -0.2, 0.2)  # Steering constraint\n",
        "            self.agent_velocities[i] += np.clip((throttle - 1) * 0.1, -0.5, 0.5)  # Velocity constraint\n",
        "            direction = np.array([np.cos(self.agent_angles[i]), np.sin(self.agent_angles[i])])\n",
        "            self.agent_positions[i] += self.agent_velocities[i] * direction\n",
        "            self.agent_positions[i] = np.clip(self.agent_positions[i], [0, 0], self.map_size)  # Keep within bounds\n",
        "\n",
        "    def _compute_rewards(self):\n",
        "        '''\n",
        "        Computes rewards for agents based on speed and other factors.\n",
        "        '''\n",
        "        speed_penalty = -np.linalg.norm(self.agent_velocities, axis=1)  # Penalize low speed\n",
        "        boundary_penalty = -np.any((self.agent_positions <= 5) | (self.agent_positions >= self.map_size - 5), axis=1) * 5  # Penalize boundary collision\n",
        "        return speed_penalty + boundary_penalty\n",
        "\n",
        "    def _get_observation(self):\n",
        "        '''\n",
        "        Returns the current observation state of all agents.\n",
        "        '''\n",
        "        return np.hstack([self.agent_positions, self.agent_velocities, self.agent_angles[:, None]]) / np.hstack([self.map_size, self.map_size, [np.pi]])\n",
        "\n",
        "    def render(self):\n",
        "        '''\n",
        "        Renders the environment using Pygame.\n",
        "        '''\n",
        "        if self.screen is None:\n",
        "            pygame.init()\n",
        "            self.screen = pygame.display.set_mode(self.map_size.astype(int))\n",
        "            self.clock = pygame.time.Clock()\n",
        "        self.screen.fill((255, 255, 255))\n",
        "\n",
        "        for pos in self.agent_positions:\n",
        "            pygame.draw.circle(self.screen, (0, 0, 255), pos.astype(int), 5)\n",
        "\n",
        "        pygame.display.flip()\n",
        "        self.clock.tick(30)\n",
        "\n",
        "    def close(self):\n",
        "        '''\n",
        "        Closes the Pygame rendering window.\n",
        "        '''\n",
        "        if self.screen:\n",
        "            pygame.quit()\n",
        "            self.screen = None\n",
        "            self.clock = None"
      ]
    }
  ]
}