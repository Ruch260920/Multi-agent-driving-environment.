
''' Multi-Agent Driving Environment This module defines a 2D multi-agent driving environment for reinforcement learning. Agents navigate within a bounded space, and their movements are determined by throttle and steering actions. Key Features: - Multi-agent support with vectorized operations - Simple physics-based movement with angle and velocity adjustments - Pygame-based visualization for debugging - Gymnasium-compatible environment '''
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

class MultiAgentDrivingEnv(gym.Env):
    '''
    A Gymnasium-compatible multi-agent driving environment.
    '''
    def _init_(self, num_agents=5, map_size=(500, 500)):
        super(MultiAgentDrivingEnv, self)._init_()
        self.num_agents = num_agents
        self.map_size = np.array(map_size)

        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([3, 3] * num_agents)  # [Throttle/Brake, Steering] per agent
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_agents, 6), dtype=np.float32)  # Position, velocity, angle

        self.reset()

        # Pygame rendering setup
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        '''
        Resets the environment to its initial state.
        '''
        super().reset(seed=seed)
        self.agent_positions = np.random.rand(self.num_agents, 2) * self.map_size
        self.agent_velocities = np.zeros((self.num_agents, 2))
        self.agent_angles = np.zeros(self.num_agents)  # Orientation angles for movement
        self.steps = 0
        return self._get_observation(), {}

    def step(self, actions):
        '''
        Executes a step in the environment based on the given actions.
        '''
        self.steps += 1
        actions = np.reshape(actions, (self.num_agents, 2))
        self._update_agents(actions)

        # Compute reward
        rewards = self._compute_rewards()
        done = self.steps > 200  # Episode ends after 200 steps
        return self._get_observation(), rewards, done, False, {}

    def _update_agents(self, actions):
        '''
        Updates agent positions, velocities, and angles based on actions.
        '''
        for i, (throttle, steering) in enumerate(actions):
            self.agent_angles[i] += np.clip((steering - 1) * 0.1, -0.2, 0.2)  # Steering constraint
            self.agent_velocities[i] += np.clip((throttle - 1) * 0.1, -0.5, 0.5)  # Velocity constraint
            direction = np.array([np.cos(self.agent_angles[i]), np.sin(self.agent_angles[i])])
            self.agent_positions[i] += self.agent_velocities[i] * direction
            self.agent_positions[i] = np.clip(self.agent_positions[i], [0, 0], self.map_size)  # Keep within bounds

    def _compute_rewards(self):
        '''
        Computes rewards for agents based on speed and other factors.
        '''
        speed_penalty = -np.linalg.norm(self.agent_velocities, axis=1)  # Penalize low speed
        boundary_penalty = -np.any((self.agent_positions <= 5) | (self.agent_positions >= self.map_size - 5), axis=1) * 5  # Penalize boundary collision
        return speed_penalty + boundary_penalty

    def _get_observation(self):
        '''
        Returns the current observation state of all agents.
        '''
        return np.hstack([self.agent_positions, self.agent_velocities, self.agent_angles[:, None]]) / np.hstack([self.map_size, self.map_size, [np.pi]])

    def render(self):
        '''
        Renders the environment using Pygame.
        '''
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.map_size.astype(int))
            self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))

        for pos in self.agent_positions:
            pygame.draw.circle(self.screen, (0, 0, 255), pos.astype(int), 5)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        '''
        Closes the Pygame rendering window.
        '''
        if self.screen:
            pygame.quit()
            self.screen = None
            self.clock = None
