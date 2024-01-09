"""
Chain environment
@author: thomas (MCTS-T+)
"""

import gym
import numpy as np


class Chain(gym.Env):
    """ Chain domain """
    
    def __init__(self, length=10, num_actions=2, loop=False, ordered=False):
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Discrete(length)
        self.length = length
        self.state = 0
        if ordered:
            self.correct = np.zeros(length)
        else:
            self.correct = np.random.randint(0, num_actions, length)
        self.counts = np.zeros((length, num_actions))
        self.terminate_on_loop = not loop

    def reset(self, *args, **kwargs):
        self.state = 0
        return self.state, {}

    def step(self, a):
        self.counts[self.state, a] += 1
        if a == self.correct[self.state]:
            # move forward
            self.state += 1
            terminal = self.state == self.length
            reward = int(terminal)  # 1 for goal, 0 otherwise
        else:
            # move back
            self.state = 0
            reward = 0
            terminal = self.terminate_on_loop
            
        return self.state, reward, terminal, False, {}

    def seed(self, seed):
        pass  # deterministic anyway

