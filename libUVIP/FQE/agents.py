import numpy as np
import random

class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.actions_list = list(range(self.n_actions))

    def policy(self, state):
        return random.choice(self.actions_list)

class EpsGreedy:
    def __init__(self, agent, n_actions, eps=0.0):
        self.agent = agent
        self.n_actions = n_actions
        self.eps = eps
        self.actions_list = list(range(self.n_actions))

    def policy(self, state):
        if random.random() < self.eps:
            action = random.choice(self.actions_list)
        else:
            action = self.agent.policy(state)

        return action

    def score(self, state):
        greedy_score = np.zeros(self.n_actions, dtype=np.float32)
        greedy_score[self.agent.policy(state)] = 1

        random_score = np.ones(self.n_actions, dtype=np.float32) / self.n_actions

        return (1 - self.eps) * greedy_score + self.eps * random_score