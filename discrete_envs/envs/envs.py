import numpy as np
import scipy as sp
from scipy import optimize
from scipy import stats
from env_base import BaseEnvironment
import utils_discrete
import chain_custom
import gym
from tqdm import tqdm
import rlberry
from rlberry.envs import Chain
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
import random

class FrozenLake(BaseEnvironment):
    def __init__(self, max_steps, seed):
        np.random.seed(seed)

        env = gym.make('FrozenLake-v0', is_slippery=True)
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n

        self.P = utils_discrete.get_dynamics(env)
        self.max_steps = max_steps

        r_sa = np.zeros((self.n_state, self.n_state, self.n_action))
        r_sa[15, 14, :] = 10
        self.R = np.sum(r_sa*self.P, axis=0)

        self.terminal_state = 15

class Chain(BaseEnvironment):
    def __init__(self, n_states, seed, p=0.5):
        np.random.seed(seed)

        env_ctor = Chain
        env_kwargs = dict(L=n_states, fail_prob=p)
        env = env_ctor(**env_kwargs)
        self.env = env

        print(env.get_params())
        assert False
        # env = chain_custom.Chain(p, n_states) # default is 15
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n

        self.P = utils_discrete.get_dynamics(env)
        self.R = np.einsum('kmn,nkm->km', env.get_rewards(), self.P)

    def is_terminal(self, state):
        return self.env.is_terminal(state)

class WrappedNRoom(BaseEnvironment):
    def __init__(self, nrooms, room_size, seed):
        np.random.seed(seed)

        self.nrooms = nrooms
        self.room_size = room_size

        env_ctor = NRoom
        env_kwargs = dict(
            nrooms=nrooms,
            remove_walls=False,
            room_size=room_size,
            initial_state_distribution="center",
            include_traps=True,
        )

        env = env_ctor(**env_kwargs)
        self.env = env
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n

        self.R = env.R.copy()
        self.P = np.transpose(env.P, (2, 0, 1))
        # self.P = env.P.copy()

        self.max_steps = self.n_state

    def is_terminal(self, state):
        return self.env.is_terminal(state)

    def corrupt_policy(self, policy, room_row, room_col, alpha=0.5):
        new_policy = policy.copy()
        for i in range(self.room_size):
            for j in range(self.room_size):
                if random.random() < alpha:
                    global_row, global_col = self.env._convert_room_coord_to_global(room_row, room_col, i, j)
                    state = self.env.coord2index[(global_row, global_col)]
                    new_policy[state, :] = np.ones(self.n_action) / self.n_action

        return new_policy

    def process_policy(self, agent):
        policy = np.zeros((self.n_state, self.n_action))
        for state in range(self.n_state):
            policy[state, agent.policy(state)] = 1.0

        return policy