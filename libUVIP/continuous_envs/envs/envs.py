import numpy as np
import torch
import scipy as sp
from scipy import optimize
from scipy import stats
import gym
from tqdm import tqdm
import rlberry
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry.wrappers.discretize_state import DiscretizeStateWrapper
from rlberry.agents.kernel_based.common import map_to_representative
import random

class WrappedTwinRooms:
    def __init__(self, params, noise_room1=0.01, noise_room2=0.01, n_bins=20, seed=42):
        np.random.seed(seed)

        self.noise_room1 = noise_room1
        self.noise_room2 = noise_room2
        self.n_bins = n_bins
        self.params = params

        env_ctor = TwinRooms
        env_kwargs = dict(
            noise_room1=noise_room1,
            noise_room2=noise_room2
        )

        env = env_ctor(**env_kwargs)
        self.env = env
        self.discr_env = DiscretizeStateWrapper(env, n_bins=n_bins)
        self.state_dim = 2
        self.n_action = 4

        self.grid = torch.zeros(n_bins**2, self.state_dim)
        for i in range(n_bins**2):
            self.grid[i, :] = torch.from_numpy(self.discr_env.get_continuous_state(i))

    def sample_discrete(self, state, action):
        next_state, reward, _, _ = self.env.sample(state, action)
        next_state_repr = map_to_representative(next_state, **self.params)
        return next_state_repr, reward

    def sample_cont(self, state, action):
        next_state, reward, _, _ = self.env.sample(state, action)

        return next_state, reward

class RenderTwinRooms(TwinRooms):
    def __init__(self, Vpi, Vup, **kwargs):
        TwinRooms.__init__(self, **kwargs)

        self.Vpi = Vpi
        self.Vup = Vup

    def get_background(self):
        pass

    def get_scene(self, state):
        pass























