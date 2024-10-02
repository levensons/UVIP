import gym
import numpy as np
import time, os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
import seaborn as sns
import pandas as pd
from IPython.display import clear_output
import sys
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
import gym
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import A2C
import rlberry
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom, get_nroom_state_coord
from rlberry.wrappers.vis2d import Vis2dWrapper
from rlberry.manager import AgentManager, MultipleManagers, evaluate_agents
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry.agents.mbqvi import MBQVIAgent
from rlberry.wrappers.discretize_state import DiscretizeStateWrapper
from rlberry.seeding import Seeder
from rlberry.agents import RSKernelUCBVIAgent, RSKernelUCBVIAgent
from libUVIP.continuous_envs.envs.envs import WrappedTwinRooms
from libUVIP.continuous_envs.algorithms import getMonteCarloUpperBounds
import logging

# plt.rcParams["mathtext.fontset"] = True
# sns.set(style="darkgrid")

def runMBQVI(env, gamma):
    env_discr = DiscretizeStateWrapper(env, n_bins=20)
    logger.info(env.observation_space)
    logger.info(env_discr.observation_space)
    init_state_discr = env_discr.reset()
    logger.info(f"reset in discrete environment gives initial state = {init_state_discr}")
    logger.info("Created DiscretizeStateWrapper!")
    agent = MBQVIAgent(env_discr, n_samples=600, gamma=gamma)
    logger.info("Created MBQVIAgent!")
    logger.info("fitting...")
    agent.fit()
    logger.info("finished!")

    n_simulations = 200
    cur_state = env.reset()
    logger.info(f"Initial state is {cur_state}")
    total_reward = 0
    for tt in range(n_simulations):
        cur_state = env.reset()
        total_gamma = 1
        for i in range(1000):
            discr_cur_state = env_discr.get_discrete_state(cur_state)
            next_state, reward, _, _ = env.step(agent.policy(discr_cur_state))
            # logger.info(f"At step {i} next state is {next_state} and reward is {reward}")
            total_reward += total_gamma * reward
            total_gamma *= gamma
            cur_state = next_state

    logger.info(f"Total discounted reward is {total_reward / n_simulations}")

    # logger.info("AVG reward at initial state: {}".format(agent.eval(n_simulations=20, gamma=gamma)))
    Vpi = agent.V

    logger.info(f"Vpi at initial state is {Vpi[init_state_discr]}")
    logger.info(Vpi)
    return Vpi

def runKernelUCBVI(env, gamma, budget):
    agent = RSKernelUCBVIAgent(env, gamma, kernel_type="gaussian", max_repr=500, beta=0.01, bandwidth=0.025, min_dist=0.05)
    logger.info("Created KernelUCBVI!")
    logger.info("fitting...")
    agent.fit(budget)
    logger.info("finished!")

    n_simulations = 10
    cur_state = env.reset()
    logger.info(f"Initial state is {cur_state}")
    total_reward = 0
    # for tt in range(n_simulations):
    #     cur_state = env.reset()
    #     logger.info(cur_state)
    #     total_gamma = 1
    #     for i in range(1000):
    #         next_state, reward, _, _ = env.step(agent.policy(cur_state))
    #         # logger.info(f"At step {i} next state is {next_state} and reward is {reward}")
    #         total_reward += total_gamma * reward
    #         total_gamma *= gamma
    #         cur_state = next_state
    total_reward = agent.eval(n_simulations=n_simulations, gamma=gamma)

    # logger.info(f"Total discounted reward is {total_reward / n_simulations}")
    logger.info(f"Total discounted reward is {total_reward}")

    repr_states = agent.representative_states
    Vpi = agent.V[0, :]
    logger.info(agent.V.shape)
    logger.info(Vpi.shape)

    init_state = env.reset()
    logger.info(f"Number of representative states is {repr_states.shape[0]}")
    logger.info(f"Init representative state is {agent._map_to_repr(init_state, False)}")
    logger.info(f"Vpi at initial state is {Vpi[agent._map_to_repr(init_state, False)]}")
    logger.info(Vpi)

    return Vpi, repr_states, agent

if __name__ == "__main__":
    exp_path = "./TwinRoomsExp5"

    logger = logging.getLogger('UVIP')
    logger.setLevel(logging.INFO)
    # os.makedirs(os.path.join(exp_path, 'logsTwinRooms.log'), exist_ok=True)
    fh = logging.FileHandler(os.path.join(exp_path, 'logsTwinRooms.log'), mode="w")
    fh.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)

    seeder = Seeder(123)
    gamma = 0.99
    budget = 2500

    env = TwinRooms()

    # Vpi = runMBQVI(env, gamma)
    Vpi, states, agent = runKernelUCBVI(env, gamma, budget)
    logger.info(states)
    logger.info(f"Number of representative states is {agent.M}")
    logger.info(f"Shape of Vpi is {Vpi.shape}")
    states = states[:agent.M]
    Vpi = Vpi[:agent.M]

    env = WrappedTwinRooms(agent, n_bins=15)

    Vup, norm_list_upper = getMonteCarloUpperBounds(env, states, Vpi, k=3, total_steps=100, M1=500, M2=200, gamma=gamma)
    np.save(os.path.join(exp_path, "Vpi.npy"), Vpi)
    np.save(os.path.join(exp_path, "Vup.npy"), Vup)
    np.save(os.path.join(exp_path, "ReprStates.npy"), states)

    logger.info(Vup)
    logger.info(f"Gap between Vup and Vpi is {np.max(np.abs(Vup - Vpi))}")

    plt.figure(figsize=(10, 10))

    plt.plot(norm_list_upper, color="red")
    plt.title("Convergence of UVIP on TwinRooms", fontsize=20)
    plt.xlabel("iteration step", fontsize=20)
    plt.ylabel(r"$\|V_k^{up} - V_{k-1}^{up}\|_{\infty}$", fontsize=20)
    plt.yscale("log")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(exp_path, "outTwinRooms.png"))
    plt.show()







