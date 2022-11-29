import gym
import numpy as np
import time, os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
import seaborn as sns
from libUVIP.utils.utils_discrete import plot_reinforce
import sys
from libUVIP.discrete_envs.envs.envs import EnvFactory
from libUVIP.algorithms import TabularReinforce, ComputeFastPreciseUVIP
import parser
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import A2C
import rlberry
from rlberry.envs.benchmarks.grid_exploration.nroom import get_nroom_state_coord
from rlberry.wrappers.vis2d import Vis2dWrapper
from rlberry.manager import AgentManager, MultipleManagers, evaluate_agents
import yaml

sns.set(style="darkgrid")

if __name__ == "__main__":
    #Args should be name of environment and kwargs

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)#TODO: read config in yaml format

    name = config['envs']['env']
    alg_conf = config['alg']

    factory = EnvFactory(name)
    env = factory.get_env(config[name])
    # env = FrozenLake(30, 42)
    scores = TabularReinforce(env, gamma=alg_conf['gamma'],
                              save_path=alg_conf['save_path_reinforce'], lr=alg_conf['lr'],
                              eps_uvip=float(alg_conf['UVIP_accuracy']))
    print("REINFORCE DONE!")
    plot_reinforce(scores, save_path=alg_conf['save_path_scores'])

    print(alg_conf['policy_accuracy'])
    _, policy = env.compute_Vstar(gamma=alg_conf['gamma'], eps=float(alg_conf['policy_accuracy']))

    Vstar, Vup, _ = ComputeFastPreciseUVIP(env, policy, gamma=alg_conf['gamma'],
                                           eps=float(alg_conf['UVIP_accuracy']),
                                           debug=False, save_path=alg_conf['save_path_UVIP'])

    print(f"For UVIP accuracy {float(alg_conf['UVIP_accuracy'])} we have |V* - Vup| = {np.max(np.abs(Vstar - Vup))}")

    # scores = TabularReinforce(env, 0.9, True, 40000, lr=0.05)












