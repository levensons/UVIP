from absl import app
from absl import flags
from ml_collections import config_flags

from time import time
from functools import reduce
import uuid
from tqdm import tqdm
import os
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor
from libUVIP.FQE.fqe import RLDatasetOffline, FQE
from libUVIP.FQE.utils import prepare_dataset
from libUVIP.FQE.agents import RandomAgent, EpsGreedy
from rlberry.agents import RSKernelUCBVIAgent
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry.seeding import Seeder, safe_reseed
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import logging
import wandb

import pandas as pd

def eval_agent(env, agent, gamma, horizon, repr_states, n_sim=30):
    total_rewards = []
    for state in tqdm(repr_states):
        total_reward = 0
        for i in range(n_sim):
            cur_state = state
            total_gamma = 1
            for h in range(horizon):
                action = agent.policy(cur_state)
                next_state, reward, done, info = env.sample(cur_state, action)
                total_reward += total_gamma * reward
                total_gamma *= gamma

                cur_state = next_state

        total_rewards.append(total_reward / n_sim)

    return total_rewards

def run_exp(args, seed):
    device = args.device
    logger = logging.getLogger("fqe")

    seeder = Seeder(seed)

    exp_path = f"./saved_values{args.config_e.budget}_{seed}/"
    os.makedirs(exp_path, exist_ok=True)

    fqe_w_path = f"./saved_fqes/fqe{args.alg_type}.pt"

    env = TwinRooms()
    safe_reseed(env, seeder)

    n_actions = 4

    # agent_b = RSKernelUCBVIAgent(env, args.config_b.gamma, **args.config_b.params)
    # agent_b.fit(args.config_b.budget)
    # agent_b = RandomAgent(n_actions)

    agent_path = Path(exp_path + "agent_e").with_suffix(".pickle")
    if os.path.exists(agent_path):
        agent_e_kwargs = {"env": env, "gamma": args.config_e.gamma}
        agent_e = RSKernelUCBVIAgent.load(agent_path, **agent_e_kwargs, **args.config_e.params)
        safe_reseed(agent_e, seeder)
    else:
        agent_e = RSKernelUCBVIAgent(env, args.config_e.gamma, **args.config_e.params)
        safe_reseed(agent_e, seeder)
        agent_e.fit(args.config_e.budget)
        agent_e.save(agent_path)

    repr_states = agent_e.representative_states[:agent_e.M]

    Vpi = agent_e.V[0, :]
    np.save(exp_path + f"Vpi_agent_e.npy", Vpi)

    agent_b = EpsGreedy(agent_e, n_actions, 0.25)
    agent_e = EpsGreedy(agent_e, n_actions, 0.1)

    rewards_agent_e = eval_agent(env, agent_e, args.gamma, args.H, repr_states)
    rewards_agent_b = eval_agent(env, agent_b, args.gamma, args.H, repr_states)
    logger.info(rewards_agent_b)
    logger.info(rewards_agent_e)
    np.save(exp_path + f"rewards_agent_b.npy", rewards_agent_b)
    np.save(exp_path + f"rewards_agent_e.npy", rewards_agent_e)

    reward_init_states = eval_agent(env, agent_e, args.gamma, args.H, [np.array([0.1, 0.1]), np.array([1.1, 0.1])], n_sim=100)
    logger.info(reward_init_states)

    states,\
    initial_states,\
    next_states,\
    next_scores,\
    actions,\
    rewards,\
    dones = prepare_dataset(env, agent_b, agent_e, args.T, args.H, n_actions, args.gamma)

    state_dim = len(states[0])

    dataset_config = {
        "states": states,
        "next_states": next_states,
        "next_scores": next_scores,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "gamma": args.gamma
    }

    dataset = RLDatasetOffline(dataset_config)

    fqe_config = {
        "dataset": dataset,
        "pi_e": agent_e,
        "optim_conf": args.optim_conf,
        "n_epochs": args.fqe_params.n_epochs,
        "initial_states": initial_states,
        "repr_states": repr_states,
        "state_dim": state_dim,
        "n_actions": n_actions,
        "hidden_size": args.fqe_params.hidden_size,
        "gamma": args.gamma,
        "tau": -1,
        "device": device
    }

    fqe = FQE(**fqe_config)

    values, repr_values, relative_err_hist, timestamps = fqe.train(batch_size=args.fqe_params.bs, plot_info=True)

    torch.save(fqe.q.state_dict(), fqe_w_path)

    np.save(exp_path + f"values.npy", values)
    np.save(exp_path + f"repr_values.npy", repr_values)
    np.save(exp_path + f"relative_err_hist.npy", relative_err_hist)
    np.save(exp_path + f"timestamps.npy", timestamps)

def main(_):
    args = FLAGS.config

    device = args.device
    seeds = args.seeds

    logger = logging.getLogger("fqe")
    fh = logging.FileHandler(f'logs/run_fqe.log')
    fh.setLevel(logging.INFO) # or any level you want
    logger.addHandler(fh)

    for seed in seeds:
        run_exp(args, seed)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config")

    app.run(main)