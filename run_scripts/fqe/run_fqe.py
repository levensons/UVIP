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

from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor
from libUVIP.FQE.fqe import RLDatasetOffline, FQE
from libUVIP.FQE.utils import prepare_dataset
from rlberry.agents import RSKernelUCBVIAgent
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import logging
import wandb

import pandas as pd

def eval_agent(env, agent, gamma, repr_states, n_sim=100):
    horizon = int(1 / (1 - gamma))
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

def main(_):
    args = FLAGS.config

    device = args.device
    seed = args.seed

    random.seed(seed)            # Python random module
    np.random.seed(seed)         # NumPy random module
    torch.manual_seed(seed)      # PyTorch CPU
    torch.cuda.manual_seed(seed) # PyTorch GPU

    logger = logging.getLogger("fqe")
    fh = logging.FileHandler(f'logs/run_fqe_{seed}.log')
    fh.setLevel(logging.INFO) # or any level you want
    logger.addHandler(fh)

    exp_path = "./saved_values/"

    fqe_w_path = f"./saved_fqes/fqe{args.alg_type}.pt"

    env = TwinRooms()

    n_actions = 4

    agent_b = RSKernelUCBVIAgent(env, args.config_b.gamma, **args.config_b.params)
    agent_b.fit(args.config_b.budget)

    agent_e = RSKernelUCBVIAgent(env, args.config_e.gamma, **args.config_e.params)
    agent_e.fit(args.config_e.budget)

    rewards_agent_b = eval_agent(env, agent_b, args.gamma, agent_b.representative_states[:agent_b.M])
    rewards_agent_e = eval_agent(env, agent_e, args.gamma, agent_b.representative_states[:agent_b.M])
    logger.info(rewards_agent_e)
    np.save(exp_path + f"rewards_agent_b_{seed}.npy", rewards_agent_b)
    np.save(exp_path + f"rewards_agent_e_{seed}.npy", rewards_agent_e)

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
        "repr_states": agent_b.representative_states[:agent_b.M],
        "state_dim": state_dim,
        "n_actions": n_actions,
        "hidden_size": args.fqe_params.hidden_size,
        "gamma": args.gamma,
        "tau": -1,
        "device": device
    }

    fqe = FQE(**fqe_config)

    values, relative_err_hist, timestamps = fqe.train(batch_size=args.fqe_params.bs, plot_info=True)

    torch.save(fqe.q.state_dict(), fqe_w_path)

    np.save(exp_path + f"values_{seed}.npy", values)
    np.save(exp_path + f"relative_err_hist_{seed}.npy", relative_err_hist)
    np.save(exp_path + f"timestamps_{seed}.npy", timestamps)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config")

    app.run(main)