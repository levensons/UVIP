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
from libUVIP.discrete_envs.algorithms import TabularReinforce, ComputeFastPreciseUVIP
import parser
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import A2C
import rlberry
from rlberry.envs.benchmarks.grid_exploration.nroom import get_nroom_state_coord
from rlberry.wrappers.vis2d import Vis2dWrapper
from rlberry.manager import AgentManager, MultipleManagers, evaluate_agents
import yaml
from rlberry.agents.dynprog import ValueIterationAgent
import logging
import matplotlib
from matplotlib import cm

sns.set(style="darkgrid")

def values_for_room(env, room_row, room_col, V):
    V_room = np.zeros((env.room_size, env.room_size))
    for i in range(env.room_size):
        for j in range(env.room_size):
            global_row, global_col = env.env._convert_room_coord_to_global(room_row, room_col, i, j)
            state = env.env.coord2index[(global_row, global_col)]
            V_room[i, j] = V[state]

    return V_room

def get_layout_img(env, state_data, colormap_name, vmin=None, vmax=None):
    wall_color = (0.0, 0.0, 0.0)
    colormap_fn = plt.get_cmap(colormap_name)
    layout = env.env.get_layout_array(state_data, fill_walls_with=np.nan)

    if vmin is None:
        vmin = state_data.min()

    if vmax is None:
        vmax = state_data.max()

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap_fn)
    img = np.zeros(layout.shape + (3,))
    for rr in range(layout.shape[0]):
        for cc in range(layout.shape[1]):
            if np.isnan(layout[rr, cc]):
                img[4 - rr, cc, :] = wall_color
            else:
                img[4 - rr, cc, :3] = scalar_map.to_rgba(
                    layout[rr, cc]
                )[:3]
    return img, scalar_map

if __name__ == "__main__":
    #Args should be name of environment and kwargs
    logger = logging.getLogger('UVIP')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('logs.log', mode="w")
    fh.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    name = config['envs']['env']
    alg_conf = config['alg']

    factory = EnvFactory(name)
    env = factory.get_env(config[name])
    nrooms = 5
    room_size = 5

    env_ctor = NRoom
    env_kwargs = dict(
        nrooms=5,
        remove_walls=False,
        room_size=5,
        initial_state_distribution="center",
        include_traps=True,
    )

    agent = ValueIterationAgent(env_ctor(**env_kwargs), gamma=0.9, epsilon=0.2)
    logger.info("fitting...")
    info = agent.fit()
    logger.info(info)

    policy_vec = env.process_policy(agent)
    corrupted_policy_vec = env.corrupt_policy(policy_vec, 0, 2, alpha=0.5)
    Vpi = env.compute_Vpi(policy_vec, alg_conf['gamma'])
    Vpi_corrupted = env.compute_Vpi(corrupted_policy_vec, alg_conf['gamma'])
    Vstar, _ = env.compute_Vstar(gamma=alg_conf['gamma'], eps=1e-9)

    fig = plt.figure(figsize=(20, 5))

    # fig.add_subplot(3, 1, 1)
    plt.imshow(env.env.get_layout_img(Vpi, colormap_name="RdBu_r"))
    # plt.title(r"$V^{\pi}$ for the constructed policy $\pi$", fontsize=20)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("out_Vpi.png")
    plt.show()

    # fig.add_subplot(3, 1, 2)
    fig = plt.figure(figsize=(20, 5))
    plt.imshow(env.env.get_layout_img(Vpi_corrupted, colormap_name="RdBu_r"))
    # plt.title(r"$V^{\hat{\pi}}$ for the corrupted policy $\hat{\pi}$", fontsize=20)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("out_Vpi_corrupted.png")
    plt.show()

    # fig.add_subplot(3, 1, 3)
    fig = plt.figure(figsize=(20, 5))
    plt.imshow(env.env.get_layout_img(Vstar, colormap_name="RdBu_r"))
    # plt.title(r"$V^*$ for the optimal policy $\pi^*$", fontsize=20)
    plt.grid(False)

    plt.tight_layout()
    plt.savefig("out_Vpi_optimal.png")
    plt.show()

    _, Vup_pi, _ = ComputeFastPreciseUVIP(env, policy_vec, alg_conf['gamma'], eps=1e-5)
    logger.info("FINISHED UVIP FOR GROUND TRUTH POLICY!")
    _, Vup_pi_corrupted, _ = ComputeFastPreciseUVIP(env, corrupted_policy_vec, alg_conf['gamma'], eps=1e-5)
    logger.info("FINISHED UVIP FOR CORRUPTED POLICY!")

    logger.info(values_for_room(env, 0, 2, Vup_pi))
    logger.info(values_for_room(env, 0, 3, Vup_pi))

    logger.info(values_for_room(env, 0, 2, Vup_pi_corrupted))
    logger.info(values_for_room(env, 0, 3, Vup_pi_corrupted))

    gap_pi = np.abs(Vup_pi - Vpi)
    gap_pi_corrupted = np.abs(Vup_pi_corrupted - Vpi_corrupted)

    mn = min(gap_pi.min(), gap_pi_corrupted.min())
    gap_pi_norm = gap_pi
    gap_pi_corrupted_norm = gap_pi_corrupted

    mx = max(gap_pi.max(), gap_pi_corrupted.max())
    gap_pi_norm = gap_pi_norm
    gap_pi_corrupted_norm = gap_pi_corrupted_norm

    fig = plt.figure(figsize=(20,5))
    # fig.add_subplot(2, 1, 1)
    img_gap_pi, scalar_map = get_layout_img(env, gap_pi_norm, "RdBu_r", mn, mx)
    logger.info(f"Image Gap Pi is {img_gap_pi.shape}")
    plt.imshow(img_gap_pi, vmin=mn, vmax=mx)
    plt.grid(False)
    cbar = plt.colorbar(scalar_map, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=35)
    # for i in range(img_gap_pi.shape[0]):
    #     for j in range(img_gap_pi.shape[1]):
    #         plt.gca().text(j, i, f"{gap_pi[env.env.coord2index[(i, j)]]:.2f}", ha="center", va="center", color="w")

    # plt.title(r"$|V^{up}(x) - V^{\pi}(x)|$ for ground truth policy $\pi$", fontsize=20)
    plt.tight_layout()
    plt.savefig("outUVIP_gapPi.png")
    plt.show()

    # fig.add_subplot(2, 1, 2)
    plt.figure(figsize=(20,5))
    img_gap_pi_corrupted, scalar_map = get_layout_img(env, gap_pi_corrupted_norm, "RdBu_r", mn, mx)
    logger.info(f"Image Gap Pi Corrupted is {img_gap_pi_corrupted.shape}")
    plt.imshow(img_gap_pi_corrupted, vmin=mn, vmax=mx)
    plt.grid(False)
    cbar = plt.colorbar(scalar_map, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=35)
    # for i in range(img_gap_pi_corrupted.shape[0]):
    #     for j in range(img_gap_pi_corrupted.shape[1]):
    #         plt.gca().text(j, i, f"{gap_pi_corrupted[env.env.coord2index[(i, j)]]:.2f}", ha="center", va="center", color="w")

    # plt.title(r"$|V^{up}(x) - V^{\pi}(x)|$ for corrupted policy $\pi$", fontsize=20)
    plt.tight_layout()

    plt.savefig("outUVIP_gapPi_corrupted.png")
    plt.show()









