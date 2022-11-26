import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from itertools import product
import os
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import utils_discrete
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
import gym
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import A2C
import rlberry
from rlberry.manager import AgentManager, MultipleManagers, evaluate_agents

def ComputePreciseUVIP(env, policy, gamma, eps=0.1, plot_norm=False, init_upper=None):
    Vpi = env.compute_Vpi(policy, gamma)

    P_Vpi_xa = np.sum(Vpi[:, None, None] * env.P, axis=0)
    M = Vpi[:, None, None] - P_Vpi_xa[None, :, :]

    assert np.all(np.abs(np.sum(M * env.P, axis=0) - np.zeros((env.n_state, env.n_action))) < 1e-14)

    states = list(range(env.n_state))
    actions = list(range(env.n_action))
    combinations = list(product(states, repeat=env.n_action))

    if init_upper is None:
        init_upper = np.zeros(env.n_state)

    upper_list = [init_upper]
    norm_list_upper = []
    norm_theta = 0

    while True:
        F_xa = env.R[None, :, :] + gamma * (upper_list[-1][:, None, None] - M)
        Vup_cur = np.zeros(env.n_state)
        for comb in combinations:
            Vup_cur += F_xa[comb, :, actions].max(axis=0) * np.exp(np.log(env.P[comb, :, actions] + 1e-12).sum(axis=0))

        upper_list.append(Vup_cur)

        if len(upper_list) > 2:
            norm_upper = np.max(np.abs(upper_list[-2] - upper_list[-1]))

            if norm_upper < eps:
                print("Norm upper:", norm_upper)
                break

            norm_list_upper.append(norm_upper)

            if plot_norm:
                clear_output(wait=True)
                fig = plt.figure(figsize=(8, 7))
                ax = fig.add_subplot(1, 1, 1)
                plt.plot(norm_list_upper)
                plt.yscale("log")
                plt.title("Upper norm")
                plt.show()

    return Vpi, upper_list[-1], norm_list_upper

def ComputeFastPreciseUVIP(env, policy, gamma, eps=0.1, plot_norm=False, init_upper=None, debug=False):
    Vpi = env.compute_Vpi(policy, gamma)

    P_Vpi_xa = np.sum(Vpi[:, None, None] * env.P, axis=0)
    M = Vpi[:, None, None] - P_Vpi_xa[None, :, :]

    assert np.all(np.abs(np.sum(M * env.P, axis=0) - np.zeros((env.n_state, env.n_action))) < 1e-7)

    states = list(range(env.n_state))
    actions = list(range(env.n_action))

    if init_upper is None:
        init_upper = np.random.normal(0, 1, size=env.n_state)
        # init_upper = np.zeros(env.n_state, dtype=np.float32)

    upper_list = [init_upper]
    norm_list_upper = []
    norm_theta = 0

    while True:
        F_xa = env.R[None, :, :] + gamma * (upper_list[-1][:, None, None] - M)

        noise = np.random.normal(1e-6, 1e-7, size=(env.n_state, env.n_state, env.n_action))
        F_xa += noise

        F_xa_sorted_idxes = np.argsort(F_xa, axis=0)
        F_xa_sorted_values = np.sort(F_xa, axis=0)
        P_permuted = env.P[F_xa_sorted_idxes, np.array(states)[None, :, None], np.array(actions)[None, None, :]]
        P_permuted_cumsum = np.cumsum(P_permuted, axis=0)

        Vup_cur = np.zeros(env.n_state, dtype=np.float32)

        for idx in range(env.n_state):
            for ak in actions:
                remain_actions = list(range(env.n_action))
                remain_actions.remove(ak)
                thd_val = F_xa[idx, :, ak]
                thd_idxes_list = []
                for i, thd in enumerate(thd_val):
                    thd_idxes = np.apply_along_axis(lambda a: a.searchsorted(thd, side='right'), axis=0, arr=F_xa_sorted_values[:, i, :])
                    thd_idxes_list.append(thd_idxes)

                thd_idxes = np.stack(thd_idxes_list, axis=0)[:, remain_actions]
                mask = np.ones(env.n_state, dtype=np.float32)
                np.putmask(mask, np.any(thd_idxes == 0, axis=1), 0)

                thd_idxes -= 1

                probs = P_permuted_cumsum[thd_idxes, np.array(states)[:, None], np.array(remain_actions)[None, :]]

                if debug:
                    for i in states:
                        for k, j in enumerate(remain_actions):
                            assert P_permuted_cumsum[thd_idxes[i, k], i, j] == probs[i, k]

                Vup_cur += F_xa[idx, :, ak] * env.P[idx, :, ak] * mask * probs.prod(axis=1)

        upper_list.append(Vup_cur)

        Vup_check = np.zeros(env.n_state, dtype=np.float32)
        err = np.max(np.abs(Vup_check - Vup_cur))

        if len(upper_list) > 2:
            norm_upper = np.max(np.abs(upper_list[-2] - upper_list[-1]))

            if norm_upper < eps:
                print("Norm upper:", norm_upper)
                break

            norm_list_upper.append(norm_upper)

            if plot_norm:
                clear_output(wait=True)
                fig = plt.figure(figsize=(8, 7))
                ax = fig.add_subplot(1, 1, 1)
                plt.plot(norm_list_upper)
                plt.yscale("log")
                plt.title("Upper norm")
                plt.show()

    return Vpi, upper_list[-1], norm_list_upper


def TabularReinforce(env, gamma, plot=False, n_iters=40000, lr=0.01):
    theta = np.ones(env.n_state * env.n_action)
    policy = utils_discrete.make_policy(env.n_state, env.n_action, theta)
    Vstar, _ = env.compute_Vstar(gamma, eps=1e-9)

    score = []
    eval_iters = list(range(0, n_iters, 200))
    norm_history = []
    norm_history_policy = []
    prev_upper = None

    for episode in tqdm(range(n_iters)):
        reward = 0
        while reward == 0:
            state = env.reset()
            trajectory = []
            rewards = []
            steps = 0
            while True:
                probs = policy[state]
                action = np.random.choice(np.arange(env.n_action), p=probs)
                new_state, reward, done = env.step(state, action)
                trajectory.append([state, action])
                rewards.append(reward)
                steps += 1
                if done:
                    break
                state = new_state
            score.append(reward)

        if done:
            theta, policy = utils_discrete.update_policy_tabular(env.n_state, env.n_action, policy, theta, rewards, trajectory, lr, gamma)
            assert np.all(policy.sum(axis=1) - np.ones(env.n_state) < 1e-12)

        if plot and (episode in eval_iters):
            Vpi, upper, _ = ComputeFastPreciseUVIP(env, policy, gamma, 0.00001, False, prev_upper)

            prev_upper = upper

            norm_history.append(np.max(np.abs(upper - Vstar)))
            norm_history_policy.append(np.max(np.abs(Vpi - Vstar)))

            clear_output(wait=True)
            fig = plt.figure(figsize=(10, 9))
            sns.set(style="darkgrid")
            ax = fig.add_subplot(1, 1, 1)
            plt.title("Accuracy of evaluation", fontsize=30)
            plt.plot(norm_history, label=r'$\|V^{up}-V^{*}\|_{\infty}$', linewidth=3)
            plt.plot(norm_history_policy, label=r'$\|V^{\pi}-V^{*}\|_{\infty}$', linewidth=3)
            plt.yscale("log")
            ax.tick_params(axis="x", labelsize=35)
            ax.tick_params(axis="y", labelsize=35)
            plt.legend(fontsize=30)
            plt.tight_layout()
            plt.savefig("norm_reinforce.png", pi=fig.dpi)
            plt.show()

    return score

def TrainA2C(env_ctor, env_kwargs, fit_budget, gamma, log_interval, eval_horizon, n_fit, seed):
    stats = AgentManager(
        StableBaselinesAgent,
        (env_ctor, env_kwargs),
        agent_name="A2C NRoom",
        init_kwargs=dict(algo_cls=A2C, policy="MlpPolicy", gamma=gamma, verbose=1),
        fit_kwargs=dict(log_interval=log_interval),
        fit_budget=fit_budget,
        eval_kwargs=dict(eval_horizon=eval_horizon),
        n_fit=n_fit,
        parallelization="process",
        output_dir="dev/stable_baselines",
        seed=seed,
    )

    stats.fit()

    evaluate_agents([stats], n_simulations=10)

    return stats.get_agent_instances()
