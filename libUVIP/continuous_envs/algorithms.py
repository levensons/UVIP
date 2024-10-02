import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import trange
import torch
from torch.distributions import Categorical
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import scipy
from numba import njit, prange
import matplotlib as mpl
import random
from tqdm import tqdm_notebook
import logging

def generate_samples_given(env, X_samples, N=1):
    device = 'cpu'
    rewards0 = []
    rewards1 = []
    for X in X_samples:
        env.reset(X)
        _, r0, _, _ = env.step(0)
        env.reset(X)
        _, r1, _, _ = env.step(1)
        rewards0.append(r0)
        rewards1.append(r1)

    rewards0 = torch.tensor(rewards0, device=device, dtype=torch.float32)
    rewards1 = torch.tensor(rewards1, device=device, dtype=torch.float32)
    rewards = torch.cat((rewards0[None, :], rewards1[None, :]), axis=0)

    return torch.FloatTensor(X_samples), rewards

def generate_samples_uniform(env, N=1):
    device = 'cpu'
    X0_samples = torch.FloatTensor(N, 1).uniform_(-2.3, 2.3)
    X1_samples = torch.FloatTensor(N, 1).uniform_(-2., 2.)
    X2_samples = torch.FloatTensor(N, 1).uniform_(-0.2, 0.2)
    X3_samples = torch.FloatTensor(N, 1).uniform_(-1.1, 1.1)
    X_samples = torch.cat((X0_samples, X1_samples, X2_samples, X3_samples), dim=-1).to(device)

    rewards0 = []
    rewards1 = []
    for X in X_samples:
        env.reset(X.numpy())
        _, r0, _, _ = env.step(0)
        env.reset(X.numpy())
        _, r1, _, _ = env.step(1)
        rewards0.append(r0)
        rewards1.append(r1)

    rewards0 = torch.tensor(rewards0, device=device, dtype=torch.float32)
    rewards1 = torch.tensor(rewards1, device=device, dtype=torch.float32)
    rewards = torch.cat((rewards0[None, :], rewards1[None, :]), axis=0)

    return X_samples, rewards

def generate_samples_normal(env, N=1):
    device = 'cpu'
    X0_samples = torch.FloatTensor(N, 1).normal_(0, 2.0 / 3.)
    X1_samples = torch.FloatTensor(N, 1).normal_(0, 2.0 / 3.)
    X2_samples = torch.FloatTensor(N, 1).normal_(0, 0.25 / 3.)
    X3_samples = torch.FloatTensor(N, 1).normal_(0, 1.1 / 3.)
    X_samples = torch.cat((X0_samples, X1_samples, X2_samples, X3_samples), dim=-1).to(device)

    rewards0 = []
    rewards1 = []
    for X in X_samples:
        env.reset(X.numpy())
        _, r0, _, _ = env.step(0)
        env.reset(X.numpy())
        _, r1, _, _ = env.step(1)
        rewards0.append(r0)
        rewards1.append(r1)

    rewards0 = torch.tensor(rewards0, device=device, dtype=torch.float32)
    rewards1 = torch.tensor(rewards1, device=device, dtype=torch.float32)
    rewards = torch.cat((rewards0[None, :], rewards1[None, :]), axis=0)

    return X_samples, rewards

def predict_probs(states, policy):
    with torch.no_grad():
        states = torch.as_tensor(states).to(torch.float32)
        probs = torch.softmax(policy(states), dim=-1)
        return probs.numpy()

def select_action(model, state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    m = Categorical(probs)

    action = m.sample()

    return action.item()

def VpiEstimation(env, X_samples, policy, alg_type, n_samples=15, gamma=0.99):
    n_actions = 2
    Vpi = []

    step = 0
    for step in tqdm_notebook(range(X_samples.shape[0])):
        average_reward = 0.
        for k in range(n_samples):
            discounted_reward = 0.
            s = env.reset(X_samples[step].numpy())
            cur_gamma = 1.
            while True:
                if alg_type == 'a2c':
                    a = select_action(policy, np.array([s]))
                elif alg_type == 'LD':
                    a = env.getBestAction(s)
                elif alg_type == 'random':
                    a = np.random.choice(range(2))

                new_s, r, done, _ = env.step(a)
                discounted_reward += cur_gamma * r
                s = new_s
                cur_gamma *= gamma
                if done:
                    break

            average_reward += discounted_reward

        Vpi.append(average_reward / n_samples)

    return torch.tensor(Vpi, dtype=torch.float32)


@njit(nopython=True, parallel=True)
def calc(Txa, X_samples_pi, cov):
    probs0 = np.empty((0, X_samples_pi.shape[0]), dtype=np.float32)
    probs1 = np.empty((0, X_samples_pi.shape[0]), dtype=np.float32)
    inv = np.linalg.inv(cov)
    for i in prange(Txa.shape[1]):
        d1 = X_samples_pi - Txa[0, i, :]
        d2 = X_samples_pi - Txa[1, i, :]
        s1 = (d1 @ inv) @ d1.T
        s2 = (d2 @ inv) @ d2.T
        p1 = np.exp(-s1 / 2.) / np.sqrt(2 * (np.pi**4) * np.linalg.det(cov))
        p2 = np.exp(-s2 / 2.) / np.sqrt(2 * (np.pi**4) * np.linalg.det(cov))
        probs0 = np.append(probs0, p1.astype(np.float32), axis=0)
        probs1 = np.append(probs1, p2.astype(np.float32), axis=0)
        if i % 1000 == 0:
            print(i)

    return probs0, probs1


def getMonteCarloUpperBounds(env, X_samples, V_pi, k=4, total_steps=50, M1=150, M2=150, gamma=0.9, save_fig=None):
    """
    0,1 means action
    """
    logger = logging.getLogger('UVIP')

    max_grad_norm = 5000
    loss_history = []
    grad_norm_history = []
    eval_freq = 1
    device = 'cpu'

    state_dim, n_actions = env.state_dim, env.n_action
    N = X_samples.shape[0] # number of samples
    neigh = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', leaf_size=30, n_jobs=-1)
    neigh.fit(X_samples.tolist())

    V_up_prev = np.copy(V_pi)
    upper_bound_sample = []
    norm_list_upper = []

    V_mean = np.zeros((n_actions, N, 1))
    for i in range(N):
        for j in range(M1):
            for action in range(n_actions):
                next_state, _ = env.sample_discrete(X_samples[i], action)
                V_mean[action, i, 0] += V_pi[next_state]

    V_mean /= M1

    step = 0
    with trange(step, total_steps + 1) as progress_bar:
        for step in progress_bar:
            Yxa_M2 = np.zeros((n_actions, N, M2, state_dim))
            rewards = np.zeros((n_actions, N, M2))
            for i in range(N):
                YX = np.zeros((n_actions, M2, state_dim))
                for j in range(M2):
                    for action in range(n_actions):
                        YX[action, j, :], reward = env.sample_cont(X_samples[i], action)
                        rewards[action, i, j] = reward

                Yxa_M2[:, i, :, :] = YX

            assert np.allclose(Yxa_M2.reshape(-1, state_dim)[0, :], Yxa_M2[0, 0, 0, :])
            _, idxes_neigh2 = neigh.kneighbors(Yxa_M2.reshape(-1, state_dim).tolist())
            V_pi2 = V_pi[idxes_neigh2].mean(axis=-1).reshape(n_actions, N, M2)
            V_k = V_up_prev[idxes_neigh2].mean(axis=-1).reshape(n_actions, N, M2)

            M = V_pi2 - V_mean
            V_up = (rewards + gamma * (V_k - M)).max(axis=0).mean(axis=1)

            logger.info(f"At step {step} we have |Vup-Vpi| = {np.max(np.abs(V_up - V_pi))}")

            if step >= 1:
                norm_upper = np.sum((V_up - V_up_prev)**2)**0.5
                norm_cur = np.sum(V_up**2)**0.5
                logger.info(f"At step {step} absolute error: {norm_upper}, relative error: {norm_upper / norm_cur}")
                norm_list_upper.append(norm_upper)
                V_up_prev = V_up

            if save_fig and step % eval_freq == 0:
                clear_output(True)
                plt.figure(figsize=(15, 10))
                plt.subplot(121)
                lower = V_pi[idx].numpy()
                upper = V_up[idx].numpy()
                upper_list.append(upper)
                upper_bound_sample.append(V_up[single_sample])
                upper_plot = np.repeat(upper.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
                lower_plot = np.repeat(lower.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
                states_plot = np.concatenate((np.arange(N_states).reshape(-1,1),
                                          (np.arange(N_states)+1).reshape(-1,1)), axis=1).reshape(-1)
                plt.fill_between(states_plot, lower_plot, upper_plot, alpha=0.3,
                            edgecolor='k', linestyle='-')
                plt.plot(states_plot, upper_plot, 'b')
                plt.legend(loc="upper left", fontsize=14)
                plt.title("Upper and Lower bounds", fontsize=14)


                plt.subplot(122)
                if step >= 1:
                    norm_upper = np.linalg.norm(upper_list[-2] - upper_list[-1])
                    norm_list_upper.append(norm_upper)
                    plt.plot(norm_list_upper)
                    plt.title("Upper norm")
                plt.show()
                if step == total_steps - 1:
                    plt.savefig('pic1.png')

    return V_up, norm_list_upper


def plotBounds(V_up, V_pi, X_grid, X_data, ax, params):
    """
    INPUT:
    V_up - upper bounds regression: type = tensor
    X_grid - elements, which are estimated with V_up: type = ndarray
    X_data - elements, which we need to estimate with V_up: type = tensor
    params - dict of parameters
    """
    mpl.style.use('seaborn')
    neigh = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', leaf_size=100, n_jobs=-1)
    neigh.fit(X_grid.tolist())
    _, idxes_neigh = neigh.kneighbors(X_data.numpy().tolist())
    V_up_data = V_up[torch.tensor(idxes_neigh)].mean(dim=-1).numpy()
    V_pi_data = V_pi
    N_states = X_data.shape[0]
    upper_plot = np.repeat(V_up_data.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
    lower_plot = np.repeat(V_pi_data.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
    states_plot = np.concatenate((np.arange(N_states).reshape(-1,1),
                                 (np.arange(N_states)+1).reshape(-1,1)), axis=1).reshape(-1)
    ax.fill_between(states_plot, lower_plot, upper_plot, alpha=0.4,
                edgecolor='green', facecolor='green', linestyle='-')
    major_ticksx = params['major_ticksx']
    major_ticksy = params['major_ticksy']
    ax.set_ylim(*params['y_lim'])
    ax.set_xticks(major_ticksx)
    ax.set_yticks(major_ticksy)
    ax.tick_params(axis="x", labelsize=params['tick_size'])
    ax.tick_params(axis="y", labelsize=params['tick_size'])