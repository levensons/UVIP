import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from itertools import product
import os
import seaborn as sns
import pandas as pd
from tqdm import trange

# making transition matrix for Garnet
def transition_matrix(N_states, N_actions, N_br, random_state=None):
    P = np.zeros((N_states, N_states, N_actions))
    for i in range(N_states):
        for j in range(N_actions):
            if random_state is not None:
                np.random.seed(i + j + random_state)
            random_states = np.random.choice(np.arange(N_states), size=N_br, replace=False)
            P[random_states, i, j] = 1/N_br
    return P

# making rewards matrix for Garnet
def get_reward(N_states, N_actions, random_state=None):
    np.random.seed(random_state)
    r_sa = np.random.random(size=(N_states, N_actions))
    random_states = np.random.choice(np.arange(N_states), N_states)
    random_actions = np.random.choice(np.arange(N_actions), N_states)
    r_sa[random_states, random_actions] *= 20
    return r_sa

# get environment matrix
def get_dynamics(env):
    p_next_state = np.zeros((env.observation_space.n, env.observation_space.n, env.action_space.n))

    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            for next_state in range(env.observation_space.n):
                for i in range(len(env.P[state][action])):
                    if next_state == env.P[state][action][i][1]:
                        p_next_state[next_state, state, action] +=\
                            env.P[state][action][i][0]
    return p_next_state

def compute_V_star(P, r_sa, gamma, eps=1e-5):
    N_states = P.shape[0]
    N_actions = P.shape[2]
    Q_prev = np.random.randn(N_states, N_actions)
    policy_prev = np.zeros((N_states, N_actions))
    norm_list = []
    j = 0

    while True:
        Q = r_sa + gamma * np.einsum("n, nkm -> km", np.max(Q_prev, axis=1), P)

        if j > 2:
            norm_list.append(np.max(np.abs(Q_prev - Q)))
            if norm_list[-1] < eps:
                break
        Q_prev = Q.copy()

        policy_determ = np.zeros((N_states, N_actions))
        for i in range(N_states):
            policy_determ[i, np.argmax(Q, axis=1)[i]] = 1

        policy_prev = policy_determ.copy()
        j += 1

    plt.figure(figsize=(10, 10))
    plt.plot(norm_list, linewidth=2)
    plt.show()

    return Q_prev.max(axis=1), policy_determ

def evaluateP(transitions, D):
    for (x, a, y) in D:
        transitions[y, x, a] += 1

    Nxa = np.sum(transitions, axis=0)
    return np.nan_to_num(transitions / Nxa[None, :, :])

def perform_kernel_iteration(P, V_star, rewards, gamma, max_steps=1000, bounds_eps=0.1, plotNorm=False, compEmp=True, path='./pics/', save=True):
    eval_freq = 1
    total_steps = 15000
    N_states = P.shape[0]
    N_actions = P.shape[2]
    transitions = np.zeros(shape=(N_states, N_states, N_actions))
    P_empirical = np.zeros(shape=(N_states, N_states, N_actions))
    states = np.arange(N_states)
    actions = np.arange(N_actions)
    probs_states = np.ones(N_states) / N_states
    probs_actions = np.ones(N_actions) / N_actions
    k = 0

    norm_history = []

    with trange(k, total_steps + 1) as progress_bar:
        for k in progress_bar:
            state = np.random.choice(a=states, p=probs_states)
            t = 0
            r = 0

            updates = []
            D = []
            while t < max_steps:
                action = np.random.choice(a=actions, p=probs_actions)
                state2 = np.random.choice(a=states, p=P[:, state, action].reshape(-1))

                if t < max_steps - 1:
                    D.append((state, action, state2))
                #add N+1

                state = state2
                t += 1

            P_empirical = evaluateP(transitions, D)

            if k % eval_freq == 0:
                _, policy_determ = compute_V_star(P_empirical, rewards, gamma)
                V, upper, _ = getAccurateBounds(policy_determ, rewards, P, gamma, bounds_eps, plotNorm, compEmp)
                norm_history.append(np.max(np.abs(upper - V_star)))

                clear_output(wait=True)
                fig = plt.figure(figsize=(10, 9))
                sns.set(style="darkgrid")
                ax = fig.add_subplot(1, 1, 1)
                plt.title(r"$\|V^{up}-V^{*}\|_{\infty}$", fontsize=30)
                plt.plot(norm_history, linewidth=2)
                plt.yscale("log")
                ax.tick_params(axis="x", labelsize=35)
                ax.tick_params(axis="y", labelsize=35)
                plt.legend(fontsize=30)
                plt.tight_layout()
                # if save:
                #     plt.savefig((path + "file_{}.png").format(iter_num), pi=fig.dpi)
                #     plt.close()
                # else:
                plt.show()

                print(np.linalg.norm(P_empirical - P))

def perform_value_iteration(P, r_sa, Y_states, p_ksi, T, gamma, bounds, is_accurate=False, eps=.00001, bounds_eps = 0.1, 
                            plotNorm=False, compEmp=True, path='./pics/', save=True):
    """
    Input:
    P - dynamics of the system size of [N_states, N_states, N_actions];
    r_sa - reward matrix size of [N_states, N_actions];
    Y_states - matrix of next states of size [N_states, N_actions, N_br],
               where N_br - number of ksi;
    p_ksi - vector of probabilities to accept ksi, which is to be in one of states [0, 1, ..., N_br - 1];
    T - number of iterations for computing next states;
    gamma - discounting factor;
    bounds - dict for storing V and V^up;
    eps - accuracy of Value iteration convergence, default eps=.00001;
    bounds_eps - accuracy of convergence of the upper bound in the norm, default bounds_eps = 0.1;
    plotNorm - plot the convergence of upper bound, bool;
    compEmp - use Monte-Carlo for expectations computation, bool;
    path - path of saving upper and lower bounds plots;
    save - save upper and lower bounds plots, bool;

    Output:
    policy_determ - deterministic policy from Value Iteration procedure, size [N_states, N_actions];
    """

    N_states = P.shape[0]
    N_actions = P.shape[2]
    Q_prev = np.random.randn(N_states, N_actions)
    policy_prev = np.zeros((N_states, N_actions))
    norm_list = []
    j = 0
    # T = 1000        #M1=M2=T

    while True:
        Q = r_sa + gamma * np.einsum("n, nkm -> km", np.max(Q_prev, axis=1), P)

        if j > 2:
            norm_list.append(np.linalg.norm(Q_prev - Q))
            if norm_list[-1] < eps:
                break
        Q_prev = Q.copy()

        policy_determ = np.zeros((N_states, N_actions))
        for i in range(N_states):
            policy_determ[i, np.argmax(Q, axis=1)[i]] = 1

        if j in [0, 2, 5]:
            if is_accurate:
                V, upper, _ = getAccurateBounds(policy_determ, r_sa, P, gamma, bounds_eps, plotNorm, compEmp)
            else:
                V, upper, _  = getDiscrStationaryBounds(policy_determ, r_sa,
                                                P, np.array(p_ksi), Y_states, gamma, int(T),
                                                bounds_eps, plotNorm, compEmp)
            bounds["policy"]["V"] = V
            bounds["policy"]["upper"] = upper

            clear_output(wait=True)
            plotBounds(bounds, j, path, save)
            bounds_eps /= 1.4

        policy_prev = policy_determ.copy()
        j += 1

    return policy_determ

# tabular reinforce
def update_policy_tabular(N_states, N_actions, policy, theta, rewards, trajectory, lr, gamma):
    T = len(trajectory)
    theta_grad = np.zeros_like(theta)
    for t in range(T):
        G = gamma**(T - t - 1)

        s, a = trajectory[t]
        theta_grad[a + N_actions * s] += lr * G * (1 - policy[s, a])

    theta += theta_grad
    policy = make_policy(N_states, N_actions, theta)
    return theta, policy

def make_policy(N_states, N_actions, theta):
    pi = np.exp(theta).reshape(N_states, N_actions)
    pi = pi/np.sum(pi, axis=1, keepdims=True)

    return pi

def plot_gap(history_Vpi, history_Vup, filename="norm_reinforce.png"):
    clear_output(wait=True)
    fig = plt.figure(figsize=(10, 9))
    sns.set(style="darkgrid")
    ax = fig.add_subplot(1, 1, 1)
    plt.title("Accuracy of evaluation", fontsize=30)
    plt.plot(history_Vup, label=r'$\|V^{up}-V^{*}\|_{\infty}$', linewidth=3)
    plt.plot(history_Vpi, label=r'$\|V^{\pi}-V^{*}\|_{\infty}$', linewidth=3)
    plt.yscale("log")
    ax.tick_params(axis="x", labelsize=35)
    ax.tick_params(axis="y", labelsize=35)
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig(filename, pi=fig.dpi)
    plt.show()

def plot_reinforce(score, save_path=None):
    moving_avg_reward = []
    window = 1000
    for i in range(window, len(score)):
        moving_avg_reward.append(sum(score[i-window:i])/window)

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.plot(range(window, len(score)), moving_avg_reward)
    ax.set(xlabel='Episode', ylabel='Success Rate', title='Expected reward with a moving average')
    if save_path:
        plt.savefig(save_path)

    plt.show()






