import numpy as np
import scipy as sp
from scipy import optimize
from scipy import stats
from IPython.display import clear_output
import matplotlib.pyplot as plt
import scipy

class BaseEnvironment:
    def __init__(self):
        '''
            Base class of RL-environment
            Fields:
                n_state : number of states of MDP
                n_action : number of actions of MDP
                P : tensor of transitions p(s'|s,a) of size (n_state, n_state, n_action)
                R : matrix of deterministic rewards r(a,s) of size (n_state, n_action)
        '''
        self.n_state = None
        self.n_action = None
        self.P = None
        self.R = None
        self.counter = None
        self.max_steps = None
        self.terminal_state = None

    def reset(self):
        self.counter = 0
        return 0

    def step(self, state, action, n_samples=1):
        '''
            Make one step corresponds to dynamics of MDP
            Args:
                state - current state
                action - current action
            Returns: next state s'
        '''
        self.counter += 1

        done = (self.counter == self.max_steps)
        next_state = np.random.choice(self.n_state, p=self.P[:,state,action], size=n_samples).item()
        done = done or self.is_terminal(next_state)

        return next_state, self.R[state, action], done

    def compute_Vpi(self, policy, gamma):
        b = np.sum(self.R * policy, axis=1)
        A = np.einsum('km,nkm->nk', policy, self.P).T

        A = np.eye(self.n_state) - gamma * A

        Vpi = np.linalg.solve(A, b)

        return Vpi


    def compute_Vstar(self, gamma, seed=42, eps=1e-5, plot_norm=False):
        Q_prev = np.random.randn(self.n_state, self.n_action)
        policy_prev = generate_pi(self.n_action, self.n_state, seed)
        norm_list = []
        j = 0

        while True:
            Q = self.R + gamma * np.einsum("n, nkm -> km", np.max(Q_prev, axis=1), self.P)

            if j > 2:
                norm_list.append(np.max(np.abs(Q_prev - Q)))
                if plot_norm and (j % 10 == 0):
                    clear_output(wait=True)
                    fig = plt.figure(figsize=(8, 7))
                    ax = fig.add_subplot(1, 1, 1)
                    plt.plot(norm_list)
                    plt.yscale("log")
                    plt.title("Norm lower")
                    plt.show()

                if norm_list[-1] < eps:
                    break
            Q_prev = Q.copy()

            policy_determ = np.zeros((self.n_state, self.n_action))
            for i in range(self.n_state):
                policy_determ[i, np.argmax(Q, axis=1)[i]] = 1.0

            policy_prev = policy_determ.copy()
            j += 1

        Vstar = self.compute_Vpi(policy_determ, gamma)

        return Vstar, policy_determ


    def state_transitions(self, policy, greedy=False):
        '''
            Return state transition matrix P_s = p_policy(s' | s)
            Args:
                policy: if greedy, vector of size (n_state)
                        else: matrix of size (n_state, n_action)
            returns:
                matrix of size (n_state, n_state)
        '''
        P_s = np.zeros((self.n_state, self.n_state),dtype = float)
        for i in range(self.n_state):
            for j in range(self.n_state):
                if greedy:
                    P_s[i,j] = self.P[i,j,policy[j]]
                else:
                    P_s[i,j] = np.dot(self.P[i,j,:], policy[:,j])
        return P_s

    def state_reward(self, policy, greedy=False):
        '''
            Return state reward vector r = p_policy(s)
            Args:
                policy: if greedy, vector of size (n_state)
                        else: matrix of size (n_state, n_action)
            returns:  P_true[s, 1][min(nState - 1, s + 1)] = 0.35
            P_true[s, 1][s] = 0.6
            P_true[s, 1][max(0, s-1)] = 0.05

                vector of size (n_state)
        '''
        r = np.zeros(self.n_state, dtype = float)
        for i in range(self.n_state):
            if greedy:
                r[i] = self.R[policy[i], i]
            else:
                r[i] = np.dot(self.R[:,i], policy[:, i])
        return r

    def is_terminal(self, state):
        return state == self.terminal_state