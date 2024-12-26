from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random
import wandb

import logging

class QNetwork(nn.Module):
    """
    A neural network model for estimating Q-values in reinforcement learning.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        gamma (float): Discount factor for future rewards.
        use_action_emb (bool, optional): Whether to use action embeddings. Defaults to False.
        hidden_size (int, optional): Number of hidden units in each layer. Defaults to 128.
    """
    def __init__(self, state_dim, n_actions, hidden_size=128):
        super().__init__()

        self.n_actions = n_actions
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = state

        q_vals = self.relu(self.fc1(x))
        q_vals = self.relu(self.fc2(q_vals))
        q_vals = self.fc3(q_vals)

        return q_vals


class RLDatasetOffline:
    def __init__(self, config):
        self.states = config["states"]
        self.next_states = config["next_states"]
        self.next_scores = config["next_scores"]
        self.actions = config["actions"]
        self.rewards = config["rewards"]
        self.dones = config["dones"]
        self.gamma = config["gamma"]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        logger = logging.getLogger("fqe")

        state = torch.FloatTensor(self.states[idx])
        next_state = torch.FloatTensor(self.next_states[idx])
        next_score = torch.FloatTensor(self.next_scores[idx])
        action = self.actions[idx]
        reward = self.rewards[idx]
        done = float(self.dones[idx])


        return reward, state, next_state, next_score, action, done


class FQE:
    def __init__(self,
                 dataset,
                 initial_states,
                 repr_states,
                 optim_conf,
                 state_dim,
                 n_actions,
                 n_epochs,
                 pi_e,
                 hidden_size,
                 gamma,
                 tau,
                 device):

        self.dataset = dataset
        self.initial_states = initial_states
        self.repr_states = repr_states
        self.optim_conf = optim_conf
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.pi_e = pi_e
        self.device = device
        self.tau = tau
        self.n_epochs = n_epochs

        self.q = QNetwork(self.state_dim,
                          self.n_actions,
                          self.hidden_size).to(self.device)

    @staticmethod
    def copy_over_to(source, target):
        """
        Copies parameters from the source network to the target network.
        """
        target.load_state_dict(source.state_dict())

    @staticmethod
    def soft_update(source, target, tau):
        """
        Performs a soft update of the target network parameters.
        """
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    tau * source_param.data + (1.0 - tau) * target_param.data
                )

    def train(self, batch_size, plot_info=False):
        logger = logging.getLogger("fqe")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.q.parameters()), **self.optim_conf)
        q_prev = QNetwork(self.state_dim,
                          self.n_actions,
                          self.hidden_size).to(self.device)

        self.copy_over_to(self.q, q_prev)

        values = []
        relative_err_hist = []
        repr_vals_prev = torch.zeros(self.repr_states.shape[0])

        total_time = 0
        timestamps = []

        for epoch in range(self.n_epochs):
            start_time = time.time()

            dataloader = DataLoader(self.dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=8,
                                    prefetch_factor=10,
                                    drop_last=True)

            loss_history = []
            cur_it = 0
            for rewards, states, next_states, next_scores, actions, dones in tqdm(dataloader, total=len(dataloader)):
                rewards = rewards.to(self.device)
                states = states.to(self.device)
                next_states = next_states.to(self.device)
                next_scores = next_scores.to(self.device)
                actions = actions.to(self.device)
                dones = dones.to(self.device)

                with torch.no_grad():
                    q_vals = q_prev(next_states)
                    q_vals = (next_scores * q_vals).sum(axis=-1)

                    y = rewards + self.gamma * q_vals * (1 - dones)

                preds = self.predict(states, actions.unsqueeze(-1)).squeeze(-1) # (bs)

                assert len(preds.shape) == 1
                assert len(y.shape) == 1

                loss = torch.mean((preds - y)**2)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
                optimizer.step()

                loss_history.append(loss.item())
                if self.tau > 0:
                    self.soft_update(self.q, q_prev, self.tau)

                cur_it += 1

            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            timestamps.append(total_time)

            actions = torch.LongTensor([self.pi_e.policy(state) for state in self.initial_states]).to(self.device)
            valid_q_vals = self.predict(torch.FloatTensor(self.initial_states).to(self.device), actions.unsqueeze(-1)).squeeze(-1).detach()
            values.append(valid_q_vals.mean().item())

            M = 10

            actions_repr = torch.LongTensor([self.pi_e.policy(state) for state in self.repr_states]).to(self.device)
            repr_vals_cur = self.predict(torch.FloatTensor(self.repr_states).float().to(self.device), actions_repr.unsqueeze(-1)).squeeze(-1).detach()
            relative_err = torch.mean(((repr_vals_cur - repr_vals_prev)/repr_vals_cur)**2)**0.5

            relative_err_hist.append(relative_err)

            logger.info(f"{values[-1]}, {np.mean(values[-M:])}, {np.abs(np.mean(values[-M:]) - np.mean(values[-(M+1):-1]))}, relative error: {relative_err}, execution time: {total_time:.6f} seconds")

            repr_vals_prev = repr_vals_cur

            if plot_info:
                self.plot_info(loss_history, values)

            if self.tau > 0:
                self.soft_update(self.q, q_prev, self.tau)
            else:
                self.copy_over_to(self.q, q_prev)

            logger.info(f"Finished Epoch {epoch}.")

        return values, relative_err_hist, timestamps


    def predict(self, states, actions):
        q_vals = torch.take_along_dim(self.q(states), actions, dim=1)

        return q_vals

    def plot_info(self, loss_history, values):
        fig = plt.figure(figsize=(20, 10))

        fig.add_subplot(1, 2, 1)
        plt.plot(loss_history[::5])
        plt.yscale("log")
        plt.grid(True)

        fig.add_subplot(1, 2, 2)
        plt.plot(values)
        plt.grid(True)

        plt.savefig("plot.png")
        plt.show()
        plt.close()





















