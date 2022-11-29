import torch
import numpy as np
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
from tqdm import tqdm_notebook
import wandb
from time import perf_counter
from utils import compute_target, update_target

class TransitionsDataset(Dataset):
    def __init__(self, buf):
        self.states = buf._train_states
        self.rewards = buf._train_rewards
        self.next_states = buf._train_next_states
        self.is_dones = buf._train_dones

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        sample = {'states': self.states[idx], 'rewards': self.rewards[idx],
                  'next_states': self.next_states[idx], 'is_dones': self.is_dones[idx]}
        return sample

def grad_step(buf, upper_net, target_net, value_net, opt, samples, batch_size=32, gamma=0.9):
    '''
        In place where done = True we should just return reward in L1_target
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = samples['states'].to(device)
    rewards = samples['rewards'].to(device)
    next_states = samples['next_states'].to(device)
    is_dones = samples['is_dones'].to(device)

    target = compute_target(target_net, value_net, buf, states, rewards, next_states, is_dones, gamma)
    predicted_upper_values = upper_net(states).squeeze()

    assert target.shape == predicted_upper_values.shape

    loss = torch.mean((target - predicted_upper_values)**2)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss, predicted_upper_values.detach(), target

def train(value_net, upper_net, target_net, buf, state_dim, action_dim, trajectory, config):
    """
        Make input parameters to be variables inside function, global parameters are recieved according
        to config file.
    """

    eval_freq = config['eval_freq']
    update_freq = config['update_freq']
    n_epochs = config['n_epochs']
    lr = config['learning_rate_outer']
    batch_size = config['batch_size_outer']
    gamma = config['gamma']

    wandb_config = dict(
        learning_rate = lr,
        batch_size = batch_size,
        architecture = "CNN",
    )
    wandb.init(project='UVIPNN', config=wandb_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = torch.optim.Adam(upper_net.parameters(), lr=lr, weight_decay=1e-5)
    wandb.watch(upper_net, log_freq=eval_freq)
    running_loss = 0.0

    dataset = TransitionsDataset(buf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start = perf_counter()

    for epoch in tqdm_notebook(range(n_epochs)):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            batch_states = data['states']
            batch_rewards = data['rewards']
            batch_next_states = data['next_states']
            batch_is_dones = data['is_dones']
            samples = {'states' : batch_states,
                       'rewards': batch_rewards,
                       'next_states': batch_next_states,
                       'is_dones': batch_is_dones}

            loss, predicted_upper_values, target = grad_step(buf, upper_net, target_net, value_net, opt, samples, \
                                                             batch_size=batch_size, gamma=gamma)
            running_loss += loss.detach().cpu().item()

            if i % eval_freq == eval_freq - 1:
                with torch.no_grad():
                    test_states, test_rewards, test_next_states, test_dones = buf.get_samples_test(batch_size)
                    test_states = test_states.to(device)
                    test_next_states = test_next_states.to(device)
                    test_rewards = test_rewards.to(device)
                    test_is_dones = test_dones.to(device)
                    test_target = compute_target(target_net, value_net, buf, test_states, test_rewards,\
                                                 test_next_states, test_is_dones, gamma)
                    test_predicted_upper_values = upper_net(test_states).detach().squeeze()
                    test_loss = torch.mean((test_target - test_predicted_upper_values)**2)
                    upper_bounds = upper_net(torch.FloatTensor(trajectory).to(device)).squeeze()
                    data = [[x, y] for (x, y) in zip(range(len(trajectory)), upper_bounds.detach().cpu().tolist())]
                    table = wandb.Table(data=data, columns =["x", "y"])

                wandb.log(
                    {
                        'batch': i + 1,
                        'epoch': epoch + 1,
                        'train_loss': running_loss / eval_freq,
                        'test_loss': test_loss,
                        'upper_bounds': wandb.plot.line(table, "x", "y", title="upper bounds"),
                        'train_upper_bounds': wandb.plot.line_series(
                                            xs=list(range(len(trajectory))), ys=[predicted_upper_values.cpu().tolist(),\
                                                                                  target.cpu().tolist()],\
                                            keys=['predicted_upper_values', 'target'],\
                                            title='train_upper_bounds',\
                                            xname='states'
                        )
                    }
                )

                running_loss = 0.0

        if i % update_freq:
            update_target(upper_net, target_net)

    end = perf_counter()
    print(f"Training time : {end-start}")

    wandb.finish()