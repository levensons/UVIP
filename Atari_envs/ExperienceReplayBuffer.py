import torch
import random
import numpy as np
import wandb
from UpperBoundsNNsoftmax import TransitionsDataset
from models import FNetwork
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm_notebook
from tqdm import tqdm
import matplotlib.pyplot as plt
from wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, wrap_deepmind, wrap_pytorch
import copy
from pickle import dumps, loads
import cv2
cv2.ocl.setUseOpenCL(False)

class ExperienceReplayBuffer:
    '''
        Generates experience replay of shape [size, action_dim, state_dim]
    '''
    def __init__(self, env, agent, state_dim, action_dim, size=10000, val=0.25):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._network = FNetwork(state_dim, action_dim).to(device)
        self.width = 84
        self.height = 84
        test_size = int(size * val)
        train_size = size - test_size
        self._test_size = test_size
        self._train_size = train_size

        train_buffer_shape = (train_size, action_dim) + state_dim
        self._train_next_states = torch.zeros(train_buffer_shape)
        self._train_states = torch.zeros((train_size, *state_dim))
        self._train_rewards = torch.zeros((train_size, action_dim))
        self._train_dones = torch.zeros((train_size, action_dim))

        test_buffer_shape = (test_size, action_dim) + state_dim
        self._test_next_states = torch.zeros(test_buffer_shape)
        self._test_states = torch.zeros((test_size, *state_dim))
        self._test_rewards = torch.zeros((test_size, action_dim))
        self._test_dones = torch.zeros((test_size, action_dim))

        self._size = train_size
        self._action_dim = action_dim
        self._state_dim = state_dim

        self.fill_buffer_train(env, agent, train_size)
        self.fill_buffer_test(env, agent, test_size)

    def fit_kernel(self, value_net, batch_size=512, n_epochs=10, log_freq=5):
        '''
            Neural network fitting for kernel approximation
        '''
        config = dict(
            learning_rate = 1e-4,
            batch_size = batch_size,
            n_epochs = n_epochs,
            architecture = "CNN"
        )
        wandb.init(project='UVIPNN', config=config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = TransitionsDataset(self)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        eval_freq = len(dataset) // batch_size
        eval_freq //= log_freq
        opt = torch.optim.Adam(self._network.parameters(), lr=1e-4)
        wandb.watch(self._network, log_freq=log_freq)
        train_loss_history = []
        test_loss_history = []
        for epoch in tqdm_notebook(range(n_epochs)):
            running_loss = 0.0
            running_test_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                batch_states = data['states'].to(device)
                batch_rewards = data['rewards'].to(device)
                batch_next_states = data['next_states'].to(device)
                predicted_values_for_actions = self._network(batch_states)
                train_target = value_net(batch_next_states.reshape(-1, *self._state_dim)).detach().max(dim=-1)[0].reshape(-1, self._action_dim)
                train_loss = torch.mean((train_target - predicted_values_for_actions)**2)
                opt.zero_grad()
                train_loss.backward()
                opt.step()

                running_loss += train_loss.item()
                if i % eval_freq == eval_freq - 1:
                    test_states, test_rewards, test_next_states, test_dones = self.get_samples_test(batch_size)
                    test_states = test_states.to(device)
                    test_next_states = test_next_states.to(device)
                    predicted_test = self._network(test_states).detach().cpu()
                    test_target = value_net(test_next_states.reshape(-1, *self._state_dim)).detach().cpu().max(dim=-1)[0].reshape(-1, self._action_dim)
                    test_loss = torch.mean((test_target - predicted_test)**2)
                    wandb.log({'epoch': epoch + 1,
                               'batch': i + 1,
                               'train_loss': running_loss,
                               'test_loss': test_loss / eval_freq}
                    )
                    # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / eval_freq}')
                    running_loss = 0.0

        print('Finished Training of kernel')
        wandb.finish()


    def get_samples_test(self, n_samples=-1):
        '''
          Generate n_samples samples of shape (action_dim, 2, state_dim)
        '''
        if n_samples == -1:
            n_samples = self._size

        idxes = np.random.randint(self._test_size, size=n_samples)
        batch_states = self._test_states[idxes]
        batch_rewards = self._test_rewards[idxes]
        batch_next_states = self._test_next_states[idxes]
        batch_dones = self._test_dones[idxes]

        return batch_states, batch_rewards, batch_next_states, batch_dones

    def process_state(self, state):
        processed_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        processed_state = cv2.resize(processed_state, (self.width, self.height), interpolation=cv2.INTER_AREA)
        processed_state = np.array(processed_state).astype(np.float32) / 255.0
        return np.swapaxes(processed_state, 0, 1)

    def fill_buffer_train(self, env, agent, size, thres=0):
        '''
            Algorithm is the following: copy the frame buffer, then get new state from unwrapped env and then push it to the buffer
            The only problem is that we should do a scaling.
        '''
        s = env.reset()
        stored_t = 0
        p = 0.5
        env_unw = copy.deepcopy(env.unwrapped)
        env_unw = NoopResetEnv(env_unw, noop_max=30)
        env_unw = MaxAndSkipEnv(env_unw, skip=4)
        with tqdm(total=size) as pbar:
            while stored_t < size:
                saved_env = env.unwrapped.clone_full_state()
                if random.random() > p:
                # if t >= thres:
                    self._train_states[stored_t] = torch.FloatTensor(s)
                    for action in range(self._action_dim):
                        env_unw.unwrapped.restore_full_state(saved_env)
                        next_s, reward, done, _ = env_unw.step(action)
                        fstack = np.stack([s[1], s[2], s[3], self.process_state(next_s)], axis=0).astype(np.float32)
                        self._train_next_states[stored_t, action] = torch.FloatTensor(fstack)
                        self._train_rewards[stored_t, action] = reward
                        self._train_dones[stored_t, action] = done

                    stored_t += 1

                    pbar.update(1)

                action = agent.act(s, 0)
                next_s, reward, done, _ = env.step(action)

                # _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 10))
                # ax1.imshow(next_s[0, :, :])
                # ax2.imshow(next_s[1, :, :])
                # ax3.imshow(next_s[2, :, :])
                # ax4.imshow(next_s[3, :, :])
                # plt.show()

                s = next_s
                if done:
                    s = env.reset()

    def fill_buffer_test(self, env, agent, size):
        s = env.reset()
        env_unw = copy.deepcopy(env.unwrapped)
        env_unw = NoopResetEnv(env_unw, noop_max=30)
        env_unw = MaxAndSkipEnv(env_unw, skip=4)
        for i in tqdm_notebook(range(size)):
            saved_env = env.unwrapped.clone_full_state()
            self._test_states[i] = torch.FloatTensor(s)
            for action in range(self._action_dim):
                env_unw.unwrapped.restore_full_state(saved_env)
                next_s, reward, done, _ = env_unw.step(action)
                fstack = np.stack([s[1], s[2], s[3], self.process_state(next_s)], axis=0).astype(np.float32)
                self._test_next_states[i, action] = torch.FloatTensor(fstack)
                self._test_rewards[i, action] = reward

            action = agent.act(s, 0)
            next_s, reward, done, _ = env.step(action)

            for action in range(self._action_dim):
                self._test_dones[i, action] = done

            s = next_s
            if done:
                s = env.reset()

    def save_net(self, path):
        torch.save(self._network.state_dict(), path)

    def load_net(self, path):
        self._network.load_state_dict(torch.load(path))

    def predict_transition_value(self, states):
        '''
            predict transition for batch of states/actions
        '''

        return self._network(states).detach()

