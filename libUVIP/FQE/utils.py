import numpy as np
import torch
import logging

def prepare_dataset(env, agent_b, agent_e, T, H, n_actions, gamma=0.99):
    logger = logging.getLogger("fqe")
    initial_states = []
    states = []
    next_states = []
    next_scores = []
    actions = []
    rewards = []
    dones = []

    total_reward = 0
    for tt in range(T):
        cur_state = env.reset()
        initial_states.append(cur_state)

        total_gamma = 1
        for h in range(H):
            action = agent_b.policy(cur_state)
            next_state, reward, done, _ = env.step(action)
            next_score = agent_e.score(next_state)
            # next_score = np.zeros(n_actions, dtype=np.float32)
            # next_score[agent_e.policy(next_state)] = 1

            states.append(cur_state)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            next_states.append(next_state)
            next_scores.append(next_score)

            if done:
                break

            total_reward += total_gamma * reward
            total_gamma *= gamma
            cur_state = next_state


    logger.info(f"Total discounted reward is {total_reward / T}")

    return states, initial_states, next_states, next_scores, actions, rewards, dones