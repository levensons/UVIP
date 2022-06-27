import torch

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def compute_target(target_net, value_net, buf, states, rewards, next_states, is_dones, gamma=0.9):
    target_upper = target_net(next_states.reshape(-1, *buf._state_dim)).detach().squeeze().reshape(-1, buf._action_dim)
    target_lower = value_net(next_states.reshape(-1, *buf._state_dim)).detach().max(dim=-1)[0].reshape(-1, buf._action_dim)
    target = torch.max(rewards + gamma * (target_upper - target_lower + buf.predict_transition_value(states)) * (1 - is_dones), dim=-1)[0]

    return target