import rlberry
import numpy as np
import matplotlib.pyplot as plt
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms

def get_background(env):
    eps = env.wall_eps

    wall = plt.Rectangle((1 - eps, 0), 2*eps, 1, fc=(0.25, 0.25, 0.25))
    plt.gca().add_patch(wall)

    for (x, y) in [
        env.base_reward_pos,
        env.base_reward_pos + np.array([1.0, 0.0]),
    ]:
        reward = plt.Circle((x, y), radius=0.1, fc=(0.0, 0.5, 0.0))
        plt.gca().add_patch(reward)

def get_scenes(states, gap_pi, colormap_name="hot"):
    colormap_fn = plt.get_cmap(colormap_name)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap_fn)
    for i, (x, y) in enumerate(states):
        state = plt.Circle((x, y), radius=0.02, fc=scalar_map.to_rgba(gap_pi[i]))
        plt.gca().add_patch(state)

env = TwinRooms()
# env = DiscretizeStateWrapper(env, n_bins=20)

# observation, info = env.reset()

plt.figure(figsize=(20, 10))

get_background(env)

plt.tight_layout()

plt.savefig("twinrooms.png")
plt.show()

# background, scenes = env._get_background_and_scenes()
# print(background)
# print(background.shapes)