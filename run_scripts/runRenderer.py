from rlberry.rendering import RenderInterface2D
import numpy as np
import matplotlib.pyplot as plt
import logging
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
# from libUVIP.continuous_envs.envs.envs import WrappedTwinRooms as TwinRooms
from matplotlib import cm
import matplotlib
from sklearn.cluster import KMeans
import matplotlib.tri as tri
import os

logger = logging.getLogger('UVIP')

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

def interpolate(points, values, colormap_name="hot", mn=None, mx=None):
    colormap_fn = plt.get_cmap(colormap_name)

    if mn is None:
        mn = np.min(values)
    if mx is None:
        mx = np.max(values)

    norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx)
    triang = tri.Triangulation(points[:, 0], points[:, 1])
    triang.set_mask(np.hypot(points[triang.triangles, 0].mean(axis=1),
                             points[triang.triangles, 1].mean(axis=1)) < 0.01)

    refiner = tri.UniformTriRefiner(triang)
    tri_refi, values_refi = refiner.refine_field(values, subdiv=3)
    plt.gca().tricontourf(tri_refi, values_refi, levels=50, cmap=colormap_fn, norm=norm)

def split_by_room(gap_pi, repr_states):
    room1_mask = (repr_states[:, 0] <= 0.95)
    room2_mask = (repr_states[:, 0] >= 1.05)

    room1_states = repr_states[room1_mask]
    room2_states = repr_states[room2_mask]
    gap_pi_r1 = gap_pi[room1_mask]
    gap_pi_r2 = gap_pi[room2_mask]

    return room1_states, gap_pi_r1, room2_states, gap_pi_r2

def load_data(path):
    Vpi = np.load(os.path.join(path, "Vpi.npy"))
    Vup = np.load(os.path.join(path, "Vup.npy"))
    repr_states = np.load(os.path.join(path, "ReprStates.npy"))

    return Vpi, Vup, repr_states

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('logs/logsRenderer.log', mode="w")
    fh.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)

    env = TwinRooms()
    Vpi_exp1, Vup_exp1, repr_states_exp1 = load_data("TwinRoomsExp2.5k")
    Vpi_exp2, Vup_exp2, repr_states_exp2 = load_data("TwinRoomsExp5k")

    repr_states = repr_states_exp1
    room1_mask = (repr_states[:, 0] <= 0.95)
    room2_mask = (repr_states[:, 0] >= 1.05)
    logger.info(Vpi_exp1[room1_mask])
    logger.info(Vpi_exp1[room2_mask])

    gap_pi_exp1 = Vup_exp1 - Vpi_exp1
    gap_pi_exp2 = Vup_exp2 - Vpi_exp2
    mn = min(gap_pi_exp1.min(), gap_pi_exp2.min())
    mx = max(gap_pi_exp1.max(), gap_pi_exp2.max())

    room1_states_exp1, gap_pi_r1_exp1,\
    room2_states_exp1, gap_pi_r2_exp1 = split_by_room(gap_pi_exp1, repr_states_exp1)
    room1_states_exp2, gap_pi_r1_exp2,\
    room2_states_exp2, gap_pi_r2_exp2 = split_by_room(gap_pi_exp2, repr_states_exp2)

    print(gap_pi_r1_exp1.mean())
    print(gap_pi_r1_exp2.mean())

    fig = plt.figure(figsize=(20, 10))
    # fig.add_subplot(2, 1, 1)
    get_background(env)
    # get_scenes(env, repr_states, gap_pi_norm)
    interpolate(room1_states_exp1, gap_pi_r1_exp1, colormap_name="RdBu_r", mn=mn, mx=mx)
    interpolate(room2_states_exp1, gap_pi_r2_exp1, colormap_name="RdBu_r", mn=mn, mx=mx)
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    colormap_fn = plt.get_cmap("RdBu_r")
    # norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap_fn)
    cbar = plt.colorbar(scalar_map, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=35)
    plt.grid(False)
    plt.xlim(0, 2)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title("Gap for policy after 2.5k iterations of KBUCBVI", fontsize=40)

    plt.tight_layout()
    plt.savefig("plotsTwinRooms/RenderedUVIP2.5k.png")
    plt.show()

    # fig.add_subplot(2, 1, 2)
    fig = plt.figure(figsize=(20, 10))
    get_background(env)
    # get_scenes(env, repr_states, gap_pi_norm)
    interpolate(room1_states_exp2, gap_pi_r1_exp2, colormap_name="RdBu_r", mn=mn, mx=mx)
    interpolate(room2_states_exp2, gap_pi_r2_exp2, colormap_name="RdBu_r", mn=mn, mx=mx)
    colormap_fn = plt.get_cmap("RdBu_r")
    norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap_fn)
    cbar = plt.colorbar(scalar_map, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=35)
    plt.grid(False)
    plt.xlim(0, 2)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title("Gap for policy after 5k iterations of KBUCBVI", fontsize=40)

    plt.tight_layout()
    plt.savefig("plotsTwinRooms/RenderedUVIP5k.png")
    plt.show()







