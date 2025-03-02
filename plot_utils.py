
from matplotlib.patches import Rectangle
import numpy as np


def box_plot(position, data, color, ax, box_width=0.1):
    ax.plot(np.array([position - box_width, position - box_width]), np.array([np.min(data), np.max(data)]), color=color)
    ax.plot(np.array([position - 1.5*box_width, position - 0.5*box_width]), np.array([np.min(data), np.min(data)]), color=color)
    ax.plot(np.array([position - 1.5*box_width, position - 0.5*box_width]), np.array([np.max(data), np.max(data)]), color=color)
    ax.plot(position - box_width, np.mean(data), "s", color=color)
    ax.add_patch(
        Rectangle((position - 1.5*box_width, np.mean(data) - np.std(data)), box_width, 2 * np.std(data), color=color, alpha=0.3)
    )

def get_q_qdot_from_data(n_shooting, nb_random, q_roots, q_joints, qdot_roots, qdot_joints):
    n_root = 3
    n_joints = q_joints.shape[0] // nb_random
    n_q = n_root + n_joints
    q = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot = np.zeros((n_q, n_shooting + 1, nb_random))
    for i_random in range(nb_random):
        for i_shooting in range(n_shooting + 1):
            q[:, i_shooting, i_random] = np.hstack(
                (
                    q_roots[i_random * n_root: (i_random + 1) * n_root, i_shooting],
                    q_joints[i_random * n_joints: (i_random + 1) * n_joints, i_shooting],
                )
            )
            qdot[:, i_shooting, i_random] = np.hstack(
                (
                    qdot_roots[i_random * n_root: (i_random + 1) * n_root, i_shooting],
                    qdot_joints[i_random * n_joints: (i_random + 1) * n_joints, i_shooting],
                )
            )
    return q, qdot
    
