import os
import pickle
from matplotlib.patches import Rectangle
import numpy as np


def box_plot(position, data, color, ax, box_width=0.05):
    ax.plot(np.array([position, position]), np.array([np.min(data), np.max(data)]), color=color)
    ax.plot(np.array([position - box_width, position + box_width]), np.array([np.min(data), np.min(data)]), color=color)
    ax.plot(np.array([position - box_width, position + box_width]), np.array([np.max(data), np.max(data)]), color=color)
    ax.plot(position, np.mean(data), "s", color=color)
    ax.add_patch(
        Rectangle((position - box_width, np.mean(data) - np.std(data)), 2 * box_width, 2 * np.std(data), color=color, alpha=0.3)
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

def define_q_mean(n_shooting, nb_random, q_roots, q_joints, qdot_roots, qdot_joints):
    q, qdot = get_q_qdot_from_data(n_shooting, nb_random, q_roots, q_joints, qdot_roots, qdot_joints)
    q_mean = np.mean(q, axis=2)
    qdot_mean = np.mean(qdot, axis=2)
    return q, qdot, q_mean, qdot_mean

def get_optimization_q_each_random(n_shooting):
    nb_randoms = [5, 10, 15]
    base_file_name = "Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02"
    file_names = ["DMS", "VARIABLE_DMS", "FEEDFORWARD_DMS", "VARIABLE_FEEDFORWARD_DMS"]
    q_socp, qdot_socp, time_vector_socp = {"5random": None, "10random": None, "15random": None}, {"5random": None, "10random": None, "15random": None}, {"5random": None, "10random": None, "15random": None}
    q_socp_variable, qdot_socp_variable, time_vector_socp_variable = {"5random": None, "10random": None, "15random": None}, {"5random": None, "10random": None, "15random": None}, {"5random": None, "10random": None, "15random": None}
    q_socp_feedforward, qdot_socp_feedforward, time_vector_socp_feedforward = {"5random": None, "10random": None, "15random": None}, {"5random": None, "10random": None, "15random": None}, {"5random": None, "10random": None, "15random": None}
    q_socp_plus, qdot_socp_plus, time_vector_socp_plus = {"5random": None, "10random": None, "15random": None}, {"5random": None, "10random": None, "15random": None}, {"5random": None, "10random": None, "15random": None}
    for current_name in file_names:
        for i_random, current_random in enumerate(nb_randoms):
            file_name = f"results/{current_random}random/{base_file_name}_{current_name}_{current_random}random_CVG_1p0e-06.pkl"
            if not os.path.exists(file_name):
                if not os.path.exists(file_name.replace("CVG", "DVG")):
                    raise RuntimeError(f"The results file {file_name} is missing")
                else:
                    continue
            with open(file_name, "rb") as f:
                data_this_random = pickle.load(f)
                q_this_nb_random, qdot_this_nb_random = get_q_qdot_from_data(
                    n_shooting,
                    current_random,
                    data_this_random["q_roots_sol"],
                    data_this_random["q_joints_sol"],
                    data_this_random["qdot_roots_sol"],
                    data_this_random["qdot_joints_sol"]
                )
            if current_name == "DMS":
                q_socp[f"{current_random}random"] = q_this_nb_random
                qdot_socp[f"{current_random}random"] = qdot_this_nb_random
                time_vector_socp[f"{current_random}random"] = np.linspace(0, float(data_this_random["time_sol"]), n_shooting + 1)
            elif current_name == "VARIABLE_DMS":
                q_socp_variable[f"{current_random}random"] = q_this_nb_random
                qdot_socp_variable[f"{current_random}random"] = qdot_this_nb_random
                time_vector_socp_variable[f"{current_random}random"] = np.linspace(0, float(data_this_random["time_sol"]),
                                                                          n_shooting + 1)
            elif current_name == "FEEDFORWARD_DMS":
                q_socp_feedforward[f"{current_random}random"] = q_this_nb_random
                qdot_socp_feedforward[f"{current_random}random"] = qdot_this_nb_random
                time_vector_socp_feedforward[f"{current_random}random"] = np.linspace(0, float(data_this_random["time_sol"]),
                                                                          n_shooting + 1)
            elif current_name == "VARIABLE_FEEDFORWARD_DMS":
                q_socp_plus[f"{current_random}random"] = q_this_nb_random
                qdot_socp_plus[f"{current_random}random"] = qdot_this_nb_random
                time_vector_socp_plus[f"{current_random}random"] = np.linspace(0, float(data_this_random["time_sol"]),
                                                                          n_shooting + 1)
    return q_socp, qdot_socp, q_socp_variable, qdot_socp_variable, q_socp_feedforward, qdot_socp_feedforward, q_socp_plus, qdot_socp_plus, time_vector_socp, time_vector_socp_variable, time_vector_socp_feedforward, time_vector_socp_plus
