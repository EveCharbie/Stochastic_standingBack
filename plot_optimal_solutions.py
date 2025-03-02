import os
import pickle

import bioviz
import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

from DMS_deterministic import prepare_ocp
from DMS_SOCP import prepare_socp
from DMS_SOCP_VARIABLE import prepare_socp_VARIABLE
from DMS_SOCP_FEEDFORWARD import prepare_socp_FEEDFORWARD
from DMS_SOCP_VARIABLE_FEEDFORWARD import prepare_socp_VARIABLE_FEEDFORWARD
from utils import (
    DMS_sensory_reference,
    motor_acuity,
    DMS_fb_noised_sensory_input_VARIABLE_no_eyes,
    DMS_ff_noised_sensory_input,
    DMS_sensory_reference_no_eyes,
    DMS_ff_sensory_input,
    visual_noise,
    vestibular_noise,
    DMS_fb_noised_sensory_input_VARIABLE,
)
from plot_utils import box_plot, get_q_qdot_from_data
from plot_reintegrate import (
    noisy_integrate_ocp,
    noisy_integrate_socp,
    noisy_integrate_socp_variable,
    noisy_integrate_socp_feedforward,
    noisy_integrate_socp_plus
)

# TODO: used the saved integration instead of recomputing each time !
    
def plot_CoM(states_integrated, model, n_shooting, save_path, nb_random=30):
    nx = states_integrated.shape[0]
    nq = int(nx / 2)
    time = np.linspace(0, final_time, n_shooting + 1)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = np.ravel(axs)
    CoM = np.zeros((nb_random, n_shooting + 1))
    CoMdot = np.zeros((nb_random, n_shooting + 1))
    PelvisRot = np.zeros((nb_random, n_shooting + 1))
    PelvisVelot = np.zeros((nb_random, n_shooting + 1))
    for j in range(nb_random):
        for k in range(n_shooting + 1):
            CoM[j, k] = model.CoM(states_integrated[:nq, j, k]).to_array()[1]
            CoMdot[j, k] = model.CoMdot(states_integrated[:nq, j, k], states_integrated[nq:, j, k]).to_array()[1]
            PelvisRot[j, k] = states_integrated[2, j, k]
            PelvisVelot[j, k] = states_integrated[nq + 2, j, k]
        axs[0].plot(time, CoM[j, :], color="tab:blue", alpha=0.2)
        axs[1].plot(time, CoMdot[j, :], color="tab:blue", alpha=0.2)
        axs[2].plot(time, PelvisRot[j, :], color="tab:blue", alpha=0.2)
        axs[3].plot(time, PelvisVelot[j, :], color="tab:blue", alpha=0.2)

    CoM_deterministic = np.zeros((n_shooting + 1))
    CoMdot_deterministic = np.zeros((n_shooting + 1))
    PelvisRot_deterministic = np.zeros((n_shooting + 1))
    PelvisVelot_deterministic = np.zeros((n_shooting + 1))
    for k in range(n_shooting + 1):
        CoM_deterministic[k] = model.CoM(q_sol[:, k]).to_array()[1]
        CoMdot_deterministic[k] = model.CoMdot(q_sol[:, k], qdot_sol[:, k]).to_array()[1]
        PelvisRot_deterministic[k] = q_sol[2, k]
        PelvisVelot_deterministic[k] = qdot_sol[2, k]
    axs[0].plot(time, CoM_deterministic, color="k")
    axs[1].plot(time, CoMdot_deterministic, color="k")
    axs[2].plot(time, PelvisRot_deterministic, color="k")
    axs[3].plot(time, PelvisVelot_deterministic, color="k")
    axs[0].set_title("CoM")
    axs[1].set_title("CoMdot")
    axs[2].set_title("PelvisRot")
    axs[3].set_title("PelvisVelot")

    plt.suptitle(f"CoM and Pelvis for {save_path}")
    plt.savefig(f"{save_path}_CoM.png")
    # plt.show()


def bioviz_animate(biorbd_model_path_with_mesh, q, name):
    b = bioviz.Viz(
        biorbd_model_path_with_mesh,
        mesh_opacity=1.0,
        background_color=(1, 1, 1),
        show_local_ref_frame=False,
        show_markers=False,
        show_segments_center_of_mass=False,
        show_global_center_of_mass=False,
        show_global_ref_frame=False,
        show_gravity_vector=False,
        # show_floor=True,
    )
    b.set_camera_zoom(0.39)
    b.maximize()
    b.update()
    b.load_movement(q)

    b.start_recording(f"videos/{result_folder}/" + name + ".ogv")
    for frame in range(q.shape[1] + 1):
        b.movement_slider[0].setValue(frame)
        b.add_frame()
    b.stop_recording()

def plot_comparison_reintegration(
    q_ocp_nominal,
    q_socp_nominal,
    q_socp_variable_nominal,
    q_socp_feedforward_nominal,
    q_socp_plus_nominal,
    q_all_socp,
    q_all_socp_variable,
    q_all_socp_feedforward,
    q_all_socp_plus,
    q_ocp_integrated,
    q_socp_integrated,
    q_socp_variable_integrated,
    q_socp_feedforward_integrated,
    q_socp_plus_integrated,
    time_vector,
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_plus_color,
    nb_random,
    nb_reintegrations,
):

    n_q = q_socp_plus_nominal.shape[0]
    fig, axs = plt.subplots(n_q, 5, figsize=(15, 10))
    for i_dof in range(n_q):

        # Reintegrated
        for i_random in range(nb_random * nb_reintegrations):
            if i_dof < 4:
                axs[i_dof, 0].plot(
                    time_vector, q_ocp_integrated[i_dof, :, i_random], color=OCP_color, alpha=0.2, linewidth=0.5
                )
                axs[i_dof, 1].plot(
                    time_vector, q_socp_integrated[i_dof, :, i_random], color=SOCP_color, alpha=0.2, linewidth=0.5
                )
                axs[i_dof, 2].plot(
                    time_vector, q_socp_variable_integrated[i_dof, :, i_random], color=SOCP_VARIABLE_color, alpha=0.2, linewidth=0.5
                )
            elif i_dof > 4:
                axs[i_dof, 0].plot(
                    time_vector, q_ocp_integrated[i_dof - 1, :, i_random], color=OCP_color, alpha=0.2, linewidth=0.5
                )
                axs[i_dof, 1].plot(
                    time_vector, q_socp_integrated[i_dof - 1, :, i_random], color=SOCP_color, alpha=0.2, linewidth=0.5
                )
                axs[i_dof, 2].plot(
                    time_vector, q_socp_variable_integrated[i_dof - 1, :, i_random], color=SOCP_VARIABLE_color, alpha=0.2, linewidth=0.5
                )
            axs[i_dof, 3].plot(
                time_vector,
                q_socp_feedforward_integrated[i_dof, :, i_random],
                color=SOCP_FEEDFORWARD_color,
                alpha=0.2,
                linewidth=0.5,
            )
            axs[i_dof, 4].plot(
                time_vector,
                q_socp_plus_integrated[i_dof, :, i_random],
                color=SOCP_plus_color,
                alpha=0.2,
                linewidth=0.5,
            )

        # Optimzation variables
        for i_random in range(nb_random):
            if i_dof < 4:
                axs[i_dof, 1].plot(time_vector, q_all_socp[i_dof, :, i_random], color="#6C165C", linewidth=0.5)
                axs[i_dof, 2].plot(time_vector, q_all_socp_variable[i_dof, :, i_random], color="#D15C02", linewidth=0.5)
            elif i_dof > 4:
                axs[i_dof, 1].plot(time_vector, q_all_socp[i_dof - 1, :, i_random], color="#6C165C", linewidth=0.5)
                axs[i_dof, 2].plot(time_vector, q_all_socp_variable[i_dof - 1, :, i_random], color="#D15C02", linewidth=0.5)
            axs[i_dof, 3].plot(time_vector, q_all_socp_feedforward[i_dof, :, i_random], color="#400191", linewidth=0.5)
            axs[i_dof, 4].plot(time_vector, q_all_socp_plus[i_dof, :, i_random], color="#016C93", linewidth=0.5)

        # Nominal
        if i_dof < 4:
            axs[i_dof, 0].plot(time_vector, q_ocp_nominal[i_dof, :], color="k", linewidth=0.5)
            axs[i_dof, 1].plot(time_vector, q_socp_nominal[i_dof, :], color="k", linewidth=0.5)
            axs[i_dof, 2].plot(time_vector, q_socp_variable_nominal[i_dof, :], color="k", linewidth=0.5)
        elif i_dof > 4:
            axs[i_dof, 0].plot(time_vector, q_ocp_nominal[i_dof - 1, :], color="k", linewidth=0.5)
            axs[i_dof, 1].plot(time_vector, q_socp_nominal[i_dof - 1, :], color="k", linewidth=0.5)
            axs[i_dof, 2].plot(time_vector, q_socp_variable_nominal[i_dof - 1, :], color="k", linewidth=0.5)
        axs[i_dof, 3].plot(time_vector, q_socp_feedforward_nominal[i_dof, :], color="k", linewidth=0.5)
        axs[i_dof, 4].plot(time_vector, q_socp_plus_nominal[i_dof, :], color="k", linewidth=0.5)

        # Box plot of the distribution of the last frame
        if i_dof < 4:
            box_plot(time_vector[-1] + 0.2, q_ocp_integrated[i_dof, -1, :], OCP_color, axs[i_dof, 0])
            box_plot(time_vector[-1] + 0.2, q_socp_integrated[i_dof, -1, :], SOCP_color, axs[i_dof, 1])
            box_plot(time_vector[-1] + 0.2, q_socp_variable_integrated[i_dof, -1, :], SOCP_VARIABLE_color, axs[i_dof, 2])
            box_plot(time_vector[-1] + 0.2, q_socp_feedforward_integrated[i_dof, -1, :], SOCP_FEEDFORWARD_color, axs[i_dof, 3])
            box_plot(time_vector[-1] + 0.2, q_socp_plus_integrated[i_dof, -1, :], SOCP_plus_color, axs[i_dof, 4])
        elif i_dof > 4:
            box_plot(time_vector[-1] + 0.2, q_ocp_integrated[i_dof - 1, -1, :], OCP_color, axs[i_dof, 0])
            box_plot(time_vector[-1] + 0.2, q_socp_integrated[i_dof - 1, -1, :], SOCP_color, axs[i_dof, 1])
            box_plot(time_vector[-1] + 0.2, q_socp_variable_integrated[i_dof - 1, -1, :], SOCP_VARIABLE_color, axs[i_dof, 2])
            box_plot(time_vector[-1] + 0.2, q_socp_feedforward_integrated[i_dof - 1, -1, :], SOCP_FEEDFORWARD_color, axs[i_dof, 3])
            box_plot(time_vector[-1] + 0.2, q_socp_plus_integrated[i_dof, -1, :], SOCP_plus_color, axs[i_dof, 4])
        box_plot(time_vector[-1] + 0.2, q_socp_feedforward_integrated[i_dof, -1, :], SOCP_FEEDFORWARD_color, axs[i_dof, 3])
        box_plot(time_vector[-1] + 0.2, q_socp_plus_integrated[i_dof, -1, :], SOCP_plus_color, axs[i_dof, 4])

    axs[0, 0].plot(0, 0, color="k", linewidth=0.5, label="OCP")
    axs[0, 0].plot(0, 0, color="k", linewidth=0.5, label="SOCP nominal")
    axs[0, 0].plot(0, 0, color="k", linewidth=0.5, label="SOCP VARIABLE nominal")
    axs[0, 0].plot(0, 0, color="k", linewidth=0.5, label="SOCP FEEDFORWARD nominal")
    axs[0, 0].plot(0, 0, color="k", linewidth=0.5, label="SOCP+ nominal")
    axs[0, 0].plot(0, 0, color=OCP_color, linewidth=0.5, label="OCP reintegrated", alpha=0.5)
    axs[0, 0].plot(0, 0, color=SOCP_color, linewidth=0.5, label="SOCP reintegrated", alpha=0.5)
    axs[0, 0].plot(0, 0, color=SOCP_VARIABLE_color, linewidth=0.5, label="SOCP VARIABLE reintegrated", alpha=0.5)
    axs[0, 0].plot(0, 0, color=SOCP_FEEDFORWARD_color, linewidth=0.5, label="SOCP FEEDFORWARD reintegrated", alpha=0.5)
    axs[0, 0].plot(0, 0, color=SOCP_plus_color, linewidth=0.5, label="SOCP+ reintegrated", alpha=0.5)
    axs[0, 0].plot(0, 0, color="#6C165C", linewidth=0.5, label=f"SOCP {nb_random} models")
    axs[0, 0].plot(0, 0, color="#D15C02", linewidth=0.5, label=f"SOCP VARIABLE {nb_random} models")
    axs[0, 0].plot(0, 0, color="#400191", linewidth=0.5, label=f"SOCP FEEDFORWARD {nb_random} models")
    axs[0, 0].plot(0, 0, color="#016C93", linewidth=0.5, label=f"SOCP+ {nb_random} models")
    fig.subplots_adjust(right=0.8)
    # axs[0, 0].legend(bbox_to_anchor=(3.35, 1), loc="upper left")

    for i_axs_2 in range(3):
        for i_axs in range(n_q-1):
            axs[i_axs, i_axs_2].get_xaxis().set_visible(False)
        axs[-1, i_axs_2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])

    plt.subplots_adjust(bottom=0.05, top=0.95, right=0.95, left=0.05)
    plt.suptitle("Comparison of nominal, integrated and reintegrated solutions")
    plt.savefig(f"graphs/comparison_reintegration_5.png")
    # plt.show()

    return

def plot_motor_command(
        nb_random,
        normalized_time_vector,
        tau_joints_ocp,
        tau_joints_socp,
        tau_joints_socp_variable,
        tau_joints_socp_feedforward,
        tau_joints_socp_plus,
        joint_friction_ocp,
        joint_frictions_socp,
        joint_frictions_socp_variable,
        joint_frictions_socp_feedforward,
        joint_frictions_socp_plus,
        OCP_color,
        SOCP_color,
        SOCP_plus_color,
        motor_noises_socp,
        motor_noises_socp_variable,
        feedbacks_socp,
        feedbacks_socp_variable,
        feedbacks_socp_feedforward,
        feedbacks_socp_plus,
        feedforwards_socp_feedforward,
        feedforwards_socp_plus,
):

    # Plot the motor command
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    for i in range(5):
        for j in range(5):
            axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
    # Head
    axs[0, 0].step(normalized_time_vector, tau_joints_ocp[0, :], color=OCP_color, label="OCP")
    for i_random in range(nb_random):
        axs[0, 0].step(
            normalized_time_vector,
            tau_joints_socp[0, :] + motor_noises_socp[0, :, i_random],
            color=SOCP_color,
            label="SOCP",
            linewidth=0.5,
            alpha=0.5,
        )
        axs[0, 0].step(
            normalized_time_vector,
            tau_joints_socp_variable[0, :] + motor_noises_socp_variable[0, :, i_random],
            color=SOCP_VARIABLE_color,
            label="SOCPV",
            linewidth=0.5,
            alpha=0.5,
        )
        axs[0, 0].step(
            normalized_time_vector,
            tau_joints_socp_feedforward[0, :] + motor_noises_socp_feedforward[0, :, i_random],
            color=SOCP_FEEDFORWARD_color,
            label="SOCPA",
            linewidth=0.5,
            alpha=0.5,
        )
        axs[0, 0].step(
            normalized_time_vector,
            tau_joints_socp_plus[0, :] + motor_noises_socp_plus[0, :, i_random],
            color=SOCP_plus_color,
            label="SOCP+",
            linewidth=0.5,
            alpha=0.5,
        )
    axs[0, 0].step(normalized_time_vector, tau_joints_socp[0, :], color=SOCP_color, label="SOCP")
    axs[0, 0].step(normalized_time_vector, tau_joints_socp_variable[0, :], color=SOCP_VARIABLE_color, label="SOCPV")
    axs[0, 0].step(normalized_time_vector, tau_joints_socp_feedforward[0, :], color=SOCP_FEEDFORWARD_color, label="SOCPA")
    axs[0, 0].step(normalized_time_vector, tau_joints_socp_plus[0, :], color=SOCP_plus_color, label="SOCP+")
    axs[0, 0].set_title("Head")
    # axs[0, 0].legend(ncol=3)
    # Eyes
    axs[0, 1].step(normalized_time_vector, tau_joints_socp_feedforward[1, :], color=SOCP_FEEDFORWARD_color, label="SOCPA")
    axs[0, 1].step(normalized_time_vector, tau_joints_socp_plus[1, :], color=SOCP_plus_color, label="SOCP+")
    axs[0, 1].set_title("Eyes")
    axs[0, 1].plot([0, 1], [0, 0], color="black", linestyle="--")
    # Other joints
    for i_dof in range(2, 5):
        axs[0, i_dof].step(normalized_time_vector, tau_joints_ocp[i_dof - 1, :], color=OCP_color)

        for i_random in range(nb_random):
            axs[0, i_dof].step(
                normalized_time_vector,
                tau_joints_socp[i_dof - 1, :] + motor_noises_socp[i_dof - 1, :, i_random],
                color=SOCP_color,
                label="SOCP",
                linewidth=0.5,
                alpha=0.5,
            )
        for i_random in range(nb_random):
            axs[0, i_dof].step(
                normalized_time_vector,
                tau_joints_socp_variable[i_dof - 1, :] + motor_noises_socp_variable[i_dof - 1, :, i_random],
                color=SOCP_VARIABLE_color,
                label="SOCPV",
                linewidth=0.5,
                alpha=0.5,
            )
        for i_random in range(nb_random):
            axs[0, i_dof].step(
                normalized_time_vector,
                tau_joints_socp_feedforward[i_dof, :] + motor_noises_socp_feedforward[i_dof, :, i_random],
                color=SOCP_FEEDFORWARD_color,
                label="SOCPA",
                linewidth=0.5,
                alpha=0.5,
            )
        for i_random in range(nb_random):
            axs[0, i_dof].step(
                normalized_time_vector,
                tau_joints_socp_plus[i_dof, :] + motor_noises_socp_plus[i_dof, :, i_random],
                color=SOCP_plus_color,
                label="SOCP+",
                linewidth=0.5,
                alpha=0.5,
            )
        axs[0, i_dof].step(normalized_time_vector, tau_joints_socp[i_dof - 1, :], color=SOCP_color, label="SOCP")
        axs[0, i_dof].step(normalized_time_vector, tau_joints_socp_variable[i_dof - 1, :], color=SOCP_VARIABLE_color, label="SOCPV")
        axs[0, i_dof].step(normalized_time_vector, tau_joints_socp_feedforward[i_dof, :], color=SOCP_FEEDFORWARD_color, label="SOCPA")
        axs[0, i_dof].step(normalized_time_vector, tau_joints_socp_plus[i_dof, :], color=SOCP_plus_color, label="SOCP+")
    axs[0, 2].set_title("Shoulder")
    axs[0, 3].set_title("Hips")
    axs[0, 4].set_title("Knees")
    axs[0, 0].set_ylabel("Nominal torques [Nm]")

    # Joint friction
    for i_dof in range(5):
        if i_dof == 0:
            axs[1, 0].step(normalized_time_vector, -joint_friction_ocp[0, :], color=OCP_color)
            for i_random in range(nb_random):
                axs[1, 0].step(
                    normalized_time_vector, -joint_frictions_socp[0, :, i_random], color=SOCP_color, linewidth=0.5
                )
                axs[1, 0].step(
                    normalized_time_vector, -joint_frictions_socp_variable[0, :, i_random], color=SOCP_VARIABLE_color,
                    linewidth=0.5
                )
                axs[1, 0].step(
                    normalized_time_vector, -joint_frictions_socp_feedforward[0, :, i_random], color=SOCP_FEEDFORWARD_color,
                    linewidth=0.5
                )
                axs[1, 0].step(
                    normalized_time_vector, -joint_frictions_socp_plus[0, :, i_random], color=SOCP_plus_color,
                    linewidth=0.5
                )
        elif i_dof == 1:
            for i_random in range(nb_random):
                axs[1, 1].step(
                    normalized_time_vector, -joint_frictions_socp_feedforward[1, :, i_random], color=SOCP_FEEDFORWARD_color,
                    linewidth=0.5
                )
                axs[1, 1].step(
                    normalized_time_vector, -joint_frictions_socp_plus[1, :, i_random], color=SOCP_plus_color,
                    linewidth=0.5
                )
        else:
            axs[1, i_dof].step(normalized_time_vector, -joint_friction_ocp[i_dof - 1, :], color=OCP_color)
            for i_random in range(nb_random):
                axs[1, i_dof].step(
                    normalized_time_vector, -joint_frictions_socp[i_dof - 1, :, i_random], color=SOCP_color,
                    linewidth=0.5
                )
                axs[1, i_dof].step(
                    normalized_time_vector,
                    -joint_frictions_socp_variable[i_dof - 1, :, i_random],
                    color=SOCP_VARIABLE_color,
                    linewidth=0.5,
                )
                axs[1, i_dof].step(
                    normalized_time_vector,
                    -joint_frictions_socp_feedforward[i_dof, :, i_random],
                    color=SOCP_FEEDFORWARD_color,
                    linewidth=0.5,
                )
                axs[1, i_dof].step(
                    normalized_time_vector,
                    -joint_frictions_socp_plus[i_dof, :, i_random],
                    color=SOCP_plus_color,
                    linewidth=0.5,
                )
    axs[1, 0].set_ylabel("Joint friction [Nm]")

    # Feedback
    for i_dof in range(5):
        for i_random in range(nb_random):
            if i_dof == 0:
                axs[2, 0].step(normalized_time_vector, feedbacks_socp[0, :, i_random], color=SOCP_color, linewidth=0.5)
                axs[2, 0].step(normalized_time_vector, feedbacks_socp_variable[0, :, i_random], color=SOCP_VARIABLE_color, linewidth=0.5)
                axs[2, 0].step(normalized_time_vector, feedbacks_socp_feedforward[0, :, i_random], color=SOCP_FEEDFORWARD_color, linewidth=0.5)
                axs[2, 0].step(
                    normalized_time_vector, feedbacks_socp_plus[0, :, i_random], color=SOCP_plus_color, linewidth=0.5
                )
            elif i_dof == 1:
                axs[2, 1].step(
                    normalized_time_vector, feedbacks_socp_feedforward[1, :, i_random], color=SOCP_FEEDFORWARD_color, linewidth=0.5
                )
                axs[2, 1].step(
                    normalized_time_vector, feedbacks_socp_plus[1, :, i_random], color=SOCP_plus_color, linewidth=0.5
                )
            else:
                axs[2, i_dof].step(
                    normalized_time_vector, feedbacks_socp[i_dof - 1, :, i_random], color=SOCP_color, linewidth=0.5
                )
                axs[2, i_dof].step(
                    normalized_time_vector, feedbacks_socp_variable[i_dof - 1, :, i_random], color=SOCP_VARIABLE_color, linewidth=0.5
                )
                axs[2, i_dof].step(
                    normalized_time_vector, feedbacks_socp_feedforward[i_dof, :, i_random], color=SOCP_FEEDFORWARD_color,
                    linewidth=0.5
                )
                axs[2, i_dof].step(
                    normalized_time_vector, feedbacks_socp_plus[i_dof, :, i_random], color=SOCP_plus_color,
                    linewidth=0.5
                )
    axs[2, 0].set_ylabel("Feedbacks [Nm]")

    # Feedforward
    for i_dof in range(5):
        for i_random in range(nb_random):
            axs[3, i_dof].step(
                normalized_time_vector, feedforwards_socp_feedforward[i_dof, :, i_random], color=SOCP_FEEDFORWARD_color, linewidth=0.5
            )
            axs[3, i_dof].step(
                normalized_time_vector, feedforwards_socp_plus[i_dof, :, i_random], color=SOCP_plus_color, linewidth=0.5
            )
    axs[3, 0].set_ylabel("Feedforwards [Nm]")

    # Sum
    for i_dof in range(5):
        if i_dof == 0:
            axs[4, i_dof].step(normalized_time_vector, tau_joints_ocp[0, :] - joint_friction_ocp[0, :], color=OCP_color)
            for i_random in range(nb_random):
                axs[4, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp[0, :]
                    - joint_frictions_socp[0, :, i_random]
                    + motor_noises_socp[0, :, i_random]
                    + feedbacks_socp[0, :, i_random],
                    color=SOCP_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[4, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_variable[0, :]
                    - joint_frictions_socp_variable[0, :, i_random]
                    + motor_noises_socp_variable[0, :, i_random]
                    + feedbacks_socp_variable[0, :, i_random],
                    color=SOCP_VARIABLE_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[4, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_feedforward[0, :]
                    - joint_frictions_socp_feedforward[0, :, i_random]
                    + motor_noises_socp_feedforward[0, :, i_random]
                    + feedbacks_socp_feedforward[0, :, i_random]
                    + feedforwards_socp_feedforward[0, :, i_random],
                    color=SOCP_FEEDFORWARD_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[4, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_plus[0, :]
                    - joint_frictions_socp_plus[0, :, i_random]
                    + motor_noises_socp_plus[0, :, i_random]
                    + feedbacks_socp_plus[0, :, i_random]
                    + feedforwards_socp_plus[0, :, i_random],
                    color=SOCP_plus_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
        elif i_dof == 1:
            axs[4, 1].step(
                normalized_time_vector,
                tau_joints_socp_feedforward[1, :]
                - joint_frictions_socp_feedforward[1, :, i_random]
                + motor_noises_socp_feedforward[1, :, i_random]
                + feedbacks_socp_feedforward[1, :, i_random]
                + feedforwards_socp_feedforward[1, :, i_random],
                color=SOCP_FEEDFORWARD_color,
            )
            axs[4, 1].step(
                normalized_time_vector,
                tau_joints_socp_plus[1, :]
                - joint_frictions_socp_plus[1, :, i_random]
                + motor_noises_socp_plus[1, :, i_random]
                + feedbacks_socp_plus[1, :, i_random]
                + feedforwards_socp_plus[1, :, i_random],
                color=SOCP_plus_color,
            )
        else:
            axs[4, i_dof].step(
                normalized_time_vector, tau_joints_ocp[i_dof - 1, :] - joint_friction_ocp[i_dof - 1, :], color=OCP_color
            )
            for i_random in range(nb_random):
                axs[4, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp[i_dof - 1, :]
                    - joint_frictions_socp[i_dof - 1, :, i_random]
                    + motor_noises_socp[i_dof - 1, :, i_random]
                    + feedbacks_socp[i_dof - 1, :, i_random],
                    color=SOCP_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[4, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_variable[i_dof - 1, :]
                    - joint_frictions_socp_variable[i_dof - 1, :, i_random]
                    + motor_noises_socp_variable[i_dof - 1, :, i_random]
                    + feedbacks_socp_variable[i_dof - 1, :, i_random],
                    color=SOCP_VARIABLE_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[4, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_feedforward[i_dof, :]
                    - joint_frictions_socp_feedforward[i_dof, :, i_random]
                    + motor_noises_socp_feedforward[i_dof, :, i_random]
                    + feedbacks_socp_feedforward[i_dof, :, i_random]
                    + feedforwards_socp_feedforward[i_dof, :, i_random],
                    color=SOCP_FEEDFORWARD_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[4, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_plus[i_dof, :]
                    - joint_frictions_socp_plus[i_dof, :, i_random]
                    + motor_noises_socp_plus[i_dof, :, i_random]
                    + feedbacks_socp_plus[i_dof, :, i_random]
                    + feedforwards_socp_plus[i_dof, :, i_random],
                    color=SOCP_plus_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
    axs[4, 0].set_ylabel(r"Total $\tau$ [Nm]", fontsize=12)

    plt.savefig("graphs/controls.png")
    # plt.show()

    return


def plot_tau_and_delta_tau(normalized_time_vector,
                            tau_joints_ocp,
                            tau_joints_socp,
                            tau_joints_socp_variable,
                            tau_joints_socp_feedforward,
                            tau_joints_socp_plus,
                            joint_friction_ocp,
                            joint_frictions_socp,
                            joint_frictions_socp_variable,
                            joint_frictions_socp_feedforward,
                            joint_frictions_socp_plus,
                            nb_random,
                            OCP_color,
                            SOCP_color,
                            SOCP_VARIABLE_color,
                            SOCP_FEEDFORWARD_color,
                            SOCP_plus_color,
                            motor_noises_socp,
                            motor_noises_socp_variable,
                            motor_noises_socp_feedforward,
                            motor_noises_socp_plus,
                            feedbacks_socp,
                            feedbacks_socp_variable,
                            feedbacks_socp_feedforward,
                            feedbacks_socp_plus,
                            feedforwards_socp_feedforward,
                            feedforwards_socp_plus,
                            ):

    # All DoFs individually -----------------------------------------------
    fig, axs = plt.subplots(2, 5, figsize=(15, 5))
    for i in range(2):
        for j in range(5):
            axs[i, j].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
    # Tau
    for i_dof in range(5):
        if i_dof == 0:
            axs[0, i_dof].step(normalized_time_vector, tau_joints_ocp[0, :] - joint_friction_ocp[0, :], color=OCP_color)
            for i_random in range(nb_random):
                axs[0, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp[0, :]
                    - joint_frictions_socp[0, :, i_random]
                    + motor_noises_socp[0, :, i_random]
                    + feedbacks_socp[0, :, i_random],
                    color=SOCP_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[0, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_variable[0, :]
                    - joint_frictions_socp_variable[0, :, i_random]
                    + motor_noises_socp_variable[0, :, i_random]
                    + feedbacks_socp_variable[0, :, i_random],
                    color=SOCP_VARIABLE_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[0, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_feedforward[0, :]
                    - joint_frictions_socp_feedforward[0, :, i_random]
                    + motor_noises_socp_feedforward[0, :, i_random]
                    + feedbacks_socp_feedforward[0, :, i_random]
                    + feedforwards_socp_feedforward[0, :, i_random],
                    color=SOCP_FEEDFORWARD_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[0, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_plus[0, :]
                    - joint_frictions_socp_plus[0, :, i_random]
                    + motor_noises_socp_plus[0, :, i_random]
                    + feedbacks_socp_plus[0, :, i_random]
                    + feedforwards_socp_plus[0, :, i_random],
                    color=SOCP_plus_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
        elif i_dof == 1:
            axs[0, 1].step(
                normalized_time_vector,
                tau_joints_socp_feedforward[1, :]
                - joint_frictions_socp_feedforward[1, :, i_random]
                + motor_noises_socp_feedforward[1, :, i_random]
                + feedbacks_socp_feedforward[1, :, i_random]
                + feedforwards_socp_feedforward[1, :, i_random],
                color=SOCP_FEEDFORWARD_color,
            )
            axs[0, 1].step(
                normalized_time_vector,
                tau_joints_socp_plus[1, :]
                - joint_frictions_socp_plus[1, :, i_random]
                + motor_noises_socp_plus[1, :, i_random]
                + feedbacks_socp_plus[1, :, i_random]
                + feedforwards_socp_plus[1, :, i_random],
                color=SOCP_plus_color,
            )
        else:
            axs[0, i_dof].step(
                normalized_time_vector, tau_joints_ocp[i_dof - 1, :] - joint_friction_ocp[i_dof - 1, :], color=OCP_color
            )
            for i_random in range(nb_random):
                axs[0, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp[i_dof - 1, :]
                    - joint_frictions_socp[i_dof - 1, :, i_random]
                    + motor_noises_socp[i_dof - 1, :, i_random]
                    + feedbacks_socp[i_dof - 1, :, i_random],
                    color=SOCP_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[0, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_variable[i_dof - 1, :]
                    - joint_frictions_socp_variable[i_dof - 1, :, i_random]
                    + motor_noises_socp_variable[i_dof - 1, :, i_random]
                    + feedbacks_socp_variable[i_dof - 1, :, i_random],
                    color=SOCP_VARIABLE_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[0, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_feedforward[i_dof, :]
                    - joint_frictions_socp_feedforward[i_dof, :, i_random]
                    + motor_noises_socp_feedforward[i_dof, :, i_random]
                    + feedbacks_socp_feedforward[i_dof, :, i_random]
                    + feedforwards_socp_feedforward[i_dof, :, i_random],
                    color=SOCP_FEEDFORWARD_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[0, i_dof].step(
                    normalized_time_vector,
                    tau_joints_socp_plus[i_dof, :]
                    - joint_frictions_socp_plus[i_dof, :, i_random]
                    + motor_noises_socp_plus[i_dof, :, i_random]
                    + feedbacks_socp_plus[i_dof, :, i_random]
                    + feedforwards_socp_plus[i_dof, :, i_random],
                    color=SOCP_plus_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
    axs[0, 0].set_ylabel(r"Total $\tau$ [Nm]", fontsize=12)

    # Delta tau
    delta_time_vector = (normalized_time_vector[1:] + normalized_time_vector[:-1]) / 2
    for i_dof in range(5):
        if i_dof == 0:
            axs[1, i_dof].step(
                delta_time_vector,
                tau_joints_ocp[0, 1:] - joint_friction_ocp[0, 1:] - (
                            tau_joints_ocp[0, :-1] + joint_friction_ocp[0, :-1]),
                color=OCP_color,
            )
            for i_random in range(nb_random):
                axs[1, i_dof].step(
                    delta_time_vector,
                    tau_joints_socp[0, 1:]
                    - joint_frictions_socp[0, 1:, i_random]
                    + motor_noises_socp[0, 1:, i_random]
                    + feedbacks_socp[0, 1:, i_random]
                    - (
                            tau_joints_socp[0, :-1]
                            - joint_frictions_socp[0, :-1, i_random]
                            + motor_noises_socp[0, :-1, i_random]
                            + feedbacks_socp[0, :-1, i_random]
                    ),
                    color=SOCP_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[1, i_dof].step(
                    delta_time_vector,
                    tau_joints_socp_variable[0, 1:]
                    - joint_frictions_socp_variable[0, 1:, i_random]
                    + motor_noises_socp_variable[0, 1:, i_random]
                    + feedbacks_socp_variable[0, 1:, i_random]
                    - (
                            tau_joints_socp_variable[0, :-1]
                            - joint_frictions_socp_variable[0, :-1, i_random]
                            + motor_noises_socp_variable[0, :-1, i_random]
                            + feedbacks_socp_variable[0, :-1, i_random]
                    ),
                    color=SOCP_VARIABLE_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[1, i_dof].step(
                    delta_time_vector,
                    tau_joints_socp_feedforward[0, 1:]
                    - joint_frictions_socp_feedforward[0, 1:, i_random]
                    + motor_noises_socp_feedforward[0, 1:, i_random]
                    + feedbacks_socp_feedforward[0, 1:, i_random]
                    + feedforwards_socp_feedforward[0, 1:, i_random]
                    - (
                            tau_joints_socp_feedforward[0, :-1]
                            - joint_frictions_socp_feedforward[0, :-1, i_random]
                            + motor_noises_socp_feedforward[0, :-1, i_random]
                            + feedbacks_socp_feedforward[0, :-1, i_random]
                            + feedforwards_socp_feedforward[0, :-1, i_random]
                    ),
                    color=SOCP_FEEDFORWARD_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[1, i_dof].step(
                    delta_time_vector,
                    tau_joints_socp_plus[0, 1:]
                    - joint_frictions_socp_plus[0, 1:, i_random]
                    + motor_noises_socp_plus[0, 1:, i_random]
                    + feedbacks_socp_plus[0, 1:, i_random]
                    + feedforwards_socp_plus[0, 1:, i_random]
                    - (
                            tau_joints_socp_plus[0, :-1]
                            - joint_frictions_socp_plus[0, :-1, i_random]
                            + motor_noises_socp_plus[0, :-1, i_random]
                            + feedbacks_socp_plus[0, :-1, i_random]
                            + feedforwards_socp_plus[0, :-1, i_random]
                    ),
                    color=SOCP_plus_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
        elif i_dof == 1:
            axs[1, 1].step(
                delta_time_vector,
                tau_joints_socp_feedforward[1, 1:]
                - joint_frictions_socp_feedforward[1, 1:, i_random]
                + motor_noises_socp_feedforward[1, 1:, i_random]
                + feedbacks_socp_feedforward[1, 1:, i_random]
                + feedforwards_socp_feedforward[1, 1:, i_random]
                - (
                        tau_joints_socp_feedforward[1, :-1]
                        - joint_frictions_socp_feedforward[1, :-1, i_random]
                        + motor_noises_socp_feedforward[1, :-1, i_random]
                        + feedbacks_socp_feedforward[1, :-1, i_random]
                        + feedforwards_socp_feedforward[1, :-1, i_random]
                ),
                color=SOCP_FEEDFORWARD_color,
            )
            axs[1, 1].step(
                delta_time_vector,
                tau_joints_socp_plus[1, 1:]
                - joint_frictions_socp_plus[1, 1:, i_random]
                + motor_noises_socp_plus[1, 1:, i_random]
                + feedbacks_socp_plus[1, 1:, i_random]
                + feedforwards_socp_plus[1, 1:, i_random]
                - (
                        tau_joints_socp_plus[1, :-1]
                        - joint_frictions_socp_plus[1, :-1, i_random]
                        + motor_noises_socp_plus[1, :-1, i_random]
                        + feedbacks_socp_plus[1, :-1, i_random]
                        + feedforwards_socp_plus[1, :-1, i_random]
                ),
                color=SOCP_plus_color,
            )
        else:
            axs[1, i_dof].step(
                delta_time_vector,
                tau_joints_ocp[i_dof - 1, 1:]
                - joint_friction_ocp[i_dof - 1, 1:]
                - (tau_joints_ocp[i_dof - 1, :-1] - joint_friction_ocp[i_dof - 1, :-1]),
                color=OCP_color,
            )
            for i_random in range(nb_random):
                axs[1, i_dof].step(
                    delta_time_vector,
                    tau_joints_socp[i_dof - 1, 1:]
                    - joint_frictions_socp[i_dof - 1, 1:, i_random]
                    + motor_noises_socp[i_dof - 1, 1:, i_random]
                    + feedbacks_socp[i_dof - 1, 1:, i_random]
                    - (
                            tau_joints_socp[i_dof - 1, :-1]
                            - joint_frictions_socp[i_dof - 1, :-1, i_random]
                            + motor_noises_socp[i_dof - 1, :-1, i_random]
                            + feedbacks_socp[i_dof - 1, :-1, i_random]
                    ),
                    color=SOCP_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[1, i_dof].step(
                    delta_time_vector,
                    tau_joints_socp_variable[i_dof - 1, 1:]
                    - joint_frictions_socp_variable[i_dof - 1, 1:, i_random]
                    + motor_noises_socp_variable[i_dof - 1, 1:, i_random]
                    + feedbacks_socp_variable[i_dof - 1, 1:, i_random]
                    - (
                            tau_joints_socp_variable[i_dof - 1, :-1]
                            - joint_frictions_socp_variable[i_dof - 1, :-1, i_random]
                            + motor_noises_socp_variable[i_dof - 1, :-1, i_random]
                            + feedbacks_socp_variable[i_dof - 1, :-1, i_random]
                    ),
                    color=SOCP_VARIABLE_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[1, i_dof].step(
                    delta_time_vector,
                    tau_joints_socp_feedforward[i_dof, 1:]
                    - joint_frictions_socp_feedforward[i_dof, 1:, i_random]
                    + motor_noises_socp_feedforward[i_dof, 1:, i_random]
                    + feedbacks_socp_feedforward[i_dof, 1:, i_random]
                    + feedforwards_socp_feedforward[i_dof, 1:, i_random]
                    - (
                            tau_joints_socp_feedforward[i_dof, :-1]
                            - joint_frictions_socp_feedforward[i_dof, :-1, i_random]
                            + motor_noises_socp_feedforward[i_dof, :-1, i_random]
                            + feedbacks_socp_feedforward[i_dof, :-1, i_random]
                            + feedforwards_socp_feedforward[i_dof, :-1, i_random]
                    ),
                    color=SOCP_FEEDFORWARD_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
                axs[1, i_dof].step(
                    delta_time_vector,
                    tau_joints_socp_plus[i_dof, 1:]
                    - joint_frictions_socp_plus[i_dof, 1:, i_random]
                    + motor_noises_socp_plus[i_dof, 1:, i_random]
                    + feedbacks_socp_plus[i_dof, 1:, i_random]
                    + feedforwards_socp_plus[i_dof, 1:, i_random]
                    - (
                            tau_joints_socp_plus[i_dof, :-1]
                            - joint_frictions_socp_plus[i_dof, :-1, i_random]
                            + motor_noises_socp_plus[i_dof, :-1, i_random]
                            + feedbacks_socp_plus[i_dof, :-1, i_random]
                            + feedforwards_socp_plus[i_dof, :-1, i_random]
                    ),
                    color=SOCP_plus_color,
                    linewidth=0.5,
                    alpha=0.5,
                )
    axs[1, 0].set_ylabel(r"Total $\Delta \tau$ [Nm]", fontsize=12)

    plt.savefig("graphs/tau_and_delta_tau.png")
    # plt.show()

    # All DoFs together ---------------------------------------------------
    # Plot tau and delta tau
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(2):
        axs[i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
    # Tau
    axs[0].step(
        normalized_time_vector,
        np.sum(np.abs(tau_joints_ocp - joint_friction_ocp), axis=0),
        color=OCP_color,
    )
    for i_random in range(nb_random):
        axs[0].step(
            normalized_time_vector,
            np.sum(np.abs(tau_joints_socp \
            - joint_frictions_socp[:, :, i_random] \
            + motor_noises_socp[:, :, i_random] \
            + feedbacks_socp[:, :, i_random]), axis=0),
            color=SOCP_color,
            linewidth=0.5,
            alpha=0.5,
        )
        axs[0].step(
            normalized_time_vector,
            np.sum(np.abs(tau_joints_socp_variable \
            - joint_frictions_socp_variable[:, :, i_random] \
            + motor_noises_socp_variable[:, :, i_random] \
            + feedbacks_socp_variable[:, :, i_random]), axis=0),
            color=SOCP_VARIABLE_color,
            linewidth=0.5,
            alpha=0.5,
        )
        axs[0].step(
            normalized_time_vector,
            np.sum(np.abs(tau_joints_socp_feedforward \
            - joint_frictions_socp_feedforward[:, :, i_random] \
            + motor_noises_socp_feedforward[:, :, i_random] \
            + feedbacks_socp_feedforward[:, :, i_random] \
            + feedforwards_socp_feedforward[:, :, i_random]), axis=0),
            color=SOCP_FEEDFORWARD_color,
            linewidth=0.5,
            alpha=0.5,
        )
        axs[0].step(
            normalized_time_vector,
            np.sum(np.abs(tau_joints_socp_plus \
            - joint_frictions_socp_plus[:, :, i_random] \
            + motor_noises_socp_plus[:, :, i_random] \
            + feedbacks_socp_plus[:, :, i_random] \
            + feedforwards_socp_plus[:, :, i_random]), axis=0),
            color=SOCP_plus_color,
            linewidth=0.5,
            alpha=0.5,
        )
    axs[0].set_ylabel(r"Total $\sum{\tau}$ [Nm]", fontsize=12)

    # Delta tau
    delta_time_vector = (normalized_time_vector[1:] + normalized_time_vector[:-1]) / 2
    axs[1].step(
        delta_time_vector,
        np.sum(
            np.abs(
                tau_joints_ocp[:, 1:] - joint_friction_ocp[:, 1:] - (
                            tau_joints_ocp[:, :-1] + joint_friction_ocp[:, :-1])
            ),
            axis=0,
        ),
        color=OCP_color,
    )
    for i_random in range(nb_random):
        axs[1].step(
            delta_time_vector,
            np.sum(
                np.abs(
                    tau_joints_socp[:, 1:]
                    - joint_frictions_socp[:, 1:, i_random]
                    + motor_noises_socp[:, 1:, i_random]
                    + feedbacks_socp[:, 1:, i_random]
                    - (
                            tau_joints_socp[:, :-1]
                            - joint_frictions_socp[:, :-1, i_random]
                            + motor_noises_socp[:, :-1, i_random]
                            + feedbacks_socp[:, :-1, i_random]
                    )
                ),
                axis=0,
            ),
            color=SOCP_color,
            linewidth=0.5,
            alpha=0.5,
        )
        axs[1].step(
            delta_time_vector,
            np.sum(
                np.abs(
                    tau_joints_socp_variable[:, 1:]
                    - joint_frictions_socp_variable[:, 1:, i_random]
                    + motor_noises_socp_variable[:, 1:, i_random]
                    + feedbacks_socp_variable[:, 1:, i_random]
                    - (
                            tau_joints_socp_variable[:, :-1]
                            - joint_frictions_socp_variable[:, :-1, i_random]
                            + motor_noises_socp_variable[:, :-1, i_random]
                            + feedbacks_socp_variable[:, :-1, i_random]
                    )
                ),
                axis=0,
            ),
            color=SOCP_VARIABLE_color,
            linewidth=0.5,
            alpha=0.5,
        )
        axs[1].step(
            delta_time_vector,
            np.sum(
                np.abs(
                    tau_joints_socp_feedforward[:, 1:]
                    - joint_frictions_socp_feedforward[:, 1:, i_random]
                    + motor_noises_socp_feedforward[:, 1:, i_random]
                    + feedbacks_socp_feedforward[:, 1:, i_random]
                    + feedforwards_socp_feedforward[:, 1:, i_random]
                    - (
                            tau_joints_socp_feedforward[:, :-1]
                            - joint_frictions_socp_feedforward[:, :-1, i_random]
                            + motor_noises_socp_feedforward[:, :-1, i_random]
                            + feedbacks_socp_feedforward[:, :-1, i_random]
                            + feedforwards_socp_feedforward[:, :-1, i_random]
                    )
                ),
                axis=0,
            ),
            color=SOCP_FEEDFORWARD_color,
            linewidth=0.5,
            alpha=0.5,
        )
        axs[1].step(
            delta_time_vector,
            np.sum(
                np.abs(
                    tau_joints_socp_plus[:, 1:]
                    - joint_frictions_socp_plus[:, 1:, i_random]
                    + motor_noises_socp_plus[:, 1:, i_random]
                    + feedbacks_socp_plus[:, 1:, i_random]
                    + feedforwards_socp_plus[:, 1:, i_random]
                    - (
                            tau_joints_socp_plus[:, :-1]
                            - joint_frictions_socp_plus[:, :-1, i_random]
                            + motor_noises_socp_plus[:, :-1, i_random]
                            + feedbacks_socp_plus[:, :-1, i_random]
                            + feedforwards_socp_plus[:, :-1, i_random]
                    )
                ),
                axis=0,
            ),
            color=SOCP_plus_color,
            linewidth=0.5,
            alpha=0.5,
        )
    axs[1].set_ylabel(r"Total $\sum{\Delta \tau}$ [Nm]", fontsize=12)

    plt.savefig("graphs/sum_tau_and_delta_tau.png")
    # plt.show()

    return


def plot_gains(
        socp_variable,
        socp_plus,
        normalized_time_vector,
        k_socp,
        k_socp_variable,
        k_socp_feedforward,
        k_socp_plus,
        SOCP_color,
        SOCP_VARIABLE_color,
        SOCP_FEEDFORWARD_color,
        SOCP_plus_color
        ):

    # Plot the gains
    n_k_fb = socp_plus.nlp[0].model.n_noised_controls + socp_plus.nlp[0].model.n_references
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    for i in range(2):
        for j in range(2):
            axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
    for i_dof in range(40):
        axs[0, 0].step(normalized_time_vector, k_socp[i_dof, :], color=SOCP_color, label="SOCP")
        axs[0, 1].step(normalized_time_vector, k_socp_variable[i_dof, :], color=SOCP_VARIABLE_color, label="SOCPV")
    for i_dof in range(n_k_fb):
        axs[0, 2].step(normalized_time_vector, k_socp_feedforward[i_dof, :], color=SOCP_FEEDFORWARD_color, label="SOCPA")
        axs[0, 3].step(normalized_time_vector, k_socp_plus[i_dof, :], color=SOCP_plus_color, label="SOCP+")
    for i_dof in range(5):
        axs[1, 2].step(normalized_time_vector, k_socp_feedforward[n_k_fb + i_dof, :], color=SOCP_FEEDFORWARD_color, label="SOCPA")
        axs[1, 3].step(normalized_time_vector, k_socp_plus[n_k_fb + i_dof, :], color=SOCP_plus_color, label="SOCP+")
    # axs[0, 0].set_ylim(-35, 35)
    # axs[0, 1].set_ylim(-35, 35)
    # axs[1, 1].set_ylim(-35, 35)
    axs[0, 0].set_ylabel("Direct feedback gains", fontsize=12)
    axs[1, 0].set_ylabel("Anticipatory feedback gains", fontsize=12)
    plt.savefig("graphs/gains.png")
    # plt.show()


    # Plot the gains
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    for i in range(2):
        for j in range(2):
            axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)

    axs[0, 0].step(normalized_time_vector, np.sum(np.abs(k_socp[:, :]), axis=0), color=SOCP_color, label="SOCP")
    axs[0, 1].step(normalized_time_vector, np.sum(np.abs(k_socp_variable[:, :]), axis=0), color=SOCP_VARIABLE_color, label="SOCPV")
    axs[0, 2].step(
        normalized_time_vector, np.sum(np.abs(k_socp_feedforward[:n_k_fb, :]), axis=0), color=SOCP_FEEDFORWARD_color, label="SOCPA"
    )
    axs[0, 3].step(
        normalized_time_vector, np.sum(np.abs(k_socp_plus[:n_k_fb, :]), axis=0), color=SOCP_plus_color, label="SOCP+"
    )
    axs[1, 2].step(
        normalized_time_vector, np.sum(np.abs(k_socp_feedforward[n_k_fb:, :]), axis=0), color=SOCP_FEEDFORWARD_color, label="SOCPA"
    )
    axs[1, 3].step(
        normalized_time_vector, np.sum(np.abs(k_socp_plus[n_k_fb:, :]), axis=0), color=SOCP_plus_color, label="SOCP+"
    )
    # axs[0, 0].set_ylim(0, 800)
    # axs[0, 1].set_ylim(0, 800)
    # axs[1, 1].set_ylim(0, 800)
    axs[0, 0].set_ylabel(r"$\sum{}$ Direct feedback gains", fontsize=12)
    axs[1, 0].set_ylabel(r"$\sum{}$ Anticipatory feedback gains", fontsize=12)
    plt.savefig("graphs/sum_gains.png")
    # plt.show()


    # Plot the delta gains
    delta_time_vector = (normalized_time_vector[1:] + normalized_time_vector[:-1]) / 2
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for i in range(2):
        for j in range(2):
            axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
    for i_dof in range(40):
        axs[0, 0].step(delta_time_vector, k_socp[i_dof, 1:] - k_socp[i_dof, :-1], color=SOCP_color, label="SOCP")
        axs[0, 0].step(delta_time_vector, k_socp_variable[i_dof, 1:] - k_socp_variable[i_dof, :-1], color=SOCP_VARIABLE_color, label="SOCPV")
    for i_dof in range(n_k_fb):
        axs[0, 1].step(
            delta_time_vector, k_socp_feedforward[i_dof, 1:] - k_socp_feedforward[i_dof, :-1], color=SOCP_FEEDFORWARD_color, label="SOCPA"
        )
        axs[0, 1].step(
            delta_time_vector, k_socp_plus[i_dof, 1:] - k_socp_plus[i_dof, :-1], color=SOCP_plus_color, label="SOCP+"
        )
    for i_dof in range(5):
        axs[1, 1].step(
            delta_time_vector,
            k_socp_feedforward[n_k_fb + i_dof, 1:] - k_socp_feedforward[n_k_fb + i_dof, :-1],
            color=SOCP_FEEDFORWARD_color,
            label="SOCPA",
        )
        axs[1, 1].step(
            delta_time_vector,
            k_socp_plus[n_k_fb + i_dof, 1:] - k_socp_plus[n_k_fb + i_dof, :-1],
            color=SOCP_plus_color,
            label="SOCP+",
        )
    # axs[0, 0].set_ylim(-40, 35)
    # axs[0, 1].set_ylim(-40, 35)
    # axs[1, 1].set_ylim(-40, 35)
    plt.savefig("graphs/delta_gains.png")


    # Plot the delta gains
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    for i in range(2):
        for j in range(2):
            axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
    axs[0, 0].step(
        delta_time_vector, np.sum(np.abs(k_socp[:, 1:] - k_socp[:, :-1]), axis=0), color=SOCP_color, label="SOCP"
    )
    axs[0, 1].step(
        delta_time_vector, np.sum(np.abs(k_socp_variable[:, 1:] - k_socp_variable[:, :-1]), axis=0), color=SOCP_VARIABLE_color, label="SOCPV"
    )
    axs[0, 2].step(
        delta_time_vector,
        np.sum(np.abs(k_socp_feedforward[:n_k_fb, 1:] - k_socp_feedforward[:n_k_fb, :-1]), axis=0),
        color=SOCP_FEEDFORWARD_color,
        label="SOCPA",
    )
    axs[0, 3].step(
        delta_time_vector,
        np.sum(np.abs(k_socp_plus[:n_k_fb, 1:] - k_socp_plus[:n_k_fb, :-1]), axis=0),
        color=SOCP_plus_color,
        label="SOCP+",
    )
    axs[1, 2].step(
        delta_time_vector,
        np.sum(np.abs(k_socp_feedforward[n_k_fb:, 1:] - k_socp_feedforward[n_k_fb:, :-1]), axis=0),
        color=SOCP_FEEDFORWARD_color,
        label="SOCPA",
    )
    axs[1, 3].step(
        delta_time_vector,
        np.sum(np.abs(k_socp_plus[n_k_fb:, 1:] - k_socp_plus[n_k_fb:, :-1]), axis=0),
        color=SOCP_plus_color,
        label="SOCP+",
    )
    # axs[0, 0].set_ylim(0, 800)
    # axs[0, 1].set_ylim(0, 800)
    # axs[1, 1].set_ylim(0, 800)
    axs[0, 0].set_ylabel(r"$\sum{} \Delta$ Direct feedback gains", fontsize=12)
    axs[1, 0].set_ylabel(r"$\sum{} \Delta$ Anticipatory feedback gains", fontsize=12)
    plt.savefig("graphs/sum_delta_gains.png")
    # plt.show()

    # Plot the FF gains vs acuity
    fig, axs = plt.subplots(4, 1, figsize=(15, 10))
    for i in range(4):
        axs[i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)

    axs[0].step(
        normalized_time_vector, np.sum(np.abs(k_socp_feedforward[:n_k_fb, :]), axis=0), color=SOCP_FEEDFORWARD_color, label="SOCPA"
    )
    axs[0].step(
        normalized_time_vector, np.sum(np.abs(k_socp_plus[:n_k_fb, :]), axis=0), color=SOCP_plus_color, label="SOCP+"
    )
    axs[0].set_ylabel("Feedback gains")
    axs[1].step(
        normalized_time_vector, np.sum(np.abs(k_socp_feedforward[n_k_fb:, :]), axis=0), color=SOCP_FEEDFORWARD_color, label="SOCPA"
    )
    axs[1].step(
        normalized_time_vector, np.sum(np.abs(k_socp_plus[n_k_fb:, :]), axis=0), color=SOCP_plus_color, label="SOCP+"
    )
    axs[1].set_ylabel("Feedforward gains")

    visual_noise_sym = cas.MX.sym("visual_noise", 1)
    vestibular_noise_sym = cas.MX.sym("vestibular_noise", 1)
    visual_noise_fcn = cas.Function(
        "visual_noise",
        [Q_8, visual_noise_sym],
        [visual_noise(socp_plus.nlp[0].model, Q_8, visual_noise_sym)],
    )
    vestibular_noise_fcn_variable = cas.Function(
        "vestibular_noise_variable",
        [Q, Qdot, vestibular_noise_sym],
        [vestibular_noise(socp_variable.nlp[0].model, Q, Qdot, vestibular_noise_sym)],
    )
    vestibular_noise_fcn_plus = cas.Function(
        "vestibular_noise_plus",
        [Q_8, Qdot_8, vestibular_noise_sym],
        [vestibular_noise(socp_plus.nlp[0].model, Q_8, Qdot_8, vestibular_noise_sym)],
    )

    visual_acuity = np.zeros((n_shooting, 1))
    vestibular_acquity_variable = np.zeros((n_shooting, 1))
    vestibular_acquity_plus = np.zeros((n_shooting, 1))
    for i_shooting in range(n_shooting):
        visual_acuity[i_shooting] = visual_noise_fcn(q_mean_socp_plus[:, i_shooting], 1)
        vestibular_acquity_variable[i_shooting] = vestibular_noise_fcn_variable(
            q_mean_socp_variable[:, i_shooting], qdot_mean_socp_variable[:, i_shooting], 1
        )
        vestibular_acquity_plus[i_shooting] = vestibular_noise_fcn_plus(
            q_mean_socp_plus[:, i_shooting], qdot_mean_socp_plus[:, i_shooting], 1
        )
    visual_acuity_normalized = (visual_acuity - np.min(visual_acuity)) / (np.max(visual_acuity) - np.min(visual_acuity))
    vestibular_acquity_normalized_variable = (vestibular_acquity_variable - np.min(vestibular_acquity_variable)) / (
        np.max(vestibular_acquity_variable) - np.min(vestibular_acquity_variable)
    )
    vestibular_acquity_normalized_plus = (vestibular_acquity_plus - np.min(vestibular_acquity_plus)) / (
            np.max(vestibular_acquity_plus) - np.min(vestibular_acquity_plus)
    )
    axs[2].plot(normalized_time_vector, np.ones_like(normalized_time_vector), color=SOCP_FEEDFORWARD_color)
    axs[2].plot(normalized_time_vector, visual_acuity_normalized, color=SOCP_plus_color)
    axs[2].set_ylabel("Visual acuity")
    axs[3].plot(normalized_time_vector, vestibular_acquity_normalized_variable, color=SOCP_VARIABLE_color)
    axs[3].plot(normalized_time_vector, np.ones_like(normalized_time_vector), color=SOCP_FEEDFORWARD_color)
    axs[3].plot(normalized_time_vector, vestibular_acquity_normalized_plus, color=SOCP_plus_color)
    axs[3].set_ylabel("Vestibular acuity")
    plt.savefig("graphs/ff_gains.png")
    # plt.show()

    return

def plot_mean_comparison(
    q_ocp,
    q_mean_socp,
    q_mean_socp_variable,
    q_mean_socp_feedforward,
    q_mean_socp_plus,
    q_socp,
    q_socp_variable,
    q_socp_feedforward,
    q_socp_plus,
    q_ocp_integrated,
    q_socp_integrated,
    q_socp_variable_integrated,
    q_socp_feedforward_integrated,
    q_socp_plus_integrated,
    time_vector,
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_plus_color,
):

    socp_mean = np.mean(q_socp, axis=2)
    socp_variable_mean = np.mean(q_socp_variable, axis=2)
    socp_feedforward_mean = np.mean(q_socp_feedforward, axis=2)
    socp_plus_mean = np.mean(q_socp_plus, axis=2)
    ocp_reintegration_mean = np.mean(q_ocp_integrated, axis=2)
    socp_reintegration_mean = np.mean(q_socp_integrated, axis=2)
    socp_variable_reintegration_mean = np.mean(q_socp_variable_integrated, axis=2)
    socp_feedforward_reintegration_mean = np.mean(q_socp_feedforward_integrated, axis=2)
    socp_plus_reintegration_mean = np.mean(q_socp_plus_integrated, axis=2)

    fig, axs = plt.subplots(7, 5, figsize=(15, 10))
    for i_dof in range(7):
        axs[i_dof, 0].plot(time_vector, q_ocp[i_dof, :], color=OCP_color, linewidth=2)
        axs[i_dof, 1].plot(time_vector, q_mean_socp[i_dof, :], color=SOCP_color, linewidth=2)
        axs[i_dof, 2].plot(time_vector, q_mean_socp_variable[i_dof, :], color=SOCP_VARIABLE_color, linewidth=2)
        axs[i_dof, 3].plot(time_vector, q_mean_socp_feedforward[i_dof, :], color=SOCP_FEEDFORWARD_color, linewidth=2)
        axs[i_dof, 4].plot(time_vector, q_mean_socp_plus[i_dof, :], color=SOCP_plus_color, linewidth=2)

        axs[i_dof, 0].plot(time_vector, ocp_reintegration_mean[i_dof, :], color=OCP_color, alpha=0.5)
        axs[i_dof, 1].plot(time_vector, socp_reintegration_mean[i_dof, :], color=SOCP_color, alpha=0.5)
        axs[i_dof, 2].plot(time_vector, socp_variable_reintegration_mean[i_dof, :], color=SOCP_VARIABLE_color, alpha=0.5)
        axs[i_dof, 3].plot(time_vector, socp_feedforward_reintegration_mean[i_dof, :], color=SOCP_FEEDFORWARD_color, alpha=0.5)
        axs[i_dof, 4].plot(time_vector, socp_plus_reintegration_mean[i_dof, :], color=SOCP_plus_color, alpha=0.5)

        axs[i_dof, 1].plot(time_vector, socp_mean[i_dof, :], color="#6C165C", linewidth=0.5)
        axs[i_dof, 2].plot(time_vector, socp_variable_mean[i_dof, :], color="#D15C02", linewidth=0.5)
        axs[i_dof, 3].plot(time_vector, socp_feedforward_mean[i_dof, :], color="#400191", linewidth=0.5)
        axs[i_dof, 4].plot(time_vector, socp_plus_mean[i_dof, :], color="#016C93", linewidth=0.5)

    return


def create_random_noise(seed, nb_random, n_shooting, n_joints, motor_noise_magnitude, sensory_noise_magnitude):
    np.random.seed(seed)
    nb_sensory = sensory_noise_magnitude.shape[0]
    # the last node deos not need motor and sensory noise
    motor_noise_numerical = np.zeros((n_joints, nb_random, n_shooting + 1))
    sensory_noise_numerical = np.zeros((nb_sensory, nb_random, n_shooting + 1))
    for i_random in range(nb_random):
        for i_shooting in range(n_shooting):
            motor_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                loc=np.zeros(motor_noise_magnitude.shape[0]),
                scale=np.reshape(np.array(motor_noise_magnitude), (n_joints,)),
                size=n_joints,
            )
            sensory_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                loc=np.zeros(sensory_noise_magnitude.shape[0]),
                scale=np.reshape(np.array(sensory_noise_magnitude), (nb_sensory,)),
                size=nb_sensory,
            )
    return motor_noise_numerical, sensory_noise_numerical


def plot_comparison_nb_random(q_ocp_integrated,
                        q_socp_integrated,
                        q_socp_variable_integrated,
                        q_socp_feedforward_integrated,
                        q_socp_plus_integrated,
                        OCP_color,
                        SOCP_color,
                        SOCP_VARIABLE_color,
                        SOCP_FEEDFORWARD_color,
                        SOCP_plus_color):

    def plot_distribution(q_integrated, current_position, color, current_file_name, ax):
        nb_randoms = [5, 10, 15]
        base_file_name = "Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02"
        for i_random, current_random in enumerate(nb_randoms):
            position_random = np.random.random((current_random, 1)) * 0.1
            file_name = f"results/{current_random}random/{base_file_name}_{current_file_name}_{current_random}random_CVG_1p0e-06.pkl"
            if not os.path.exists(file_name):
                if not os.path.exists(file_name.replace("CVG", "DVG")):
                    raise RuntimeError(f"The results file {file_name} is missing")
                else:
                    current_position += 2
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
                ax.plot(current_position + position_random, q_this_nb_random[2, -1, :], 'o', color=color)
            current_position += 2
        position_all_random = np.random.random((q_socp_integrated.shape[2], 1)) * 0.1
        ax.plot(current_position + position_all_random, q_integrated[2, -1, :], '.', color=color, alpha=0.5)
        current_position += 1
        box_plot(current_position, q_integrated[2, -1, :], color, ax, box_width=0.3)
        return

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    position_all_random = np.random.random((q_socp_integrated.shape[2], 1)) * 0.1
    ax.plot(position_all_random, q_ocp_integrated[2, -1, :], 'o', color=OCP_color)
    box_plot(1, q_ocp_integrated[2, -1, :], OCP_color, ax)
    plot_distribution(q_socp_integrated, 10, SOCP_color, "DMS", ax)
    plot_distribution(q_socp_variable_integrated, 20, SOCP_VARIABLE_color, "VARIABLE_DMS", ax)
    plot_distribution(q_socp_feedforward_integrated, 30, SOCP_FEEDFORWARD_color, "FEEDFORWARD_DMS", ax)
    plot_distribution(q_socp_plus_integrated, 40, SOCP_plus_color, "VARIABLE_FEEDFORWARD_DMS", ax)
    ax.set_ylabel("Final somersault angle [rad]")
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xticklabels(["OCP", "SOCP", "SOCPV", "SOCPA", "SOCP+"])
    plt.savefig("graphs/comparison_nb_random.png")
    # plt.show()
    return

def define_q_mean(n_q, n_shooting, nb_random, q_roots, q_joints, qdot_roots, qdot_joints):
    q, qdot = get_q_qdot_from_data(n_shooting, nb_random, q_roots, q_joints, qdot_roots, qdot_joints)
    q_mean = np.mean(q, axis=2)
    qdot_mean = np.mean(qdot, axis=2)
    return q, qdot, q_mean, qdot_mean

FLAG_GENERATE_VIDEOS = False
model_name = "Model2D_7Dof_0C_3M"

OCP_color = "#5DC962"
SOCP_color = "#AC2594"
SOCP_VARIABLE_color = "#F18F01"
SOCP_FEEDFORWARD_color = "#A469F1"
SOCP_plus_color = "#06b0f0"

biorbd_model_path = f"models/{model_name}.bioMod"
biorbd_model_path_with_mesh_ocp = f"models/{model_name}_with_mesh_ocp.bioMod"
biorbd_model_path_with_mesh_all = f"models/{model_name}_with_mesh_all.bioMod"

biorbd_model_path_with_mesh_socp = f"models/{model_name}_with_mesh_socp.bioMod"
biorbd_model_path_with_mesh_all_socp = f"models/{model_name}_with_mesh_all_socp.bioMod"

biorbd_model_path_with_mesh_socp_variable = f"models/{model_name}_with_mesh_socp_variable.bioMod"
biorbd_model_path_with_mesh_all_socp_variable = f"models/{model_name}_with_mesh_all_socp_variable.bioMod"

biorbd_model_path_with_mesh_socp_feedforward = f"models/{model_name}_with_mesh_socp_feedforward.bioMod"
biorbd_model_path_with_mesh_all_socp_feedforward = f"models/{model_name}_with_mesh_all_socp_feedforward.bioMod"

biorbd_model_path_vision = f"models/{model_name}_vision.bioMod"
biorbd_model_path_vision_with_mesh = f"models/{model_name}_with_mesh_socp_plus.bioMod"
biorbd_model_path_vision_with_mesh_all = f"models/{model_name}_with_mesh_all_socp_plus.bioMod"

biorbd_model_path_comparison = f"models/{model_name}_comparison_5versions.bioMod"


n_q = 7
n_root = 3
n_joints = n_q - n_root
n_ref = 2 * n_joints + 2

dt = 0.05
final_time = 0.8
n_shooting = int(final_time / dt)
tol = 1e-6
nb_random = 15
nb_reintegrations = 3  # TODO: change to 100

motor_noise_std = 0.05 * 10
wPq_std = 0.001 * 5
wPqdot_std = 0.003 * 5
motor_noise_magnitude = cas.DM(np.array([motor_noise_std**2 / dt for _ in range(n_q - n_root)]))  # All DoFs except root

# ------------- result paths ------------- #
result_folder = f"{nb_random}random"
ocp_path_to_results = f"results/deterministic/{model_name}_ocp_DMS_CVG_1e-8.pkl"
socp_path_to_results = (
    f"results/{result_folder}/Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02_DMS_{nb_random}random_CVG_1p0e-06.pkl"
)
socp_variable_path_to_results = (
    f"results/{result_folder}/Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02_VARIABLE_DMS_{nb_random}random_CVG_1p0e-06.pkl"
)
socp_feedforward_path_to_results = f"results/{result_folder}/Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02_FEEDFORWARD_DMS_{nb_random}random_CVG_1p0e-06.pkl"
socp_plus_path_to_results = f"results/{result_folder}/Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02_VARIABLE_FEEDFORWARD_DMS_{nb_random}random_CVG_1p0e-06.pkl"


# ------------- symbolics ------------- #
Q = cas.MX.sym("Q", n_q)
Qdot = cas.MX.sym("Qdot", n_q)
Tau = cas.MX.sym("Tau", n_joints)
MotorNoise = cas.MX.sym("Motor_noise", n_joints)
SensoryNoise = cas.MX.sym("Sensory_noise", n_ref)
FF_SensoryNoise = cas.MX.sym("FF_Sensory_noise", 1)

Q_8 = cas.MX.sym("Q", n_q + 1)
Qdot_8 = cas.MX.sym("Qdot", n_q + 1)
Tau_8 = cas.MX.sym("Tau", n_joints + 1)
MotorNoise_8 = cas.MX.sym("Motor_noise", n_joints + 1)
SensoryNoise_8 = cas.MX.sym("Sensory_noise", 2 * n_joints + 3)

q_sym = cas.MX.sym("Q", n_q, nb_random)
qdot_sym = cas.MX.sym("Qdot", n_q, nb_random)
tau_sym = cas.MX.sym("Tau", n_joints)
k_matrix_sym = cas.MX.sym("k_matrix", n_joints, n_ref)
ref_sym = cas.MX.sym("Ref", 2 * n_joints + 2)
motor_noise_sym = cas.MX.sym("Motor_noise", n_joints, nb_random)
sensory_noise_sym = cas.MX.sym("sensory_noise", n_ref, nb_random)
time_sym = cas.MX.sym("Time", 1)
tf_sym = cas.MX.sym("Tf", 1)

q_8_sym = cas.MX.sym("Q", n_q + 1, nb_random)
qdot_8_sym = cas.MX.sym("Qdot", n_q + 1, nb_random)
tau_8_sym = cas.MX.sym("Tau", n_joints + 1)
k_fb_matrix_sym = cas.MX.sym("k_matrix_fb", n_joints + 1, n_ref)
k_ff_matrix_sym = cas.MX.sym("k_ff_matrix", n_joints + 1, 1)
fb_ref_sym = cas.MX.sym("fb_ref", 2 * n_joints + 2)
ff_ref_sym = cas.MX.sym("ff_ref", 1)
motor_noise_8_sym = cas.MX.sym("Motor_noise", n_joints + 1, nb_random)
sensory_noise_8_sym = cas.MX.sym("sensory_noise", 2 * n_joints + 3, nb_random)

# ------------------------------------- #


# OCP
with open(ocp_path_to_results, "rb") as file:
    data = pickle.load(file)
    q_roots_ocp = data["q_roots_sol"]
    q_joints_ocp = data["q_joints_sol"]
    qdot_roots_ocp = data["qdot_roots_sol"]
    qdot_joints_ocp = data["qdot_joints_sol"]
    tau_joints_ocp = data["tau_joints_sol"]
    time_ocp = data["time_sol"]

ocp = prepare_ocp(biorbd_model_path=biorbd_model_path, time_last=final_time, n_shooting=n_shooting)

forward_dynamics_func = cas.Function("forward_dynamics", [Q, Qdot, Tau], [ocp.nlp[0].model.forward_dynamics(Q, Qdot, cas.vertcat(cas.MX.zeros(3), Tau))])

time_vector_ocp = np.linspace(0, float(time_ocp), n_shooting + 1)

q_ocp = np.vstack((q_roots_ocp, q_joints_ocp))
qdot_ocp = np.vstack((qdot_roots_ocp, qdot_joints_ocp))
# if FLAG_GENERATE_VIDEOS:
#     print("Generating OCP_one : ", ocp_path_to_results)
#     bioviz_animate(biorbd_model_path_with_mesh_ocp, np.vstack((q_roots_ocp, q_joints_ocp)), "OCP_one")

q_ocp_integrated, qdot_ocp_integrated, q_all_ocp, joint_frictions_ocp, motor_noises_ocp = noisy_integrate_ocp(
    n_shooting=n_shooting,
    nb_random=nb_random,
    nb_reintegrations=nb_reintegrations,
    q_roots_ocp = q_roots_ocp,
    q_joints_ocp = q_joints_ocp,
    motor_noise_magnitude=motor_noise_magnitude,
    tau_joints_ocp=tau_joints_ocp,
    time_vector_ocp=time_vector_ocp,
    ocp=ocp,
    forward_dynamics_func=forward_dynamics_func,
)

ocp_out_path_to_results = ocp_path_to_results.replace(".pkl", "_integrated.pkl")
with open(ocp_out_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_ocp_integrated,
        "time_vector": time_vector_ocp,
        "q_nominal": cas.vertcat(q_roots_ocp, q_joints_ocp),
    }
    pickle.dump(data, file)


# if FLAG_GENERATE_VIDEOS:
#     print("Generating OCP_all : ", ocp_path_to_results)
#     bioviz_animate(biorbd_model_path_with_mesh_all, q_all_ocp, "OCP_all")

joint_friction_ocp = np.zeros((n_q - 3, n_shooting))
for i_shooting in range(n_shooting):
    joint_friction_ocp[:, i_shooting] = np.reshape(
        ocp.nlp[0].model.friction_coefficients @ qdot_joints_ocp[:, i_shooting], (-1,)
    )


# SOCP
sensory_noise_magnitude = cas.DM(
    cas.vertcat(
        np.array([wPq_std**2 / dt for _ in range(n_q - n_root + 1)]),
        np.array([wPqdot_std**2 / dt for _ in range(n_q - n_root + 1)]),
    )
)

_, _, socp, _ = prepare_socp(
    biorbd_model_path=biorbd_model_path,
    time_last=time_ocp,
    n_shooting=n_shooting,
    motor_noise_magnitude=motor_noise_magnitude,
    sensory_noise_magnitude=sensory_noise_magnitude,
    q_roots_last=q_roots_ocp,
    q_joints_last=q_joints_ocp,
    qdot_roots_last=qdot_roots_ocp,
    qdot_joints_last=qdot_joints_ocp,
    tau_joints_last=tau_joints_ocp,
    k_last=None,
    ref_last=None,
    nb_random=nb_random,
)

with open(socp_path_to_results, "rb") as file:
    data = pickle.load(file)
    q_roots_socp = data["q_roots_sol"]
    q_joints_socp = data["q_joints_sol"]
    qdot_roots_socp = data["qdot_roots_sol"]
    qdot_joints_socp = data["qdot_joints_sol"]
    tau_joints_socp = data["tau_joints_sol"]
    time_socp = data["time_sol"]
    k_socp = data["k_sol"]
    ref_socp = data["ref_sol"]
    motor_noise_numerical_socp = data["motor_noise_numerical"]
    sensory_noise_numerical_socp = data["sensory_noise_numerical"]

DMS_sensory_reference_func = cas.Function(
    "DMS_sensory_reference", [Q, Qdot], [DMS_sensory_reference(socp.nlp[0].model, n_root, Q, Qdot)]
)

forward_dynamics_func = cas.Function("forward_dynamics", [Q, Qdot, Tau], [socp.nlp[0].model.forward_dynamics(Q, Qdot, cas.vertcat(cas.MX.zeros(3), Tau))])

time_vector_socp = np.linspace(0, float(time_socp), n_shooting + 1)

q_socp, qdot_socp, q_mean_socp, qdot_mean_socp = define_q_mean(n_q, n_shooting, nb_random, q_roots_socp, q_joints_socp, qdot_roots_socp, qdot_joints_socp)

q_socp_integrated, qdot_socp_integrated, q_all_socp, joint_frictions_socp, motor_noises_socp, feedbacks_socp = (
    noisy_integrate_socp(
        socp,
        motor_noise_magnitude,
        sensory_noise_magnitude,
        n_shooting,
        nb_random,
        nb_reintegrations,
        q_socp,
        tau_joints_socp,
        k_socp,
        ref_socp,
        time_vector_socp,
        q_mean_socp,
        DMS_sensory_reference_func,
        forward_dynamics_func,
    )
)

q_mean_socp = np.mean(q_socp, axis=2)
# if FLAG_GENERATE_VIDEOS:
#     print("Generating SOCP_one : ", socp_path_to_results)
#     bioviz_animate(biorbd_model_path_with_mesh_socp, q_mean_socp, "SOCP_one")

socp_out_path_to_results = socp_path_to_results.replace(".pkl", "_integrated.pkl")
with open(socp_out_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_socp_integrated,
        "qdot_integrated": qdot_socp_integrated,
        "time_vector": time_vector_socp,
        "q_mean_integrated": np.mean(q_socp_integrated, axis=2),
        "q_mean": q_mean_socp,
    }
    pickle.dump(data, file)

# if FLAG_GENERATE_VIDEOS:
#     print("Generating SOCP_all : ", socp_path_to_results)
#     bioviz_animate(biorbd_model_path_with_mesh_all_socp, q_all_socp, "SOCP_all")


# SOCP VARIABLE
sensory_noise_magnitude = cas.DM(
    np.array(
        [
            wPq_std**2 / dt,  # Proprioceptive position
            wPq_std**2 / dt,
            wPq_std**2 / dt,
            wPq_std**2 / dt,
            wPqdot_std**2 / dt,  # Proprioceptive velocity
            wPqdot_std**2 / dt,
            wPqdot_std**2 / dt,
            wPqdot_std**2 / dt,
            wPq_std**2 / dt,  # Vestibular position
            wPq_std**2 / dt,  # Vestibular velocity
        ]
    )
)
_, _, socp_variable, _ = prepare_socp_VARIABLE(
    biorbd_model_path=biorbd_model_path,
    time_last=time_ocp,
    n_shooting=n_shooting,
    motor_noise_magnitude=motor_noise_magnitude,
    sensory_noise_magnitude=sensory_noise_magnitude,
    q_roots_last=q_roots_ocp,
    q_joints_last=q_joints_ocp,
    qdot_roots_last=qdot_roots_ocp,
    qdot_joints_last=qdot_joints_ocp,
    tau_joints_last=tau_joints_ocp,
    k_last=None,
    ref_last=None,
    nb_random=nb_random,
)

with open(socp_variable_path_to_results, "rb") as file:
    data = pickle.load(file)
    q_roots_socp_variable = data["q_roots_sol"]
    q_joints_socp_variable = data["q_joints_sol"]
    qdot_roots_socp_variable = data["qdot_roots_sol"]
    qdot_joints_socp_variable = data["qdot_joints_sol"]
    tau_joints_socp_variable = data["tau_joints_sol"]
    time_socp_variable = data["time_sol"]
    k_socp_variable = data["k_sol"]
    ref_socp_variable = data["ref_sol"]
    motor_noise_numerical_socp_variable = data["motor_noise_numerical"]
    sensory_noise_numerical_socp_variable = data["sensory_noise_numerical"]

DMS_fb_noised_sensory_input_VARIABLE_func = cas.Function(
    "DMS_fb_noised_sensory_input_VARIABLE",
    [Q, Qdot, SensoryNoise],
    [DMS_fb_noised_sensory_input_VARIABLE(socp_variable.nlp[0].model, Q[:n_root], Q[n_root:], Qdot[:n_root], Qdot[n_root:], SensoryNoise)],
)
time_vector_socp_variable = np.linspace(0, float(time_socp_variable), n_shooting + 1)

q_socp_variable, qdot_socp_variable, q_mean_socp_variable, qdot_mean_socp_variable = define_q_mean(n_q, n_shooting, nb_random, q_roots_socp_variable, q_joints_socp_variable,
                                               qdot_roots_socp_variable, qdot_joints_socp_variable)

q_socp_variable_integrated, qdot_socp_variable_integrated, q_all_socp_variable, joint_frictions_socp_variable, motor_noises_socp_variable, feedbacks_socp_variable = (
    noisy_integrate_socp_variable(
        socp_variable,
        motor_noise_magnitude,
        sensory_noise_magnitude,
        n_shooting,
        nb_random,
        nb_reintegrations,
        q_socp_variable,
        tau_joints_socp_variable,
        k_socp_variable,
        ref_socp_variable,
        time_vector_socp_variable,
        q_mean_socp_variable,
        DMS_fb_noised_sensory_input_VARIABLE_func,
        forward_dynamics_func,
    )
)

q_mean_socp_variable = np.mean(q_socp_variable, axis=2)
# if FLAG_GENERATE_VIDEOS:
#     print("Generating SOCP_VARIABLE_one : ", socp_variable_path_to_results)
#     bioviz_animate(biorbd_model_path_with_mesh_socp_variable, q_mean_socp_variable, "SOCP_VARIABLE_one")

socp_variable_out_path_to_results = socp_variable_path_to_results.replace(".pkl", "_integrated.pkl")
with open(socp_variable_out_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_socp_variable_integrated,
        "qdot_integrated": qdot_socp_variable_integrated,
        "time_vector": time_vector_socp_variable,
        "q_mean_integrated": np.mean(q_socp_variable_integrated, axis=2),
        "q_mean": q_mean_socp_variable,
    }
    pickle.dump(data, file)

# if FLAG_GENERATE_VIDEOS:
#     print("Generating SOCP_VARIABLE_all : ", socp_variable_path_to_results)
#     bioviz_animate(biorbd_model_path_with_mesh_all_socp_variable, q_all_socp_variable, "SOCP_VARIABLE_all")


# SOCP FEEDFORWARD
n_q = 8
n_joints = n_q - 3
motor_noise_magnitude = cas.DM(
    np.array(
        [
            motor_noise_std**2 / dt,
            0.0,
            motor_noise_std**2 / dt,
            motor_noise_std**2 / dt,
            motor_noise_std**2 / dt,
        ]
    )
)  # All DoFs except root
sensory_noise_magnitude = cas.DM(
    np.array(
        [
            wPq_std**2 / dt,  # Proprioceptive position
            wPq_std**2 / dt,
            wPq_std**2 / dt,
            wPq_std**2 / dt,
            wPqdot_std**2 / dt,  # Proprioceptive velocity
            wPqdot_std**2 / dt,
            wPqdot_std**2 / dt,
            wPqdot_std**2 / dt,
            wPq_std**2 / dt,  # Vestibular position
            wPq_std**2 / dt,  # Vestibular velocity
            wPq_std**2 / dt,  # Visual
        ]
    )
)

q_joints_last = np.vstack((q_joints_ocp[0, :], np.zeros((1, q_joints_ocp.shape[1])), q_joints_ocp[1:, :]))
q_joints_last[1, :5] = -0.5
q_joints_last[1, 5:-5] = np.linspace(-0.5, 0.3, n_shooting + 1 - 10)
q_joints_last[1, -5:] = 0.3

qdot_joints_last = np.vstack(
    (qdot_joints_ocp[0, :], np.ones((1, qdot_joints_ocp.shape[1])) * 0.01, qdot_joints_ocp[1:, :])
)
tau_joints_last = np.vstack(
    (tau_joints_ocp[0, :], np.ones((1, tau_joints_ocp.shape[1])) * 0.01, tau_joints_ocp[1:, :])
)

_, _, socp_feedforward, noised_states = prepare_socp_FEEDFORWARD(
    biorbd_model_path=biorbd_model_path_vision,
    time_last=time_ocp,
    n_shooting=n_shooting,
    motor_noise_magnitude=motor_noise_magnitude,
    sensory_noise_magnitude=sensory_noise_magnitude,
    q_roots_last=q_roots_ocp,
    q_joints_last=q_joints_last,
    qdot_roots_last=qdot_roots_ocp,
    qdot_joints_last=qdot_joints_last,
    tau_joints_last=tau_joints_last,
    k_last=None,
    ref_last=None,
    nb_random=nb_random,
)

with open(socp_feedforward_path_to_results, "rb") as file:
    data = pickle.load(file)
    q_roots_socp_feedforward = data["q_roots_sol"]
    q_joints_socp_feedforward = data["q_joints_sol"]
    qdot_roots_socp_feedforward = data["qdot_roots_sol"]
    qdot_joints_socp_feedforward = data["qdot_joints_sol"]
    tau_joints_socp_feedforward = data["tau_joints_sol"]
    time_socp_feedforward = data["time_sol"]
    k_socp_feedforward = data["k_sol"]
    ref_fb_socp_feedforward = data["ref_fb_sol"]
    ref_ff_socp_feedforward = data["ref_ff_sol"]
    motor_noise_numerical_socp_feedforward = data["motor_noise_numerical"]
    sensory_noise_numerical_socp_feedforward = data["sensory_noise_numerical"]

DMS_ff_sensory_input_func = cas.Function(
    "DMS_fb_noised_sensory_input_no_eyes",
    [tf_sym, time_sym, Q_8, Qdot_8],
    [DMS_ff_sensory_input(socp_feedforward.nlp[0].model, tf_sym, time_sym, Q_8, Qdot_8)],
)

DMS_sensory_reference_no_eyes_func = cas.Function(
    "DMS_sensory_reference_no_eyes",
    [Q_8, Qdot_8],
    [DMS_sensory_reference_no_eyes(socp_feedforward.nlp[0].model, n_root, Q_8, Qdot_8)]
)

forward_dynamics_func = cas.Function("forward_dynamics", [Q_8, Qdot_8, Tau_8], [socp_feedforward.nlp[0].model.forward_dynamics(Q_8, Qdot_8, cas.vertcat(cas.MX.zeros(3), Tau_8))])

time_vector_socp_feedforward = np.linspace(0, float(time_socp_feedforward), n_shooting + 1)

q_socp_feedforward, qdot_socp_feedforward, q_mean_socp_feedforward, qdot_mean_socp_feedforward = define_q_mean(n_q, n_shooting, nb_random, q_roots_socp_feedforward, q_joints_socp_feedforward, qdot_roots_socp_feedforward, qdot_joints_socp_feedforward)

(
    q_socp_feedforward_integrated,
    qdot_socp_feedforward_integrated,
    q_all_socp_feedforward,
    joint_frictions_socp_feedforward,
    motor_noises_socp_feedforward,
    feedbacks_socp_feedforward,
    feedforwards_socp_feedforward,
) = noisy_integrate_socp_feedforward(
    socp_feedforward,
    motor_noise_magnitude,
    sensory_noise_magnitude,
    n_shooting,
    nb_random,
    nb_reintegrations,
    q_socp_feedforward,
    tau_joints_socp_feedforward,
    k_socp_feedforward,
    ref_fb_socp_feedforward,
    ref_ff_socp_feedforward,
    time_vector_socp_feedforward,
    q_mean_socp_feedforward,
    DMS_sensory_reference_no_eyes_func,
    DMS_ff_sensory_input_func,
    forward_dynamics_func,
)

q_mean_socp_feedforward = np.mean(q_socp_feedforward, axis=2)
if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_FEEDFORWARD_one : ", socp_feedforward_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_socp_feedforward, q_mean_socp_feedforward, "SOCP_FEEDFORWARD_one")

socp_feedforward_out_path_to_results = socp_feedforward_path_to_results.replace(".pkl", "_integrated.pkl")
with open(socp_feedforward_out_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_socp_feedforward_integrated,
        "qdot_integrated": qdot_socp_feedforward_integrated,
        "time_vector": time_vector_socp_feedforward,
        "q_mean_integrated": np.mean(q_socp_feedforward_integrated, axis=2),
        "q_mean": q_mean_socp_feedforward,
    }
    pickle.dump(data, file)

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_FEEDFORWARD_all : ", socp_feedforward_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_all_socp_feedforward, q_all_socp_feedforward, "SOCP_FEEDFORWARD_all")


# SOCP+
n_q = 8
n_root = 3
n_joints = n_q - n_root
motor_noise_magnitude = cas.DM(
    np.array(
        [
            motor_noise_std**2 / dt,
            0.0,
            motor_noise_std**2 / dt,
            motor_noise_std**2 / dt,
            motor_noise_std**2 / dt,
        ]
    )
)  # All DoFs except root
sensory_noise_magnitude = cas.DM(
    np.array(
        [
            wPq_std**2 / dt,  # Proprioceptive position
            wPq_std**2 / dt,
            wPq_std**2 / dt,
            wPq_std**2 / dt,
            wPqdot_std**2 / dt,  # Proprioceptive velocity
            wPqdot_std**2 / dt,
            wPqdot_std**2 / dt,
            wPqdot_std**2 / dt,
            wPq_std**2 / dt,  # Vestibular position
            wPq_std**2 / dt,  # Vestibular velocity
            wPq_std**2 / dt,  # Visual
        ]
    )
)

q_joints_last = np.vstack((q_joints_ocp[0, :], np.zeros((1, q_joints_ocp.shape[1])), q_joints_ocp[1:, :]))
qdot_joints_last = np.vstack(
    (qdot_joints_ocp[0, :], np.ones((1, qdot_joints_ocp.shape[1])) * 0.01, qdot_joints_ocp[1:, :])
)
tau_joints_last = np.vstack((tau_joints_ocp[0, :], np.ones((1, tau_joints_ocp.shape[1])) * 0.01, tau_joints_ocp[1:, :]))

_, _, socp_plus, _ = prepare_socp_VARIABLE_FEEDFORWARD(
    biorbd_model_path=biorbd_model_path_vision,
    time_last=time_ocp,
    n_shooting=n_shooting,
    motor_noise_magnitude=motor_noise_magnitude,
    sensory_noise_magnitude=sensory_noise_magnitude,
    q_roots_last=q_roots_ocp,
    q_joints_last=q_joints_last,
    qdot_roots_last=qdot_roots_ocp,
    qdot_joints_last=qdot_joints_last,
    tau_joints_last=tau_joints_last,
    k_last=None,
    ref_last=None,
    nb_random=nb_random,
)

with open(socp_plus_path_to_results, "rb") as file:
    data = pickle.load(file)
    q_roots_socp_plus = data["q_roots_sol"]
    q_joints_socp_plus = data["q_joints_sol"]
    qdot_roots_socp_plus = data["qdot_roots_sol"]
    qdot_joints_socp_plus = data["qdot_joints_sol"]
    tau_joints_socp_plus = data["tau_joints_sol"]
    time_socp_plus = data["time_sol"]
    k_socp_plus = data["k_sol"]
    ref_fb_socp_plus = data["ref_fb_sol"]
    ref_ff_socp_plus = data["ref_ff_sol"]
    motor_noise_numerical_socp_plus = data["motor_noise_numerical"]
    sensory_noise_numerical_socp_plus = data["sensory_noise_numerical"]


DMS_sensory_reference_no_eyes_func = cas.Function(
    "DMS_fb_sensory_reference", [Q_8, Qdot_8], [DMS_sensory_reference_no_eyes(socp_plus.nlp[0].model, n_root, Q_8, Qdot_8)]
)
DMS_ff_noised_sensory_input_func = cas.Function(
    "DMS_ff_sensory_reference", [tf_sym, time_sym, Q_8, Qdot_8, FF_SensoryNoise], [DMS_ff_noised_sensory_input(socp_plus.nlp[0].model, tf_sym, time_sym, Q_8, Qdot_8, FF_SensoryNoise)]
)

DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func = cas.Function(
    "DMS_fb_noised_sensory_input_VARIABLE_no_eyes",
    [Q_8, Qdot_8, SensoryNoise_8],
    [DMS_fb_noised_sensory_input_VARIABLE_no_eyes(socp_plus.nlp[0].model, Q_8[:n_root], Q_8[n_root:], Qdot_8[:n_root], Qdot_8[n_root:], SensoryNoise_8)],
)

forward_dynamics_func = cas.Function("forward_dynamics", [Q_8, Qdot_8, Tau_8], [socp_plus.nlp[0].model.forward_dynamics(Q_8, Qdot_8, cas.vertcat(cas.MX.zeros(3), Tau_8))])

time_vector_socp_plus = np.linspace(0, float(time_socp_plus), n_shooting + 1)

q_socp_plus, qdot_socp_plus, q_mean_socp_plus, qdot_mean_socp_plus = define_q_mean(n_q, n_shooting, nb_random, q_roots_socp_plus, q_joints_socp_plus, qdot_roots_socp_plus, qdot_joints_socp_plus)

(
    q_socp_plus_integrated,
    qdot_socp_plus_integrated,
    q_all_socp_plus,
    joint_frictions_socp_plus,
    motor_noises_socp_plus,
    feedbacks_socp_plus,
    feedforwards_socp_plus,
) = noisy_integrate_socp_plus(
    socp_plus,
    motor_noise_magnitude,
    sensory_noise_magnitude,
    n_shooting,
    nb_random,
    nb_reintegrations,
    q_socp_plus,
    tau_joints_socp_plus,
    k_socp_plus,
    ref_fb_socp_plus,
    ref_ff_socp_plus,
    time_vector_socp_plus,
    q_mean_socp_plus,
    DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func,
    DMS_ff_noised_sensory_input_func,
    forward_dynamics_func,
)

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_plus_one : ", socp_plus_path_to_results)
    bioviz_animate(biorbd_model_path_vision_with_mesh, q_mean_socp_plus, "SOCP_plus_one")

socp_plus_out_path_to_results = socp_plus_path_to_results.replace(".pkl", "_integrated.pkl")
with open(socp_plus_out_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_socp_plus_integrated,
        "qdot_integrated": qdot_socp_plus_integrated,
        "time_vector": time_vector_socp_plus,
        "q_mean_integrated": np.mean(q_socp_plus_integrated, axis=2),
        "q_mean": q_mean_socp_plus,
    }
    pickle.dump(data, file)

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_plus_all : ", socp_plus_path_to_results)
    bioviz_animate(biorbd_model_path_vision_with_mesh_all, q_all_socp_plus, "SOCP_plus_all")


# Comparison ----------------------------------------------------------------------------------------------------------
q_mean_comparison = np.zeros((7 + 7 + 7 + 8 + 8, n_shooting + 1))
q_mean_comparison[:7, :] = q_ocp
q_mean_comparison[7:7+7, :] = q_mean_socp
q_mean_comparison[7+7: 7+7+7, :] = q_mean_socp_variable
q_mean_comparison[7+7+7: 7+7+7+8, :] = q_mean_socp_feedforward
q_mean_comparison[7+7+7+8:, :] = q_mean_socp_plus
if FLAG_GENERATE_VIDEOS:
    print("Generating comparison")
    bioviz_animate(biorbd_model_path_comparison, q_mean_comparison, "Comparison")


# Plots ---------------------------------------------------------------------------------------------------------------
time_vector_ocp = time_vector_ocp[:-1]
time_vector_socp = time_vector_socp[:-1]
time_vector_socp_variable = time_vector_socp_variable[:-1]
time_vector_socp_feedforward = time_vector_socp_feedforward[:-1]
time_vector_socp_plus = time_vector_socp_plus[:-1]
normalized_time_vector = np.linspace(0, 1, n_shooting)

plot_motor_command(nb_random,
                    normalized_time_vector,
                    tau_joints_ocp,
                    tau_joints_socp,
                    tau_joints_socp_variable,
                    tau_joints_socp_feedforward,
                    tau_joints_socp_plus,
                    joint_friction_ocp,
                    joint_frictions_socp,
                    joint_frictions_socp_variable,
                    joint_frictions_socp_feedforward,
                    joint_frictions_socp_plus,
                    OCP_color,
                    SOCP_color,
                    SOCP_plus_color,
                    motor_noises_socp,
                    motor_noises_socp_variable,
                    feedbacks_socp,
                    feedbacks_socp_variable,
                    feedbacks_socp_feedforward,
                    feedbacks_socp_plus,
                    feedforwards_socp_feedforward,
                    feedforwards_socp_plus)

plot_tau_and_delta_tau(normalized_time_vector,
                        tau_joints_ocp,
                        tau_joints_socp,
                        tau_joints_socp_variable,
                        tau_joints_socp_feedforward,
                        tau_joints_socp_plus,
                        joint_friction_ocp,
                        joint_frictions_socp,
                        joint_frictions_socp_variable,
                        joint_frictions_socp_feedforward,
                        joint_frictions_socp_plus,
                        nb_random,
                        OCP_color,
                        SOCP_color,
                        SOCP_VARIABLE_color,
                        SOCP_FEEDFORWARD_color,
                        SOCP_plus_color,
                        motor_noises_socp,
                        motor_noises_socp_variable,
                        motor_noises_socp_feedforward,
                        motor_noises_socp_plus,
                        feedbacks_socp,
                        feedbacks_socp_variable,
                        feedbacks_socp_feedforward,
                        feedbacks_socp_plus,
                        feedforwards_socp_feedforward,
                        feedforwards_socp_plus,)

plt.figure()
plt.bar(0, float(time_ocp), color=OCP_color, label="OCP")
plt.bar(1, float(time_socp), color=SOCP_color, label="SOCP")
plt.bar(2, float(time_socp_variable), color=SOCP_VARIABLE_color, label="SOCPV")
plt.bar(3, float(time_socp_feedforward), color=SOCP_FEEDFORWARD_color, label="SOCPA")
plt.bar(4, float(time_socp_plus), color=SOCP_plus_color, label="SOCP+")
plt.xticks([0, 1, 2, 3, 4], ["OCP", "SOCP", "SOCPV", "SOCPA", "SOCP+"])
print("Movement durations: ", time_ocp, time_socp, time_socp_variable, time_socp_feedforward, time_socp_plus)
plt.savefig("graphs/movement_durations.png")
# plt.show()

plot_gains(socp_variable,
        socp_plus,
        normalized_time_vector,
        k_socp,
        k_socp_variable,
        k_socp_feedforward,
        k_socp_plus,
        SOCP_color,
        SOCP_VARIABLE_color,
        SOCP_FEEDFORWARD_color,
        SOCP_plus_color)

# Plot the landing variability
CoM_y_fcn = cas.Function("CoM_y", [Q], [socp.nlp[0].model.model.CoM(Q).to_mx()[1]])
CoM_y_8_fcn = cas.Function("CoM_y", [Q_8], [socp_plus.nlp[0].model.model.CoM(Q_8).to_mx()[1]])
CoM_y_dot_fcn = cas.Function("CoM_y_dot", [Q, Qdot], [socp.nlp[0].model.model.CoMdot(Q, Qdot).to_mx()[1]])
CoM_dot_8_fcn = cas.Function("CoM_dot", [Q_8, Qdot_8], [socp_plus.nlp[0].model.model.CoMdot(Q_8, Qdot_8).to_mx()[1]])
BodyVelocity_fcn = cas.Function(
    "BodyVelocity", [Q, Qdot], [socp.nlp[0].model.model.bodyAngularVelocity(Q, Qdot).to_mx()[0]]
)
BodyVelocity_8_fcn = cas.Function(
    "BodyVelocity", [Q_8, Qdot_8], [socp_plus.nlp[0].model.model.bodyAngularVelocity(Q_8, Qdot_8).to_mx()[0]]
)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
CoM_y_ocp = np.zeros((nb_random, 1))
CoM_y_socp = np.zeros((nb_random, 1))
CoM_y_socp_variable = np.zeros((nb_random, 1))
CoM_y_socp_feedforward = np.zeros((nb_random, 1))
CoM_y_socp_plus = np.zeros((nb_random, 1))
CoM_dot_y_ocp = np.zeros((nb_random, 1))
CoM_dot_y_socp = np.zeros((nb_random, 1))
CoM_dot_y_socp_variable = np.zeros((nb_random, 1))
CoM_dot_y_socp_feedforward = np.zeros((nb_random, 1))
CoM_dot_y_socp_plus = np.zeros((nb_random, 1))
BodyVelocity_ocp = np.zeros((nb_random, 1))
BodyVelocity_socp = np.zeros((nb_random, 1))
BodyVelocity_socp_variable = np.zeros((nb_random, 1))
BodyVelocity_socp_feedforward = np.zeros((nb_random, 1))
BodyVelocity_socp_plus = np.zeros((nb_random, 1))
for i_random in range(nb_random):
    CoM_y_ocp[i_random] = CoM_y_fcn(q_ocp_integrated[:, -1, i_random])
    CoM_y_socp[i_random] = CoM_y_fcn(q_socp[:, -1, i_random])
    CoM_y_socp_variable[i_random] = CoM_y_fcn(q_socp_variable[:, -1, i_random])
    CoM_y_socp_feedforward[i_random] = CoM_y_8_fcn(q_socp_feedforward[:, -1, i_random])
    CoM_y_socp_plus[i_random] = CoM_y_8_fcn(q_socp_plus[:, -1, i_random])
    axs[0].plot(0 + np.random.random(1) * 0.1, CoM_y_ocp[i_random], ".", color=OCP_color)
    axs[0].plot(0.5 + np.random.random(1) * 0.1, CoM_y_socp[i_random], ".", color=SOCP_color)
    axs[0].plot(1 + np.random.random(1) * 0.1, CoM_y_socp_variable[i_random], ".", color=SOCP_VARIABLE_color)
    axs[0].plot(1.5 + np.random.random(1) * 0.1, CoM_y_socp_feedforward[i_random], ".", color=SOCP_FEEDFORWARD_color)
    axs[0].plot(2 + np.random.random(1) * 0.1, CoM_y_socp_plus[i_random], ".", color=SOCP_plus_color)

    CoM_dot_y_ocp[i_random] = CoM_y_dot_fcn(q_ocp_integrated[:, -1, i_random], qdot_ocp_integrated[:, -1, i_random])
    CoM_dot_y_socp[i_random] = CoM_y_dot_fcn(q_socp[:, -1, i_random], qdot_socp[:, -1, i_random])
    CoM_dot_y_socp_variable[i_random] = CoM_y_dot_fcn(q_socp_variable[:, -1, i_random], qdot_socp_variable[:, -1, i_random])
    CoM_dot_y_socp_feedforward[i_random] = CoM_dot_8_fcn(q_socp_feedforward[:, -1, i_random], qdot_socp_feedforward[:, -1, i_random])
    CoM_dot_y_socp_plus[i_random] = CoM_dot_8_fcn(q_socp_plus[:, -1, i_random], qdot_socp_plus[:, -1, i_random])
    axs[1].plot(0 + np.random.random(1) * 0.1, CoM_dot_y_ocp[i_random], ".", color=OCP_color)
    axs[1].plot(0.5 + np.random.random(1) * 0.1, CoM_dot_y_socp[i_random], ".", color=SOCP_color)
    axs[1].plot(1 + np.random.random(1) * 0.1, CoM_dot_y_socp_variable[i_random], ".", color=SOCP_VARIABLE_color)
    axs[1].plot(1.5 + np.random.random(1) * 0.1, CoM_dot_y_socp_feedforward[i_random], ".", color=SOCP_FEEDFORWARD_color)
    axs[1].plot(2 + np.random.random(1) * 0.1, CoM_dot_y_socp_plus[i_random], ".", color=SOCP_plus_color)

    BodyVelocity_ocp[i_random] = BodyVelocity_fcn(
        q_ocp_integrated[:, -1, i_random], qdot_ocp_integrated[:, -1, i_random]
    )
    BodyVelocity_socp[i_random] = BodyVelocity_fcn(q_socp[:, -1, i_random], qdot_socp[:, -1, i_random])
    BodyVelocity_socp_variable[i_random] = BodyVelocity_fcn(q_socp_variable[:, -1, i_random], qdot_socp_variable[:, -1, i_random])
    BodyVelocity_socp_feedforward[i_random] = BodyVelocity_8_fcn(q_socp_feedforward[:, -1, i_random], qdot_socp_feedforward[:, -1, i_random])
    BodyVelocity_socp_plus[i_random] = BodyVelocity_8_fcn(q_socp_plus[:, -1, i_random], qdot_socp_plus[:, -1, i_random])
    axs[2].plot(0 + np.random.random(1) * 0.1, BodyVelocity_ocp[i_random], ".", color=OCP_color)
    axs[2].plot(0.5 + np.random.random(1) * 0.1, BodyVelocity_socp[i_random], ".", color=SOCP_color)
    axs[2].plot(1 + np.random.random(1) * 0.1, BodyVelocity_socp_variable[i_random], ".", color=SOCP_VARIABLE_color)
    axs[2].plot(1.5 + np.random.random(1) * 0.1, BodyVelocity_socp_feedforward[i_random], ".", color=SOCP_FEEDFORWARD_color)
    axs[2].plot(2 + np.random.random(1) * 0.1, BodyVelocity_socp_plus[i_random], ".", color=SOCP_plus_color)

axs[0].set_title("CoM y")
axs[1].set_title("CoM y dot")
axs[2].set_title("Body velocity")


box_plot(0, CoM_y_ocp, OCP_color, axs[0])
box_plot(0.5, CoM_y_socp, SOCP_color, axs[0])
box_plot(1, CoM_y_socp_variable, SOCP_VARIABLE_color, axs[0])
box_plot(1.5, CoM_y_socp_feedforward, SOCP_FEEDFORWARD_color, axs[0])
box_plot(2, CoM_y_socp_plus, SOCP_plus_color, axs[0])

box_plot(0, CoM_dot_y_ocp, OCP_color, axs[1])
box_plot(0.5, CoM_dot_y_socp, SOCP_color, axs[1])
box_plot(1, CoM_dot_y_socp_variable, SOCP_VARIABLE_color, axs[1])
box_plot(1.5, CoM_dot_y_socp_feedforward, SOCP_FEEDFORWARD_color, axs[1])
box_plot(2, CoM_dot_y_socp_plus, SOCP_plus_color, axs[1])

box_plot(0, BodyVelocity_ocp, OCP_color, axs[2])
box_plot(0.5, BodyVelocity_socp, SOCP_color, axs[2])
box_plot(1, BodyVelocity_socp_variable, SOCP_VARIABLE_color, axs[2])
box_plot(1.5, BodyVelocity_socp_feedforward, SOCP_FEEDFORWARD_color, axs[2])
box_plot(2, BodyVelocity_socp_plus, SOCP_plus_color, axs[2])

plt.savefig("graphs/landing_variability.png")
# plt.show()


time_vector = np.linspace(0, 1, n_shooting + 1)
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs = axs.ravel()
for i_dof in range(8):
    if i_dof < 4:
        axs[i_dof].plot(time_vector, q_ocp[i_dof, :], color=OCP_color)
    elif i_dof > 4:
        axs[i_dof].plot(time_vector, q_ocp[i_dof - 1, :], color=OCP_color)
    for i_random in range(nb_random):
        if i_dof < 4:
            axs[i_dof].plot(time_vector, q_socp[i_dof, :, i_random], color=SOCP_color, linewidth=0.5)
            axs[i_dof].plot(time_vector, q_socp_variable[i_dof, :, i_random], color=SOCP_VARIABLE_color, linewidth=0.5)
        elif i_dof > 4:
            axs[i_dof].plot(time_vector, q_socp[i_dof - 1, :, i_random], color=SOCP_color, linewidth=0.5)
            axs[i_dof].plot(time_vector, q_socp_variable[i_dof - 1, :, i_random], color=SOCP_VARIABLE_color, linewidth=0.5)
    for i_random in range(nb_random):
        axs[i_dof].plot(time_vector, q_socp_feedforward[i_dof, :, i_random], color=SOCP_FEEDFORWARD_color, linewidth=0.5)
        axs[i_dof].plot(time_vector, q_socp_plus[i_dof, :, i_random], color=SOCP_plus_color, linewidth=0.5)
plt.savefig("graphs/kinematics.png")
# plt.show()


plot_comparison_reintegration(
    q_ocp,
    q_mean_socp,
    q_mean_socp_variable,
    q_mean_socp_feedforward,
    q_mean_socp_plus,
    q_socp,
    q_socp_variable,
    q_socp_feedforward,
    q_socp_plus,
    q_ocp_integrated,
    q_socp_integrated,
    q_socp_variable_integrated,
    q_socp_feedforward_integrated,
    q_socp_plus_integrated,
    time_vector,
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_plus_color,
    nb_random,
    nb_reintegrations,
)

plot_mean_comparison(
    q_ocp,
    q_mean_socp,
    q_mean_socp_variable,
    q_mean_socp_feedforward,
    q_mean_socp_plus,
    q_socp,
    q_socp_variable,
    q_socp_feedforward,
    q_socp_plus,
    q_ocp_integrated,
    q_socp_integrated,
    q_socp_variable_integrated,
    q_socp_feedforward_integrated,
    q_socp_plus_integrated,
    time_vector,
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_plus_color,
)

plot_comparison_nb_random(q_ocp_integrated,
                        q_socp_integrated,
                        q_socp_variable_integrated,
                        q_socp_feedforward_integrated,
                        q_socp_plus_integrated,
                        OCP_color,
                        SOCP_color,
                        SOCP_VARIABLE_color,
                        SOCP_FEEDFORWARD_color,
                        SOCP_plus_color)

plt.show()