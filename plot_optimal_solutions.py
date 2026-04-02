import os
import pickle

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
import spm1d
from bioptim import StochasticBioModel

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
from plot_utils import (
    box_plot,
    add_std_to_box_plot,
    step_plot,
    get_q_qdot_from_data,
    define_q_mean,
    get_optimization_q_each_random,
)
from plot_reintegrate import (
    noisy_integrate_ocp,
    integrate_MS_ocp,
    noisy_integrate_socp,
    integrate_socp_MS,
    noisy_integrate_socp_variable,
    integrate_socp_variable_MS,
    noisy_integrate_socp_feedforward,
    integrate_socp_feedforward_MS,
    noisy_integrate_socp_plus,
    integrate_socp_plus_MS,
)
from animation_utils import bioviz_animate



def plot_comparison_reintegration(
    normalized_time_vector,
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
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_PLUS_color,
    nb_random,
    nb_reintegrations,
):

    n_q = q_socp_plus_nominal.shape[0]
    fig, axs = plt.subplots(n_q-2, 5, figsize=(15, 10))
    for i_ax, i_dof in enumerate(range(2, n_q)):

        # Reintegrated
        for i_random in range(nb_random * nb_reintegrations):
            if i_dof < 4 and i_dof > 1:
                axs[i_ax, 0].plot(
                    normalized_time_vector, q_ocp_integrated[i_dof, :, i_random], color=OCP_color, alpha=0.2, linewidth=0.5
                )
                axs[i_ax, 1].plot(
                    normalized_time_vector, q_socp_integrated["20random"][i_dof, :, i_random], color=SOCP_color, alpha=0.2, linewidth=0.5
                )
                axs[i_ax, 2].plot(
                    normalized_time_vector, q_socp_variable_integrated["20random"][i_dof, :, i_random], color=SOCP_VARIABLE_color, alpha=0.2, linewidth=0.5
                )
            elif i_dof > 4:
                axs[i_ax, 0].plot(
                    normalized_time_vector, q_ocp_integrated[i_dof - 1, :, i_random], color=OCP_color, alpha=0.2, linewidth=0.5
                )
                axs[i_ax, 1].plot(
                    normalized_time_vector, q_socp_integrated["20random"][i_dof - 1, :, i_random], color=SOCP_color, alpha=0.2, linewidth=0.5
                )
                axs[i_ax, 2].plot(
                    normalized_time_vector, q_socp_variable_integrated["20random"][i_dof - 1, :, i_random], color=SOCP_VARIABLE_color, alpha=0.2, linewidth=0.5
                )
            axs[i_ax, 3].plot(
                normalized_time_vector,
                q_socp_feedforward_integrated["20random"][i_dof, :, i_random],
                color=SOCP_FEEDFORWARD_color,
                alpha=0.2,
                linewidth=0.5,
            )
            axs[i_ax, 4].plot(
                normalized_time_vector,
                q_socp_plus_integrated["20random"][i_dof, :, i_random],
                color=SOCP_PLUS_color,
                alpha=0.2,
                linewidth=0.5,
            )

        # Optimzation variables
        for i_random in range(nb_random):
            if i_dof < 4 and i_dof > 1:
                axs[i_ax, 1].plot(normalized_time_vector, q_all_socp[i_dof, :, i_random], color="#6C165C", linewidth=0.5)
                axs[i_ax, 2].plot(normalized_time_vector, q_all_socp_variable[i_dof, :, i_random], color="#D15C02", linewidth=0.5)
            elif i_dof > 4:
                axs[i_ax, 1].plot(normalized_time_vector, q_all_socp[i_dof - 1, :, i_random], color="#6C165C", linewidth=0.5)
                axs[i_ax, 2].plot(normalized_time_vector, q_all_socp_variable[i_dof - 1, :, i_random], color="#D15C02", linewidth=0.5)
            axs[i_ax, 3].plot(normalized_time_vector, q_all_socp_feedforward[i_dof, :, i_random], color="#400191", linewidth=0.5)
            axs[i_ax, 4].plot(normalized_time_vector, q_all_socp_plus[i_dof, :, i_random], color="#016C93", linewidth=0.5)

        # Nominal
        if i_dof < 4 and i_dof > 1:
            axs[i_ax, 0].plot(normalized_time_vector, q_ocp_nominal[i_dof, :], color="k", linewidth=0.5)
            axs[i_ax, 1].plot(normalized_time_vector, q_socp_nominal[i_dof, :], color="k", linewidth=0.5)
            axs[i_ax, 2].plot(normalized_time_vector, q_socp_variable_nominal[i_dof, :], color="k", linewidth=0.5)
        elif i_dof > 4:
            axs[i_ax, 0].plot(normalized_time_vector, q_ocp_nominal[i_dof - 1, :], color="k", linewidth=0.5)
            axs[i_ax, 1].plot(normalized_time_vector, q_socp_nominal[i_dof - 1, :], color="k", linewidth=0.5)
            axs[i_ax, 2].plot(normalized_time_vector, q_socp_variable_nominal[i_dof - 1, :], color="k", linewidth=0.5)
        axs[i_ax, 3].plot(normalized_time_vector, q_socp_feedforward_nominal[i_dof, :], color="k", linewidth=0.5)
        axs[i_ax, 4].plot(normalized_time_vector, q_socp_plus_nominal[i_dof, :], color="k", linewidth=0.5)

        # Box plot of the distribution of the last frame
        if i_dof < 4 and i_dof > 1:
            box_plot(normalized_time_vector[-1] + 0.075, q_ocp_integrated[i_dof, -1, :], OCP_color, axs[i_ax, 0], box_width=0.025)
            box_plot(normalized_time_vector[-1] + 0.075, q_socp_integrated["20random"][i_dof, -1, :], SOCP_color, axs[i_ax, 1], box_width=0.025)
            box_plot(normalized_time_vector[-1] + 0.075, q_socp_variable_integrated["20random"][i_dof, -1, :], SOCP_VARIABLE_color, axs[i_ax, 2], box_width=0.025)
            box_plot(normalized_time_vector[-1] + 0.075, q_socp_feedforward_integrated["20random"][i_dof, -1, :], SOCP_FEEDFORWARD_color, axs[i_ax, 3], box_width=0.025)
            box_plot(normalized_time_vector[-1] + 0.075, q_socp_plus_integrated["20random"][i_dof, -1, :], SOCP_PLUS_color, axs[i_ax, 4], box_width=0.025)
        elif i_dof > 4:
            box_plot(normalized_time_vector[-1] + 0.075, q_ocp_integrated[i_dof - 1, -1, :], OCP_color, axs[i_ax, 0], box_width=0.025)
            box_plot(normalized_time_vector[-1] + 0.075, q_socp_integrated["20random"][i_dof - 1, -1, :], SOCP_color, axs[i_ax, 1], box_width=0.025)
            box_plot(normalized_time_vector[-1] + 0.075, q_socp_variable_integrated["20random"][i_dof - 1, -1, :], SOCP_VARIABLE_color, axs[i_ax, 2], box_width=0.025)
            box_plot(normalized_time_vector[-1] + 0.075, q_socp_feedforward_integrated["20random"][i_dof, -1, :], SOCP_FEEDFORWARD_color, axs[i_ax, 3], box_width=0.025)
            box_plot(normalized_time_vector[-1] + 0.075, q_socp_plus_integrated["20random"][i_dof, -1, :], SOCP_PLUS_color, axs[i_ax, 4], box_width=0.025)
        elif i_dof == 4:
            box_plot(normalized_time_vector[-1] + 0.075, q_socp_feedforward_integrated["20random"][i_dof, -1, :], SOCP_FEEDFORWARD_color, axs[i_ax, 3], box_width=0.025)
            box_plot(normalized_time_vector[-1] + 0.075, q_socp_plus_integrated["20random"][i_dof, -1, :], SOCP_PLUS_color, axs[i_ax, 4], box_width=0.025)

    axs[0, 0].plot(0, 0, color="k", linewidth=0.5, label="OCP")
    axs[0, 0].plot(0, 0, color="k", linewidth=0.5, label="SOCP nominal")
    axs[0, 0].plot(0, 0, color="k", linewidth=0.5, label="SOCP VARIABLE nominal")
    axs[0, 0].plot(0, 0, color="k", linewidth=0.5, label="SOCP FEEDFORWARD nominal")
    axs[0, 0].plot(0, 0, color="k", linewidth=0.5, label="SOCP+ nominal")
    axs[0, 0].plot(0, 0, color=OCP_color, linewidth=0.5, label="OCP reintegrated", alpha=0.5)
    axs[0, 0].plot(0, 0, color=SOCP_color, linewidth=0.5, label="SOCP reintegrated", alpha=0.5)
    axs[0, 0].plot(0, 0, color=SOCP_VARIABLE_color, linewidth=0.5, label="SOCP VARIABLE reintegrated", alpha=0.5)
    axs[0, 0].plot(0, 0, color=SOCP_FEEDFORWARD_color, linewidth=0.5, label="SOCP FEEDFORWARD reintegrated", alpha=0.5)
    axs[0, 0].plot(0, 0, color=SOCP_PLUS_color, linewidth=0.5, label="SOCP+ reintegrated", alpha=0.5)
    axs[0, 0].plot(0, 0, color="#6C165C", linewidth=0.5, label=f"SOCP {nb_random} models")
    axs[0, 0].plot(0, 0, color="#D15C02", linewidth=0.5, label=f"SOCP VARIABLE {nb_random} models")
    axs[0, 0].plot(0, 0, color="#400191", linewidth=0.5, label=f"SOCP FEEDFORWARD {nb_random} models")
    axs[0, 0].plot(0, 0, color="#016C93", linewidth=0.5, label=f"SOCP+ {nb_random} models")
    fig.subplots_adjust(right=0.8)

    axs[0, 0].set_ylabel("Somersault")
    axs[1, 0].set_ylabel("Neck")
    axs[2, 0].set_ylabel("Eyes")
    axs[3, 0].set_ylabel("Shoulders")
    axs[4, 0].set_ylabel("Hips")
    axs[5, 0].set_ylabel("Knees")

    axs[0, 0].set_title("OCP")
    axs[0, 1].set_title("SOCP")
    axs[0, 2].set_title(r"SOCP$_{\text{VN}}$")
    axs[0, 3].set_title(r"SOCP$^{\text{AF}}$")
    axs[0, 4].set_title(r"SOCP$_{\text{VN}}^{\text{AF}}$")

    for i_axs_2 in range(5):
        for i_axs in range(n_q-3):
            axs[i_axs, i_axs_2].get_xaxis().set_visible(False)
        axs[-1, i_axs_2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
        axs[-1, i_axs_2].set_xlabel("Normalized time")

    axs[2, 0].get_yaxis().set_visible(False)
    axs[2, 1].get_yaxis().set_visible(False)
    axs[2, 2].get_yaxis().set_visible(False)

    plt.subplots_adjust(bottom=0.05, top=0.95, right=0.95, left=0.05)
    # plt.suptitle("Comparison of nominal, integrated and reintegrated solutions")
    plt.savefig(f"graphs/comparison_reintegration.png")
    # plt.show()

    return

def plot_motor_command(
        nb_random,
        normalized_time_vector,
        time_vector_ocp,
        time_vector_socp,
        time_vector_socp_variable,
        time_vector_socp_feedforward,
        time_vector_socp_plus,
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
        SOCP_PLUS_color,
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
    step_plot(normalized_time_vector, tau_joints_ocp[0, :], color=OCP_color, ax=axs[0, 0], label="OCP")
    step_plot(normalized_time_vector, tau_joints_socp[0, :], color=SOCP_color, ax=axs[0, 0], label="SOCP")
    step_plot(normalized_time_vector, tau_joints_socp_variable[0, :], color=SOCP_VARIABLE_color, ax=axs[0, 0], label=r"SOCP$_{\text{VN}}$")
    step_plot(normalized_time_vector, tau_joints_socp_feedforward[0, :], color=SOCP_FEEDFORWARD_color, ax=axs[0, 0], label=r"SOCP$^{\text{AF}}$")
    step_plot(normalized_time_vector, tau_joints_socp_plus[0, :], color=SOCP_PLUS_color, ax=axs[0, 0], label=r"SOCP$_{\text{VN}}^{\text{AF}}$")
    axs[0, 0].set_title("Neck")
    # axs[0, 0].legend(ncol=3)
    # Eyes
    step_plot(normalized_time_vector, tau_joints_socp_feedforward[1, :], color=SOCP_FEEDFORWARD_color, ax=axs[0, 1], label=r"SOCP$^{\text{AF}}$")
    step_plot(normalized_time_vector, tau_joints_socp_plus[1, :], color=SOCP_PLUS_color, ax=axs[0, 1], label=r"SOCP$_{\text{VN}}^{\text{AF}}$")
    axs[0, 1].set_title("Eyes")
    axs[0, 1].plot([0, 1], [0, 0], color="black", linestyle="--")
    # Other joints
    for i_dof in range(2, 5):
        step_plot(normalized_time_vector, tau_joints_ocp[i_dof - 1, :], color=OCP_color, ax=axs[0, i_dof], label="OCP")
        step_plot(normalized_time_vector, tau_joints_socp[i_dof - 1, :], color=SOCP_color, ax=axs[0, i_dof], label="SOCP")
        step_plot(normalized_time_vector, tau_joints_socp_variable[i_dof - 1, :], color=SOCP_VARIABLE_color, ax=axs[0, i_dof], label=r"SOCP$_{\text{VN}}$")
        step_plot(normalized_time_vector, tau_joints_socp_feedforward[i_dof, :], color=SOCP_FEEDFORWARD_color, ax=axs[0, i_dof], label=r"SOCP$^{\text{AF}}$")
        step_plot(normalized_time_vector, tau_joints_socp_plus[i_dof, :], color=SOCP_PLUS_color, ax=axs[0, i_dof], label=r"SOCP$_{\text{VN}}^{\text{AF}}$")

    axs[0, 2].set_title("Shoulder")
    axs[0, 3].set_title("Hips")
    axs[0, 4].set_title("Knees")
    axs[0, 0].set_ylabel("Open-loop\n" + r"($\tau_{voluntary}$) [Nm]")

    # Joint friction
    for i_dof in range(5):
        if i_dof == 0:
            step_plot(normalized_time_vector, -joint_friction_ocp[0, :], color=OCP_color, ax=axs[1, 0])
            step_plot(normalized_time_vector, -np.mean(joint_frictions_socp[0, :, :], axis=1), color=SOCP_color, ax=axs[1, 0])
            step_plot(normalized_time_vector, -np.mean(joint_frictions_socp_variable[0, :, :], axis=1), color=SOCP_VARIABLE_color, ax=axs[1, 0])
            step_plot(normalized_time_vector, -np.mean(joint_frictions_socp_feedforward[0, :, :], axis=1), color=SOCP_FEEDFORWARD_color, ax=axs[1, 0])
            step_plot(normalized_time_vector, -np.mean(joint_frictions_socp_plus[0, :, :], axis=1), color=SOCP_PLUS_color, ax=axs[1, 0])
        elif i_dof == 1:
            step_plot(normalized_time_vector, -np.mean(joint_frictions_socp_feedforward[1, :, :], axis=1), color=SOCP_FEEDFORWARD_color, ax=axs[1, 1])
            step_plot(normalized_time_vector, -np.mean(joint_frictions_socp_plus[1, :, :], axis=1), color=SOCP_PLUS_color, ax=axs[1, 1])
        else:
            step_plot(normalized_time_vector, -joint_friction_ocp[i_dof - 1, :], color=OCP_color, ax=axs[1, i_dof])
            step_plot(normalized_time_vector, -np.mean(joint_frictions_socp[i_dof - 1, :, :], axis=1), color=SOCP_color, ax=axs[1, i_dof])
            step_plot(normalized_time_vector, -np.mean(joint_frictions_socp_variable[i_dof - 1, :, :], axis=1), color=SOCP_VARIABLE_color, ax=axs[1, i_dof])
            step_plot(normalized_time_vector, -np.mean(joint_frictions_socp_feedforward[i_dof, :, :], axis=1), color=SOCP_FEEDFORWARD_color, ax=axs[1, i_dof])
            step_plot(normalized_time_vector, -np.mean(joint_frictions_socp_plus[i_dof, :, :], axis=1), color=SOCP_PLUS_color, ax=axs[1, i_dof])

    axs[1, 0].set_ylabel("Joint friction\n[Nm]")

    # Feedback
    for i_dof in range(5):
        if i_dof == 0:
            step_plot(normalized_time_vector, np.mean(feedbacks_socp[0, :, :], axis=1), color=SOCP_color, ax=axs[2, 0])
            step_plot(normalized_time_vector, np.mean(feedbacks_socp_variable[0, :, :], axis=1), color=SOCP_VARIABLE_color, ax=axs[2, 0])
            step_plot(normalized_time_vector, np.mean(feedbacks_socp_feedforward[0, :, :], axis=1), color=SOCP_FEEDFORWARD_color, ax=axs[2, 0])
            step_plot(normalized_time_vector, np.mean(feedbacks_socp_plus[0, :, :], axis=1), color=SOCP_PLUS_color, ax=axs[2, 0])
        elif i_dof == 1:
            step_plot(normalized_time_vector, np.mean(feedbacks_socp_feedforward[1, :, :], axis=1), color=SOCP_FEEDFORWARD_color, ax=axs[2, 1])
            step_plot(normalized_time_vector, np.mean(feedbacks_socp_plus[1, :, :], axis=1), color=SOCP_PLUS_color, ax=axs[2, 1])
        else:
            step_plot(normalized_time_vector, np.mean(feedbacks_socp[i_dof - 1, :, :], axis=1), color=SOCP_color, ax=axs[2, i_dof])
            step_plot(normalized_time_vector, np.mean(feedbacks_socp_variable[i_dof - 1, :, :], axis=1), color=SOCP_VARIABLE_color, ax=axs[2, i_dof])
            step_plot(normalized_time_vector, np.mean(feedbacks_socp_feedforward[i_dof, :, :], axis=1), color=SOCP_FEEDFORWARD_color, ax=axs[2, i_dof])
            step_plot(normalized_time_vector, np.mean(feedbacks_socp_plus[i_dof, :, :], axis=1), color=SOCP_PLUS_color, ax=axs[2, i_dof])
    axs[2, 0].set_ylabel("Direct\nfeedback\n" + r"($\tau_{dfb}$) [Nm]")

    # Feedforward
    for i_dof in range(5):
        step_plot(normalized_time_vector, np.mean(feedforwards_socp_feedforward[i_dof, :, :], axis=1), color=SOCP_FEEDFORWARD_color,
                  ax=axs[3, i_dof])
        step_plot(normalized_time_vector, np.mean(feedforwards_socp_plus[i_dof, :, :], axis=1), color=SOCP_PLUS_color,
                  ax=axs[3, i_dof])
    axs[3, 0].set_ylabel("Anticipatory\nfeedback\n" + r"($\tau_{afb}$) [Nm]")

    # Sum
    for i_dof in range(5):
        if i_dof == 0:
            step_plot(
                normalized_time_vector,
                tau_joints_ocp[0, :]
                - joint_friction_ocp[0, :],
                # + motor_noises_ocp[0, :, :],            ,
                color=OCP_color,
                ax=axs[4, i_dof],
            )
            step_plot(
                normalized_time_vector,
                np.mean(tau_joints_socp[0, :, np.newaxis]
                - joint_frictions_socp[0, :, :]
                # + motor_noises_socp[0, :, :]
                + feedbacks_socp[0, :, :], axis=1),
                color=SOCP_color,
                ax=axs[4, i_dof],
            )
            step_plot(
                normalized_time_vector,
                np.mean(tau_joints_socp_variable[0, :, np.newaxis]
                - joint_frictions_socp_variable[0, :, :]
                # + motor_noises_socp_variable[0, :, :]
                + feedbacks_socp_variable[0, :, :], axis=1),
                color=SOCP_VARIABLE_color,
                ax=axs[4, i_dof],
            )
            step_plot(
                normalized_time_vector,
                np.mean(tau_joints_socp_feedforward[0, :, np.newaxis]
                - joint_frictions_socp_feedforward[0, :, :]
                # + motor_noises_socp_feedforward[0, :, :]
                + feedbacks_socp_feedforward[0, :, :]
                + feedforwards_socp_feedforward[0, :, :], axis=1),
                color=SOCP_FEEDFORWARD_color,
                ax=axs[4, i_dof],
            )
            step_plot(
                normalized_time_vector,
                np.mean(tau_joints_socp_plus[0, :, np.newaxis]
                - joint_frictions_socp_plus[0, :, :]
                # + motor_noises_socp_plus[0, :, :]
                + feedbacks_socp_plus[0, :, :]
                + feedforwards_socp_plus[0, :, :], axis=1),
                color=SOCP_PLUS_color,
                ax=axs[4, i_dof],
            )
        elif i_dof == 1:
            step_plot(
                normalized_time_vector,
                np.mean(tau_joints_socp_feedforward[1, :, np.newaxis]
                - joint_frictions_socp_feedforward[1, :, :]
                # + motor_noises_socp_feedforward[1, :, :]
                + feedbacks_socp_feedforward[1, :, :]
                + feedforwards_socp_feedforward[1, :, :], axis=1),
                color=SOCP_FEEDFORWARD_color,
                ax=axs[4, i_dof],
            )
            step_plot(
                normalized_time_vector,
                np.mean(tau_joints_socp_plus[1, :, np.newaxis]
                - joint_frictions_socp_plus[1, :, :]
                # + motor_noises_socp_plus[1, :, :]
                + feedbacks_socp_plus[1, :, :]
                + feedforwards_socp_plus[1, :, :], axis=1),
                color=SOCP_PLUS_color,
                ax=axs[4, i_dof],
            )
        else:
            step_plot(
                normalized_time_vector,
                tau_joints_ocp[i_dof - 1, :] - joint_friction_ocp[i_dof - 1, :],
                color=OCP_color,
                ax=axs[4, i_dof],
            )
            for i_random in range(nb_random):
                step_plot(
                    normalized_time_vector,
                    np.mean(tau_joints_socp[i_dof - 1, :, np.newaxis]
                    - joint_frictions_socp[i_dof - 1, :, :]
                    # + motor_noises_socp[i_dof - 1, :, :]
                    + feedbacks_socp[i_dof - 1, :, :], axis=1),
                    color=SOCP_color,
                    ax=axs[4, i_dof],
                )
                step_plot(
                    normalized_time_vector,
                    np.mean(tau_joints_socp_variable[i_dof - 1, :, np.newaxis]
                    - joint_frictions_socp_variable[i_dof - 1, :, :]
                    # + motor_noises_socp_variable[i_dof - 1, :, :]
                    + feedbacks_socp_variable[i_dof - 1, :, :], axis=1),
                    color=SOCP_VARIABLE_color,
                    ax=axs[4, i_dof],
                )
                step_plot(
                    normalized_time_vector,
                    np.mean(tau_joints_socp_feedforward[i_dof, :, np.newaxis]
                    - joint_frictions_socp_feedforward[i_dof, :, :]
                    # + motor_noises_socp_feedforward[i_dof, :, :]
                    + feedbacks_socp_feedforward[i_dof, :, :]
                    + feedforwards_socp_feedforward[i_dof, :, :], axis=1),
                    color=SOCP_FEEDFORWARD_color,
                    ax=axs[4, i_dof],
                )
                step_plot(
                    normalized_time_vector,
                    np.mean(tau_joints_socp_plus[i_dof, :, np.newaxis]
                    - joint_frictions_socp_plus[i_dof, :, :]
                    # + motor_noises_socp_plus[i_dof, :, :]
                    + feedbacks_socp_plus[i_dof, :, :]
                    + feedforwards_socp_plus[i_dof, :, :], axis=1),
                    color=SOCP_PLUS_color,
                    ax=axs[4, i_dof],
                )
    axs[4, 0].set_ylabel("Total\n" + r"($\tau_{total}$) [Nm]")

    for i_ax in range(5):
        for i_ax2 in range(4):
            axs[i_ax2, i_ax].get_xaxis().set_visible(False)
        axs[4, i_ax].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[4, i_ax].set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        axs[4, i_ax].set_xlabel("Normalized time")

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    plt.savefig("graphs/controls.png", dpi=300)
    # plt.show()


    nb_tests = feedbacks_socp_plus.shape[2]
    total_tau_joint_ocp = np.sum(np.trapezoid(np.abs(tau_joints_ocp), time_vector_ocp[:-1], axis=1))
    total_joint_friction_ocp = np.sum(np.trapezoid(np.abs(joint_friction_ocp), time_vector_ocp[:-1], axis=1))
    total_ocp = total_tau_joint_ocp + total_joint_friction_ocp
    print(f"Total torque OCP: {np.sum(np.trapezoid(np.abs(tau_joints_ocp - joint_friction_ocp), time_vector_ocp[:-1], axis=1))} : "
          f"(tau: {total_tau_joint_ocp/total_ocp * 100}%, friction: {total_joint_friction_ocp/total_ocp * 100}%)")
    total_tau_joint_socp = np.sum(np.trapezoid(np.abs(tau_joints_socp), time_vector_socp[:-1], axis=1))
    total_joint_friction_socp = np.sum(np.trapezoid(np.abs(joint_frictions_socp), time_vector_socp[:-1], axis=1)) / nb_tests
    total_feedbacks_socp = np.sum(np.trapezoid(np.abs(feedbacks_socp), time_vector_socp[:-1], axis=1)) / nb_tests
    total_socp = total_tau_joint_socp + total_joint_friction_socp + total_feedbacks_socp
    print(f"Total torque SOCP: {np.sum(np.trapezoid(np.abs(tau_joints_socp[:, :, np.newaxis]
                                        - joint_frictions_socp[:, :, :]
                                        + feedbacks_socp[:, :, :]), time_vector_socp[:-1], axis=1)) / nb_tests} : "
          f"(tau: {total_tau_joint_socp/total_socp * 100}%, friction: {total_joint_friction_socp/total_socp * 100}%, "
          f"feedback: {total_feedbacks_socp/total_socp * 100}")
    total_tau_joint_socp_variable = np.sum(np.trapezoid(np.abs(tau_joints_socp_variable), time_vector_socp_variable[:-1], axis=1))
    total_joint_friction_socp_variable = np.sum(np.trapezoid(np.abs(joint_frictions_socp_variable), time_vector_socp_variable[:-1], axis=1)) / nb_tests
    total_feedbacks_socp_variable = np.sum(np.trapezoid(np.abs(feedbacks_socp_variable), time_vector_socp_variable[:-1], axis=1)) / nb_tests
    total_socp_variable = total_tau_joint_socp_variable + total_joint_friction_socp_variable + total_feedbacks_socp_variable
    print(f"Total torque SOCP VARIABLE: {np.sum(np.trapezoid(np.abs(tau_joints_socp_variable[:, :, np.newaxis]
                     - joint_frictions_socp_variable[:, :, :]
                     + feedbacks_socp_variable[:, :, :]), time_vector_socp_variable[:-1], axis=1)) / nb_tests} : "
          f"(tau: {total_tau_joint_socp_variable/total_socp_variable * 100}%, friction: {total_joint_friction_socp_variable/total_socp_variable * 100}%, "
            f"feedback: {total_feedbacks_socp_variable/total_socp_variable * 100}")
    total_tau_joint_socp_feedforward = np.sum(np.trapezoid(np.abs(tau_joints_socp_feedforward), time_vector_socp_feedforward[:-1], axis=1))
    total_joint_friction_socp_feedforward = np.sum(np.trapezoid(np.abs(joint_frictions_socp_feedforward), time_vector_socp_feedforward[:-1], axis=1)) / nb_tests
    total_feedbacks_socp_feedforward = np.sum(np.trapezoid(np.abs(feedbacks_socp_feedforward), time_vector_socp_feedforward[:-1], axis=1)) / nb_tests
    total_feedforwards_socp_feedforward = np.sum(np.trapezoid(np.abs(feedforwards_socp_feedforward), time_vector_socp_feedforward[:-1], axis=1)) / nb_tests
    total_socp_feedforward = (total_tau_joint_socp_feedforward + total_joint_friction_socp_feedforward
                              + total_feedbacks_socp_feedforward + total_feedforwards_socp_feedforward)
    print(f"Total torque SOCP FEEDFORWARD: {np.sum(np.trapezoid(np.abs(tau_joints_socp_feedforward[:, :, np.newaxis]
                    - joint_frictions_socp_feedforward[:, :, :]
                    + feedbacks_socp_feedforward[:, :, :]
                    + feedforwards_socp_feedforward[:, :, :]), time_vector_socp_feedforward[:-1], axis=1)) / nb_tests} : "
          f"(tau: {total_tau_joint_socp_feedforward/total_socp_feedforward * 100}%, "
          f"friction: {total_joint_friction_socp_feedforward/total_socp_feedforward * 100}%, "
          f"feedback: {total_feedbacks_socp_feedforward/total_socp_feedforward * 100}%, "
          f"feedforward: {total_feedforwards_socp_feedforward/total_socp_feedforward * 100}%")
    total_tau_joint_socp_plus = np.sum(np.trapezoid(np.abs(tau_joints_socp_plus), time_vector_socp_plus[:-1], axis=1))
    total_joint_friction_socp_plus = np.sum(np.trapezoid(np.abs(joint_frictions_socp_plus), time_vector_socp_plus[:-1], axis=1)) / nb_tests
    total_feedbacks_socp_plus = np.sum(np.trapezoid(np.abs(feedbacks_socp_plus), time_vector_socp_plus[:-1], axis=1)) / nb_tests
    total_feedforwards_socp_plus = np.sum(np.trapezoid(np.abs(feedforwards_socp_plus), time_vector_socp_plus[:-1], axis=1)) / nb_tests
    total_socp_plus = (total_tau_joint_socp_plus + total_joint_friction_socp_plus
                       + total_feedbacks_socp_plus + total_feedforwards_socp_plus)
    print(f"Total torque SOCP+: {np.sum(np.trapezoid(np.abs(tau_joints_socp_plus[:, :, np.newaxis]
                    - joint_frictions_socp_plus[:, :, :]
                    + feedbacks_socp_plus[:, :, :]
                    + feedforwards_socp_plus[:, :, :]), time_vector_socp_plus[:-1], axis=1)) / nb_tests} : "
          f"(tau: {total_tau_joint_socp_plus/total_socp_plus * 100}%, "
          f"friction: {total_joint_friction_socp_plus/total_socp_plus * 100}%, "
          f"feedback: {total_feedbacks_socp_plus/total_socp_plus * 100}%, "
          f"feedforward: {total_feedforwards_socp_plus/total_socp_plus * 100}%")


    plt.figure(figsize=(15, 3))

    total = 0
    plt.bar(0, total_tau_joint_ocp, bottom=total, width=0.4, color="tab:red") #, label="Open-loop")
    plt.text(-0.2, total + total_tau_joint_ocp / 2, f"{(total_tau_joint_ocp/total_ocp) * 100:.2f}%", ha="right", va="center")
    total += total_tau_joint_ocp
    plt.bar(0, total_joint_friction_ocp, bottom=total, width=0.4, color="tab:blue") #, label="Joint friction")
    plt.text(-0.2, total + total_joint_friction_ocp / 2, f"{(total_joint_friction_ocp/total_ocp) * 100:.2f}%", ha="right", va="center")

    total = 0
    plt.bar(1, total_tau_joint_socp, bottom=total, width=0.4, color="tab:red") #, label="Open-loop")
    plt.text(0.8, total + total_tau_joint_socp / 2, f"{(total_tau_joint_socp/total_socp) * 100:.2f}%", ha="right", va="center")
    total += total_tau_joint_socp
    plt.bar(1, total_joint_friction_socp, bottom=total, width=0.4, color="tab:blue") #, label="Joint friction")
    plt.text(0.8, total + total_joint_friction_socp / 2, f"{(total_joint_friction_socp/total_socp) * 100:.2f}%", ha="right", va="center")
    total += total_joint_friction_socp
    plt.bar(1, total_feedbacks_socp, bottom=total, width=0.4, color="tab:green") #, label="Direct feedback")
    plt.text(0.8, total + total_feedbacks_socp / 2, f"{(total_feedbacks_socp/total_socp) * 100:.2f}%", ha="right", va="center")

    total = 0
    plt.bar(2, total_tau_joint_socp_variable, bottom=total, width=0.4, color="tab:red") #, label="Open-loop")
    plt.text(1.8, total + total_tau_joint_socp_variable / 2, f"{(total_tau_joint_socp_variable/total_socp_variable) * 100:.2f}%", ha="right", va="center")
    total += total_tau_joint_socp_variable
    plt.bar(2, total + total_joint_friction_socp_variable, bottom=total, width=0.4, color="tab:blue") #, label="Joint friction")
    plt.text(1.8, total + total_joint_friction_socp_variable / 2, f"{(total_joint_friction_socp_variable/total_socp_variable) * 100:.2f}%", ha="right", va="center")
    total += total_joint_friction_socp_variable
    plt.bar(2, total_feedbacks_socp_variable, bottom=total, width=0.4, color="tab:green") #, label="Direct feedback")
    plt.text(1.8, total + total_feedbacks_socp_variable / 2, f"{(total_feedbacks_socp_variable/total_socp_variable) * 100:.2f}%", ha="right", va="center")

    total = 0
    plt.bar(3, total_tau_joint_socp_feedforward, bottom=total, width=0.4, color="tab:red") #, label="Open-loop")
    plt.text(2.8, total + total_tau_joint_socp_feedforward / 2, f"{(total_tau_joint_socp_feedforward/total_socp_feedforward) * 100:.2f}%", ha="right", va="center")
    total += total_tau_joint_socp_feedforward
    plt.bar(3, total_joint_friction_socp_feedforward, bottom=total, width=0.4, color="tab:blue") #, label="Joint friction")
    plt.text(2.8, total + total_joint_friction_socp_feedforward / 2, f"{(total_joint_friction_socp_feedforward/total_socp_feedforward) * 100:.2f}%", ha="right", va="center")
    total += total_joint_friction_socp_feedforward
    plt.bar(3, total_feedbacks_socp_feedforward, bottom=total, width=0.4, color="tab:green") #, label="Direct feedback")
    plt.text(2.8, total + total_feedbacks_socp_feedforward / 2, f"{(total_feedbacks_socp_feedforward/total_socp_feedforward) * 100:.2f}%", ha="right", va="center")
    total += total_feedbacks_socp_feedforward
    plt.bar(3, total_feedforwards_socp_feedforward, bottom=total, width=0.4, color="tab:pink") #, label="Anticipatory feedback")
    plt.text(2.8, total + total_feedforwards_socp_feedforward / 2, f"{(total_feedforwards_socp_feedforward/total_socp_feedforward) * 100:.2f}%", ha="right", va="center")

    total = 0
    plt.bar(4, total_tau_joint_socp_plus, bottom=total, width=0.4, color="tab:red", label="Open-loop")
    plt.text(3.8, total + total_tau_joint_socp_plus / 2, f"{(total_tau_joint_socp_plus/total_socp_plus) * 100:.2f}%", ha="right", va="center")
    total += total_tau_joint_socp_plus
    plt.bar(4, total_joint_friction_socp_plus, bottom=total, width=0.4, color="tab:blue", label="Joint friction")
    plt.text(3.8, total + total_joint_friction_socp_plus / 2, f"{(total_joint_friction_socp_plus/total_socp_plus) * 100:.2f}%", ha="right", va="center")
    total += total_joint_friction_socp_plus
    plt.bar(4, total_feedbacks_socp_plus, bottom=total, width=0.4, color="tab:green", label="Direct feedback")
    plt.text(3.8, total + total_feedbacks_socp_plus / 2, f"{(total_feedbacks_socp_plus/total_socp_plus) * 100:.2f}%", ha="right", va="center")
    total += total_feedbacks_socp_plus
    plt.bar(4, total_feedforwards_socp_plus, bottom=total, width=0.4, color="tab:pink", label="Anticipatory feedback")
    plt.text(3.8, total + total_feedforwards_socp_plus / 2, f"{(total_feedforwards_socp_plus/total_socp_plus) * 100:.2f}%", ha="right", va="center")

    plt.ylabel("Sum of absolute integrated torques [Nm.s]")
    plt.xlim(-0.7, 4.3)
    plt.legend(bbox_to_anchor=(1.025, 0.5), loc='center left', frameon=False)
    plt.xticks([0, 1, 2, 3, 4], ["OCP", "SOCP", r"SOCP$_{\text{VN}}$", r"SOCP$^{\text{AF}}$", r"SOCP$_{\text{VN}}^{\text{AF}}$"])
    plt.subplots_adjust(right=0.75, left=0.1)
    plt.savefig("graphs/controls_contributions.png", dpi=300)
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
                            SOCP_PLUS_color,
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
            step_plot(
                normalized_time_vector,
                tau_joints_ocp[0, :] - joint_friction_ocp[0, :],
                color=OCP_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                tau_joints_socp[0, :]
                - np.mean(joint_frictions_socp[0, :, :], axis=1)
                + np.mean(motor_noises_socp[0, :, :], axis=1)
                + np.mean(feedbacks_socp[0, :, :], axis=1),
                color=SOCP_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                tau_joints_socp_variable[0, :]
                - np.mean(joint_frictions_socp_variable[0, :, :], axis=1)
                + np.mean(motor_noises_socp_variable[0, :, :], axis=1)
                + np.mean(feedbacks_socp_variable[0, :, :], axis=1),
                color=SOCP_VARIABLE_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                tau_joints_socp_feedforward[0, :]
                - np.mean(joint_frictions_socp_feedforward[0, :, :], axis=1)
                + np.mean(motor_noises_socp_feedforward[0, :, :], axis=1)
                + np.mean(feedbacks_socp_feedforward[0, :, :], axis=1)
                + np.mean(feedforwards_socp_feedforward[0, :, :], axis=1),
                color=SOCP_FEEDFORWARD_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                tau_joints_socp_plus[0, :]
                - np.mean(joint_frictions_socp_plus[0, :, :], axis=1)
                + np.mean(motor_noises_socp_plus[0, :, :], axis=1)
                + np.mean(feedbacks_socp_plus[0, :, :], axis=1)
                + np.mean(feedforwards_socp_plus[0, :, :], axis=1),
                color=SOCP_PLUS_color,
                ax=axs[0, i_dof],
            )
        elif i_dof == 1:
            step_plot(
                normalized_time_vector,
                tau_joints_socp_feedforward[1, :]
                - np.mean(joint_frictions_socp_feedforward[1, :, :], axis=1)
                + np.mean(motor_noises_socp_feedforward[1, :, :], axis=1)
                + np.mean(feedbacks_socp_feedforward[1, :, :], axis=1)
                + np.mean(feedforwards_socp_feedforward[1, :, :], axis=1),
                color=SOCP_FEEDFORWARD_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                tau_joints_socp_plus[1, :]
                - np.mean(joint_frictions_socp_plus[1, :, :], axis=1)
                + np.mean(motor_noises_socp_plus[1, :, :], axis=1)
                + np.mean(feedbacks_socp_plus[1, :, :], axis=1)
                + np.mean(feedforwards_socp_plus[1, :, :], axis=1),
                color=SOCP_PLUS_color,
                ax=axs[0, i_dof],
            )
        else:
            step_plot(
                normalized_time_vector,
                tau_joints_ocp[i_dof - 1, :] - joint_friction_ocp[i_dof - 1, :],
                color=OCP_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                tau_joints_socp[i_dof - 1, :]
                - np.mean(joint_frictions_socp[i_dof - 1, :, :], axis=1)
                + np.mean(motor_noises_socp[i_dof - 1, :, :], axis=1)
                + np.mean(feedbacks_socp[i_dof - 1, :, :], axis=1),
                color=SOCP_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                tau_joints_socp_variable[i_dof - 1, :]
                - np.mean(joint_frictions_socp_variable[i_dof - 1, :, :], axis=1)
                + np.mean(motor_noises_socp_variable[i_dof - 1, :, :], axis=1)
                + np.mean(feedbacks_socp_variable[i_dof - 1, :, :], axis=1),
                color=SOCP_VARIABLE_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                tau_joints_socp_feedforward[i_dof, :]
                - np.mean(joint_frictions_socp_feedforward[i_dof, :, :], axis=1)
                + np.mean(motor_noises_socp_feedforward[i_dof, :, :], axis=1)
                + np.mean(feedbacks_socp_feedforward[i_dof, :, :], axis=1)
                + np.mean(feedforwards_socp_feedforward[i_dof, :, :], axis=1),
                color=SOCP_FEEDFORWARD_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                tau_joints_socp_plus[i_dof, :]
                - np.mean(joint_frictions_socp_plus[i_dof, :, :], axis=1)
                + np.mean(motor_noises_socp_plus[i_dof, :, :], axis=1)
                + np.mean(feedbacks_socp_plus[i_dof, :, :], axis=1)
                + np.mean(feedforwards_socp_plus[i_dof, :, :], axis=1),
                color=SOCP_PLUS_color,
                ax=axs[0, i_dof],
            )
    axs[0, 0].set_ylabel(r"Total $\tau$ [Nm]", fontsize=12)

    # Delta tau
    delta_time_vector = (normalized_time_vector[1:] + normalized_time_vector[:-1]) / 2
    for i_dof in range(5):
        if i_dof == 0:
            step_plot(
                delta_time_vector,
                tau_joints_ocp[0, 1:] - joint_friction_ocp[0, 1:] - (
                        tau_joints_ocp[0, :-1] + joint_friction_ocp[0, :-1]),
                color=OCP_color,
                ax=axs[1, i_dof],
            )
            step_plot(
                delta_time_vector,
                tau_joints_socp[0, 1:]
                - np.mean(joint_frictions_socp[0, 1:, :], axis=1)
                + np.mean(motor_noises_socp[0, 1:, :], axis=1)
                + np.mean(feedbacks_socp[0, 1:, :], axis=1)
                - (
                        tau_joints_socp[0, :-1]
                        - np.mean(joint_frictions_socp[0, :-1, :], axis=1)
                        + np.mean(motor_noises_socp[0, :-1, :], axis=1)
                        + np.mean(feedbacks_socp[0, :-1, :], axis=1)
                ),
                color=SOCP_color,
                ax=axs[1, i_dof],
            )
            step_plot(
                delta_time_vector,
                tau_joints_socp_variable[0, 1:]
                - np.mean(joint_frictions_socp_variable[0, 1:, :], axis=1)
                + np.mean(motor_noises_socp_variable[0, 1:, :], axis=1)
                + np.mean(feedbacks_socp_variable[0, 1:, :], axis=1)
                - (
                        tau_joints_socp_variable[0, :-1]
                        - np.mean(joint_frictions_socp_variable[0, :-1, :], axis=1)
                        + np.mean(motor_noises_socp_variable[0, :-1, :], axis=1)
                        + np.mean(feedbacks_socp_variable[0, :-1, :], axis=1)
                ),
                color=SOCP_VARIABLE_color,
                ax=axs[1, i_dof],
            )
            step_plot(
                delta_time_vector,
                tau_joints_socp_feedforward[0, 1:]
                - np.mean(joint_frictions_socp_feedforward[0, 1:, :], axis=1)
                + np.mean(motor_noises_socp_feedforward[0, 1:, :], axis=1)
                + np.mean(feedbacks_socp_feedforward[0, 1:, :], axis=1)
                + np.mean(feedforwards_socp_feedforward[0, 1:, :], axis=1)
                - (
                        tau_joints_socp_feedforward[0, :-1]
                        - np.mean(joint_frictions_socp_feedforward[0, :-1, :], axis=1)
                        + np.mean(motor_noises_socp_feedforward[0, :-1, :], axis=1)
                        + np.mean(feedbacks_socp_feedforward[0, :-1, :], axis=1)
                        + np.mean(feedforwards_socp_feedforward[0, :-1, :], axis=1)
                ),
                color=SOCP_FEEDFORWARD_color,
                ax=axs[1, i_dof],
            )
            step_plot(
                delta_time_vector,
                tau_joints_socp_plus[0, 1:]
                - np.mean(joint_frictions_socp_plus[0, 1:, :], axis=1)
                + np.mean(motor_noises_socp_plus[0, 1:, :], axis=1)
                + np.mean(feedbacks_socp_plus[0, 1:, :], axis=1)
                + np.mean(feedforwards_socp_plus[0, 1:, :], axis=1)
                - (
                        tau_joints_socp_plus[0, :-1]
                        - np.mean(joint_frictions_socp_plus[0, :-1, :], axis=1)
                        + np.mean(motor_noises_socp_plus[0, :-1, :], axis=1)
                        + np.mean(feedbacks_socp_plus[0, :-1, :], axis=1)
                        + np.mean(feedforwards_socp_plus[0, :-1, :], axis=1)
                ),
                color=SOCP_PLUS_color,
                ax=axs[1, i_dof],
            )
        elif i_dof == 1:
            step_plot(
                delta_time_vector,
                tau_joints_socp_feedforward[1, 1:]
                - np.mean(joint_frictions_socp_feedforward[1, 1:, :], axis=1)
                + np.mean(motor_noises_socp_feedforward[1, 1:, :], axis=1)
                + np.mean(feedbacks_socp_feedforward[1, 1:, :], axis=1)
                + np.mean(feedforwards_socp_feedforward[1, 1:, :], axis=1)
                - (
                        tau_joints_socp_feedforward[1, :-1]
                        - np.mean(joint_frictions_socp_feedforward[1, :-1, :], axis=1)
                        + np.mean(motor_noises_socp_feedforward[1, :-1, :], axis=1)
                        + np.mean(feedbacks_socp_feedforward[1, :-1, :], axis=1)
                        + np.mean(feedforwards_socp_feedforward[1, :-1, :], axis=1)
                ),
                color=SOCP_FEEDFORWARD_color,
                ax=axs[1, 1],
            )
            step_plot(
                delta_time_vector,
                tau_joints_socp_plus[1, 1:]
                - np.mean(joint_frictions_socp_plus[1, 1:, :], axis=1)
                + np.mean(motor_noises_socp_plus[1, 1:, :], axis=1)
                + np.mean(feedbacks_socp_plus[1, 1:, :], axis=1)
                + np.mean(feedforwards_socp_plus[1, 1:, :], axis=1)
                - (
                        tau_joints_socp_plus[1, :-1]
                        - np.mean(joint_frictions_socp_plus[1, :-1, :], axis=1)
                        + np.mean(motor_noises_socp_plus[1, :-1, :], axis=1)
                        + np.mean(feedbacks_socp_plus[1, :-1, :], axis=1)
                        + np.mean(feedforwards_socp_plus[1, :-1, :], axis=1)
                ),
                color=SOCP_PLUS_color,
                ax=axs[1, 1],
            )
        else:
            step_plot(
                delta_time_vector,
                tau_joints_ocp[i_dof - 1, 1:]
                - joint_friction_ocp[i_dof - 1, 1:]
                - (tau_joints_ocp[i_dof - 1, :-1] - joint_friction_ocp[i_dof - 1, :-1]),
                color=OCP_color,
                ax=axs[1, i_dof],
            )
            step_plot(
                delta_time_vector,
                tau_joints_socp[i_dof - 1, 1:]
                - np.mean(joint_frictions_socp[i_dof - 1, 1:, :], axis=1)
                + np.mean(motor_noises_socp[i_dof - 1, 1:, :], axis=1)
                + np.mean(feedbacks_socp[i_dof - 1, 1:, :], axis=1)
                - (
                        tau_joints_socp[i_dof - 1, :-1]
                        - np.mean(joint_frictions_socp[i_dof - 1, :-1, :], axis=1)
                        + np.mean(motor_noises_socp[i_dof - 1, :-1, :], axis=1)
                        + np.mean(feedbacks_socp[i_dof - 1, :-1, :], axis=1)
                ),
                color=SOCP_color,
                ax=axs[1, i_dof],
            )
            step_plot(
                delta_time_vector,
                tau_joints_socp_variable[i_dof - 1, 1:]
                - np.mean(joint_frictions_socp_variable[i_dof - 1, 1:, :], axis=1)
                + np.mean(motor_noises_socp_variable[i_dof - 1, 1:, :], axis=1)
                + np.mean(feedbacks_socp_variable[i_dof - 1, 1:, :], axis=1)
                - (
                        tau_joints_socp_variable[i_dof - 1, :-1]
                        - np.mean(joint_frictions_socp_variable[i_dof - 1, :-1, :], axis=1)
                        + np.mean(motor_noises_socp_variable[i_dof - 1, :-1, :], axis=1)
                        + np.mean(feedbacks_socp_variable[i_dof - 1, :-1, :], axis=1)
                ),
                color=SOCP_VARIABLE_color,
                ax=axs[1, i_dof],
            )
            step_plot(
                delta_time_vector,
                tau_joints_socp_feedforward[i_dof, 1:]
                - np.mean(joint_frictions_socp_feedforward[i_dof, 1:, :], axis=1)
                + np.mean(motor_noises_socp_feedforward[i_dof, 1:, :], axis=1)
                + np.mean(feedbacks_socp_feedforward[i_dof, 1:, :], axis=1)
                + np.mean(feedforwards_socp_feedforward[i_dof, 1:, :], axis=1)
                - (
                        tau_joints_socp_feedforward[i_dof, :-1]
                        - np.mean(joint_frictions_socp_feedforward[i_dof, :-1, :], axis=1)
                        + np.mean(motor_noises_socp_feedforward[i_dof, :-1, :], axis=1)
                        + np.mean(feedbacks_socp_feedforward[i_dof, :-1, :], axis=1)
                        + np.mean(feedforwards_socp_feedforward[i_dof, :-1, :], axis=1)
                ),
                color=SOCP_FEEDFORWARD_color,
                ax=axs[1, i_dof],
            )
            step_plot(
                delta_time_vector,
                tau_joints_socp_plus[i_dof, 1:]
                - np.mean(joint_frictions_socp_plus[i_dof, 1:, :], axis=1)
                + np.mean(motor_noises_socp_plus[i_dof, 1:, :], axis=1)
                + np.mean(feedbacks_socp_plus[i_dof, 1:, :], axis=1)
                + np.mean(feedforwards_socp_plus[i_dof, 1:, :], axis=1)
                - (
                        tau_joints_socp_plus[i_dof, :-1]
                        - np.mean(joint_frictions_socp_plus[i_dof, :-1, :], axis=1)
                        + np.mean(motor_noises_socp_plus[i_dof, :-1, :], axis=1)
                        + np.mean(feedbacks_socp_plus[i_dof, :-1, :], axis=1)
                        + np.mean(feedforwards_socp_plus[i_dof, :-1, :], axis=1)
                ),
                color=SOCP_PLUS_color,
                ax=axs[1, i_dof],
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
    step_plot(
        normalized_time_vector,
        np.sum(np.abs(tau_joints_ocp - joint_friction_ocp), axis=0),
        color=OCP_color,
        ax=axs[0],
    )
    step_plot(
        normalized_time_vector,
        np.sum(np.abs(tau_joints_socp \
                      - np.mean(joint_frictions_socp[:, :, :], axis=2) \
                      + np.mean(motor_noises_socp[:, :, :], axis=2) \
                      + np.mean(feedbacks_socp[:, :, :], axis=2)), axis=0),
        color=SOCP_color,
        ax=axs[0],
    )
    step_plot(
        normalized_time_vector,
        np.sum(np.abs(tau_joints_socp_variable \
        - np.mean(joint_frictions_socp_variable[:, :, :], axis=2) \
        + np.mean(motor_noises_socp_variable[:, :, :], axis=2) \
        + np.mean(feedbacks_socp_variable[:, :, :], axis=2)), axis=0),
        color=SOCP_VARIABLE_color,
        ax=axs[0],
    )
    step_plot(
        normalized_time_vector,
        np.sum(np.abs(tau_joints_socp_feedforward \
        - np.mean(joint_frictions_socp_feedforward[:, :, :], axis=2) \
        + np.mean(motor_noises_socp_feedforward[:, :, :], axis=2) \
        + np.mean(feedbacks_socp_feedforward[:, :, :], axis=2) \
        + np.mean(feedforwards_socp_feedforward[:, :, :], axis=2)), axis=0),
        color=SOCP_FEEDFORWARD_color,
        ax=axs[0],
    )
    step_plot(
        normalized_time_vector,
        np.sum(np.abs(tau_joints_socp_plus \
        - np.mean(joint_frictions_socp_plus[:, :, :], axis=2) \
        + np.mean(motor_noises_socp_plus[:, :, :], axis=2) \
        + np.mean(feedbacks_socp_plus[:, :, :], axis=2) \
        + np.mean(feedforwards_socp_plus[:, :, :], axis=2)), axis=0),
        color=SOCP_PLUS_color,
        ax=axs[0],
    )
    axs[0].set_ylabel(r"Total $\sum{\tau}$ [Nm]", fontsize=12)

    # Delta tau
    delta_time_vector = (normalized_time_vector[1:] + normalized_time_vector[:-1]) / 2
    step_plot(
        delta_time_vector,
        np.sum(
            np.abs(
                tau_joints_ocp[:, 1:] - joint_friction_ocp[:, 1:] - (
                            tau_joints_ocp[:, :-1] + joint_friction_ocp[:, :-1])
            ),
            axis=0,
        ),
        color=OCP_color,
        ax=axs[1],
    )
    step_plot(
        delta_time_vector,
        np.sum(
            np.abs(
                tau_joints_socp[:, 1:]
                - np.mean(joint_frictions_socp[:, 1:, :], axis=2)
                + np.mean(motor_noises_socp[:, 1:, :], axis=2)
                + np.mean(feedbacks_socp[:, 1:, :], axis=2)
                - (
                        tau_joints_socp[:, :-1]
                        - np.mean(joint_frictions_socp[:, :-1, :], axis=2)
                        + np.mean(motor_noises_socp[:, :-1, :], axis=2)
                        + np.mean(feedbacks_socp[:, :-1, :], axis=2)
                )
            ),
            axis=0,
        ),
        color=SOCP_color,
        ax=axs[1],
    )
    step_plot(
        delta_time_vector,
        np.sum(
            np.abs(
                tau_joints_socp_variable[:, 1:]
                - np.mean(joint_frictions_socp_variable[:, 1:, :], axis=2)
                + np.mean(motor_noises_socp_variable[:, 1:, :], axis=2)
                + np.mean(feedbacks_socp_variable[:, 1:, :], axis=2)
                - (
                        tau_joints_socp_variable[:, :-1]
                        - np.mean(joint_frictions_socp_variable[:, :-1, :], axis=2)
                        + np.mean(motor_noises_socp_variable[:, :-1, :], axis=2)
                        + np.mean(feedbacks_socp_variable[:, :-1, :], axis=2)
                )
            ),
            axis=0,
        ),
        color=SOCP_VARIABLE_color,
        ax=axs[1],
    )
    step_plot(
        delta_time_vector,
        np.sum(
            np.abs(
                tau_joints_socp_feedforward[:, 1:]
                - np.mean(joint_frictions_socp_feedforward[:, 1:, :], axis=2)
                + np.mean(motor_noises_socp_feedforward[:, 1:, :], axis=2)
                + np.mean(feedbacks_socp_feedforward[:, 1:, :], axis=2)
                + np.mean(feedforwards_socp_feedforward[:, 1:, :], axis=2)
                - (
                        tau_joints_socp_feedforward[:, :-1]
                        - np.mean(joint_frictions_socp_feedforward[:, :-1, :], axis=2)
                        + np.mean(motor_noises_socp_feedforward[:, :-1, :], axis=2)
                        + np.mean(feedbacks_socp_feedforward[:, :-1, :], axis=2)
                        + np.mean(feedforwards_socp_feedforward[:, :-1, :], axis=2)
                )
            ),
            axis=0,
        ),
        color=SOCP_FEEDFORWARD_color,
        ax=axs[1],
    )
    step_plot(
        delta_time_vector,
        np.sum(
            np.abs(
                tau_joints_socp_plus[:, 1:]
                - np.mean(joint_frictions_socp_plus[:, 1:, :], axis=2)
                + np.mean(motor_noises_socp_plus[:, 1:, :], axis=2)
                + np.mean(feedbacks_socp_plus[:, 1:, :], axis=2)
                + np.mean(feedforwards_socp_plus[:, 1:, :], axis=2)
                - (
                        tau_joints_socp_plus[:, :-1]
                        - np.mean(joint_frictions_socp_plus[:, :-1, :], axis=2)
                        + np.mean(motor_noises_socp_plus[:, :-1, :], axis=2)
                        + np.mean(feedbacks_socp_plus[:, :-1, :], axis=2)
                        + np.mean(feedforwards_socp_plus[:, :-1, :], axis=2)
                )
            ),
            axis=0,
        ),
        color=SOCP_PLUS_color,
        ax=axs[1],
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
        SOCP_PLUS_color
        ):

    # Plot the gains
    n_k_fb = socp_plus.nlp[0].model.n_noised_controls + socp_plus.nlp[0].model.n_references
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    for i in range(2):
        for j in range(2):
            axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
    for i_dof in range(40):
        step_plot(normalized_time_vector, k_socp[i_dof, :], color=SOCP_color, ax=axs[0, 0], label="SOCP")
        step_plot(normalized_time_vector, k_socp_variable[i_dof, :], color=SOCP_VARIABLE_color, ax=axs[0, 1], label=r"SOCP$_{\text{VN}}$")
    for i_dof in range(n_k_fb):
        step_plot(normalized_time_vector, k_socp_feedforward[i_dof, :], color=SOCP_FEEDFORWARD_color, ax=axs[0, 2], label=r"SOCP$^{\text{AF}}$")
        step_plot(normalized_time_vector, k_socp_plus[i_dof, :], color=SOCP_PLUS_color, ax=axs[0, 3], label=r"SOCP$_{\text{VN}}^{\text{AF}}$")
    for i_dof in range(5):
        step_plot(normalized_time_vector, k_socp_feedforward[n_k_fb + i_dof, :], color=SOCP_FEEDFORWARD_color, ax=axs[1, 2], label=r"SOCP$^{\text{AF}}$")
        step_plot(normalized_time_vector, k_socp_plus[n_k_fb + i_dof, :], color=SOCP_PLUS_color, ax=axs[1, 3], label=r"SOCP$_{\text{VN}}^{\text{AF}}$")
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

    step_plot(normalized_time_vector, np.sum(np.abs(k_socp[:, :]), axis=0), color=SOCP_color, ax=axs[0, 0], label=r"SOCP")
    step_plot(normalized_time_vector, np.sum(np.abs(k_socp_variable[:, :]), axis=0), color=SOCP_VARIABLE_color, ax=axs[0, 1], label=r"SOCP$_{\text{VN}}$")
    step_plot(normalized_time_vector, np.sum(np.abs(k_socp_feedforward[:n_k_fb, :]), axis=0), color=SOCP_FEEDFORWARD_color, ax=axs[0, 2], label=r"SOCP$^{\text{AF}}$")
    step_plot(normalized_time_vector, np.sum(np.abs(k_socp_plus[:n_k_fb, :]), axis=0), color=SOCP_PLUS_color, ax=axs[0, 3], label=r"SOCP$_{\text{VN}}^{\text{AF}}$")
    step_plot(normalized_time_vector, np.sum(np.abs(k_socp_feedforward[n_k_fb:, :]), axis=0), color=SOCP_FEEDFORWARD_color, ax=axs[1, 2], label=r"SOCP$^{\text{AF}}$")
    step_plot(normalized_time_vector, np.sum(np.abs(k_socp_plus[n_k_fb:, :]), axis=0), color=SOCP_PLUS_color, ax=axs[1, 3], label=r"SOCP$_{\text{VN}}^{\text{AF}}$")

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
        step_plot(delta_time_vector, k_socp[i_dof, 1:] - k_socp[i_dof, :-1], color=SOCP_color, ax=axs[0, 0], label=r"SOCP")
        step_plot(delta_time_vector, k_socp_variable[i_dof, 1:] - k_socp_variable[i_dof, :-1], color=SOCP_VARIABLE_color, ax=axs[0, 0], label=r"SOCP$_{\text{VN}}$")
    for i_dof in range(n_k_fb):
        step_plot(delta_time_vector, k_socp_feedforward[i_dof, 1:] - k_socp_feedforward[i_dof, :-1], color=SOCP_FEEDFORWARD_color, ax=axs[0, 1], label=r"SOCP$^{\text{AF}}$")
        step_plot(delta_time_vector, k_socp_plus[i_dof, 1:] - k_socp_plus[i_dof, :-1], color=SOCP_PLUS_color, ax=axs[0, 1], label=r"SOCP$_{\text{VN}}^{\text{AF}}$")
    for i_dof in range(5):
        step_plot(
            delta_time_vector,
            k_socp_feedforward[n_k_fb + i_dof, 1:] - k_socp_feedforward[n_k_fb + i_dof, :-1],
            color=SOCP_FEEDFORWARD_color,
            label=r"SOCP$^{\text{AF}}$",
            ax=axs[1, 1],
        )
        step_plot(
            delta_time_vector,
            k_socp_plus[n_k_fb + i_dof, 1:] - k_socp_plus[n_k_fb + i_dof, :-1],
            color=SOCP_PLUS_color,
            label=r"SOCP$_{\text{VN}}^{\text{AF}}$",
            ax=axs[1, 1],
        )

    # axs[0, 0].set_ylim(-40, 35)
    # axs[0, 1].set_ylim(-40, 35)
    # axs[1, 1].set_ylim(-40, 35)

    for i_ax in range(2):
        axs[0, i_ax].get_xaxis().set_visible(False)
        axs[1, i_ax].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[1, i_ax].set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        axs[1, i_ax].set_xlabel("Normalized time")

    plt.savefig("graphs/delta_gains.png")


    # Plot the delta gains
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    for i in range(2):
        for j in range(2):
            axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)

    step_plot(
        delta_time_vector,
        np.sum(np.abs(k_socp[:, 1:] - k_socp[:, :-1]), axis=0),
        color=SOCP_color,
        label=r"SOCP",
        ax=axs[0, 0],
    )
    step_plot(
        delta_time_vector,
        np.sum(np.abs(k_socp_variable[:, 1:] - k_socp_variable[:, :-1]), axis=0),
        color=SOCP_VARIABLE_color,
        label=r"SOCP$_{\text{VN}}$",
        ax=axs[0, 1],
    )
    step_plot(
        delta_time_vector,
        np.sum(np.abs(k_socp_feedforward[:n_k_fb, 1:] - k_socp_feedforward[:n_k_fb, :-1]), axis=0),
        color=SOCP_FEEDFORWARD_color,
        label=r"SOCP$^{\text{AF}}$",
        ax=axs[0, 2],
    )
    step_plot(
        delta_time_vector,
        np.sum(np.abs(k_socp_plus[:n_k_fb, 1:] - k_socp_plus[:n_k_fb, :-1]), axis=0),
        color=SOCP_PLUS_color,
        label=r"SOCP$_{\text{VN}}^{\text{AF}}$",
        ax=axs[0, 3],
    )
    step_plot(
        delta_time_vector,
        np.sum(np.abs(k_socp_feedforward[n_k_fb:, 1:] - k_socp_feedforward[n_k_fb:, :-1]), axis=0),
        color=SOCP_FEEDFORWARD_color,
        label=r"SOCP$^{\text{AF}}$",
        ax=axs[1, 2],
    )
    step_plot(
        delta_time_vector,
        np.sum(np.abs(k_socp_plus[n_k_fb:, 1:] - k_socp_plus[n_k_fb:, :-1]), axis=0),
        color=SOCP_PLUS_color,
        label=r"SOCP$_{\text{VN}}^{\text{AF}}$",
        ax=axs[1, 3],
    )
    # axs[0, 0].set_ylim(0, 800)
    # axs[0, 1].set_ylim(0, 800)
    # axs[1, 1].set_ylim(0, 800)
    axs[0, 0].set_ylabel(r"$\sum{} \Delta$ Direct feedback gains", fontsize=12)
    axs[1, 0].set_ylabel(r"$\sum{} \Delta$ Anticipatory feedback gains", fontsize=12)

    for i_ax in range(4):
        axs[0, i_ax].get_xaxis().set_visible(False)
        axs[1, i_ax].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[1, i_ax].set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        axs[1, i_ax].set_xlabel("Normalized time")

    axs[0, 0].set_title("SOCP")
    axs[0, 1].set_title(r"SOCP$_{\text{VN}}$")
    axs[0, 2].set_title(r"SOCP$^{\text{AF}}$")
    axs[0, 3].set_title(r"SOCP$_{\text{VN}}^{\text{AF}}$")

    plt.savefig("graphs/sum_delta_gains.png")
    # plt.show()

    # Plot the FF gains vs acuity
    fig, axs = plt.subplots(4, 1, figsize=(15, 10))
    for i in range(4):
        axs[i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)

    step_plot(
        normalized_time_vector,
        np.sum(np.abs(k_socp_feedforward[:n_k_fb, :]), axis=0),
        color=SOCP_FEEDFORWARD_color,
        label=r"SOCP$^{\text{AF}}$",
        ax=axs[0],
    )
    step_plot(
        normalized_time_vector,
        np.sum(np.abs(k_socp_plus[:n_k_fb, :]), axis=0),
        color=SOCP_PLUS_color,
        label=r"SOCP$_{\text{VN}}^{\text{AF}}$",
        ax=axs[0],
    )
    axs[0].set_ylabel(r"$\sum{}$ Direct feedback gains")
    step_plot(
        normalized_time_vector,
        np.sum(np.abs(k_socp_feedforward[n_k_fb:, :]), axis=0),
        color=SOCP_FEEDFORWARD_color,
        label=r"SOCP$^{\text{AF}}$",
        ax=axs[1],
    )
    step_plot(
        normalized_time_vector,
        np.sum(np.abs(k_socp_plus[n_k_fb:, :]), axis=0),
        color=SOCP_PLUS_color,
        label=r"SOCP$_{\text{VN}}^{\text{AF}}$",
        ax=axs[1],
    )
    axs[1].set_ylabel(r"$\sum{}$ Anticipatory feedback gains")

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
        [vestibular_noise(socp_variable.nlp[0].model, Q, Qdot, vestibular_noise_sym, [])],
    )
    vestibular_noise_fcn_plus = cas.Function(
        "vestibular_noise_plus",
        [Q_8, Qdot_8, vestibular_noise_sym],
        [vestibular_noise(socp_plus.nlp[0].model, Q_8, Qdot_8, vestibular_noise_sym, [])],
    )

    visual_acuity = np.zeros((n_shooting + 1, 1))
    vestibular_acquity_variable = np.zeros((n_shooting + 1, 1))
    vestibular_acquity_plus = np.zeros((n_shooting + 1, 1))
    for i_shooting in range(n_shooting + 1):
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
    axs[2].plot(normalized_time_vector, vestibular_acquity_normalized_variable * -1 + 1, color=SOCP_VARIABLE_color)
    axs[2].plot(normalized_time_vector, np.zeros_like(normalized_time_vector), color=SOCP_FEEDFORWARD_color)
    axs[2].plot(normalized_time_vector, vestibular_acquity_normalized_plus * -1 + 1, color=SOCP_PLUS_color)
    axs[2].set_ylabel("Vestibular acuity")
    axs[3].plot(normalized_time_vector, np.zeros_like(normalized_time_vector), color=SOCP_FEEDFORWARD_color)
    axs[3].plot(normalized_time_vector, visual_acuity_normalized * -1 + 1, color=SOCP_PLUS_color)
    axs[3].set_ylabel("Visual acuity")

    axs[0].get_xaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[2].get_xaxis().set_visible(False)
    axs[3].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[3].set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    axs[3].set_xlabel("Normalized time")

    plt.savefig("graphs/ff_gains.png")
    # plt.show()

    return

def plot_gains_per_dof(
        normalized_time_vector,
        normalized_time_vector_MS,
        socp,
        socp_variable,
        socp_feedforward,
        socp_plus,
        k_socp,
        k_socp_variable,
        k_socp_feedforward,
        k_socp_plus,
        q_ocp,
        q_mean_socp,
        q_mean_socp_variable,
        q_mean_socp_feedforward,
        q_mean_socp_plus,
        qdot_ocp,
        qdot_mean_socp,
        qdot_mean_socp_variable,
        qdot_mean_socp_feedforward,
        qdot_mean_socp_plus,
        OCP_color,
        SOCP_color,
        SOCP_VARIABLE_color,
        SOCP_FEEDFORWARD_color,
        SOCP_PLUS_color,
        head_velocity_fcn,
        head_velocity_fcn_8,
        eye_orientation_fcn_8,
):

    n_shooting = socp.nlp[0].ns
    k_socp_matrix = np.zeros((socp.nlp[0].model.matrix_shape_k[0], socp.nlp[0].model.matrix_shape_k[1], n_shooting))
    k_socp_variable_matrix = np.zeros((socp_variable.nlp[0].model.matrix_shape_k[0], socp_variable.nlp[0].model.matrix_shape_k[1], n_shooting))
    k_socp_feedforward_matrix = np.zeros((socp_feedforward.nlp[0].model.matrix_shape_k[0], socp_feedforward.nlp[0].model.matrix_shape_k[1], n_shooting))
    k_socp_plus_matrix = np.zeros((socp_plus.nlp[0].model.matrix_shape_k[0], socp_plus.nlp[0].model.matrix_shape_k[1], n_shooting))
    for i_shooting in range(n_shooting):
        k_socp_matrix[:, :, i_shooting] = StochasticBioModel.reshape_to_matrix(k_socp[:, i_shooting], socp.nlp[0].model.matrix_shape_k)
        k_socp_variable_matrix[:, :, i_shooting] = StochasticBioModel.reshape_to_matrix(k_socp_variable[:, i_shooting], socp_variable.nlp[0].model.matrix_shape_k)
        k_socp_feedforward_matrix[:, :, i_shooting] = StochasticBioModel.reshape_to_matrix(k_socp_feedforward[:, i_shooting], socp_feedforward.nlp[0].model.matrix_shape_k)
        k_socp_plus_matrix[:, :, i_shooting] = StochasticBioModel.reshape_to_matrix(k_socp_plus[:, i_shooting], socp_plus.nlp[0].model.matrix_shape_k)

    k_fb_socp_feedforward = k_socp_feedforward_matrix[:, : socp_feedforward.nlp[0].model.n_feedbacks, :]
    k_ff_socp_feedforward = k_socp_feedforward_matrix[:, socp_feedforward.nlp[0].model.n_feedbacks :, :]
    k_fb_socp_plus = k_socp_plus_matrix[:, : socp_plus.nlp[0].model.n_feedbacks, :]
    k_ff_socp_plus = k_socp_plus_matrix[:, socp_plus.nlp[0].model.n_feedbacks :, :]

    fig, axs = plt.subplots(2, 5, figsize=(15, 4))
    for i in range(5):
        for j in range(2):
            axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)

    for i_dof, dof in enumerate(range(5)):
        if i_dof < 1:
            step_plot(
                normalized_time_vector,
                np.sum(np.abs(k_socp_matrix[dof, :, :]), axis=0),
                color=SOCP_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                np.sum(np.abs(k_socp_variable_matrix[dof, :, :]), axis=0),
                color=SOCP_VARIABLE_color,
                ax=axs[0, i_dof],
            )
        elif i_dof > 1:
            step_plot(
                normalized_time_vector,
                np.sum(np.abs(k_socp_matrix[dof - 1, :, :]), axis=0),
                color=SOCP_color,
                ax=axs[0, i_dof],
            )
            step_plot(
                normalized_time_vector,
                np.sum(np.abs(k_socp_variable_matrix[dof - 1, :, :]), axis=0),
                color=SOCP_VARIABLE_color,
                ax=axs[0, i_dof],
            )

        step_plot(
            normalized_time_vector,
            np.sum(np.abs(k_fb_socp_feedforward[dof, :, :]), axis=0),
            color=SOCP_FEEDFORWARD_color,
            ax=axs[0, i_dof],
        )
        step_plot(
            normalized_time_vector,
            np.sum(np.abs(k_ff_socp_feedforward[dof, :, :]), axis=0),
            color=SOCP_FEEDFORWARD_color,
            ax=axs[1, i_dof],
        )
        step_plot(
            normalized_time_vector,
            np.sum(np.abs(k_fb_socp_plus[dof, :, :]), axis=0),
            color=SOCP_PLUS_color,
            ax=axs[0, i_dof],
        )
        step_plot(
            normalized_time_vector,
            np.sum(np.abs(k_ff_socp_plus[dof, :, :]), axis=0),
            color=SOCP_PLUS_color,
            ax=axs[1, i_dof],
        )

    axs[0, 0].set_ylabel(r"$\sum$ Direct" + "\nfeedback gains")
    axs[1, 0].set_ylabel(r"$\sum$ Anticipatory" + "\nfeedback gains")

    axs[0, 0].set_title("Neck")
    axs[0, 1].set_title("Eyes")
    axs[0, 2].set_title("Shoulder")
    axs[0, 3].set_title("Hips")
    axs[0, 4].set_title("Knees")

    for i_ax in range(5):
        axs[0, i_ax].get_xaxis().set_visible(False)
        axs[1, i_ax].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[1, i_ax].set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        axs[1, i_ax].set_xlabel("Normalized time")

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    plt.savefig("graphs/gains_per_dof.png", dpi=300)
    # plt.show()

    n_tests = q_ocp.shape[1]
    head_velocity_ocp = np.zeros((n_tests, ))
    head_velocity_socp = np.zeros((n_tests, ))
    head_velocity_socp_variable = np.zeros((n_tests, ))
    head_velocity_socp_feedforward = np.zeros((n_tests, ))
    head_velocity_socp_plus = np.zeros((n_tests, ))
    eye_orientation_socp_feedforward = np.zeros((n_tests, ))
    eye_orientation_socp_plus = np.zeros((n_tests, ))
    for i_shooting in range(n_tests):
        head_velocity_ocp[i_shooting] = head_velocity_fcn(q_ocp[:, i_shooting], qdot_ocp[:, i_shooting])
        head_velocity_socp[i_shooting] = head_velocity_fcn(q_mean_socp[:, i_shooting], qdot_mean_socp[:, i_shooting])
        head_velocity_socp_variable[i_shooting] = head_velocity_fcn(q_mean_socp_variable[:, i_shooting], qdot_mean_socp_variable[:, i_shooting])
        head_velocity_socp_feedforward[i_shooting] = head_velocity_fcn_8(q_mean_socp_feedforward[:, i_shooting], qdot_mean_socp_feedforward[:, i_shooting])
        head_velocity_socp_plus[i_shooting] = head_velocity_fcn_8(q_mean_socp_plus[:, i_shooting], qdot_mean_socp_plus[:, i_shooting])

        eye_orientation_socp_feedforward[i_shooting] = eye_orientation_fcn_8(q_mean_socp_feedforward[:, i_shooting], qdot_mean_socp_feedforward[:, i_shooting])
        eye_orientation_socp_plus[i_shooting] = eye_orientation_fcn_8(q_mean_socp_plus[:, i_shooting], qdot_mean_socp_plus[:, i_shooting])

    fig, axs = plt.subplots(2, 1, figsize=(15, 4))
    axs[0].plot(normalized_time_vector_MS, np.zeros_like(normalized_time_vector_MS), '--k', alpha=0.5)
    axs[0].plot(normalized_time_vector_MS, head_velocity_ocp * 180 / np.pi, color=OCP_color, label="OCP")
    axs[0].plot(normalized_time_vector_MS, head_velocity_socp * 180 / np.pi, color=SOCP_color, label="SOCP")
    axs[0].plot(normalized_time_vector_MS, head_velocity_socp_variable * 180 / np.pi, color=SOCP_VARIABLE_color, label=r"SOCP$_{\text{VN}}$")
    axs[0].plot(normalized_time_vector_MS, head_velocity_socp_feedforward * 180 / np.pi, color=SOCP_FEEDFORWARD_color, label=r"SOCP$^{\text{AF}}$")
    axs[0].plot(normalized_time_vector_MS, head_velocity_socp_plus * 180 / np.pi, color=SOCP_PLUS_color, label=r"SOCP$_{\text{VN}}^{\text{AF}}$")
    axs[0].set_ylabel(r"Head velocity [$^\circ/s$]")
    axs[1].plot(normalized_time_vector_MS, np.zeros_like(normalized_time_vector_MS), '--k', alpha=0.5)
    axs[1].fill_between(
        normalized_time_vector_MS,
        np.ones_like(normalized_time_vector_MS) * np.pi/4 * 180 / np.pi,
        np.ones_like(normalized_time_vector_MS) * 180,
        color="gray",
        alpha=0.2,
        linewidth=0,
        )
    axs[1].plot(normalized_time_vector_MS, 180-eye_orientation_socp_feedforward * 180 / np.pi, color=SOCP_FEEDFORWARD_color, label=r"SOCP$^{\text{AF}}$")
    axs[1].plot(normalized_time_vector_MS, 180-eye_orientation_socp_plus * 180 / np.pi, color=SOCP_PLUS_color, label=r"SOCP$_{\text{VN}}^{\text{AF}}$")
    axs[1].set_ylabel(r"Gaze orientation [$^\circ$]")
    axs[1].set_ylim(-10, 180)

    axs[0].get_xaxis().set_visible(False)
    axs[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1].set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    axs[1].set_xlabel("Normalized time")

    plt.subplots_adjust(hspace=0.1)
    plt.savefig("graphs/head_velocity_eye_orientation.png", dpi=300)
    # plt.show()

    print("Mean absolute head velocity OCP:", np.mean(np.abs(head_velocity_ocp)))
    print("Mean absolute head velocity SOCP:", np.mean(np.abs(head_velocity_socp)))
    print("Mean absolute head velocity SOCPV:", np.mean(np.abs(head_velocity_socp_variable)))
    print("Mean absolute head velocity SOCPA:", np.mean(np.abs(head_velocity_socp_feedforward)))
    print("Mean absolute head velocity SOCP+:", np.mean(np.abs(head_velocity_socp_plus)))

    print("Mean absolute eye orientation SOCPA:", np.mean(np.abs(180 - eye_orientation_socp_feedforward * 180 / np.pi)))
    print("Mean absolute eye orientation SOCP+:", np.mean(np.abs(180 - eye_orientation_socp_plus * 180 / np.pi)))

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
    time_vector_ocp,
    time_vector_socp,
    time_vector_socp_variable,
    time_vector_socp_feedforward,
    time_vector_socp_plus,
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_PLUS_color,
):

    socp_mean = np.mean(q_socp, axis=2)
    socp_variable_mean = np.mean(q_socp_variable, axis=2)
    socp_feedforward_mean = np.mean(q_socp_feedforward, axis=2)
    socp_plus_mean = np.mean(q_socp_plus, axis=2)
    ocp_reintegration_mean = np.mean(q_ocp_integrated, axis=2)
    socp_reintegration_mean = np.mean(q_socp_integrated["20random"], axis=2)
    socp_variable_reintegration_mean = np.mean(q_socp_variable_integrated["20random"], axis=2)
    socp_feedforward_reintegration_mean = np.mean(q_socp_feedforward_integrated["20random"], axis=2)
    socp_plus_reintegration_mean = np.mean(q_socp_plus_integrated["20random"], axis=2)

    fig, axs = plt.subplots(7, 5, figsize=(15, 10))
    for i_dof in range(7):
        axs[i_dof, 0].plot(time_vector_ocp, q_ocp[i_dof, :], color=OCP_color, linewidth=2)
        axs[i_dof, 1].plot(time_vector_socp, q_mean_socp[i_dof, :], color=SOCP_color, linewidth=2)
        axs[i_dof, 2].plot(time_vector_socp_variable, q_mean_socp_variable[i_dof, :], color=SOCP_VARIABLE_color, linewidth=2)
        axs[i_dof, 3].plot(time_vector_socp_feedforward, q_mean_socp_feedforward[i_dof, :], color=SOCP_FEEDFORWARD_color, linewidth=2)
        axs[i_dof, 4].plot(time_vector_socp_plus, q_mean_socp_plus[i_dof, :], color=SOCP_PLUS_color, linewidth=2)

        axs[i_dof, 0].plot(time_vector_ocp, ocp_reintegration_mean[i_dof, :], color=OCP_color, alpha=0.5)
        axs[i_dof, 1].plot(time_vector_socp, socp_reintegration_mean[i_dof, :], color=SOCP_color, alpha=0.5)
        axs[i_dof, 2].plot(time_vector_socp_variable, socp_variable_reintegration_mean[i_dof, :], color=SOCP_VARIABLE_color, alpha=0.5)
        axs[i_dof, 3].plot(time_vector_socp_feedforward, socp_feedforward_reintegration_mean[i_dof, :], color=SOCP_FEEDFORWARD_color, alpha=0.5)
        axs[i_dof, 4].plot(time_vector_socp_plus, socp_plus_reintegration_mean[i_dof, :], color=SOCP_PLUS_color, alpha=0.5)

        axs[i_dof, 1].plot(time_vector_socp, socp_mean[i_dof, :], color="#6C165C", linewidth=0.5)
        axs[i_dof, 2].plot(time_vector_socp_variable, socp_variable_mean[i_dof, :], color="#D15C02", linewidth=0.5)
        axs[i_dof, 3].plot(time_vector_socp_feedforward, socp_feedforward_mean[i_dof, :], color="#400191", linewidth=0.5)
        axs[i_dof, 4].plot(time_vector_socp_plus, socp_plus_mean[i_dof, :], color="#016C93", linewidth=0.5)
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
                        SOCP_PLUS_color):

    def plot_distribution(q_integrated, current_position, color, current_file_name, ax):
        nb_randoms = [5, 10, 15, 20]
        base_file_name = "Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02"
        for i_random, current_random in enumerate(nb_randoms):
            position_random = np.random.random((current_random, 1)) * 0.1
            file_name = f"results/{current_random}random-seed0/{base_file_name}_{current_file_name}_{current_random}random_CVG_1p0e-06.pkl"
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
        position_all_random = np.random.random((q_this_nb_random.shape[2], 1)) * 0.1
        ax.plot(current_position + position_all_random, q_integrated[2, -1, :], '.', color=color, alpha=0.2)
        current_position += 1
        box_plot(current_position, q_integrated[2, -1, :], color, ax, box_width=0.3)
        return

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    position_all_random = np.random.random((q_socp_integrated.shape[2], 1)) * 0.1
    ax.plot(position_all_random - 3, q_ocp_integrated[2, -1, :], 'o', color=OCP_color)
    box_plot(-2, q_ocp_integrated[2, -1, :], OCP_color, ax)
    plot_distribution(q_socp_integrated, 0, SOCP_color, "DMS", ax)
    plot_distribution(q_socp_variable_integrated, 10, SOCP_VARIABLE_color, "VARIABLE_DMS", ax)
    plot_distribution(q_socp_feedforward_integrated, 20, SOCP_FEEDFORWARD_color, "FEEDFORWARD_DMS", ax)
    plot_distribution(q_socp_plus_integrated, 30, SOCP_PLUS_color, "VARIABLE_FEEDFORWARD_DMS", ax)
    ax.set_ylabel("Final somersault angle [rad]")
    ax.set_xticks([-2, 5, 15, 25, 35])
    ax.set_xticklabels(["OCP", "SOCP", r"SOCP$_{\text{VN}}$", r"SOCP$^{\text{AF}}$", r"SOCP$_{\text{VN}}^{\text{AF}}$"])
    plt.savefig("graphs/comparison_nb_random.png")
    # plt.show()
    return


def plot_comparison_kinematics_nb_random(
    q_ocp,
    time_vector_ocp,
):
    n_q = 8
    nb_random_list = [5, 10, 15, 20]
    colors_random = ["tab:red", "tab:green", "tab:blue", "tab:orange"]

    fig, axs = plt.subplots(n_q-2, 5, figsize=(15, 10))

    # # Reintegrated
    # for i_dof in range(n_q):
    #     for i_random in range(15 * nb_reintegrations):
    #         if i_dof < 4:
    #             axs[i_dof, 0].plot(
    #                 time_vector, q_ocp_integrated[i_dof, :, i_random], color=OCP_color, alpha=0.2, linewidth=0.5
    #             )
    #             axs[i_dof, 1].plot(
    #                 time_vector, q_socp_integrated[i_dof, :, i_random], color=SOCP_color, alpha=0.2, linewidth=0.5
    #             )
    #             axs[i_dof, 2].plot(
    #                 time_vector, q_socp_variable_integrated[i_dof, :, i_random], color=SOCP_VARIABLE_color, alpha=0.2,
    #                 linewidth=0.5
    #             )
    #         elif i_dof > 4:
    #             axs[i_dof, 0].plot(
    #                 time_vector, q_ocp_integrated[i_dof - 1, :, i_random], color=OCP_color, alpha=0.2, linewidth=0.5
    #             )
    #             axs[i_dof, 1].plot(
    #                 time_vector, q_socp_integrated[i_dof - 1, :, i_random], color=SOCP_color, alpha=0.2, linewidth=0.5
    #             )
    #             axs[i_dof, 2].plot(
    #                 time_vector, q_socp_variable_integrated[i_dof - 1, :, i_random], color=SOCP_VARIABLE_color,
    #                 alpha=0.2, linewidth=0.5
    #             )
    #         axs[i_dof, 3].plot(
    #             time_vector,
    #             q_socp_feedforward_integrated[i_dof, :, i_random],
    #             color=SOCP_FEEDFORWARD_color,
    #             alpha=0.2,
    #             linewidth=0.5,
    #         )
    #         axs[i_dof, 4].plot(
    #             time_vector,
    #             q_socp_plus_integrated[i_dof, :, i_random],
    #             color=SOCP_PLUS_color,
    #             alpha=0.2,
    #             linewidth=0.5,
    #         )

    # # Nominal
    # for i_dof in range(n_q):
    #     if i_dof < 4:
    #         axs[i_dof, 0].plot(time_vector, q_ocp_nominal[i_dof, :], color="k", linewidth=0.5)
    #         axs[i_dof, 1].plot(time_vector, q_socp_nominal[i_dof, :], color="k", linewidth=0.5)
    #         axs[i_dof, 2].plot(time_vector, q_socp_variable_nominal[i_dof, :], color="k", linewidth=0.5)
    #     elif i_dof > 4:
    #         axs[i_dof, 0].plot(time_vector, q_ocp_nominal[i_dof - 1, :], color="k", linewidth=0.5)
    #         axs[i_dof, 1].plot(time_vector, q_socp_nominal[i_dof - 1, :], color="k", linewidth=0.5)
    #         axs[i_dof, 2].plot(time_vector, q_socp_variable_nominal[i_dof - 1, :], color="k", linewidth=0.5)
    #     axs[i_dof, 3].plot(time_vector, q_socp_feedforward_nominal[i_dof, :], color="k", linewidth=0.5)
    #     axs[i_dof, 4].plot(time_vector, q_socp_plus_nominal[i_dof, :], color="k", linewidth=0.5)

    # Optimization variables
    for i_dof, dof in enumerate(range(2, n_q)):
        if dof < 4:
            axs[i_dof, 0].plot(time_vector_ocp, q_ocp[dof, :], color="k")
        elif dof > 4:
            axs[i_dof, 0].plot(time_vector_ocp, q_ocp[dof - 1, :], color="k")

    q_socp, qdot_socp, q_socp_variable, qdot_socp_variable, q_socp_feedforward, qdot_socp_feedforward, q_socp_plus, qdot_socp_plus, time_vector_socp, time_vector_socp_variable, time_vector_socp_feedforward, time_vector_socp_plus = get_optimization_q_each_random(
        n_shooting)
    for i_rand_for_color, current_random in enumerate(nb_random_list):
        for i_dof, dof in enumerate(range(2, n_q)):
            for i_random in range(current_random):
                if dof < 4:
                    if q_socp[f"{current_random}random"] is not None:
                        axs[i_dof, 1].plot(time_vector_socp[f"{current_random}random"], q_socp[f"{current_random}random"][dof, :, i_random], color=colors_random[i_rand_for_color], linewidth=0.5, alpha=0.3)
                    if q_socp_variable[f"{current_random}random"] is not None:
                        axs[i_dof, 2].plot(time_vector_socp_variable[f"{current_random}random"], q_socp_variable[f"{current_random}random"][dof, :, i_random], color=colors_random[i_rand_for_color], linewidth=0.5, alpha=0.3)
                elif dof > 4:
                    if q_socp[f"{current_random}random"] is not None:
                        axs[i_dof, 1].plot(time_vector_socp[f"{current_random}random"], q_socp[f"{current_random}random"][dof-1, :, i_random], color=colors_random[i_rand_for_color], linewidth=0.5, alpha=0.3)
                    if q_socp_variable[f"{current_random}random"] is not None:
                        axs[i_dof, 2].plot(time_vector_socp_variable[f"{current_random}random"], q_socp_variable[f"{current_random}random"][dof-1, :, i_random],
                                           color=colors_random[i_rand_for_color], linewidth=0.5, alpha=0.3)
                if q_socp_feedforward[f"{current_random}random"] is not None:
                    axs[i_dof, 3].plot(time_vector_socp_feedforward[f"{current_random}random"], q_socp_feedforward[f"{current_random}random"][dof, :, i_random], color=colors_random[i_rand_for_color], linewidth=0.5, alpha=0.3)
                if q_socp_plus[f"{current_random}random"] is not None:
                    axs[i_dof, 4].plot(time_vector_socp_plus[f"{current_random}random"], q_socp_plus[f"{current_random}random"][dof, :, i_random], color=colors_random[i_rand_for_color], linewidth=0.5, alpha=0.3)

                # # Box plot of the distribution of the last frame
                # if i_dof < 4:
                #     box_plot(time_vector[-1] + 0.2, q_ocp_integrated[i_dof, -1, :], OCP_color, axs[i_dof, 0])
                #     box_plot(time_vector[-1] + 0.2, q_socp_integrated[i_dof, -1, :], SOCP_color, axs[i_dof, 1])
                #     box_plot(time_vector[-1] + 0.2, q_socp_variable_integrated[i_dof, -1, :], SOCP_VARIABLE_color, axs[i_dof, 2])
                #     box_plot(time_vector[-1] + 0.2, q_socp_feedforward_integrated[i_dof, -1, :], SOCP_FEEDFORWARD_color, axs[i_dof, 3])
                #     box_plot(time_vector[-1] + 0.2, q_socp_plus_integrated[i_dof, -1, :], SOCP_PLUS_color, axs[i_dof, 4])
                # elif i_dof > 4:
                #     box_plot(time_vector[-1] + 0.2, q_ocp_integrated[i_dof - 1, -1, :], OCP_color, axs[i_dof, 0])
                #     box_plot(time_vector[-1] + 0.2, q_socp_integrated[i_dof - 1, -1, :], SOCP_color, axs[i_dof, 1])
                #     box_plot(time_vector[-1] + 0.2, q_socp_variable_integrated[i_dof - 1, -1, :], SOCP_VARIABLE_color, axs[i_dof, 2])
                #     box_plot(time_vector[-1] + 0.2, q_socp_feedforward_integrated[i_dof - 1, -1, :], SOCP_FEEDFORWARD_color, axs[i_dof, 3])
                #     box_plot(time_vector[-1] + 0.2, q_socp_plus_integrated[i_dof, -1, :], SOCP_PLUS_color, axs[i_dof, 4])
                # box_plot(time_vector[-1] + 0.2, q_socp_feedforward_integrated[i_dof, -1, :], SOCP_FEEDFORWARD_color, axs[i_dof, 3])
                # box_plot(time_vector[-1] + 0.2, q_socp_plus_integrated[i_dof, -1, :], SOCP_PLUS_color, axs[i_dof, 4])

            # Means
            if dof < 4:
                if q_socp[f"{current_random}random"] is not None:
                    axs[i_dof, 1].plot(time_vector_socp[f"{current_random}random"], np.mean(q_socp[f"{current_random}random"][dof, :, :], axis=1),
                                       color=colors_random[i_rand_for_color])
                if q_socp_variable[f"{current_random}random"] is not None:
                    axs[i_dof, 2].plot(time_vector_socp_variable[f"{current_random}random"], np.mean(q_socp_variable[f"{current_random}random"][dof, :, :], axis=1),
                                       color=colors_random[i_rand_for_color])
            if dof > 4:
                if q_socp[f"{current_random}random"] is not None:
                    axs[i_dof, 1].plot(time_vector_socp[f"{current_random}random"], np.mean(q_socp[f"{current_random}random"][dof-1, :, :], axis=1),
                                       color=colors_random[i_rand_for_color])
                if q_socp_variable[f"{current_random}random"] is not None:
                    axs[i_dof, 2].plot(time_vector_socp_variable[f"{current_random}random"], np.mean(q_socp_variable[f"{current_random}random"][dof-1, :, :], axis=1),
                                       color=colors_random[i_rand_for_color])
            if q_socp_feedforward[f"{current_random}random"] is not None:
                axs[i_dof, 3].plot(time_vector_socp_feedforward[f"{current_random}random"], np.mean(q_socp_feedforward[f"{current_random}random"][dof, :, :], axis=1),
                                   color=colors_random[i_rand_for_color])
            if q_socp_plus[f"{current_random}random"] is not None:
                axs[i_dof, 4].plot(time_vector_socp_plus[f"{current_random}random"], np.mean(q_socp_plus[f"{current_random}random"][dof, :, :], axis=1),
                                   color=colors_random[i_rand_for_color])

    axs[0, 0].set_title("OCP")
    axs[0, 1].set_title("SOCP")
    axs[0, 2].set_title(r"SOCP$_{\text{VN}}$")
    axs[0, 3].set_title(r"SOCP$^{\text{AF}}$")
    axs[0, 4].set_title(r"SOCP$_{\text{VN}}^{\text{AF}}$")
    for i_rand_for_color, current_random in enumerate(nb_random_list):
        axs[0, 0].plot(0, 0, color=colors_random[i_rand_for_color], linewidth=0.5, label=f"{current_random} episodes", alpha=0.3)
        axs[0, 0].plot(0, 0, color=colors_random[i_rand_for_color], label=f"Mean of {current_random} episodes")
    fig.subplots_adjust(right=0.8)
    axs[0, 0].legend(bbox_to_anchor=(1.4, -5.65), loc="upper left", ncol=4)

    for i_axs_2 in range(5):
        for i_axs in range(n_q-3):
            axs[i_axs, i_axs_2].get_xaxis().set_visible(False)
    #     axs[-1, i_axs_2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
        axs[-1, i_axs_2].set_xlabel("Time [s]")

    rotation_names = ["Somersault", "Neck", "Eyes", "Shoulders", "Hips", "Knees"]
    for i_axs_2 in range(5):
        for i_axs in range(n_q-2):
            min_y = np.min([axs[i_axs, i_axs_2].get_ylim()[0] for i_axs_2 in range(5)])
            max_y = np.max([axs[i_axs, i_axs_2].get_ylim()[1] for i_axs_2 in range(5)])
            axs[i_axs, i_axs_2].set_ylim(min_y, max_y)
            if i_axs_2 != 0:
                axs[i_axs, i_axs_2].get_yaxis().set_visible(False)
            else:
                axs[i_axs, i_axs_2].set_ylabel(f"{rotation_names[i_axs]}\n[rad]")
    #     axs[-1, i_axs_2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])

    plt.subplots_adjust(bottom=0.15, top=0.9, right=0.95, left=0.05, wspace=0.02, hspace=0.02)
    plt.suptitle("Comparison of solutions with different number of episodes")
    plt.savefig(f"graphs/comparison_kinematics_nb_random.png")
    # plt.show()
    return

def plot_movement_duration(time_ocp, time_socp, time_socp_variable, time_socp_feedforward, time_socp_plus, OCP_color, SOCP_color, SOCP_VARIABLE_color, SOCP_FEEDFORWARD_color, SOCP_PLUS_color):
    plt.figure()
    plt.bar(0, float(time_ocp), color=OCP_color, label="OCP")
    plt.bar(1, float(time_socp), color=SOCP_color, label="SOCP")
    plt.bar(2, float(time_socp_variable), color=SOCP_VARIABLE_color, label=r"SOCP$_{\text{VN}}$")
    plt.bar(3, float(time_socp_feedforward), color=SOCP_FEEDFORWARD_color, label=r"SOCP$^{\text{AF}}$")
    plt.bar(4, float(time_socp_plus), color=SOCP_PLUS_color, label=r"SOCP$_{\text{VN}}^{\text{AF}}$")
    plt.xticks([0, 1, 2, 3, 4], ["OCP", "SOCP", r"SOCP$_{\text{VN}}$", r"SOCP$^{\text{AF}}$", r"SOCP$_{\text{VN}}^{\text{AF}}$"])
    print("Movement durations: ", time_ocp, time_socp, time_socp_variable, time_socp_feedforward, time_socp_plus)
    plt.savefig("graphs/movement_durations.png")
    # plt.show()

def plot_landing_variability(
        CoM_y_fcn,
        CoM_y_8_fcn,
        CoM_y_dot_fcn,
        CoM_dot_8_fcn,
        BodyVelocity_fcn,  # Already in degrees in the casadi function
        BodyVelocity_8_fcn,  # Already in degrees in the casadi function
        nb_random,
        q_ocp_integrated,
        qdot_ocp_integrated,
        q_socp,
        qdot_socp,
        q_socp_variable,
        qdot_socp_variable,
        q_socp_feedforward,
        qdot_socp_feedforward,
        q_socp_plus,
        qdot_socp_plus,
        OCP_color,
        SOCP_color,
        SOCP_VARIABLE_color,
        SOCP_FEEDFORWARD_color,
        SOCP_PLUS_color,
):

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
        m_to_mm_factor = 1000
        CoM_y_ocp[i_random] = CoM_y_fcn(q_ocp_integrated[:, -1, i_random]) * m_to_mm_factor
        CoM_y_socp[i_random] = CoM_y_fcn(q_socp[:, -1, i_random]) * m_to_mm_factor
        CoM_y_socp_variable[i_random] = CoM_y_fcn(q_socp_variable[:, -1, i_random]) * m_to_mm_factor
        CoM_y_socp_feedforward[i_random] = CoM_y_8_fcn(q_socp_feedforward[:, -1, i_random]) * m_to_mm_factor
        CoM_y_socp_plus[i_random] = CoM_y_8_fcn(q_socp_plus[:, -1, i_random]) * m_to_mm_factor
        axs[0].plot(0 + np.random.random(1) * 0.2-0.1, CoM_y_ocp[i_random], ".", color=OCP_color)
        axs[0].plot(0.5 + np.random.random(1) * 0.2-0.1, CoM_y_socp[i_random], ".", color=SOCP_color)
        axs[0].plot(1 + np.random.random(1) * 0.2-0.1, CoM_y_socp_variable[i_random], ".", color=SOCP_VARIABLE_color)
        axs[0].plot(1.5 + np.random.random(1) * 0.2-0.1, CoM_y_socp_feedforward[i_random], ".",
                    color=SOCP_FEEDFORWARD_color)
        axs[0].plot(2 + np.random.random(1) * 0.2-0.1, CoM_y_socp_plus[i_random], ".", color=SOCP_PLUS_color)

        CoM_dot_y_ocp[i_random] = CoM_y_dot_fcn(
            q_ocp_integrated[:, -1, i_random],
            qdot_ocp_integrated[:, -1, i_random],
        ) * m_to_mm_factor
        CoM_dot_y_socp[i_random] = CoM_y_dot_fcn(
            q_socp[:, -1, i_random],
            qdot_socp[:, -1, i_random],
        ) * m_to_mm_factor
        CoM_dot_y_socp_variable[i_random] = CoM_y_dot_fcn(
            q_socp_variable[:, -1, i_random],
            qdot_socp_variable[:, -1, i_random],
        ) * m_to_mm_factor
        CoM_dot_y_socp_feedforward[i_random] = CoM_dot_8_fcn(
            q_socp_feedforward[:, -1, i_random],
            qdot_socp_feedforward[:, -1, i_random],
        ) * m_to_mm_factor
        CoM_dot_y_socp_plus[i_random] = CoM_dot_8_fcn(
            q_socp_plus[:, -1, i_random],
            qdot_socp_plus[:, -1, i_random],
        ) * m_to_mm_factor
        axs[1].plot(0 + np.random.random(1) * 0.2-0.1, CoM_dot_y_ocp[i_random], ".", color=OCP_color)
        axs[1].plot(0.5 + np.random.random(1) * 0.2-0.1, CoM_dot_y_socp[i_random], ".", color=SOCP_color)
        axs[1].plot(1 + np.random.random(1) * 0.2-0.1, CoM_dot_y_socp_variable[i_random], ".", color=SOCP_VARIABLE_color)
        axs[1].plot(1.5 + np.random.random(1) * 0.2-0.1, CoM_dot_y_socp_feedforward[i_random], ".",
                    color=SOCP_FEEDFORWARD_color)
        axs[1].plot(2 + np.random.random(1) * 0.2-0.1, CoM_dot_y_socp_plus[i_random], ".", color=SOCP_PLUS_color)

        BodyVelocity_ocp[i_random] = BodyVelocity_fcn(
            q_ocp_integrated[:, -1, i_random],
            qdot_ocp_integrated[:, -1, i_random]
        )
        BodyVelocity_socp[i_random] = BodyVelocity_fcn(
            q_socp[:, -1, i_random],
            qdot_socp[:, -1, i_random],
        )
        BodyVelocity_socp_variable[i_random] = BodyVelocity_fcn(
            q_socp_variable[:, -1, i_random],

            qdot_socp_variable[:, -1, i_random])
        BodyVelocity_socp_feedforward[i_random] = BodyVelocity_8_fcn(
            q_socp_feedforward[:, -1, i_random],
            qdot_socp_feedforward[:, -1, i_random],
        )
        BodyVelocity_socp_plus[i_random] = BodyVelocity_8_fcn(
            q_socp_plus[:, -1, i_random],
            qdot_socp_plus[:, -1, i_random],
        )
        axs[2].plot(0 + np.random.random(1) * 0.2-0.1, BodyVelocity_ocp[i_random], ".", color=OCP_color)
        axs[2].plot(0.5 + np.random.random(1) * 0.2-0.1, BodyVelocity_socp[i_random], ".", color=SOCP_color)
        axs[2].plot(1 + np.random.random(1) * 0.2-0.1, BodyVelocity_socp_variable[i_random], ".", color=SOCP_VARIABLE_color)
        axs[2].plot(1.5 + np.random.random(1) * 0.2-0.1, BodyVelocity_socp_feedforward[i_random], ".",
                    color=SOCP_FEEDFORWARD_color)
        axs[2].plot(2 + np.random.random(1) * 0.2-0.1, BodyVelocity_socp_plus[i_random], ".", color=SOCP_PLUS_color)

    axs[0].set_title(r"Center of mass horizontal position ($CoM$) [mm]")
    axs[1].set_title(r"Center of mass horizontal velocity ($\dot{CoM}$) [mm/s]")
    axs[2].set_title(r"Body angular velocity [$^\circ$]")

    box_plot(0, CoM_y_ocp, OCP_color, axs[0], box_width=0.1)
    axs[0].text(0, 0.995 * np.max(CoM_y_ocp), f"{np.std(CoM_y_ocp):.4f}", horizontalalignment='center')
    box_plot(0.5, CoM_y_socp, SOCP_color, axs[0], box_width=0.1)
    axs[0].text(0.5, 0.995 * np.max(CoM_y_socp), f"{np.std(CoM_y_socp):.4f}", horizontalalignment='center')
    box_plot(1, CoM_y_socp_variable, SOCP_VARIABLE_color, axs[0], box_width=0.1)
    axs[0].text(1, 0.995 * np.max(CoM_y_socp_variable), f"{np.std(CoM_y_socp_variable):.4f}", horizontalalignment='center')
    box_plot(1.5, CoM_y_socp_feedforward, SOCP_FEEDFORWARD_color, axs[0], box_width=0.1)
    axs[0].text(1.5, 0.995 * np.max(CoM_y_socp_feedforward), f"{np.std(CoM_y_socp_feedforward):.4f}", horizontalalignment='center')
    box_plot(2, CoM_y_socp_plus, SOCP_PLUS_color, axs[0], box_width=0.1)
    axs[0].text(2, 0.995 * np.max(CoM_y_socp_plus), f"{np.std(CoM_y_socp_plus):.4f}", horizontalalignment='center')

    add_std_to_box_plot(0, CoM_y_ocp, axs[0])
    add_std_to_box_plot(0.5, CoM_y_socp, axs[0])
    add_std_to_box_plot(1, CoM_y_socp_variable, axs[0])
    add_std_to_box_plot(1.5, CoM_y_socp_feedforward, axs[0])
    add_std_to_box_plot(2, CoM_y_socp_plus, axs[0])

    box_plot(0, CoM_dot_y_ocp, OCP_color, axs[1], box_width=0.1)
    axs[1].text(0, 0.999 * np.max(CoM_dot_y_ocp), f"{np.std(CoM_dot_y_ocp):.4f}", horizontalalignment='center')
    box_plot(0.5, CoM_dot_y_socp, SOCP_color, axs[1], box_width=0.1)
    axs[1].text(0.5, 0.999 * np.max(CoM_dot_y_socp), f"{np.std(CoM_dot_y_socp):.4f}", horizontalalignment='center')
    box_plot(1, CoM_dot_y_socp_variable, SOCP_VARIABLE_color, axs[1], box_width=0.1)
    axs[1].text(1, 0.999 * np.max(CoM_dot_y_socp_variable), f"{np.std(CoM_dot_y_socp_variable):.4f}", horizontalalignment='center')
    box_plot(1.5, CoM_dot_y_socp_feedforward, SOCP_FEEDFORWARD_color, axs[1], box_width=0.1)
    axs[1].text(1.5, 0.999 * np.max(CoM_dot_y_socp_feedforward), f"{np.std(CoM_dot_y_socp_feedforward):.4f}", horizontalalignment='center')
    box_plot(2, CoM_dot_y_socp_plus, SOCP_PLUS_color, axs[1], box_width=0.1)
    axs[1].text(2, 0.999 * np.max(CoM_dot_y_socp_plus), f"{np.std(CoM_dot_y_socp_plus):.4f}", horizontalalignment='center')

    add_std_to_box_plot(0, CoM_dot_y_ocp, axs[1])
    add_std_to_box_plot(0.5, CoM_dot_y_socp, axs[1])
    add_std_to_box_plot(1, CoM_dot_y_socp_variable, axs[1])
    add_std_to_box_plot(1.5, CoM_dot_y_socp_feedforward, axs[1])
    add_std_to_box_plot(2, CoM_dot_y_socp_plus, axs[1])

    box_plot(0, BodyVelocity_ocp, OCP_color, axs[2], box_width=0.1)
    axs[2].text(0, 1.005 * np.max(BodyVelocity_ocp), f"{np.std(BodyVelocity_ocp):.3f}", horizontalalignment='center')
    box_plot(0.5, BodyVelocity_socp, SOCP_color, axs[2], box_width=0.1)
    axs[2].text(0.5, 1.005 * np.max(BodyVelocity_socp), f"{np.std(BodyVelocity_socp):.3f}", horizontalalignment='center')
    box_plot(1, BodyVelocity_socp_variable, SOCP_VARIABLE_color, axs[2], box_width=0.1)
    axs[2].text(1, 1.005 * np.max(BodyVelocity_socp_variable), f"{np.std(BodyVelocity_socp_variable):.3f}", horizontalalignment='center')
    box_plot(1.5, BodyVelocity_socp_feedforward, SOCP_FEEDFORWARD_color, axs[2], box_width=0.1)
    axs[2].text(1.5, 1.005 * np.max(BodyVelocity_socp_feedforward), f"{np.std(BodyVelocity_socp_feedforward):.3f}", horizontalalignment='center')
    box_plot(2, BodyVelocity_socp_plus, SOCP_PLUS_color, axs[2], box_width=0.1)
    axs[2].text(2, 1.005 * np.max(BodyVelocity_socp_plus), f"{np.std(BodyVelocity_socp_plus):.3f}", horizontalalignment='center')

    add_std_to_box_plot(0, BodyVelocity_ocp, axs[2])
    add_std_to_box_plot(0.5, BodyVelocity_socp, axs[2])
    add_std_to_box_plot(1, BodyVelocity_socp_variable, axs[2])
    add_std_to_box_plot(1.5, BodyVelocity_socp_feedforward, axs[2])
    add_std_to_box_plot(2, BodyVelocity_socp_plus, axs[2])

    axs[0].set_xticks([0, 0.5, 1, 1.5, 2], ["OCP", "SOCP", r"SOCP$_{\text{VN}}$", r"SOCP$^{\text{AF}}$", r"SOCP$_{\text{VN}}^{\text{AF}}$"])
    axs[1].set_xticks([0, 0.5, 1, 1.5, 2], ["OCP", "SOCP", r"SOCP$_{\text{VN}}$", r"SOCP$^{\text{AF}}$", r"SOCP$_{\text{VN}}^{\text{AF}}$"])
    axs[2].set_xticks([0, 0.5, 1, 1.5, 2], ["OCP", "SOCP", r"SOCP$_{\text{VN}}$", r"SOCP$^{\text{AF}}$", r"SOCP$_{\text{VN}}^{\text{AF}}$"])

    axs[0].set_ylim(-265, -205)
    axs[1].set_ylim(-416, -395)
    axs[2].set_ylim(650, 795)

    plt.tight_layout()
    plt.savefig("graphs/landing_variability.png")
    # plt.show()

    OCP_phi_1 = (1e4 * np.sum((CoM_y_ocp/m_to_mm_factor - np.mean(CoM_y_ocp/m_to_mm_factor))**2) +
            1e4 * np.sum((CoM_dot_y_ocp/m_to_mm_factor - np.mean(CoM_dot_y_ocp/m_to_mm_factor))**2) +
            1e4 * np.sum((BodyVelocity_ocp*np.pi/180 - np.mean(BodyVelocity_ocp*np.pi/180 ))**2))
    print(f"OCP landing variability: {OCP_phi_1}")
    print(f"SOCP landing variability: {
        1e4 * np.sum((CoM_y_socp/m_to_mm_factor - np.mean(CoM_y_socp/m_to_mm_factor))**2) + 
        1e4 * np.sum((CoM_dot_y_socp/m_to_mm_factor - np.mean(CoM_dot_y_socp/m_to_mm_factor))**2) +
        1e4 * np.sum((BodyVelocity_socp*np.pi/180  - np.mean(BodyVelocity_socp*np.pi/180 ))**2)
    }")
    print(f"SOCP VARIABLE landing variability: {
        1e4 * np.sum((CoM_y_socp_variable/m_to_mm_factor - np.mean(CoM_y_socp_variable/m_to_mm_factor))**2) + 
        1e4 * np.sum((CoM_dot_y_socp_variable/m_to_mm_factor - np.mean(CoM_dot_y_socp_variable/m_to_mm_factor))**2) +
        1e4 * np.sum((BodyVelocity_socp_variable*np.pi/180  - np.mean(BodyVelocity_socp_variable*np.pi/180 ))**2)
    }")
    print(f"SOCP FEEDFORWARD landing variability: {
        1e4 * np.sum((CoM_y_socp_feedforward/m_to_mm_factor - np.mean(CoM_y_socp_feedforward/m_to_mm_factor))**2) + 
        1e4 * np.sum((CoM_dot_y_socp_feedforward/m_to_mm_factor - np.mean(CoM_dot_y_socp_feedforward/m_to_mm_factor))**2) +
        1e4 * np.sum((BodyVelocity_socp_feedforward*np.pi/180  - np.mean(BodyVelocity_socp_feedforward*np.pi/180 ))**2)
    }")
    SOCP_plus_phi_1 = (1e4 * np.sum((CoM_y_socp_plus/m_to_mm_factor - np.mean(CoM_y_socp_plus/m_to_mm_factor))**2) +
        1e4 * np.sum((CoM_dot_y_socp_plus/m_to_mm_factor - np.mean(CoM_dot_y_socp_plus/m_to_mm_factor))**2) +
        1e4 * np.sum((BodyVelocity_socp_plus*np.pi/180  - np.mean(BodyVelocity_socp_plus*np.pi/180 ))**2))
    print(f"SOCP+ landing variability: {SOCP_plus_phi_1}")

    print(f"OCP / SOCP+ landing variability: {OCP_phi_1 / SOCP_plus_phi_1}")
    return

def plot_inertia_ang_mom(
        normalized_time_vector,
        inertia_fcn,
        inertia_8_fcn,
        ang_mom_fcn,
        ang_mom_8_fcn,
        BodyVelocity_fcn,  # Already in degrees in the casadi function
        BodyVelocity_8_fcn,  # Already in degrees in the casadi function
        nb_random,
        q_ocp_integrated,
        qdot_ocp_integrated,
        q_socp,
        qdot_socp,
        q_socp_variable,
        qdot_socp_variable,
        q_socp_feedforward,
        qdot_socp_feedforward,
        q_socp_plus,
        qdot_socp_plus,
        OCP_color,
        SOCP_color,
        SOCP_VARIABLE_color,
        SOCP_FEEDFORWARD_color,
        SOCP_PLUS_color,
):
    n_shooting = normalized_time_vector.shape[0]
    fig, axs = plt.subplots(2, 1, figsize=(15, 6))

    inertia_ocp = np.zeros((nb_random, n_shooting))
    inertia_socp = np.zeros((nb_random, n_shooting))
    inertia_socp_variable = np.zeros((nb_random, n_shooting))
    inertia_socp_feedforward = np.zeros((nb_random, n_shooting))
    inertia_socp_plus = np.zeros((nb_random, n_shooting))
    ang_mom_ocp = np.zeros((nb_random, n_shooting))
    ang_mom_socp = np.zeros((nb_random, n_shooting))
    ang_mom_socp_variable = np.zeros((nb_random, n_shooting))
    ang_mom_socp_feedforward = np.zeros((nb_random, n_shooting))
    ang_mom_socp_plus = np.zeros((nb_random, n_shooting))
    body_rotation_rate_ocp = np.zeros((nb_random, n_shooting))
    body_rotation_rate_socp = np.zeros((nb_random, n_shooting))
    body_rotation_rate_socp_variable = np.zeros((nb_random, n_shooting))
    body_rotation_rate_socp_feedforward = np.zeros((nb_random, n_shooting))
    body_rotation_rate_socp_plus = np.zeros((nb_random, n_shooting))
    for i_random in range(nb_random):
        for i_shooting in range(n_shooting):
            inertia_ocp[i_random, i_shooting] = inertia_fcn(q_ocp_integrated[:, i_shooting, i_random])
            inertia_socp[i_random, i_shooting] = inertia_fcn(q_socp[:, i_shooting, i_random])
            inertia_socp_variable[i_random, i_shooting] = inertia_fcn(q_socp_variable[:, i_shooting, i_random])
            inertia_socp_feedforward[i_random, i_shooting] = inertia_8_fcn(q_socp_feedforward[:, i_shooting, i_random])
            inertia_socp_plus[i_random, i_shooting] = inertia_8_fcn(q_socp_plus[:, i_shooting, i_random])

        axs[0].plot(normalized_time_vector, inertia_ocp[i_random], ".", linestyle="-", color=OCP_color)
        axs[0].plot(normalized_time_vector, inertia_socp[i_random], ".", linestyle="-", color=SOCP_color)
        axs[0].plot(normalized_time_vector, inertia_socp_variable[i_random], ".", linestyle="-", color=SOCP_VARIABLE_color)
        axs[0].plot(normalized_time_vector, inertia_socp_feedforward[i_random], ".", linestyle="-", color=SOCP_FEEDFORWARD_color)
        axs[0].plot(normalized_time_vector, inertia_socp_plus[i_random], ".", linestyle="-", color=SOCP_PLUS_color)

        for i_shooting in range(n_shooting):
            ang_mom_ocp[i_random, i_shooting] = ang_mom_fcn(
                q_ocp_integrated[:, i_shooting, i_random],
                qdot_ocp_integrated[:, i_shooting, i_random],
            )
            ang_mom_socp[i_random, i_shooting] = ang_mom_fcn(
                q_socp[:, i_shooting, i_random],
                qdot_socp[:, i_shooting, i_random],
            )
            ang_mom_socp_variable[i_random, i_shooting] = ang_mom_fcn(
                q_socp_variable[:, i_shooting, i_random],
                qdot_socp_variable[:, i_shooting, i_random],
            )
            ang_mom_socp_feedforward[i_random, i_shooting] = ang_mom_8_fcn(
                q_socp_feedforward[:, i_shooting, i_random],
                qdot_socp_feedforward[:, i_shooting, i_random],
            )
            ang_mom_socp_plus[i_random, i_shooting] = ang_mom_8_fcn(
                q_socp_plus[:, i_shooting, i_random],
                qdot_socp_plus[:, i_shooting, i_random],
            )

        # axs[1].plot(normalized_time_vector, ang_mom_ocp[i_random], ".", linestyle="-", color=OCP_color)
        # axs[1].plot(normalized_time_vector, ang_mom_socp[i_random], ".", linestyle="-", color=SOCP_color)
        # axs[1].plot(normalized_time_vector, ang_mom_socp_variable[i_random], ".", linestyle="-",
        #             color=SOCP_VARIABLE_color)
        # axs[1].plot(normalized_time_vector, ang_mom_socp_feedforward[i_random], ".", linestyle="-",
        #             color=SOCP_FEEDFORWARD_color)
        # axs[1].plot(normalized_time_vector, ang_mom_socp_plus[i_random], ".", linestyle="-", color=SOCP_PLUS_color)

        for i_shooting in range(n_shooting):
            body_rotation_rate_ocp[i_random, i_shooting] = BodyVelocity_fcn(
                q_ocp_integrated[:, i_shooting, i_random],
                qdot_ocp_integrated[:, i_shooting, i_random],
            )
            body_rotation_rate_socp[i_random, i_shooting] = BodyVelocity_fcn(
                q_socp[:, i_shooting, i_random],
                qdot_socp[:, i_shooting, i_random],
            )
            body_rotation_rate_socp_variable[i_random,  i_shooting] = BodyVelocity_fcn(
                q_socp_variable[:, i_shooting, i_random],
                qdot_socp_variable[:, i_shooting, i_random],
            )
            body_rotation_rate_socp_feedforward[i_random, i_shooting] = BodyVelocity_8_fcn(
                q_socp_feedforward[:, i_shooting, i_random],
                qdot_socp_feedforward[:, i_shooting, i_random],
            )
            body_rotation_rate_socp_plus[i_random, i_shooting] = BodyVelocity_8_fcn(
                q_socp_plus[:, i_shooting, i_random],
                qdot_socp_plus[:, i_shooting, i_random],
            )

        axs[1].plot(normalized_time_vector, body_rotation_rate_ocp[i_random], ".", linestyle="-", color=OCP_color)
        axs[1].plot(normalized_time_vector, body_rotation_rate_socp[i_random], ".", linestyle="-", color=SOCP_color)
        axs[1].plot(normalized_time_vector, body_rotation_rate_socp_variable[i_random], ".", linestyle="-",
                    color=SOCP_VARIABLE_color)
        axs[1].plot(normalized_time_vector, body_rotation_rate_socp_feedforward[i_random], ".", linestyle="-",
                    color=SOCP_FEEDFORWARD_color)
        axs[1].plot(normalized_time_vector, body_rotation_rate_socp_plus[i_random], ".", linestyle="-", color=SOCP_PLUS_color)


    print(f"OCP Ang Mom: {np.mean(ang_mom_ocp)}")
    print(f"SOCP Ang Mom: {np.mean(ang_mom_socp)}")
    print(f"SOCP VARIABLE Ang Mom: {np.mean(ang_mom_socp_variable)}")
    print(f"SOCP FEEDFORWARD Ang Mom: {np.mean(ang_mom_socp_feedforward)}")
    print(f"SOCP+ Ang Mom: {np.mean(ang_mom_socp_plus)}")

    axs[0].get_xaxis().set_visible(False)
    axs[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1].set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    axs[1].set_xlabel("Normalized time")

    axs[0].set_ylabel("Transverse moment\nof inertia " +  r"[kg.m$^2$]")
    # axs[1].set_ylabel("Transverse angular\nmomentum " + r"[kg.m$^2$/s]")
    axs[1].set_ylabel("Transverse body\nrotation rate " + r"[^\circ/s]")

    plt.subplots_adjust(hspace=0.1)
    plt.savefig("graphs/inertia_ang_mom.png", dpi=300)
    # plt.show()
    return

def plot_significant_timing_blocks(dof1, dof2, q1, q2, color1, color2, num_combinaition, ax):
    significant_timings = []

    t = spm1d.stats.ttest2(q1[dof1, :, :].T,
                           q2[dof2, :, :].T)
    ti = t.inference(alpha=0.05, two_tailed=True)
    if ti.h0reject == True:
        if ti.clusters != []:
            significant_timing_list = ti.clusters
            for k in range(len(significant_timing_list)):
                significant_timing_1_x, _ = significant_timing_list[k].get_patch_vertices()
                significant_timings += list(range(int(significant_timing_1_x[1]+1), int(significant_timing_1_x[-2]+2)))

    # Remove duplicates from list and sort
    significant_timings = list(set(significant_timings))
    # Find blocks of consecutive indices
    significant_timings_array = np.array(significant_timings)
    significant_timings_diff = significant_timings_array[1:] - significant_timings_array[:-1]
    significant_timings_index = np.hstack((-1, np.where(significant_timings_diff > 1)[0], len(significant_timings_array)-1))
    timings = []
    for i in range(len(significant_timings_index)-1):
        timings += [range(significant_timings_array[significant_timings_index[i]+1], significant_timings_array[significant_timings_index[i+1]])]

    for timing_this_time in timings:
        start = timing_this_time.start
        end = timing_this_time.stop if timing_this_time.stop == 17 else timing_this_time.stop + 1
        ax.fill_between([start/17, end/17],
                            [-num_combinaition, -num_combinaition],
                            [-num_combinaition + 0.25, -num_combinaition + 0.25], color=color1, alpha=0.5)
        ax.fill_between([start/17, end/17],
                            [-num_combinaition - 0.25, -num_combinaition - 0.25],
                            [-num_combinaition, -num_combinaition], color=color2, alpha=0.5)

    print(f"There is {(len(significant_timings)/17) * 100}% of the movement with significant difference for DOF {dof1} and {dof2}.")
    return


def plot_kinematics(normalized_time_vector,
                    normalized_time_vector_MS,
                    q_ocp_integrated_MS,
                    q_socp_integrated_MS,
                    q_socp_variable_integrated_MS,
                    q_socp_feedforward_integrated_MS,
                    q_socp_plus_integrated_MS,
                    q_ocp,
                    q_socp,
                    q_socp_variable,
                    q_socp_feedforward,
                    q_socp_plus,
                    OCP_color,
                    SOCP_color,
                    SOCP_VARIABLE_color,
                    SOCP_FEEDFORWARD_color,
                    SOCP_PLUS_color):

    fig1, axs1 = plt.subplots(2, 3, figsize=(10, 4), gridspec_kw={'height_ratios': [5, 2]})
    fig2, axs2 = plt.subplots(2, 3, figsize=(10, 4), gridspec_kw={'height_ratios': [5, 2]})
    axs_dof_ordered = [axs1[0, 0], axs1[0, 1], axs1[0, 2], axs2[0, 0], axs2[0, 1], axs2[0, 2]]
    axs_spm_ordered = [axs1[1, 0], axs1[1, 1], axs1[1, 2], axs2[1, 0], axs2[1, 1], axs2[1, 2]]
    for i_dof, dof in enumerate(range(2, n_q)):
        dof_ocp = dof
        dof_socp = dof
        dof_socp_variable = dof
        dof_socp_feedforward = dof
        dof_socp_plus = dof
        if dof < 4:
            axs_dof_ordered[i_dof].plot(normalized_time_vector, q_ocp[dof, :] * 180 / np.pi, '.', color=OCP_color)
            axs_dof_ordered[i_dof].plot(normalized_time_vector_MS, q_ocp_integrated_MS[dof, :] * 180 / np.pi, '-', color=OCP_color)
        elif dof > 4:
            axs_dof_ordered[i_dof].plot(normalized_time_vector, q_ocp[dof - 1, :] * 180 / np.pi, '.', color=OCP_color)
            axs_dof_ordered[i_dof].plot(normalized_time_vector_MS, q_ocp_integrated_MS[dof - 1, :] * 180 / np.pi, '-', color=OCP_color)
            dof_ocp = dof - 1
        for i_random in range(nb_random):
            if dof < 4:
                axs_dof_ordered[i_dof].plot(normalized_time_vector, q_socp[dof, :, i_random] * 180 / np.pi, '.', color=SOCP_color, linewidth=0.5)
                axs_dof_ordered[i_dof].plot(normalized_time_vector_MS, q_socp_integrated_MS[dof, i_random, :] * 180 / np.pi, '-', color=SOCP_color, linewidth=0.5)
                axs_dof_ordered[i_dof].plot(normalized_time_vector, q_socp_variable[dof, :, i_random] * 180 / np.pi, '.', color=SOCP_VARIABLE_color,
                                linewidth=0.5)
                axs_dof_ordered[i_dof].plot(normalized_time_vector_MS, q_socp_variable_integrated_MS[dof, i_random, :] * 180 / np.pi, '-', color=SOCP_VARIABLE_color,
                                linewidth=0.5)
            elif dof > 4:
                axs_dof_ordered[i_dof].plot(normalized_time_vector, q_socp[dof - 1, :, i_random] * 180 / np.pi, '.', color=SOCP_color, linewidth=0.5)
                axs_dof_ordered[i_dof].plot(normalized_time_vector_MS, q_socp_integrated_MS[dof - 1, i_random, :] * 180 / np.pi, '-', color=SOCP_color, linewidth=0.5)
                axs_dof_ordered[i_dof].plot(normalized_time_vector, q_socp_variable[dof - 1, :, i_random] * 180 / np.pi, '.', color=SOCP_VARIABLE_color,
                                linewidth=0.5)
                axs_dof_ordered[i_dof].plot(normalized_time_vector_MS, q_socp_variable_integrated_MS[dof - 1, i_random, :] * 180 / np.pi, '-', color=SOCP_VARIABLE_color,
                                linewidth=0.5)
                dof_socp = dof - 1
                dof_socp_variable = dof - 1
        for i_random in range(nb_random):
            axs_dof_ordered[i_dof].plot(normalized_time_vector, q_socp_feedforward[dof, :, i_random] * 180 / np.pi, '.', color=SOCP_FEEDFORWARD_color,
                            linewidth=0.5)
            axs_dof_ordered[i_dof].plot(normalized_time_vector_MS, q_socp_feedforward_integrated_MS[dof, i_random, :] * 180 / np.pi, '-', color=SOCP_FEEDFORWARD_color,
                            linewidth=0.5)
            axs_dof_ordered[i_dof].plot(normalized_time_vector, q_socp_plus[dof, :, i_random] * 180 / np.pi, '.', color=SOCP_PLUS_color, linewidth=0.5)
            axs_dof_ordered[i_dof].plot(normalized_time_vector_MS, q_socp_plus_integrated_MS[dof, i_random, :] * 180 / np.pi, '-', color=SOCP_PLUS_color, linewidth=0.5)

        q_ocp_with_variance = q_ocp[:, :, np.newaxis] + np.random.normal(7, 17, nb_random) * 0.0000001

        num_combinaition = 0
        plot_significant_timing_blocks(dof_ocp, dof_socp, q_ocp_with_variance, q_socp, OCP_color, SOCP_color, num_combinaition, axs_spm_ordered[i_dof])
        num_combinaition += 1
        plot_significant_timing_blocks(dof_ocp, dof_socp_variable, q_ocp_with_variance, q_socp_variable, OCP_color, SOCP_VARIABLE_color, num_combinaition, axs_spm_ordered[i_dof])
        num_combinaition += 1
        plot_significant_timing_blocks(dof_ocp, dof_socp_feedforward, q_ocp_with_variance, q_socp_feedforward, OCP_color, SOCP_FEEDFORWARD_color, num_combinaition, axs_spm_ordered[i_dof])
        num_combinaition += 1
        plot_significant_timing_blocks(dof_ocp, dof_socp_plus, q_ocp_with_variance, q_socp_plus, OCP_color, SOCP_PLUS_color, num_combinaition, axs_spm_ordered[i_dof])
        num_combinaition += 1
        plot_significant_timing_blocks(dof_socp, dof_socp_variable, q_socp, q_socp_variable, SOCP_color, SOCP_VARIABLE_color, num_combinaition, axs_spm_ordered[i_dof])
        num_combinaition += 1
        plot_significant_timing_blocks(dof_socp, dof_socp_feedforward, q_socp, q_socp_feedforward, SOCP_color, SOCP_FEEDFORWARD_color, num_combinaition, axs_spm_ordered[i_dof])
        num_combinaition += 1
        plot_significant_timing_blocks(dof_socp, dof_socp_plus, q_socp, q_socp_plus, SOCP_color, SOCP_PLUS_color, num_combinaition, axs_spm_ordered[i_dof])
        num_combinaition += 1
        plot_significant_timing_blocks(dof_socp_variable, dof_socp_feedforward, q_socp_variable, q_socp_feedforward, SOCP_VARIABLE_color, SOCP_FEEDFORWARD_color, num_combinaition, axs_spm_ordered[i_dof])
        num_combinaition += 1
        plot_significant_timing_blocks(dof_socp_variable, dof_socp_plus, q_socp_variable, q_socp_plus, SOCP_VARIABLE_color, SOCP_PLUS_color, num_combinaition, axs_spm_ordered[i_dof])
        num_combinaition += 1
        plot_significant_timing_blocks(dof_socp_feedforward, dof_socp_plus, q_socp_feedforward, q_socp_plus, SOCP_FEEDFORWARD_color, SOCP_PLUS_color, num_combinaition, axs_spm_ordered[i_dof])

    axs_dof_ordered[0].set_title("Somersault")
    axs_dof_ordered[1].set_title("Neck")
    axs_dof_ordered[2].set_title("Eyes")
    axs_dof_ordered[3].set_title("Shoulders")
    axs_dof_ordered[4].set_title("Hips")
    axs_dof_ordered[5].set_title("Knees")

    axs_dof_ordered[0].set_ylabel(r"Angle [$^\circ$]")
    axs_dof_ordered[3].set_ylabel(r"Angle [$^\circ$]")

    for i_ax in range(6):
        axs_dof_ordered[i_ax].get_xaxis().set_visible(False)
        axs_spm_ordered[i_ax].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs_spm_ordered[i_ax].set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        axs_spm_ordered[i_ax].set_xlabel("Time [s]")
        axs_spm_ordered[i_ax].spines['top'].set_visible(False)
        axs_spm_ordered[i_ax].spines['right'].set_visible(False)
        axs_spm_ordered[i_ax].spines['left'].set_visible(False)
        axs_spm_ordered[i_ax].set_ylim(-10, 1)
        axs_spm_ordered[i_ax].get_yaxis().set_ticks([])

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.subplots_adjust(hspace=0.05)
    fig2.subplots_adjust(hspace=0.05)
    fig1.savefig("graphs/kinematics1.png")
    fig2.savefig("graphs/kinematics2.png")
    # plt.show()
    return


FLAG_GENERATE_VIDEOS = False
model_name = "Model2D_7Dof_0C_3M"

OCP_color = "#5DC962"
SOCP_color = "#AC2594"
SOCP_VARIABLE_color = "#F18F01"
SOCP_FEEDFORWARD_color = "#A469F1"
SOCP_PLUS_color = "#06b0f0"

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
nb_random = 20
nb_reintegrations = 5

motor_noise_std = 0.05 * 10
wPq_std = 0.001 * 5
wPqdot_std = 0.003 * 5
motor_noise_magnitude = cas.DM(np.array([motor_noise_std**2 / dt for _ in range(n_q - n_root)]))  # All DoFs except root

# ------------- result paths ------------- #
result_folder = f"{nb_random}random-seed0"
ocp_path_to_results = f"results/deterministic/{model_name}_ocp_DMS_CVG_1e-8.pkl"
# socp_path_to_results = (
#     f"results/{result_folder}/Model2D_7Dof_0C_3M_socp_DMS_{nb_random}random_5p0e-01_5p0e-03_1p5e-02_CVG_1p0e-06.pkl"
# )
socp_path_to_results = (
    f"results/20random-seed1/Model2D_7Dof_0C_3M_socp_DMS_{nb_random}random_5p0e-01_5p0e-03_1p5e-02_CVG_1p0e-06.pkl"
)
socp_variable_path_to_results = (
    f"results/{result_folder}/Model2D_7Dof_0C_3M_socp_DMS_VARIABLE_{nb_random}random_5p0e-01_5p0e-03_1p5e-02_CVG_1p0e-06.pkl"
)
socp_feedforward_path_to_results = f"results/{result_folder}/Model2D_7Dof_0C_3M_socp_DMS_FEEDFORWARD_{nb_random}random_5p0e-01_5p0e-03_1p5e-02_CVG_1p0e-06.pkl"
socp_plus_path_to_results = f"results/{result_folder}/Model2D_7Dof_0C_3M_socp_DMS_VARIABLE_FEEDFORWARD_{nb_random}random_5p0e-01_5p0e-03_1p5e-02_CVG_1p0e-06.pkl"


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

nb_random_list = [5, 10, 15, 20]

q_ocp_integrated = None
q_socp_integrated = {f"{nb}random": None for nb in nb_random_list}
q_socp_variable_integrated = {f"{nb}random": None for nb in nb_random_list}
q_socp_feedforward_integrated = {f"{nb}random": None for nb in nb_random_list}
q_socp_plus_integrated = {f"{nb}random": None for nb in nb_random_list}

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

forward_dynamics_func = cas.Function("forward_dynamics", [Q, Qdot, Tau], [ocp.nlp[0].model.forward_dynamics()(Q, Qdot, cas.vertcat(cas.MX.zeros(3), Tau), cas.MX.zeros(), cas.MX.zeros())])

time_vector_ocp = np.linspace(0, float(time_ocp), n_shooting + 1)

q_ocp = np.vstack((q_roots_ocp, q_joints_ocp))
qdot_ocp = np.vstack((qdot_roots_ocp, qdot_joints_ocp))

if FLAG_GENERATE_VIDEOS:
    print("Generating OCP_one : ", ocp_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_ocp, np.vstack((q_roots_ocp, q_joints_ocp)), result_folder, "OCP_one")

ocp_out_path_to_results = ocp_path_to_results.replace(".pkl", "_integrated.pkl")
if not os.path.exists(ocp_out_path_to_results):
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
    q_ocp_integrated_MS, qdot_ocp_integrated_MS, time_vector_ocp_integrated_MS = integrate_MS_ocp(
        n_shooting=n_shooting,
        q_ocp = q_ocp,
        qdot_ocp=qdot_ocp,
        tau_joints_ocp=tau_joints_ocp,
        time_vector_ocp=time_vector_ocp,
        ocp=ocp,
        forward_dynamics_func=forward_dynamics_func,
    )

    joint_friction_ocp = np.zeros((n_q - 3, n_shooting))
    for i_shooting in range(n_shooting):
        joint_friction_ocp[:, i_shooting] = np.reshape(
            ocp.nlp[0].model.friction_coefficients @ qdot_joints_ocp[:, i_shooting], (-1,)
        )

    with open(ocp_out_path_to_results, "wb") as file:
        data = {
            "q_ocp_integrated": q_ocp_integrated,
            "qdot_ocp_integrated": qdot_ocp_integrated,
            "q_ocp_integrated_MS": q_ocp_integrated_MS,
            "qdot_ocp_integrated_MS": qdot_ocp_integrated_MS,
            "time_vector_ocp_integrated_MS": time_vector_ocp_integrated_MS,
            "q_all_ocp": q_all_ocp,
            "joint_frictions_ocp": joint_frictions_ocp,
            "motor_noises_ocp": motor_noises_ocp,
            "time_vector_ocp": time_vector_ocp,
            "q_nominal_ocp": cas.vertcat(q_roots_ocp, q_joints_ocp),
            "q_mean_ocp_integrated": np.mean(q_ocp_integrated, axis=2),
        }
        pickle.dump(data, file)
else:
    with open(ocp_out_path_to_results, "rb") as file:
        data = pickle.load(file)
        q_ocp_integrated = data["q_ocp_integrated"]
        qdot_ocp_integrated = data["qdot_ocp_integrated"]
        q_ocp_integrated_MS = data["q_ocp_integrated_MS"]
        qdot_ocp_integrated_MS = data["qdot_ocp_integrated_MS"]
        time_vector_ocp_integrated_MS = data["time_vector_ocp_integrated_MS"]
        q_all_ocp = data["q_all_ocp"]
        joint_frictions_ocp = data["joint_frictions_ocp"]
        motor_noises_ocp = data["motor_noises_ocp"]
        time_vector_ocp = data["time_vector_ocp"]
        q_mean_ocp_integrated = data["q_mean_ocp_integrated"]

    joint_friction_ocp = np.zeros((n_q - 3, n_shooting))
    for i_shooting in range(n_shooting):
        joint_friction_ocp[:, i_shooting] = np.reshape(
            ocp.nlp[0].model.friction_coefficients @ qdot_joints_ocp[:, i_shooting], (-1,)
        )

if FLAG_GENERATE_VIDEOS:
    print("Generating OCP_all : ", ocp_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_all, q_all_ocp, result_folder, "OCP_all")


with open(ocp_path_to_results.replace(".pkl", "_sol.pkl"), "rb") as file:
    sol_ocp = pickle.load(file)
    # sol_ocp.ocp = ocp
    # sol_ocp.detailed_cost  # Not implemented for n_thread > 1
print("OCP cost: ", sol_ocp.cost)

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

socp_out_path_to_results = socp_path_to_results.replace(".pkl", "_integrated.pkl")


DMS_sensory_reference_func = cas.Function(
    "DMS_sensory_reference", [Q, Qdot], [DMS_sensory_reference(socp.nlp[0].model, n_root, Q, Qdot, cas.MX.zeros())]
)

forward_dynamics_func = cas.Function("forward_dynamics", [Q, Qdot, Tau], [socp.nlp[0].model.forward_dynamics()(Q, Qdot, cas.vertcat(cas.MX.zeros(3), Tau), cas.MX.zeros(), cas.MX.zeros())])

time_vector_socp = np.linspace(0, float(time_socp), n_shooting + 1)

q_socp, qdot_socp, q_mean_socp, qdot_mean_socp = define_q_mean(n_shooting, nb_random, q_roots_socp, q_joints_socp, qdot_roots_socp, qdot_joints_socp)

if not os.path.exists(socp_out_path_to_results):
    q_integrated, qdot_socp_integrated, q_all_socp, joint_frictions_socp, motor_noises_socp, feedbacks_socp = (
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
    q_socp_integrated[f"{nb_random}random"] = q_integrated

    q_socp_integrated_MS, qdot_socp_integrated_MS, time_vector_socp_integrated_MS = integrate_socp_MS(
        motor_noise_numerical_socp,
        sensory_noise_numerical_socp,
        n_shooting,
        nb_random,
        q_socp,
        qdot_socp,
        tau_joints_socp,
        k_socp,
        ref_socp,
        time_vector_socp,
        socp.nlp[0],
        DMS_sensory_reference_func,
        forward_dynamics_func,
    )

    with open(socp_out_path_to_results, "wb") as file:
        data = {
            "q_socp_integrated": q_integrated,
            "qdot_socp_integrated": qdot_socp_integrated,
            "q_socp_integrated_MS": q_socp_integrated_MS,
            "qdot_socp_integrated_MS": qdot_socp_integrated_MS,
            "time_vector_socp_integrated_MS": time_vector_socp_integrated_MS,
            "q_all_socp": q_all_socp,
            "joint_frictions_socp": joint_frictions_socp,
            "motor_noises_socp": motor_noises_socp,
            "feedbacks_socp": feedbacks_socp,
            "time_vector_socp": time_vector_socp,
            "q_mean_integrated": np.mean(q_integrated, axis=2),
            "q_mean_socp": np.mean(q_socp, axis=2),
        }
        pickle.dump(data, file)
else:
    with open(socp_out_path_to_results, "rb") as file:
        data = pickle.load(file)
        q_socp_integrated[f"{nb_random}random"] = data["q_socp_integrated"]
        qdot_socp_integrated = data["qdot_socp_integrated"]
        q_socp_integrated_MS = data["q_socp_integrated_MS"]
        qdot_socp_integrated_MS = data["qdot_socp_integrated_MS"]
        time_vector_socp_integrated_MS = data["time_vector_socp_integrated_MS"]
        q_all_socp = data["q_all_socp"]
        joint_frictions_socp = data["joint_frictions_socp"]
        motor_noises_socp = data["motor_noises_socp"]
        feedbacks_socp = data["feedbacks_socp"]
        time_vector_socp = data["time_vector_socp"]
        q_mean_socp = data["q_mean_socp"]

with open(socp_path_to_results.replace(".pkl", "_sol.pkl"), "rb") as file:
    sol_socp = pickle.load(file)
print("SOCP cost: ", sol_socp.cost)

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_one : ", socp_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_socp, q_mean_socp, result_folder, "SOCP_one")

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_all : ", socp_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_all_socp, q_all_socp, result_folder, "SOCP_all")


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

socp_variable_out_path_to_results = socp_variable_path_to_results.replace(".pkl", "_integrated.pkl")

DMS_fb_noised_sensory_input_VARIABLE_func = cas.Function(
    "DMS_fb_noised_sensory_input_VARIABLE",
    [Q, Qdot, SensoryNoise],
    [DMS_fb_noised_sensory_input_VARIABLE(socp_variable.nlp[0].model, Q[:n_root], Q[n_root:], Qdot[:n_root], Qdot[n_root:], SensoryNoise, cas.MX.zeros())],
)
time_vector_socp_variable = np.linspace(0, float(time_socp_variable), n_shooting + 1)

q_socp_variable, qdot_socp_variable, q_mean_socp_variable, qdot_mean_socp_variable = define_q_mean(n_shooting, nb_random, q_roots_socp_variable, q_joints_socp_variable,
                                               qdot_roots_socp_variable, qdot_joints_socp_variable)

if not os.path.exists(socp_variable_out_path_to_results):
    q_integrated, qdot_socp_variable_integrated, q_all_socp_variable, joint_frictions_socp_variable, motor_noises_socp_variable, feedbacks_socp_variable = (
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
    q_socp_variable_integrated[f"{nb_random}random"] = q_integrated

    q_socp_variable_integrated_MS, qdot_socp_variable_integrated_MS, time_vector_socp_variable_integrated_MS = integrate_socp_variable_MS(
        motor_noise_numerical_socp_variable,
        sensory_noise_numerical_socp_variable,
        n_shooting,
        nb_random,
        q_socp_variable,
        qdot_socp_variable,
        tau_joints_socp_variable,
        k_socp_variable,
        ref_socp_variable,
        time_vector_socp_variable,
        socp_variable.nlp[0],
        DMS_fb_noised_sensory_input_VARIABLE_func,
        forward_dynamics_func,
    )

    with open(socp_variable_out_path_to_results, "wb") as file:
        data = {
            "q_socp_variable_integrated": q_integrated,
            "qdot_socp_variable_integrated": qdot_socp_variable_integrated,
            "q_all_socp_variable": q_all_socp_variable,
            "q_socp_variable_integrated_MS": q_socp_variable_integrated_MS,
            "qdot_socp_variable_integrated_MS": qdot_socp_variable_integrated_MS,
            "time_vector_socp_variable_integrated_MS": time_vector_socp_variable_integrated_MS,
            "joint_frictions_socp_variable": joint_frictions_socp_variable,
            "motor_noises_socp_variable": motor_noises_socp_variable,
            "feedbacks_socp_variable": feedbacks_socp_variable,
            "time_vector_socp_variable": time_vector_socp_variable,
            "q_mean_socp_variable_integrated": np.mean(q_integrated, axis=2),
            "q_mean_socp_variable": np.mean(q_socp_variable, axis=2),
        }
        pickle.dump(data, file)
else:
    with open(socp_variable_out_path_to_results, "rb") as file:
        data = pickle.load(file)
        q_socp_variable_integrated[f"{nb_random}random"] = data["q_socp_variable_integrated"]
        qdot_socp_variable_integrated = data["qdot_socp_variable_integrated"]
        q_socp_variable_integrated_MS = data["q_socp_variable_integrated_MS"]
        qdot_socp_variable_integrated_MS = data["qdot_socp_variable_integrated_MS"]
        time_vector_socp_variable_integrated_MS = data["time_vector_socp_variable_integrated_MS"]
        q_all_socp_variable = data["q_all_socp_variable"]
        joint_frictions_socp_variable = data["joint_frictions_socp_variable"]
        motor_noises_socp_variable = data["motor_noises_socp_variable"]
        feedbacks_socp_variable = data["feedbacks_socp_variable"]
        time_vector_socp_variable = data["time_vector_socp_variable"]
        q_mean_socp_variable_integrated = data["q_mean_socp_variable_integrated"]
        q_mean_socp_variable = data["q_mean_socp_variable"]

with open(socp_variable_path_to_results.replace(".pkl", "_sol.pkl"), "rb") as file:
    sol_socp_variable = pickle.load(file)
print("SOCP VARIABLE cost: ", sol_socp_variable.cost)

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_VARIABLE_one : ", socp_variable_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_socp_variable, q_mean_socp_variable, result_folder, "SOCP_VARIABLE_one")

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_VARIABLE_all : ", socp_variable_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_all_socp_variable, q_all_socp_variable, result_folder, "SOCP_VARIABLE_all")


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

socp_feedforward_out_path_to_results = socp_feedforward_path_to_results.replace(".pkl", "_integrated.pkl")

DMS_ff_sensory_input_func = cas.Function(
    "DMS_fb_noised_sensory_input_no_eyes",
    [tf_sym, time_sym, Q_8, Qdot_8],
    [DMS_ff_sensory_input(socp_feedforward.nlp[0].model, tf_sym, time_sym, Q_8, Qdot_8, cas.MX.zeros())],
)

DMS_sensory_reference_no_eyes_func = cas.Function(
    "DMS_sensory_reference_no_eyes",
    [Q_8, Qdot_8],
    [DMS_sensory_reference_no_eyes(socp_feedforward.nlp[0].model, n_root, Q_8, Qdot_8, cas.MX.zeros())]
)

forward_dynamics_func = cas.Function("forward_dynamics", [Q_8, Qdot_8, Tau_8], [socp_feedforward.nlp[0].model.forward_dynamics()(Q_8, Qdot_8, cas.vertcat(cas.MX.zeros(3), Tau_8), cas.MX.zeros(), cas.MX.zeros())])

time_vector_socp_feedforward = np.linspace(0, float(time_socp_feedforward), n_shooting + 1)

q_socp_feedforward, qdot_socp_feedforward, q_mean_socp_feedforward, qdot_mean_socp_feedforward = define_q_mean(n_shooting, nb_random, q_roots_socp_feedforward, q_joints_socp_feedforward, qdot_roots_socp_feedforward, qdot_joints_socp_feedforward)

if not os.path.exists(socp_feedforward_out_path_to_results):
    (
        q_integrated,
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
    q_socp_feedforward_integrated[f"{nb_random}random"] = q_integrated

    q_socp_feedforward_integrated_MS, qdot_socp_feedforward_integrated_MS, time_vector_socp_feedforward_integrated_MS = integrate_socp_feedforward_MS(
        motor_noise_numerical_socp_feedforward,
        sensory_noise_numerical_socp_feedforward,
        n_shooting,
        nb_random,
        q_socp_feedforward,
        qdot_socp_feedforward,
        tau_joints_socp_feedforward,
        k_socp_feedforward,
        ref_fb_socp_feedforward,
        ref_ff_socp_feedforward,
        time_vector_socp_feedforward,
        socp_feedforward.nlp[0],
        DMS_sensory_reference_no_eyes_func,
        DMS_ff_sensory_input_func,
        forward_dynamics_func,
    )

    with open(socp_feedforward_out_path_to_results, "wb") as file:
        data = {
            "q_socp_feedforward_integrated": q_integrated,
            "qdot_socp_feedforward_integrated": qdot_socp_feedforward_integrated,
            "q_socp_feedforward_integrated_MS": q_socp_feedforward_integrated_MS,
            "qdot_socp_feedforward_integrated_MS": qdot_socp_feedforward_integrated_MS,
            "time_vector_socp_feedforward_integrated_MS": time_vector_socp_feedforward_integrated_MS,
            "q_all_socp_feedforward": q_all_socp_feedforward,
            "joint_frictions_socp_feedforward": joint_frictions_socp_feedforward,
            "motor_noises_socp_feedforward": motor_noises_socp_feedforward,
            "feedbacks_socp_feedforward": feedbacks_socp_feedforward,
            "feedforwards_socp_feedforward": feedforwards_socp_feedforward,
            "time_vector_socp_feedforward": time_vector_socp_feedforward,
            "q_mean_socp_feedforward_integrated": np.mean(q_integrated, axis=2),
            "q_mean_socp_feedforward": np.mean(q_socp_feedforward, axis=2),
        }
        pickle.dump(data, file)

else:
    with open(socp_feedforward_out_path_to_results, "rb") as file:
        data = pickle.load(file)
        q_socp_feedforward_integrated[f"{nb_random}random"] = data["q_socp_feedforward_integrated"]
        qdot_socp_feedforward_integrated = data["qdot_socp_feedforward_integrated"]
        q_socp_feedforward_integrated_MS = data["q_socp_feedforward_integrated_MS"]
        qdot_socp_feedforward_integrated_MS = data["qdot_socp_feedforward_integrated_MS"]
        time_vector_socp_feedforward_integrated_MS = data["time_vector_socp_feedforward_integrated_MS"]
        q_all_socp_feedforward = data["q_all_socp_feedforward"]
        joint_frictions_socp_feedforward = data["joint_frictions_socp_feedforward"]
        motor_noises_socp_feedforward = data["motor_noises_socp_feedforward"]
        feedbacks_socp_feedforward = data["feedbacks_socp_feedforward"]
        feedforwards_socp_feedforward = data["feedforwards_socp_feedforward"]
        time_vector_socp_feedforward = data["time_vector_socp_feedforward"]
        q_mean_socp_feedforward_integrated = data["q_mean_socp_feedforward_integrated"]
        q_mean_socp_feedforward = data["q_mean_socp_feedforward"]

with open(socp_feedforward_path_to_results.replace(".pkl", "_sol.pkl"), "rb") as file:
    sol_socp_feedforward = pickle.load(file)
print("SOCP FEEDFORWARD cost: ", sol_socp_feedforward.cost)

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_FEEDFORWARD_one : ", socp_feedforward_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_socp_feedforward, q_mean_socp_feedforward, result_folder, "SOCP_FEEDFORWARD_one")

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_FEEDFORWARD_all : ", socp_feedforward_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_all_socp_feedforward, q_all_socp_feedforward, result_folder, "SOCP_FEEDFORWARD_all")


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

socp_plus_out_path_to_results = socp_plus_path_to_results.replace(".pkl", "_integrated.pkl")

DMS_sensory_reference_no_eyes_func = cas.Function(
    "DMS_fb_sensory_reference", [Q_8, Qdot_8, ff_ref_sym], [DMS_sensory_reference_no_eyes(socp_plus.nlp[0].model, n_root, Q_8, Qdot_8, ff_ref_sym)]
)
DMS_ff_noised_sensory_input_func = cas.Function(
    "DMS_ff_sensory_reference", [tf_sym, time_sym, Q_8, Qdot_8, FF_SensoryNoise, ff_ref_sym], [DMS_ff_noised_sensory_input(socp_plus.nlp[0].model, tf_sym, time_sym, Q_8, Qdot_8, FF_SensoryNoise, ff_ref_sym)]
)

DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func = cas.Function(
    "DMS_fb_noised_sensory_input_VARIABLE_no_eyes",
    [Q_8, Qdot_8, SensoryNoise_8, ff_ref_sym],
    [DMS_fb_noised_sensory_input_VARIABLE_no_eyes(socp_plus.nlp[0].model, Q_8[:n_root], Q_8[n_root:], Qdot_8[:n_root], Qdot_8[n_root:], SensoryNoise_8, ff_ref_sym)],
)

forward_dynamics_func = cas.Function("forward_dynamics", [Q_8, Qdot_8, Tau_8], [socp_plus.nlp[0].model.forward_dynamics()(Q_8, Qdot_8, cas.vertcat(cas.MX.zeros(3), Tau_8), cas.MX.zeros(), cas.MX.zeros())])

time_vector_socp_plus = np.linspace(0, float(time_socp_plus), n_shooting + 1)

q_socp_plus, qdot_socp_plus, q_mean_socp_plus, qdot_mean_socp_plus = define_q_mean(n_shooting, nb_random, q_roots_socp_plus, q_joints_socp_plus, qdot_roots_socp_plus, qdot_joints_socp_plus)

if not os.path.exists(socp_plus_out_path_to_results):
    (
        q_integrated,
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
    q_socp_plus_integrated[f"{nb_random}random"] = q_integrated

    q_socp_plus_integrated_MS, qdot_socp_plus_integrated_MS, time_vector_socp_plus_integrated_MS = integrate_socp_plus_MS(
        motor_noise_numerical_socp_plus,
        sensory_noise_numerical_socp_plus,
        n_shooting,
        nb_random,
        q_socp_plus,
        qdot_socp_plus,
        tau_joints_socp_plus,
        k_socp_plus,
        ref_fb_socp_plus,
        ref_ff_socp_plus,
        time_vector_socp_plus,
        socp_plus.nlp[0],
        DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func,
        DMS_ff_noised_sensory_input_func,
        forward_dynamics_func,
    )

    with open(socp_plus_out_path_to_results, "wb") as file:
        data = {
            "q_socp_plus_integrated": q_integrated,
            "qdot_socp_plus_integrated": qdot_socp_plus_integrated,
            "q_socp_plus_integrated_MS": q_socp_plus_integrated_MS,
            "qdot_socp_plus_integrated_MS": qdot_socp_plus_integrated_MS,
            "time_vector_socp_plus_integrated_MS": time_vector_socp_plus_integrated_MS,
            "q_all_socp_plus": q_all_socp_plus,
            "joint_frictions_socp_plus": joint_frictions_socp_plus,
            "motor_noises_socp_plus": motor_noises_socp_plus,
            "feedbacks_socp_plus": feedbacks_socp_plus,
            "feedforwards_socp_plus": feedforwards_socp_plus,
            "time_vector_socp_plus": time_vector_socp_plus,
            "q_mean_socp_plus_integrated": np.mean(q_integrated, axis=2),
            "q_mean_socp_plus": np.mean(q_socp_plus, axis=2),
        }
        pickle.dump(data, file)

else:
    with open(socp_plus_out_path_to_results, "rb") as file:
        data = pickle.load(file)
        q_socp_plus_integrated[f"{nb_random}random"] = data["q_socp_plus_integrated"]
        qdot_socp_plus_integrated = data["qdot_socp_plus_integrated"]
        q_socp_plus_integrated_MS = data["q_socp_plus_integrated_MS"]
        qdot_socp_plus_integrated_MS = data["qdot_socp_plus_integrated_MS"]
        time_vector_socp_plus_integrated_MS = data["time_vector_socp_plus_integrated_MS"]
        q_all_socp_plus = data["q_all_socp_plus"]
        joint_frictions_socp_plus = data["joint_frictions_socp_plus"]
        motor_noises_socp_plus = data["motor_noises_socp_plus"]
        feedbacks_socp_plus = data["feedbacks_socp_plus"]
        feedforwards_socp_plus = data["feedforwards_socp_plus"]
        time_vector_socp_plus = data["time_vector_socp_plus"]
        q_mean_socp_plus_integrated = data["q_mean_socp_plus_integrated"]
        q_mean_socp_plus = data["q_mean_socp_plus"]

with open(socp_plus_path_to_results.replace(".pkl", "_sol.pkl"), "rb") as file:
    sol_socp_plus = pickle.load(file)
print("SOCP+ cost: ", sol_socp_plus.cost)

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_plus_one : ", socp_plus_path_to_results)
    bioviz_animate(biorbd_model_path_vision_with_mesh, q_mean_socp_plus, result_folder, "SOCP_plus_one")

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_plus_all : ", socp_plus_path_to_results)
    bioviz_animate(biorbd_model_path_vision_with_mesh_all, q_all_socp_plus, result_folder, "SOCP_plus_all")


# Comparison ----------------------------------------------------------------------------------------------------------
q_mean_comparison = np.zeros((7 + 7 + 7 + 8 + 8, n_shooting + 1))
q_mean_comparison[:7, :] = q_ocp
q_mean_comparison[7:7+7, :] = q_mean_socp
q_mean_comparison[7+7: 7+7+7, :] = q_mean_socp_variable
q_mean_comparison[7+7+7: 7+7+7+8, :] = q_mean_socp_feedforward
q_mean_comparison[7+7+7+8:, :] = q_mean_socp_plus
if FLAG_GENERATE_VIDEOS:
    print("Generating comparison")
    bioviz_animate(biorbd_model_path_comparison, q_mean_comparison, result_folder, "Comparison")


# Plots ---------------------------------------------------------------------------------------------------------------
normalized_time_vector = np.linspace(0, 1, n_shooting + 1)
normalized_time_vector_MS = np.linspace(0, 1, time_vector_ocp_integrated_MS.shape[0])

# Landing variability functions
CoM_y_fcn = cas.Function("CoM_y", [Q], [socp.nlp[0].model.model.CoM(Q).to_mx()[1]])
CoM_y_8_fcn = cas.Function("CoM_y", [Q_8], [socp_plus.nlp[0].model.model.CoM(Q_8).to_mx()[1]])
CoM_y_dot_fcn = cas.Function("CoM_y_dot", [Q, Qdot], [socp.nlp[0].model.model.CoMdot(Q, Qdot).to_mx()[1]])
CoM_dot_8_fcn = cas.Function("CoM_dot", [Q_8, Qdot_8],
                             [socp_plus.nlp[0].model.model.CoMdot(Q_8, Qdot_8).to_mx()[1]])
inertia_fcn = cas.Function(
    "inertia", [Q], [socp.nlp[0].model.model.bodyInertia(Q).to_mx()[0]]
)
inertia_8_fcn = cas.Function(
    "inertia", [Q_8], [socp_plus.nlp[0].model.model.bodyInertia(Q_8).to_mx()[0]]
)
ang_mom_fcn = cas.Function(
    "ang_mom", [Q, Qdot], [socp.nlp[0].model.model.angularMomentum(Q, Qdot).to_mx()[0]]
)
ang_mom_8_fcn = cas.Function(
    "ang_mom", [Q_8, Qdot_8], [socp_plus.nlp[0].model.model.angularMomentum(Q_8, Qdot_8).to_mx()[0]]
)
BodyVelocity_fcn = cas.Function(
    "BodyVelocity", [Q, Qdot], [socp.nlp[0].model.model.bodyAngularVelocity(Q, Qdot).to_mx()[0] * 180 / np.pi]
)
BodyVelocity_8_fcn = cas.Function(
    "BodyVelocity", [Q_8, Qdot_8], [socp_plus.nlp[0].model.model.bodyAngularVelocity(Q_8, Qdot_8).to_mx()[0] * 180 / np.pi]
)

# Perform plots
plot_motor_command(nb_random,
                    normalized_time_vector,
                    time_vector_ocp,
                    time_vector_socp,
                    time_vector_socp_variable,
                    time_vector_socp_feedforward,
                    time_vector_socp_plus,
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
                    SOCP_PLUS_color,
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
                        SOCP_PLUS_color,
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

plot_movement_duration(time_ocp, time_socp, time_socp_variable, time_socp_feedforward, time_socp_plus, OCP_color, SOCP_color, SOCP_VARIABLE_color, SOCP_FEEDFORWARD_color, SOCP_PLUS_color)

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
        SOCP_PLUS_color)

head_idx = socp_variable.nlp[0].model.segment_index("Head")
head_velocity_fcn = cas.Function("head_velocity", [Q, Qdot], [socp_variable.nlp[0].model.segment_angular_velocity(head_idx)(Q, Qdot, [])[0]])
head_idx_8 = socp_plus.nlp[0].model.segment_index("Head")
head_velocity_fcn_8 = cas.Function("head_velocity", [Q_8, Qdot_8], [socp_plus.nlp[0].model.segment_angular_velocity(head_idx_8)(Q_8, Qdot_8, [])[0]])

floor_normal_vector = cas.MX.zeros(3, 1)
floor_normal_vector[2] = 1
eyes_vect_start_8 = socp_plus.nlp[0].model.marker(socp_plus.nlp[0].model.marker_index("eyes_vect_start"))(Q_8, cas.MX.zeros())
eyes_vect_end_8 = socp_plus.nlp[0].model.marker(socp_plus.nlp[0].model.marker_index("eyes_vect_end"))(Q_8, cas.MX.zeros())
gaze_vector_8 = eyes_vect_end_8 - eyes_vect_start_8
angle_8 = cas.acos(
    cas.dot(gaze_vector_8, floor_normal_vector) / (cas.norm_fro(gaze_vector_8) * cas.norm_fro(floor_normal_vector))
)
eye_orientation_fcn_8 = cas.Function("eye_orientation", [Q_8, Qdot_8], [angle_8])

plot_gains_per_dof(
    normalized_time_vector,
    normalized_time_vector_MS,
    socp,
    socp_variable,
    socp_feedforward,
    socp_plus,
    k_socp,
    k_socp_variable,
    k_socp_feedforward,
    k_socp_plus,
    q_ocp_integrated_MS,
    np.mean(q_socp_integrated_MS, axis=1),
    np.mean(q_socp_variable_integrated_MS, axis=1),
    np.mean(q_socp_feedforward_integrated_MS, axis=1),
    np.mean(q_socp_plus_integrated_MS, axis=1),
    qdot_ocp_integrated_MS,
    np.mean(qdot_socp_integrated_MS, axis=1),
    np.mean(qdot_socp_variable_integrated_MS, axis=1),
    np.mean(qdot_socp_feedforward_integrated_MS, axis=1),
    np.mean(qdot_socp_plus_integrated_MS, axis=1),
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_PLUS_color,
    head_velocity_fcn,
    head_velocity_fcn_8,
    eye_orientation_fcn_8,
)

plot_landing_variability(
    CoM_y_fcn,
    CoM_y_8_fcn,
    CoM_y_dot_fcn,
    CoM_dot_8_fcn,
    BodyVelocity_fcn,
    BodyVelocity_8_fcn,
    nb_random,
    q_ocp_integrated,
    qdot_ocp_integrated,
    q_socp,
    qdot_socp,
    q_socp_variable,
    qdot_socp_variable,
    q_socp_feedforward,
    qdot_socp_feedforward,
    q_socp_plus,
    qdot_socp_plus,
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_PLUS_color,
)

plot_inertia_ang_mom(
    normalized_time_vector,
    inertia_fcn,
    inertia_8_fcn,
    ang_mom_fcn,
    ang_mom_8_fcn,
    BodyVelocity_fcn,
    BodyVelocity_8_fcn,
    nb_random,
    q_ocp_integrated,
    qdot_ocp_integrated,
    q_socp,
    qdot_socp,
    q_socp_variable,
    qdot_socp_variable,
    q_socp_feedforward,
    qdot_socp_feedforward,
    q_socp_plus,
    qdot_socp_plus,
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_PLUS_color,
)

plot_kinematics(
    normalized_time_vector,
    normalized_time_vector_MS,
    q_ocp_integrated_MS,
    q_socp_integrated_MS,
    q_socp_variable_integrated_MS,
    q_socp_feedforward_integrated_MS,
    q_socp_plus_integrated_MS,
    q_ocp,
    q_socp,
    q_socp_variable,
    q_socp_feedforward,
    q_socp_plus,
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_PLUS_color,
)

plot_comparison_reintegration(
    normalized_time_vector,
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
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_PLUS_color,
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
    time_vector_ocp,
    time_vector_socp,
    time_vector_socp_variable,
    time_vector_socp_feedforward,
    time_vector_socp_plus,
    OCP_color,
    SOCP_color,
    SOCP_VARIABLE_color,
    SOCP_FEEDFORWARD_color,
    SOCP_PLUS_color,
)

# plot_comparison_nb_random(q_ocp_integrated,
#                         q_socp_integrated,
#                         q_socp_variable_integrated,
#                         q_socp_feedforward_integrated,
#                         q_socp_plus_integrated,
#                         OCP_color,
#                         SOCP_color,
#                         SOCP_VARIABLE_color,
#                         SOCP_FEEDFORWARD_color,
#                         SOCP_PLUS_color)

# plot_comparison_kinematics_nb_random(
#     q_ocp,
#     time_vector_ocp,
# )

plt.show()
