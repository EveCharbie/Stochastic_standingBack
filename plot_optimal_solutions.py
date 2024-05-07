import pickle

import bioviz
import casadi as cas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from DMS_SOCP import prepare_socp
from DMS_SOCP_VARIABLE_FEEDFORWARD import prepare_socp_VARIABLE_FEEDFORWARD
from DMS_deterministic import prepare_ocp
from bioptim import StochasticBioModel, Solution, SolutionMerge
from utils import (
    DMS_sensory_reference,
    motor_acuity,
    DMS_fb_noised_sensory_input_VARIABLE_no_eyes,
    DMS_ff_noised_sensory_input,
    visual_noise,
    gaussian_function,
    DMS_sensory_reference_no_eyes,
)


def plot_CoM(states_integrated, model, n_shooting, name, nb_random=30):
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

    plt.suptitle(f"CoM and Pelvis for {name}")
    plt.savefig(f"{save_path}_CoM.png")
    plt.show()


def OCP_dynamics(q, qdot, tau, motor_noise_numerical, ocp):

    nb_root = 3
    nb_q = q.shape[0]

    dxdt = cas.MX.zeros((2 * nb_q, 1))
    dxdt[:nb_q, :] = qdot

    # Joint friction
    tau_this_time = tau - ocp.nlp[0].model.friction_coefficients @ qdot[nb_root:]

    # Motor noise
    tau_this_time += motor_noise_numerical

    tau_this_time = cas.vertcat(cas.MX.zeros(nb_root), tau_this_time)

    dxdt[nb_q:] = ocp.nlp[0].model.forward_dynamics(q, qdot, tau_this_time)

    return dxdt


def SOCP_dynamics(nb_random, q, qdot, tau, k_matrix, ref, motor_noise_numerical, sensory_noise_numerical, socp):

    nb_root = 3
    nb_q = 7
    nb_joints = nb_q - nb_root

    dxdt = cas.MX.zeros((2 * nb_q, nb_random))
    dxdt[:nb_q, :] = qdot
    joint_friction = cas.MX.zeros(nb_joints, nb_random)
    motor_noise = cas.MX.zeros(nb_joints, nb_random)
    feedback = cas.MX.zeros(nb_joints, nb_random)
    for i_random in range(nb_random):

        q_this_time = q[:, i_random]
        qdot_this_time = qdot[:, i_random]

        tau_this_time = tau[:]

        # Joint friction
        joint_friction[:, i_random] = socp.nlp[0].model.friction_coefficients @ qdot_this_time[nb_root:]
        tau_this_time -= joint_friction[:, i_random]

        # Motor noise
        motor_noise[:, i_random] = motor_noise_numerical[:, i_random]
        tau_this_time += motor_noise[:, i_random]

        # Feedback
        feedback[:, i_random] = k_matrix @ (
            ref
            - DMS_sensory_reference(socp.nlp[0].model, nb_root, q_this_time, qdot_this_time)
            + sensory_noise_numerical[:, i_random]
        )
        tau_this_time += feedback[:, i_random]

        tau_this_time = cas.vertcat(cas.MX.zeros(nb_root), tau_this_time)

        dxdt[nb_q:, i_random] = socp.nlp[0].model.forward_dynamics(q_this_time, qdot_this_time, tau_this_time)

    return dxdt, joint_friction, motor_noise, feedback


def SOCP_PLUS_dynamics(
    nb_random,
    q,
    qdot,
    tau,
    k_fb,
    k_ff,
    fb_ref,
    ff_ref,
    tf,
    time,
    motor_noise_numerical,
    sensory_noise_numerical,
    socp_plus,
):

    nb_root = 3
    nb_q = 8
    nb_joints = nb_q - nb_root

    dxdt = cas.MX.zeros((2 * nb_q, nb_random))
    dxdt[:nb_q, :] = qdot
    joint_friction = cas.MX.zeros(nb_joints, nb_random)
    motor_noise = cas.MX.zeros(nb_joints, nb_random)
    feedback = cas.MX.zeros(nb_joints, nb_random)
    feedforward = cas.MX.zeros(nb_joints, nb_random)
    for i_random in range(nb_random):

        q_this_time = q[:, i_random]
        qdot_this_time = qdot[:, i_random]

        tau_this_time = tau[:]

        # Joint friction
        joint_friction[:, i_random] = socp_plus.nlp[0].model.friction_coefficients @ qdot_this_time[nb_root:]
        tau_this_time -= joint_friction[:, i_random]

        # Motor noise
        motor_noise[:, i_random] = motor_acuity(motor_noise_numerical[:, i_random], tau)
        motor_noise[1, i_random] = 0
        tau_this_time += motor_noise[:, i_random]

        # Feedback
        feedback[:, i_random] = k_fb @ (
            fb_ref
            - DMS_fb_noised_sensory_input_VARIABLE_no_eyes(
                socp_plus.nlp[0].model,
                q_this_time[:nb_root],
                q_this_time[nb_root:],
                qdot_this_time[:nb_root],
                qdot_this_time[nb_root:],
                sensory_noise_numerical[: socp_plus.nlp[0].model.n_feedbacks, i_random],
            )
        )
        tau_this_time += feedback[:, i_random]

        # Feedforward
        feedforward[:, i_random] = k_ff @ (
            ff_ref
            - DMS_ff_noised_sensory_input(
                socp_plus.nlp[0].model,
                tf,
                time,
                q_this_time,
                qdot_this_time,
                sensory_noise_numerical[socp_plus.nlp[0].model.n_feedbacks :, i_random],
            )
        )
        tau_this_time += feedforward[:, i_random]

        tau_this_time = cas.vertcat(cas.MX.zeros(nb_root), tau_this_time)

        dxdt[nb_q:, i_random] = socp_plus.nlp[0].model.forward_dynamics(q_this_time, qdot_this_time, tau_this_time)

    return dxdt, joint_friction, motor_noise, feedback, feedforward


def RK4_OCP(q, qdot, tau, dt, motor_noise_numerical, dyn_fun):

    nb_q = q.shape[0]
    states = np.zeros((2 * nb_q, 6))
    states[:nb_q, 0] = np.reshape(q, (-1,))
    states[nb_q:, 0] = np.reshape(qdot, (-1,))
    h = dt / 5
    for i in range(1, 6):
        k1 = dyn_fun(
            states[:nb_q, i - 1],
            states[nb_q:, i - 1],
            tau,
            motor_noise_numerical,
        )
        k2 = dyn_fun(
            states[:nb_q, i - 1] + h / 2 * k1[:nb_q],
            states[nb_q:, i - 1] + h / 2 * k1[nb_q:],
            tau,
            motor_noise_numerical,
        )
        k3 = dyn_fun(
            states[:nb_q, i - 1] + h / 2 * k2[:nb_q],
            states[nb_q:, i - 1] + h / 2 * k2[nb_q:],
            tau,
            motor_noise_numerical,
        )
        k4 = dyn_fun(
            states[:nb_q, i - 1] + h * k3[:nb_q],
            states[nb_q:, i - 1] + h * k3[nb_q:],
            tau,
            motor_noise_numerical,
        )
        states[:, i] = np.reshape(states[:, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (2 * nb_q,))
    return states[:, -1]


def RK4_SOCP(q, qdot, tau, dt, k_matrix, ref, motor_noise_numerical, sensory_noise_numerical, n_random, dyn_fun):

    _, joint_friction_out, motor_noise_out, feedback_out = dyn_fun(
        q, qdot, tau, k_matrix, ref, motor_noise_numerical, sensory_noise_numerical
    )

    nb_q = q.shape[0]
    states = np.zeros((2 * nb_q, n_random, 6))
    states[:nb_q, :, 0] = q
    states[nb_q:, :, 0] = qdot
    h = dt / 5
    for i in range(1, 6):
        k1, _, _, _ = dyn_fun(
            states[:nb_q, :, i - 1],
            states[nb_q:, :, i - 1],
            tau,
            k_matrix,
            ref,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        k2, _, _, _ = dyn_fun(
            states[:nb_q, :, i - 1] + h / 2 * k1[:nb_q, :],
            states[nb_q:, :, i - 1] + h / 2 * k1[nb_q:, :],
            tau,
            k_matrix,
            ref,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        k3, _, _, _ = dyn_fun(
            states[:nb_q, :, i - 1] + h / 2 * k2[:nb_q, :],
            states[nb_q:, :, i - 1] + h / 2 * k2[nb_q:, :],
            tau,
            k_matrix,
            ref,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        k4, _, _, _ = dyn_fun(
            states[:nb_q, :, i - 1] + h * k3[:nb_q, :],
            states[nb_q:, :, i - 1] + h * k3[nb_q:, :],
            tau,
            k_matrix,
            ref,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        states[:, :, i] = states[:, :, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return states[:, :, -1], joint_friction_out, motor_noise_out, feedback_out


def RK4_SOCP_PLUS(
    q,
    qdot,
    tau,
    dt,
    k_fb_matrix,
    k_ff_matrix,
    ref_fb,
    ref_ff,
    motor_noise_numerical,
    sensory_noise_numerical,
    n_random,
    dyn_fun,
    node_idx,
):

    _, joint_friction_out, motor_noise_out, feedback_out, feedforward_out = dyn_fun(
        q,
        qdot,
        tau,
        k_fb_matrix,
        k_ff_matrix,
        ref_fb,
        ref_ff,
        node_idx * dt,
        motor_noise_numerical,
        sensory_noise_numerical,
    )

    nb_q = q.shape[0]
    states = np.zeros((2 * nb_q, n_random, 6))
    states[:nb_q, :, 0] = q
    states[nb_q:, :, 0] = qdot
    h = dt / 5
    for i in range(1, 6):
        k1, _, _, _, _ = dyn_fun(
            states[:nb_q, :, i - 1],
            states[nb_q:, :, i - 1],
            tau,
            k_fb_matrix,
            k_ff_matrix,
            ref_fb,
            ref_ff,
            node_idx * dt + h * (i - 1),
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        k2, _, _, _, _ = dyn_fun(
            states[:nb_q, :, i - 1] + h / 2 * k1[:nb_q, :],
            states[nb_q:, :, i - 1] + h / 2 * k1[nb_q:, :],
            tau,
            k_fb_matrix,
            k_ff_matrix,
            ref_fb,
            ref_ff,
            node_idx * dt + h * (i - 1) + h / 2,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        k3, _, _, _, _ = dyn_fun(
            states[:nb_q, :, i - 1] + h / 2 * k2[:nb_q, :],
            states[nb_q:, :, i - 1] + h / 2 * k2[nb_q:, :],
            tau,
            k_fb_matrix,
            k_ff_matrix,
            ref_fb,
            ref_ff,
            node_idx * dt + h * (i - 1) + h / 2,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        k4, _, _, _, _ = dyn_fun(
            states[:nb_q, :, i - 1] + h * k3[:nb_q, :],
            states[nb_q:, :, i - 1] + h * k3[nb_q:, :],
            tau,
            k_fb_matrix,
            k_ff_matrix,
            ref_fb,
            ref_ff,
            node_idx * dt + h * (i - 1) + h,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        states[:, :, i] = states[:, :, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return states[:, :, -1], joint_friction_out, motor_noise_out, feedback_out, feedforward_out


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


def noisy_integrate(time_vector, q, qdot, tau_joints, n_shooting, motor_noise_magnitude, dyn_fun, nb_random):

    n_q = 7

    dt_last = time_vector[1] - time_vector[0]
    q_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
    q_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))

    # initial variability
    np.random.seed(0)
    initial_cov = np.eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P
    noised_states = np.random.multivariate_normal(np.zeros((n_q * 2,)), initial_cov, nb_random).T

    motor_noise_numerical = np.zeros((n_joints, nb_random, n_shooting + 1))
    for i_random in range(nb_random):
        q_integrated[:, 0, i_random] = np.reshape(q[:, 0] + noised_states[:n_q, i_random], (-1,))
        qdot_integrated[:, 0, i_random] = np.reshape(qdot[:, 0] + noised_states[n_q:, i_random], (-1,))
        q_multiple_shooting[:, 0, i_random] = np.reshape(q[:, 0] + noised_states[:n_q, i_random], (-1,))
        qdot_multiple_shooting[:, 0, i_random] = np.reshape(qdot[:, 0] + noised_states[n_q:, i_random], (-1,))
        for i_shooting in range(n_shooting):
            motor_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                loc=np.zeros(motor_noise_magnitude.shape[0]),
                scale=np.reshape(np.array(motor_noise_magnitude), (n_joints,)),
                size=n_joints,
            )

    for i_random in range(nb_random):
        for i_shooting in range(n_shooting):
            states_integrated = RK4_OCP(
                q_integrated[:, i_shooting, i_random],
                qdot_integrated[:, i_shooting, i_random],
                tau_joints[:, i_shooting],
                dt_last,
                motor_noise_numerical[:, i_random, i_shooting],
                dyn_fun,
            )
            q_integrated[:, i_shooting + 1, i_random] = states_integrated[:n_q]
            qdot_integrated[:, i_shooting + 1, i_random] = states_integrated[n_q:]

            states_integrated_multiple = RK4_OCP(
                q[:, i_shooting],
                qdot[:, i_shooting],
                tau_joints[:, i_shooting],
                dt_last,
                motor_noise_numerical[:, i_random, i_shooting],
                dyn_fun,
            )
            q_multiple_shooting[:, i_shooting + 1, i_random] = states_integrated_multiple[:n_q]
            qdot_multiple_shooting[:, i_shooting + 1, i_random] = states_integrated_multiple[n_q:]

    return q_integrated, qdot_integrated, q_multiple_shooting, qdot_multiple_shooting, motor_noise_numerical


def integrate_socp(
    time_vector,
    q_socp,
    qdot_socp,
    tau_joints_socp,
    k_socp,
    ref_socp,
    motor_noise_numerical,
    sensory_noise_numerical,
    dyn_fun,
    socp,
):

    dt_socp = time_vector[1] - time_vector[0]
    n_q = q_socp.shape[0]
    n_shooting = q_socp.shape[1] - 1
    nb_random = q_socp.shape[2]

    # single shooting
    q_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
    q_integrated[:, 0, :] = q_socp[:, 0, :]
    qdot_integrated[:, 0, :] = qdot_socp[:, 0, :]

    # multiple shooting
    q_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))
    q_multiple_shooting[:, 0, :] = q_socp[:, 0, :]
    qdot_multiple_shooting[:, 0, :] = qdot_socp[:, 0, :]

    joint_frictions = np.zeros((n_q - 3, nb_random, n_shooting))
    motor_noises = np.zeros((n_q - 3, nb_random, n_shooting))
    feedbacks = np.zeros((n_q - 3, nb_random, n_shooting))

    for i_shooting in range(n_shooting):
        k_matrix = StochasticBioModel.reshape_to_matrix(k_socp[:, i_shooting], socp.nlp[0].model.matrix_shape_k)
        states_integrated, joint_friction_out, motor_noise_out, feedback_out = RK4_SOCP(
            q_integrated[:, i_shooting, :],
            qdot_integrated[:, i_shooting, :],
            tau_joints_socp[:, i_shooting],
            dt_socp,
            k_matrix,
            ref_socp[:, i_shooting],
            motor_noise_numerical[:, :, i_shooting],
            sensory_noise_numerical[:, :, i_shooting],
            nb_random,
            dyn_fun,
        )
        q_integrated[:, i_shooting + 1, :] = states_integrated[:n_q, :]
        qdot_integrated[:, i_shooting + 1, :] = states_integrated[n_q:, :]

        if joint_friction_out.shape != (0, 0):
            joint_frictions[:, :, i_shooting] = joint_friction_out
            motor_noises[:, :, i_shooting] = motor_noise_out
            feedbacks[:, :, i_shooting] = feedback_out

        states_integrated_multiple, _, _, _ = RK4_SOCP(
            q_socp[:, i_shooting, :],
            qdot_socp[:, i_shooting, :],
            tau_joints_socp[:, i_shooting],
            dt_socp,
            k_matrix,
            ref_socp[:, i_shooting],
            motor_noise_numerical[:, :, i_shooting],
            sensory_noise_numerical[:, :, i_shooting],
            nb_random,
            dyn_fun,
        )
        q_multiple_shooting[:, i_shooting + 1, :] = states_integrated_multiple[:n_q, :]
        qdot_multiple_shooting[:, i_shooting + 1, :] = states_integrated_multiple[n_q:, :]

    return (
        q_integrated,
        qdot_integrated,
        q_multiple_shooting,
        qdot_multiple_shooting,
        joint_frictions,
        motor_noises,
        feedbacks,
    )


def integrate_socp_plus(
    time_vector,
    q_socp_plus,
    qdot_socp_plus,
    tau_joints_socp_plus,
    k_socp_plus,
    ref_fb_socp_plus,
    ref_ff_socp_plus,
    motor_noise_numerical,
    sensory_noise_numerical,
    dyn_fun,
    socp,
):

    dt_socp = time_vector[1] - time_vector[0]
    n_q = q_socp_plus.shape[0]
    n_shooting = q_socp_plus.shape[1] - 1
    nb_random = q_socp_plus.shape[2]

    # single shooting
    q_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
    q_integrated[:, 0, :] = q_socp_plus[:, 0, :]
    qdot_integrated[:, 0, :] = qdot_socp_plus[:, 0, :]

    # multiple shooting
    q_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))
    q_multiple_shooting[:, 0, :] = q_socp_plus[:, 0, :]
    qdot_multiple_shooting[:, 0, :] = qdot_socp_plus[:, 0, :]

    joint_frictions = np.zeros((n_q - 3, nb_random, n_shooting))
    motor_noises = np.zeros((n_q - 3, nb_random, n_shooting))
    feedbacks = np.zeros((n_q - 3, nb_random, n_shooting))
    feedforwards = np.zeros((n_q - 3, nb_random, n_shooting))
    for i_shooting in range(n_shooting):
        k_matrix = StochasticBioModel.reshape_to_matrix(k_socp_plus[:, i_shooting], socp.nlp[0].model.matrix_shape_k)
        k_fb_matrix = k_matrix[:, :-1]
        k_ff_matrix = k_matrix[:, -1]
        states_integrated, joint_friction_out, motor_noise_out, feedback_out, feedforward_out = RK4_SOCP_PLUS(
            q_integrated[:, i_shooting, :],
            qdot_integrated[:, i_shooting, :],
            tau_joints_socp_plus[:, i_shooting],
            dt_socp,
            k_fb_matrix,
            k_ff_matrix,
            ref_fb_socp_plus[:, i_shooting],
            ref_ff_socp_plus,
            motor_noise_numerical[:, :, i_shooting],
            sensory_noise_numerical[:, :, i_shooting],
            nb_random,
            dyn_fun,
            i_shooting,
        )

        q_integrated[:, i_shooting + 1, :] = states_integrated[:n_q, :]
        qdot_integrated[:, i_shooting + 1, :] = states_integrated[n_q:, :]

        if joint_friction_out.shape != (0, 0):
            joint_frictions[:, :, i_shooting] = joint_friction_out
            motor_noises[:, :, i_shooting] = motor_noise_out
            feedbacks[:, :, i_shooting] = feedback_out
            feedforwards[:, :, i_shooting] = feedforward_out

        states_integrated_multiple, _, _, _, _ = RK4_SOCP_PLUS(
            q_socp_plus[:, i_shooting, :],
            qdot_socp_plus[:, i_shooting, :],
            tau_joints_socp_plus[:, i_shooting],
            dt_socp,
            k_fb_matrix,
            k_ff_matrix,
            ref_fb_socp_plus[:, i_shooting],
            ref_ff_socp_plus,
            motor_noise_numerical[:, :, i_shooting],
            sensory_noise_numerical[:, :, i_shooting],
            nb_random,
            dyn_fun,
            i_shooting,
        )
        q_multiple_shooting[:, i_shooting + 1, :] = states_integrated_multiple[:n_q, :]
        qdot_multiple_shooting[:, i_shooting + 1, :] = states_integrated_multiple[n_q:, :]

    return (
        q_integrated,
        qdot_integrated,
        q_multiple_shooting,
        qdot_multiple_shooting,
        joint_frictions,
        motor_noises,
        feedbacks,
        feedforwards,
    )

def plot_comparison_reintegration(q_ocp_nominal, q_socp_nominal, q_socp_plus_nominal,
                                  q_all_socp, q_all_socp_plus,
                                  q_ocp_reintegrated, q_socp_reintegrated, q_socp_plus_reintegrated,
                                  time_vector_ocp, time_vector_socp, time_vector_socp_plus,
                                  OCP_color, SOCP_color, SOCP_plus_color,
                                  nb_random, nb_reintegrations):

    n_q = q_socp_plus_nominal.shape[0]
    fig, axs = plt.subplots(n_q, 3, figsize=(15, 10))
    for i_dof in range(n_q):

        # Reintegrated
        for i_random in range(nb_random*nb_reintegrations):
            if i_dof < 4:
                axs[i_dof, 0].plot(time_vector_ocp, q_ocp_reintegrated[i_dof, :, i_random], color=OCP_color, label="OCP reintegration")
                axs[i_dof, 1].plot(time_vector_socp, q_socp_reintegrated[i_dof, :, i_random], color=SOCP_color,
                                   label="SOCP reintegration")
            elif i_dof > 4:
                axs[i_dof, 0].plot(time_vector_ocp, q_ocp_reintegrated[i_dof-1, :, i_random], color=OCP_color)
                axs[i_dof, 1].plot(time_vector_socp, q_socp_reintegrated[i_dof - 1, :, i_random], color=SOCP_color)
            axs[i_dof, 2].plot(time_vector_socp_plus, q_socp_plus_reintegrated[i_dof, :, i_random], color=SOCP_plus_color,
                               label="SOCP+ reintegration")

        # Optimzation variables
        for i_random in range(nb_random):
            if i_dof < 4:
                axs[i_dof, 1].plot(time_vector_socp, q_all_socp[i_dof, :, i_random], color=SOCP_color, label="SOCP 15 models")
            elif i_dof > 4:
                axs[i_dof, 1].plot(time_vector_socp, q_all_socp[i_dof-1, :, i_random], color=SOCP_color)
            axs[i_dof, 2].plot(time_vector_socp_plus, q_all_socp_plus[i_dof, :, i_random], color=SOCP_plus_color, label="SOCP+ 15 models")

        # Nominal
        if i_dof < 4:
            axs[i_dof, 0].plot(time_vector_ocp, q_ocp_nominal[i_dof, :], color=OCP_color, label="OCP", linewidth=2)
            axs[i_dof, 1].plot(time_vector_socp, q_socp_nominal[i_dof, :], color=SOCP_color, label="SOCP nominal", linewidth=2)
        elif i_dof > 4:
            axs[i_dof, 0].plot(time_vector_ocp, q_ocp_nominal[i_dof-1, :], color=OCP_color, linewidth=2)
            axs[i_dof, 1].plot(time_vector_socp, q_socp_nominal[i_dof-1, :], color=SOCP_color, linewidth=2)
        axs[i_dof, 2].plot(time_vector_socp_plus, q_socp_plus_nominal[i_dof, :], color=SOCP_plus_color, label="SOCP+ nominal", linewidth=2)

    axs[0, 0].legend()
    plt.suptitle("Comparison of nominal, integrated and reintegrated solutions")
    plt.savefig(f"graphs/comparison_reintegration.png")
    plt.show()

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



FLAG_GENERATE_VIDEOS = False

OCP_color = "#5dc962"
SOCP_color = "#ac2594"
SOCP_plus_color = "#06b0f0"
model_name = "Model2D_7Dof_0C_3M"
biorbd_model_path = f"models/{model_name}.bioMod"
biorbd_model_path_with_mesh_ocp = f"models/{model_name}_with_mesh_ocp.bioMod"
biorbd_model_path_with_mesh_socp = f"models/{model_name}_with_mesh_socp.bioMod"
biorbd_model_path_with_mesh_all = f"models/{model_name}_with_mesh_all.bioMod"
biorbd_model_path_with_mesh_all_socp = f"models/{model_name}_with_mesh_all_socp.bioMod"

biorbd_model_path_vision = f"models/{model_name}_vision.bioMod"
biorbd_model_path_vision_with_mesh = f"models/{model_name}_vision_with_mesh.bioMod"
biorbd_model_path_vision_with_mesh_all = f"models/{model_name}_vision_with_mesh_all.bioMod"
biorbd_model_path_comparison = f"models/{model_name}_comparison.bioMod"


result_folder = "good"
ocp_path_to_results = f"results/{result_folder}/{model_name}_ocp_DMS_CVG_1e-8.pkl"
socp_path_to_results = f"results/{result_folder}/Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02_DMS_15random_CVG_1p0e-06.pkl"
socp_plus_path_to_results = (
    f"results/{result_folder}/Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02_VARIABLE_FEEDFORWARD_DMS_15random_CVG_1p0e-06.pkl"
)

n_q = 7
n_root = 3
n_joints = n_q - n_root
n_ref = 2 * n_joints + 2

dt = 0.05
final_time = 0.8
n_shooting = int(final_time / dt)
tol = 1e-6
nb_random = 15
nb_reintegrations = 10

motor_noise_std = 0.05 * 10
wPq_std = 0.001 * 5
wPqdot_std = 0.003 * 5
motor_noise_magnitude = cas.DM(np.array([motor_noise_std**2 / dt for _ in range(n_q - n_root)]))  # All DoFs except root

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

q_ocp = np.vstack((q_roots_ocp, q_joints_ocp))
qdot_ocp = np.vstack((qdot_roots_ocp, qdot_joints_ocp))
# if FLAG_GENERATE_VIDEOS:
#     print("Generating OCP_one : ", ocp_path_to_results)
#     bioviz_animate(biorbd_model_path_with_mesh_ocp, np.vstack((q_roots_ocp, q_joints_ocp)), "OCP_one")

time_vector_ocp = np.linspace(0, float(time_ocp), n_shooting + 1)
dyn_fun_ocp = cas.Function("dynamics", [Q, Qdot, Tau, MotorNoise], [OCP_dynamics(Q, Qdot, Tau, MotorNoise, ocp)])
q_ocp_integrated, qdot_ocp_integrated, q_ocp_multiple_shooting, qdot_ocp_multiple_shooting, motor_noise_numerical = (
    noisy_integrate(
        time_vector_ocp,
        cas.vertcat(q_roots_ocp, q_joints_ocp),
        cas.vertcat(qdot_roots_ocp, qdot_joints_ocp),
        tau_joints_ocp,
        n_shooting,
        motor_noise_magnitude,
        dyn_fun_ocp,
        nb_random,
    )
)
q_ocp_reintegrated, _, _, _, _ = (
    noisy_integrate(
        time_vector_ocp,
        cas.vertcat(q_roots_ocp, q_joints_ocp),
        cas.vertcat(qdot_roots_ocp, qdot_joints_ocp),
        tau_joints_ocp,
        n_shooting,
        motor_noise_magnitude,
        dyn_fun_ocp,
        nb_random*nb_reintegrations,
    )
)

ocp_out_path_to_results = ocp_path_to_results.replace(".pkl", "_integrated.pkl")
with open(ocp_out_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_ocp_integrated,
        "qdot_integrated": qdot_ocp_integrated,
        "q_multiple_shooting": q_ocp_multiple_shooting,
        "qdot_multiple_shooting": qdot_ocp_multiple_shooting,
        "motor_noise_numerical": motor_noise_numerical,
        "time_vector": time_vector_ocp,
        "q_nominal": cas.vertcat(q_roots_ocp, q_joints_ocp),
    }
    pickle.dump(data, file)

q_all_ocp = np.zeros((n_q * (nb_random + 1), n_shooting + 1))
for i_shooting in range(n_shooting + 1):
    for i_random in range(nb_random):
        q_all_ocp[i_random * n_q : (i_random + 1) * n_q, i_shooting] = q_ocp_integrated[:, i_shooting, i_random]
    q_all_ocp[(i_random + 1) * n_q : (i_random + 2) * n_q, i_shooting] = np.hstack(
        (np.array(q_roots_ocp[:, i_shooting]), np.array(q_joints_ocp[:, i_shooting]))
    )

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

_, _, socp = prepare_socp(
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

time_vector_socp = np.linspace(0, float(time_socp), n_shooting + 1)

out = DMS_sensory_reference(socp.nlp[0].model, n_root, Q, Qdot)
DMS_sensory_reference_func = cas.Function("DMS_sensory_reference", [Q, Qdot], [out])
dyn_fun_out, joint_friction_out, motor_noise_out, feedback_out = SOCP_dynamics(
    nb_random, q_sym, qdot_sym, tau_sym, k_matrix_sym, ref_sym, motor_noise_sym, sensory_noise_sym, socp
)
dyn_fun_socp = cas.Function(
    "dynamics",
    [q_sym, qdot_sym, tau_sym, k_matrix_sym, ref_sym, motor_noise_sym, sensory_noise_sym],
    [dyn_fun_out, joint_friction_out, motor_noise_out, feedback_out],
)
dyn_fun_socp_nominal = cas.Function(
    "nominal_dyn",
    [Q, Qdot, Tau, k_matrix_sym, ref_sym, MotorNoise, SensoryNoise],
    [OCP_dynamics(Q, Qdot, Tau, np.zeros(MotorNoise.shape), ocp), cas.MX(), cas.MX(), cas.MX()],
)


q_socp = np.zeros((n_q, n_shooting + 1, nb_random))
qdot_socp = np.zeros((n_q, n_shooting + 1, nb_random))
for i_random in range(nb_random):
    for i_shooting in range(n_shooting + 1):
        q_socp[:, i_shooting, i_random] = np.hstack(
            (
                q_roots_socp[i_random * n_root : (i_random + 1) * n_root, i_shooting],
                q_joints_socp[i_random * n_joints : (i_random + 1) * n_joints, i_shooting],
            )
        )
        qdot_socp[:, i_shooting, i_random] = np.hstack(
            (
                qdot_roots_socp[i_random * n_root : (i_random + 1) * n_root, i_shooting],
                qdot_joints_socp[i_random * n_joints : (i_random + 1) * n_joints, i_shooting],
            )
        )
(
    q_socp_integrated,
    qdot_socp_integrated,
    q_socp_multiple_shooting,
    qdot_socp_multiple_shooting,
    joint_frictions_socp,
    motor_noises_socp,
    feedbacks_socp,
) = integrate_socp(
    time_vector_socp,
    q_socp,
    qdot_socp,
    tau_joints_socp,
    k_socp,
    ref_socp,
    motor_noise_numerical_socp,
    sensory_noise_numerical_socp,
    dyn_fun_socp,
    socp,
)

motor_noise_numerical_socp_random = np.zeros((motor_noise_magnitude.shape[0], nb_random * nb_reintegrations, n_shooting + 1))
sensory_noise_numerical_socp_random = np.zeros((sensory_noise_magnitude.shape[0], nb_random * nb_reintegrations, n_shooting + 1))
q_socp_reintegrated = np.zeros((n_q, n_shooting + 1, nb_random * nb_reintegrations))
for i_reintegration in range(nb_reintegrations):
    tempo_motor, tempo_sensory = create_random_noise(i_reintegration, nb_random, n_shooting, n_joints, motor_noise_magnitude, sensory_noise_magnitude)
    motor_noise_numerical_socp_random[:, i_reintegration * nb_random : (i_reintegration + 1) * nb_random, :] = tempo_motor
    sensory_noise_numerical_socp_random[:, i_reintegration * nb_random : (i_reintegration + 1) * nb_random, :] = tempo_sensory

    (
        q,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = integrate_socp(
        time_vector_socp,
        q_socp,
        qdot_socp,
        tau_joints_socp,
        k_socp,
        ref_socp,
        motor_noise_numerical_socp_random[:, i_reintegration * nb_random : (i_reintegration + 1) * nb_random, :],
        sensory_noise_numerical_socp_random[:, i_reintegration * nb_random : (i_reintegration + 1) * nb_random, :],
        dyn_fun_socp,
        socp,
    )
    q_socp_reintegrated[:, :, i_reintegration * nb_random : (i_reintegration + 1) * nb_random] = q

(q_socp_nominal, qdot_socp_nominal, _, _, _, _, _) = integrate_socp(
    time_vector_socp,
    np.mean(q_socp, axis=2)[:, :, np.newaxis],
    np.mean(qdot_socp, axis=2)[:, :, np.newaxis],
    tau_joints_socp,
    np.zeros(np.shape(k_socp)),
    np.zeros(np.shape(ref_socp)),
    np.zeros((4, 1, n_shooting)),
    np.zeros((10, 1, n_shooting)),
    dyn_fun_socp_nominal,
    socp,
)

# q_mean_socp = np.mean(q_socp_integrated, axis=2)
q_mean_socp = np.mean(q_socp, axis=2)
# if FLAG_GENERATE_VIDEOS:
#    # TODO: fix this integration issue ?
#     print("Generating SOCP_one : ", socp_path_to_results)
#     # bioviz_animate(biorbd_model_path_with_mesh_socp, q_socp_nominal[:, :, 0], "SOCP_one")
#     bioviz_animate(biorbd_model_path_with_mesh_socp, q_mean_socp, "SOCP_one")

socp_out_path_to_results = socp_path_to_results.replace(".pkl", "_integrated.pkl")
with open(socp_out_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_socp_integrated,
        "qdot_integrated": qdot_socp_integrated,
        "q_multiple_shooting": q_socp_multiple_shooting,
        "qdot_multiple_shooting": qdot_socp_multiple_shooting,
        "motor_noise_numerical": motor_noise_numerical_socp,
        "time_vector": time_vector_socp,
        "q_mean_integrated": np.mean(q_socp_integrated, axis=2),
        "q_nominal": q_socp_nominal,
    }
    pickle.dump(data, file)

q_all_socp = np.zeros((n_q * (nb_random + 1), n_shooting + 1))
for i_shooting in range(n_shooting + 1):
    for i_random in range(nb_random):
        q_all_socp[i_random * n_q : (i_random + 1) * n_q, i_shooting] = np.reshape(
            q_socp[:, i_shooting, i_random], (-1,)
        )
    q_all_socp[(i_random + 1) * n_q : (i_random + 2) * n_q, i_shooting] = np.reshape(q_mean_socp[:, i_shooting], (-1,))

# if FLAG_GENERATE_VIDEOS:
#     print("Generating SOCP_all : ", socp_path_to_results)
#     bioviz_animate(biorbd_model_path_with_mesh_all_socp, q_all_socp, "SOCP_all")


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

_, _, socp_plus = prepare_socp_VARIABLE_FEEDFORWARD(
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
    ref_fb_socp_plus = data["ref_sol"]
    ref_ff_socp_plus = data["ref_ff_sol"]
    motor_noise_numerical_socp_plus = data["motor_noise_numerical"]
    sensory_noise_numerical_socp_plus = data["sensory_noise_numerical"]

time_vector_socp_plus = np.linspace(0, float(time_socp_plus), n_shooting + 1)

out = DMS_sensory_reference_no_eyes(socp_plus.nlp[0].model, n_root, Q_8, Qdot_8)
DMS_sensory_reference_func = cas.Function("DMS_sensory_reference", [Q_8, Qdot_8], [out])
dyn_fun_out, joint_friction_out, motor_noise_out, feedback_out, feedforward_out = SOCP_PLUS_dynamics(
    nb_random,
    q_8_sym,
    qdot_8_sym,
    tau_8_sym,
    k_fb_matrix_sym,
    k_ff_matrix_sym,
    fb_ref_sym,
    ff_ref_sym,
    float(time_socp_plus),
    time_sym,
    motor_noise_8_sym,
    sensory_noise_8_sym,
    socp_plus,
)
dyn_fun_socp_plus = cas.Function(
    "dynamics",
    [
        q_8_sym,
        qdot_8_sym,
        tau_8_sym,
        k_fb_matrix_sym,
        k_ff_matrix_sym,
        fb_ref_sym,
        ff_ref_sym,
        time_sym,
        motor_noise_8_sym,
        sensory_noise_8_sym,
    ],
    [dyn_fun_out, joint_friction_out, motor_noise_out, feedback_out, feedforward_out],
)
dyn_fun_socp_plus_nominal = cas.Function(
    "nominal_dyn",
    [
        Q_8,
        Qdot_8,
        Tau_8,
        k_fb_matrix_sym,
        k_ff_matrix_sym,
        fb_ref_sym,
        ff_ref_sym,
        time_sym,
        MotorNoise_8,
        SensoryNoise_8,
    ],
    [OCP_dynamics(Q_8, Qdot_8, Tau_8, np.zeros(MotorNoise_8.shape), socp_plus), cas.MX(), cas.MX(), cas.MX(), cas.MX()],
)

visual_noise_fcn = cas.Function(
    "visual_noise", [Q_8, FF_SensoryNoise], [visual_noise(socp_plus.nlp[0].model, Q_8, FF_SensoryNoise)]
)
head_angular_velocity = socp_plus.nlp[0].model.segment_angular_velocity(
    Q_8, Qdot_8, socp_plus.nlp[0].model.segment_index("Head")
)[0]
vestibular_noise = gaussian_function(
    x=head_angular_velocity,
    sigma=10,
    offset=FF_SensoryNoise,
    scaling_factor=10,
    flip=True,
)
vestibular_noise_fcn = cas.Function("vestibular_noise", [Q_8, Qdot_8, FF_SensoryNoise], [vestibular_noise])

q_socp_plus = np.zeros((n_q, n_shooting + 1, nb_random))
qdot_socp_plus = np.zeros((n_q, n_shooting + 1, nb_random))
for i_random in range(nb_random):
    for i_shooting in range(n_shooting + 1):
        q_socp_plus[:, i_shooting, i_random] = np.hstack(
            (
                q_roots_socp_plus[i_random * n_root : (i_random + 1) * n_root, i_shooting],
                q_joints_socp_plus[i_random * n_joints : (i_random + 1) * n_joints, i_shooting],
            )
        )
        qdot_socp_plus[:, i_shooting, i_random] = np.hstack(
            (
                qdot_roots_socp_plus[i_random * n_root : (i_random + 1) * n_root, i_shooting],
                qdot_joints_socp_plus[i_random * n_joints : (i_random + 1) * n_joints, i_shooting],
            )
        )

(
    q_socp_plus_integrated,
    qdot_socp_plus_integrated,
    q_socp_plus_multiple_shooting,
    qdot_socp_plus_multiple_shooting,
    joint_frictions_socp_plus,
    motor_noises_socp_plus,
    feedbacks_socp_plus,
    feedforwards_socp_plus,
) = integrate_socp_plus(
    time_vector_socp_plus,
    q_socp_plus,
    qdot_socp_plus,
    tau_joints_socp_plus,
    k_socp_plus,
    ref_fb_socp_plus,
    ref_ff_socp_plus,
    motor_noise_numerical_socp_plus,
    sensory_noise_numerical_socp_plus,
    dyn_fun_socp_plus,
    socp_plus,
)

q_socp_plus_reintegrated = np.zeros((n_q, n_shooting + 1, nb_random * nb_reintegrations))
motor_noise_numerical_socp_plus_random = np.zeros((motor_noise_magnitude.shape[0], nb_random * nb_reintegrations, n_shooting + 1))
sensory_noise_numerical_socp_plus_random = np.zeros((sensory_noise_magnitude.shape[0], nb_random * nb_reintegrations, n_shooting + 1))
for i_reintegration in range(nb_reintegrations):
    tempo_motor, tempo_sensory = create_random_noise(i_reintegration, nb_random, n_shooting, n_joints, motor_noise_magnitude, sensory_noise_magnitude)
    tempo_motor[1, :, :] = 0  # No noise on the eyes
    motor_noise_numerical_socp_plus_random[:, i_reintegration * nb_random: (i_reintegration + 1) * nb_random, :] = tempo_motor
    sensory_noise_numerical_socp_plus_random[:, i_reintegration * nb_random: (i_reintegration + 1) * nb_random, :] = tempo_sensory

    (
        q,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = integrate_socp_plus(
        time_vector_socp_plus,
        q_socp_plus,
        qdot_socp_plus,
        tau_joints_socp_plus,
        k_socp_plus,
        ref_fb_socp_plus,
        ref_ff_socp_plus,
        motor_noise_numerical_socp_plus_random[:, i_reintegration * nb_random : (i_reintegration + 1) * nb_random, :],
        sensory_noise_numerical_socp_plus_random[:, i_reintegration * nb_random : (i_reintegration + 1) * nb_random, :],
        dyn_fun_socp_plus,
        socp_plus,
    )
    q_socp_plus_reintegrated[:, :, i_reintegration * nb_random : (i_reintegration + 1) * nb_random] = q

(
    q_socp_plus_nominal,
    qdot_socp_plus_nominal,
    _,
    _,
    _,
    _,
    _,
    _,
) = integrate_socp_plus(
    time_vector_socp_plus,
    np.mean(q_socp_plus, axis=2)[:, :, np.newaxis],
    np.mean(qdot_socp_plus, axis=2)[:, :, np.newaxis],
    tau_joints_socp_plus,
    np.zeros(np.shape(k_socp_plus)),
    np.zeros(np.shape(ref_fb_socp_plus)),
    0,
    np.zeros((5, 1, n_shooting)),
    np.zeros((11, 1, n_shooting)),
    dyn_fun_socp_plus_nominal,
    socp_plus,
)

# q_mean_socp_plus = np.mean(q_socp_plus_integrated, axis=2)
q_mean_socp_plus = np.mean(q_socp_plus, axis=2)
qdot_mean_socp_plus = np.mean(qdot_socp_plus, axis=2)
# if FLAG_GENERATE_VIDEOS:
#     # TODO: fix this integration issue ?
#     print("Generating SOCP_plus_one : ", socp_plus_path_to_results)
#     # bioviz_animate(biorbd_model_path_with_mesh_socp, q_socp_plus_nominal, "SOCP_plus_one")
#     bioviz_animate(biorbd_model_path_vision_with_mesh, q_mean_socp_plus, "SOCP_plus_one")


socp_plus_out_path_to_results = socp_plus_path_to_results.replace(".pkl", "_integrated.pkl")
with open(socp_plus_out_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_socp_plus_integrated,
        "qdot_integrated": qdot_socp_plus_integrated,
        "q_multiple_shooting": q_socp_plus_multiple_shooting,
        "qdot_multiple_shooting": qdot_socp_plus_multiple_shooting,
        "motor_noise_numerical": motor_noise_numerical_socp_plus,
        "time_vector": time_vector_socp_plus,
        "q_mean_integrated": np.mean(q_socp_plus_integrated, axis=2),
        "q_nominal": q_socp_plus_nominal,
    }
    pickle.dump(data, file)

q_all_socp_plus = np.zeros((n_q * (nb_random + 1), n_shooting + 1))
for i_shooting in range(n_shooting + 1):
    for i_random in range(nb_random):
        q_all_socp_plus[i_random * n_q : (i_random + 1) * n_q, i_shooting] = np.reshape(
            q_socp_plus[:, i_shooting, i_random], (-1,)
        )
    # q_all_socp_plus[(i_random + 1) * n_q: (i_random + 2) * n_q, i_shooting] = np.reshape(q_socp_plus_nominal[:, i_shooting], (-1, ))
    q_all_socp_plus[(i_random + 1) * n_q : (i_random + 2) * n_q, i_shooting] = np.reshape(
        q_mean_socp_plus[:, i_shooting], (-1,)
    )

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_plus_all : ", socp_plus_path_to_results)
    bioviz_animate(biorbd_model_path_vision_with_mesh_all, q_all_socp_plus, "SOCP_plus_all")


# Comparison ----------------------------------------------------------------------------------------------------------
q_mean_comparison = np.zeros((7 + 7 + 8, n_shooting + 1))
q_mean_comparison[:7, :] = q_ocp
q_mean_comparison[7:14, :] = q_mean_socp
q_mean_comparison[14:, :] = q_mean_socp_plus
if FLAG_GENERATE_VIDEOS:
    print("Generating comparison")
    bioviz_animate(biorbd_model_path_comparison, q_mean_comparison, "Comparison")


# Plots ---------------------------------------------------------------------------------------------------------------
time_vector_ocp = time_vector_ocp[:-1]
time_vector_socp = time_vector_socp[:-1]
time_vector_socp_plus = time_vector_socp_plus[:-1]
normalized_time_vector = np.linspace(0, 1, n_shooting)

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
        tau_joints_socp[0, :] + motor_noises_socp[0, i_random, :],
        color=SOCP_color,
        label="SOCP",
        linewidth=0.5,
        alpha=0.5,
    )
    axs[0, 0].step(
        normalized_time_vector,
        tau_joints_socp_plus[0, :] + motor_noises_socp_plus[0, i_random, :],
        color=SOCP_plus_color,
        label="SOCP+",
        linewidth=0.5,
        alpha=0.5,
    )
axs[0, 0].step(normalized_time_vector, tau_joints_socp[0, :], color=SOCP_color, label="SOCP")
axs[0, 0].step(normalized_time_vector, tau_joints_socp_plus[0, :], color=SOCP_plus_color, label="SOCP+")
axs[0, 0].set_title("Head")
# axs[0, 0].legend(ncol=3)
# Eyes
axs[0, 1].step(normalized_time_vector, tau_joints_socp_plus[1, :], color=SOCP_plus_color, label="SOCP+")
axs[0, 1].set_title("Eyes")
axs[0, 1].plot([0, 1], [0, 0], color="black", linestyle="--")
# Other joints
for i_dof in range(2, 5):
    axs[0, i_dof].step(normalized_time_vector, tau_joints_ocp[i_dof - 1, :], color=OCP_color)

    for i_random in range(nb_random):
        axs[0, i_dof].step(
            normalized_time_vector,
            tau_joints_socp[i_dof - 1, :] + motor_noises_socp[i_dof - 1, i_random, :],
            color=SOCP_color,
            label="OCP",
            linewidth=0.5,
            alpha=0.5,
        )
    for i_random in range(nb_random):
        axs[0, i_dof].step(
            normalized_time_vector,
            tau_joints_socp_plus[i_dof, :] + motor_noises_socp_plus[i_dof, i_random, :],
            color=SOCP_plus_color,
            label="SOCP+",
            linewidth=0.5,
            alpha=0.5,
        )
    axs[0, i_dof].step(normalized_time_vector, tau_joints_socp[i_dof - 1, :], color=SOCP_color, label="SOCP")
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
                normalized_time_vector, -joint_frictions_socp[0, i_random, :], color=SOCP_color, linewidth=0.5
            )
            axs[1, 0].step(
                normalized_time_vector, -joint_frictions_socp_plus[0, i_random, :], color=SOCP_plus_color, linewidth=0.5
            )
    elif i_dof == 1:
        for i_random in range(nb_random):
            axs[1, 1].step(
                normalized_time_vector, -joint_frictions_socp_plus[1, i_random, :], color=SOCP_plus_color, linewidth=0.5
            )
    else:
        axs[1, i_dof].step(normalized_time_vector, -joint_friction_ocp[i_dof - 1, :], color=OCP_color)
        for i_random in range(nb_random):
            axs[1, i_dof].step(
                normalized_time_vector, -joint_frictions_socp[i_dof - 1, i_random, :], color=SOCP_color, linewidth=0.5
            )
            axs[1, i_dof].step(
                normalized_time_vector,
                -joint_frictions_socp_plus[i_dof, i_random, :],
                color=SOCP_plus_color,
                linewidth=0.5,
            )
axs[1, 0].set_ylabel("Joint friction [Nm]")

# Feedback
for i_dof in range(5):
    for i_random in range(nb_random):
        if i_dof == 0:
            axs[2, 0].step(normalized_time_vector, feedbacks_socp[0, i_random, :], color=SOCP_color, linewidth=0.5)
            axs[2, 0].step(
                normalized_time_vector, feedbacks_socp_plus[0, i_random, :], color=SOCP_plus_color, linewidth=0.5
            )
        elif i_dof == 1:
            axs[2, 1].step(
                normalized_time_vector, feedbacks_socp_plus[1, i_random, :], color=SOCP_plus_color, linewidth=0.5
            )
        else:
            axs[2, i_dof].step(
                normalized_time_vector, feedbacks_socp[i_dof - 1, i_random, :], color=SOCP_color, linewidth=0.5
            )
            axs[2, i_dof].step(
                normalized_time_vector, feedbacks_socp_plus[i_dof, i_random, :], color=SOCP_plus_color, linewidth=0.5
            )
axs[2, 0].set_ylabel("Feedbacks [Nm]")

# Feedforward
for i_dof in range(5):
    for i_random in range(nb_random):
        axs[3, i_dof].step(
            normalized_time_vector, feedforwards_socp_plus[i_dof, i_random, :], color=SOCP_plus_color, linewidth=0.5
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
                - joint_frictions_socp[0, i_random, :]
                + motor_noises_socp[0, i_random, :]
                + feedbacks_socp[0, i_random, :],
                color=SOCP_color,
                linewidth=0.5,
                alpha=0.5,
            )
            axs[4, i_dof].step(
                normalized_time_vector,
                tau_joints_socp_plus[0, :]
                - joint_frictions_socp_plus[0, i_random, :]
                + motor_noises_socp_plus[0, i_random, :]
                + feedbacks_socp_plus[0, i_random, :]
                + feedforwards_socp_plus[0, i_random, :],
                color=SOCP_plus_color,
                linewidth=0.5,
                alpha=0.5,
            )
    elif i_dof == 1:
        axs[4, 1].step(
            normalized_time_vector,
            tau_joints_socp_plus[1, :]
            - joint_frictions_socp_plus[1, i_random, :]
            + motor_noises_socp_plus[1, i_random, :]
            + feedbacks_socp_plus[1, i_random, :]
            + feedforwards_socp_plus[1, i_random, :],
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
                - joint_frictions_socp[i_dof - 1, i_random, :]
                + motor_noises_socp[i_dof - 1, i_random, :]
                + feedbacks_socp[i_dof - 1, i_random, :],
                color=SOCP_color,
                linewidth=0.5,
                alpha=0.5,
            )
            axs[4, i_dof].step(
                normalized_time_vector,
                tau_joints_socp_plus[i_dof, :]
                - joint_frictions_socp_plus[i_dof, i_random, :]
                + motor_noises_socp_plus[i_dof, i_random, :]
                + feedbacks_socp_plus[i_dof, i_random, :]
                + feedforwards_socp_plus[i_dof, i_random, :],
                color=SOCP_plus_color,
                linewidth=0.5,
                alpha=0.5,
            )
axs[4, 0].set_ylabel(r"\tau (sum) [Nm]")

plt.savefig("graphs/controls.png")
plt.show()


# Plot tau and delta tau
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
                - joint_frictions_socp[0, i_random, :]
                + motor_noises_socp[0, i_random, :]
                + feedbacks_socp[0, i_random, :],
                color=SOCP_color,
                linewidth=0.5,
                alpha=0.5,
            )
            axs[0, i_dof].step(
                normalized_time_vector,
                tau_joints_socp_plus[0, :]
                - joint_frictions_socp_plus[0, i_random, :]
                + motor_noises_socp_plus[0, i_random, :]
                + feedbacks_socp_plus[0, i_random, :]
                + feedforwards_socp_plus[0, i_random, :],
                color=SOCP_plus_color,
                linewidth=0.5,
                alpha=0.5,
            )
    elif i_dof == 1:
        axs[0, 1].step(
            normalized_time_vector,
            tau_joints_socp_plus[1, :]
            - joint_frictions_socp_plus[1, i_random, :]
            + motor_noises_socp_plus[1, i_random, :]
            + feedbacks_socp_plus[1, i_random, :]
            + feedforwards_socp_plus[1, i_random, :],
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
                - joint_frictions_socp[i_dof - 1, i_random, :]
                + motor_noises_socp[i_dof - 1, i_random, :]
                + feedbacks_socp[i_dof - 1, i_random, :],
                color=SOCP_color,
                linewidth=0.5,
                alpha=0.5,
            )
            axs[0, i_dof].step(
                normalized_time_vector,
                tau_joints_socp_plus[i_dof, :]
                - joint_frictions_socp_plus[i_dof, i_random, :]
                + motor_noises_socp_plus[i_dof, i_random, :]
                + feedbacks_socp_plus[i_dof, i_random, :]
                + feedforwards_socp_plus[i_dof, i_random, :],
                color=SOCP_plus_color,
                linewidth=0.5,
                alpha=0.5,
            )
axs[0, 0].set_ylabel(r"$\tau$ (sum) [Nm]")

# Delta tau
delta_time_vector = (normalized_time_vector[1:] + normalized_time_vector[:-1]) / 2
for i_dof in range(5):
    if i_dof == 0:
        axs[1, i_dof].step(
            delta_time_vector,
            tau_joints_ocp[0, 1:] - joint_friction_ocp[0, 1:] - (tau_joints_ocp[0, :-1] + joint_friction_ocp[0, :-1]),
            color=OCP_color,
        )
        for i_random in range(nb_random):
            axs[1, i_dof].step(
                delta_time_vector,
                tau_joints_socp[0, 1:]
                - joint_frictions_socp[0, i_random, 1:]
                + motor_noises_socp[0, i_random, 1:]
                + feedbacks_socp[0, i_random, 1:]
                - (
                    tau_joints_socp[0, :-1]
                    - joint_frictions_socp[0, i_random, :-1]
                    + motor_noises_socp[0, i_random, :-1]
                    + feedbacks_socp[0, i_random, :-1]
                ),
                color=SOCP_color,
                linewidth=0.5,
                alpha=0.5,
            )
            axs[1, i_dof].step(
                delta_time_vector,
                tau_joints_socp_plus[0, 1:]
                - joint_frictions_socp_plus[0, i_random, 1:]
                + motor_noises_socp_plus[0, i_random, 1:]
                + feedbacks_socp_plus[0, i_random, 1:]
                + feedforwards_socp_plus[0, i_random, 1:]
                - (
                    tau_joints_socp_plus[0, :-1]
                    - joint_frictions_socp_plus[0, i_random, :-1]
                    + motor_noises_socp_plus[0, i_random, :-1]
                    + feedbacks_socp_plus[0, i_random, :-1]
                    + feedforwards_socp_plus[0, i_random, :-1]
                ),
                color=SOCP_plus_color,
                linewidth=0.5,
                alpha=0.5,
            )
    elif i_dof == 1:
        axs[1, 1].step(
            delta_time_vector,
            tau_joints_socp_plus[1, 1:]
            - joint_frictions_socp_plus[1, i_random, 1:]
            + motor_noises_socp_plus[1, i_random, 1:]
            + feedbacks_socp_plus[1, i_random, 1:]
            + feedforwards_socp_plus[1, i_random, 1:]
            - (
                tau_joints_socp_plus[1, :-1]
                - joint_frictions_socp_plus[1, i_random, :-1]
                + motor_noises_socp_plus[1, i_random, :-1]
                + feedbacks_socp_plus[1, i_random, :-1]
                + feedforwards_socp_plus[1, i_random, :-1]
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
                - joint_frictions_socp[i_dof - 1, i_random, 1:]
                + motor_noises_socp[i_dof - 1, i_random, 1:]
                + feedbacks_socp[i_dof - 1, i_random, 1:]
                - (
                    tau_joints_socp[i_dof - 1, :-1]
                    - joint_frictions_socp[i_dof - 1, i_random, :-1]
                    + motor_noises_socp[i_dof - 1, i_random, :-1]
                    + feedbacks_socp[i_dof - 1, i_random, :-1]
                ),
                color=SOCP_color,
                linewidth=0.5,
                alpha=0.5,
            )
            axs[1, i_dof].step(
                delta_time_vector,
                tau_joints_socp_plus[i_dof, 1:]
                - joint_frictions_socp_plus[i_dof, i_random, 1:]
                + motor_noises_socp_plus[i_dof, i_random, 1:]
                + feedbacks_socp_plus[i_dof, i_random, 1:]
                + feedforwards_socp_plus[i_dof, i_random, 1:]
                - (
                    tau_joints_socp_plus[i_dof, :-1]
                    - joint_frictions_socp_plus[i_dof, i_random, :-1]
                    + motor_noises_socp_plus[i_dof, i_random, :-1]
                    + feedbacks_socp_plus[i_dof, i_random, :-1]
                    + feedforwards_socp_plus[i_dof, i_random, :-1]
                ),
                color=SOCP_plus_color,
                linewidth=0.5,
                alpha=0.5,
            )
axs[1, 0].set_ylabel(r"$\Delta \tau$ (sum) [Nm]")

plt.savefig("graphs/tau_and_delta_tau.png")
plt.show()


# Plot tau and delta tau
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
for i in range(2):
    axs[i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
# Tau
axs[0].step(
    normalized_time_vector, np.sum(np.abs(tau_joints_ocp), axis=0) - np.sum(np.abs(joint_friction_ocp), axis=0), color=OCP_color
)
for i_random in range(nb_random):
    axs[0].step(
        normalized_time_vector,
        np.sum(np.abs(tau_joints_socp), axis=0)
        - np.sum(np.abs(joint_frictions_socp[:, i_random, :]), axis=0)
        + np.sum(np.abs(motor_noises_socp[:, i_random, :]), axis=0)
        + np.sum(np.abs(feedbacks_socp[:, i_random, :]), axis=0),
        color=SOCP_color,
        linewidth=0.5,
        alpha=0.5,
    )
    axs[0].step(
        normalized_time_vector,
        np.sum(np.abs(tau_joints_socp_plus), axis=0)
        - np.sum(np.abs(joint_frictions_socp_plus[:, i_random, :]), axis=0)
        + np.sum(np.abs(motor_noises_socp_plus[:, i_random, :]), axis=0)
        + np.sum(np.abs(feedbacks_socp_plus[:, i_random, :]), axis=0)
        + np.sum(np.abs(feedforwards_socp_plus[:, i_random, :]), axis=0),
        color=SOCP_plus_color,
        linewidth=0.5,
        alpha=0.5,
    )
axs[0].set_ylabel(r"$\tau$ (sum) [Nm]")

# Delta tau
delta_time_vector = (normalized_time_vector[1:] + normalized_time_vector[:-1]) / 2
axs[1].step(
    delta_time_vector,
    np.sum(np.abs(tau_joints_ocp[:, 1:] - joint_friction_ocp[:, 1:] - (tau_joints_ocp[:, :-1] + joint_friction_ocp[:, :-1])), axis=0),
    color=OCP_color,
)
for i_random in range(nb_random):
    axs[1].step(
        delta_time_vector,
        np.sum(np.abs(tau_joints_socp[:, 1:]
        - joint_frictions_socp[:, i_random, 1:]
        + motor_noises_socp[:, i_random, 1:]
        + feedbacks_socp[:, i_random, 1:]
        - (
            tau_joints_socp[:, :-1]
            - joint_frictions_socp[:, i_random, :-1]
            + motor_noises_socp[:, i_random, :-1]
            + feedbacks_socp[:, i_random, :-1]
        )), axis=0),
        color=SOCP_color,
        linewidth=0.5,
        alpha=0.5,
    )
    axs[1].step(
        delta_time_vector,
        np.sum(np.abs(tau_joints_socp_plus[:, 1:]
        - joint_frictions_socp_plus[:, i_random, 1:]
        + motor_noises_socp_plus[:, i_random, 1:]
        + feedbacks_socp_plus[:, i_random, 1:]
        + feedforwards_socp_plus[:, i_random, 1:]
        - (
            tau_joints_socp_plus[:, :-1]
            - joint_frictions_socp_plus[:, i_random, :-1]
            + motor_noises_socp_plus[:, i_random, :-1]
            + feedbacks_socp_plus[:, i_random, :-1]
            + feedforwards_socp_plus[:, i_random, :-1]
        )), axis=0),
        color=SOCP_plus_color,
        linewidth=0.5,
        alpha=0.5,
    )
axs[1].set_ylabel(r"$\Delta \tau$ (sum) [Nm]")

plt.savefig("graphs/sum_tau_and_delta_tau.png")
plt.show()

plt.figure()
plt.bar(0, float(time_ocp), color=OCP_color, label="OCP")
plt.bar(1, float(time_socp), color=SOCP_color, label="SOCP")
plt.bar(2, float(time_socp_plus), color=SOCP_plus_color, label="SOCP+")
plt.xticks([0, 1, 2], ["OCP", "SOCP", "SOCP+"])
print("Movement durations: ", time_ocp, time_socp, time_socp_plus)
plt.savefig("graphs/movement_durations.png")
plt.show()


# Plot the gains
n_k_fb = socp_plus.nlp[0].model.n_noised_controls + socp_plus.nlp[0].model.n_references
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i in range(2):
    for j in range(2):
        axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
for i_dof in range(40):
    axs[0, 0].step(normalized_time_vector, k_socp[i_dof, :], color=SOCP_color, label="SOCP")
for i_dof in range(n_k_fb):
    axs[0, 1].step(normalized_time_vector, k_socp_plus[i_dof, :], color=SOCP_plus_color, label="SOCP+")
for i_dof in range(5):
    axs[1, 1].step(normalized_time_vector, k_socp_plus[n_k_fb + i_dof, :], color=SOCP_plus_color, label="SOCP+")
axs[0, 0].set_ylim(-35, 35)
axs[0, 1].set_ylim(-35, 35)
axs[1, 1].set_ylim(-35, 35)
plt.savefig("graphs/gains.png")
plt.show()


# Plot the gains
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i in range(2):
    for j in range(2):
        axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)

axs[0, 0].step(normalized_time_vector, np.sum(np.abs(k_socp[:, :]), axis=0), color=SOCP_color, label="SOCP")
axs[0, 1].step(normalized_time_vector, np.sum(np.abs(k_socp_plus[:n_k_fb, :]), axis=0), color=SOCP_plus_color, label="SOCP+")
axs[1, 1].step(normalized_time_vector, np.sum(np.abs(k_socp_plus[n_k_fb:, :]), axis=0), color=SOCP_plus_color, label="SOCP+")
axs[0, 0].set_ylim(0, 800)
axs[0, 1].set_ylim(0, 800)
axs[1, 1].set_ylim(0, 800)
plt.savefig("graphs/sum_gains.png")
plt.show()



# Plot the delta gains
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i in range(2):
    for j in range(2):
        axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
for i_dof in range(40):
    axs[0, 0].step(delta_time_vector, k_socp[i_dof, 1:] - k_socp[i_dof, :-1], color=SOCP_color, label="SOCP")
for i_dof in range(n_k_fb):
    axs[0, 1].step(
        delta_time_vector, k_socp_plus[i_dof, 1:] - k_socp_plus[i_dof, :-1], color=SOCP_plus_color, label="SOCP+"
    )
for i_dof in range(5):
    axs[1, 1].step(
        delta_time_vector,
        k_socp_plus[n_k_fb + i_dof, 1:] - k_socp_plus[n_k_fb + i_dof, :-1],
        color=SOCP_plus_color,
        label="SOCP+",
    )
axs[0, 0].set_ylim(-40, 35)
axs[0, 1].set_ylim(-40, 35)
axs[1, 1].set_ylim(-40, 35)
plt.savefig("graphs/delta_gains.png")


# Plot the delta gains
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i in range(2):
    for j in range(2):
        axs[j, i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)
axs[0, 0].step(delta_time_vector, np.sum(np.abs(k_socp[:, 1:] - k_socp[:, :-1]), axis=0), color=SOCP_color, label="SOCP")
axs[0, 1].step(
    delta_time_vector, np.sum(np.abs(k_socp_plus[:n_k_fb, 1:] - k_socp_plus[:n_k_fb, :-1]), axis=0), color=SOCP_plus_color, label="SOCP+"
)
axs[1, 1].step(
    delta_time_vector, np.sum(np.abs(k_socp_plus[n_k_fb:, 1:] - k_socp_plus[n_k_fb:, :-1]), axis=0), color=SOCP_plus_color, label="SOCP+"
)
axs[0, 0].set_ylim(0, 800)
axs[0, 1].set_ylim(0, 800)
axs[1, 1].set_ylim(0, 800)
plt.savefig("graphs/sum_delta_gains.png")
plt.show()

# Plot the FF gains vs acuity
fig, axs = plt.subplots(4, 1, figsize=(15, 10))
for i in range(4):
    axs[i].plot([0, 1], [0, 0], color="black", linestyle="--", alpha=0.5)

axs[0].step(normalized_time_vector, np.sum(np.abs(k_socp_plus[:n_k_fb, :]), axis=0), color=SOCP_plus_color, label="SOCP+")
axs[0].set_ylabel("Feedback gains")
axs[1].step(normalized_time_vector, np.sum(np.abs(k_socp_plus[n_k_fb:, :]), axis=0), color=SOCP_plus_color, label="SOCP+")
axs[1].set_ylabel("Feedforward gains")

visual_acuity = np.zeros((n_shooting, 1))
vestibular_acquity = np.zeros((n_shooting, 1))
for i_shooting in range(n_shooting):
    visual_acuity[i_shooting] = visual_noise_fcn(q_mean_socp_plus[:, i_shooting], 1)
    vestibular_acquity[i_shooting] = vestibular_noise_fcn(
        q_mean_socp_plus[:, i_shooting], qdot_mean_socp_plus[:, i_shooting], 1
    )
visual_acuity_normalized = (visual_acuity - np.min(visual_acuity)) / (np.max(visual_acuity) - np.min(visual_acuity))
vestibular_acquity_normalized = (vestibular_acquity - np.min(vestibular_acquity)) / (
    np.max(vestibular_acquity) - np.min(vestibular_acquity)
)
axs[2].plot(normalized_time_vector, visual_acuity_normalized, color="turquoise", label="Visual acuity")
axs[2].set_ylabel("Visual acuity")
axs[3].plot(normalized_time_vector, vestibular_acquity_normalized, color="hotpink", label="Vestibular acuity")
axs[3].set_ylabel("Vestibular acuity")
plt.savefig("graphs/ff_gains.png")
plt.show()


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
CoM_y_socp_plus = np.zeros((nb_random, 1))
CoM_dot_y_ocp = np.zeros((nb_random, 1))
CoM_dot_y_socp = np.zeros((nb_random, 1))
CoM_dot_y_socp_plus = np.zeros((nb_random, 1))
BodyVelocity_ocp = np.zeros((nb_random, 1))
BodyVelocity_socp = np.zeros((nb_random, 1))
BodyVelocity_socp_plus = np.zeros((nb_random, 1))
for i_random in range(nb_random):
    CoM_y_ocp[i_random] = CoM_y_fcn(q_ocp_integrated[:, -1, i_random])
    axs[0].plot(0 + np.random.random(1) * 0.1, CoM_y_ocp[i_random], ".", color=OCP_color)
    CoM_y_socp[i_random] = CoM_y_fcn(q_socp[:, -1, i_random])
    axs[0].plot(0.5 + np.random.random(1) * 0.1, CoM_y_socp[i_random], ".", color=SOCP_color)
    CoM_y_socp_plus[i_random] = CoM_y_8_fcn(q_socp_plus[:, -1, i_random])
    axs[0].plot(1 + np.random.random(1) * 0.1, CoM_y_socp_plus[i_random], ".", color=SOCP_plus_color)
    CoM_dot_y_ocp[i_random] = CoM_y_dot_fcn(q_ocp_integrated[:, -1, i_random], qdot_ocp_integrated[:, -1, i_random])
    axs[1].plot(0 + np.random.random(1) * 0.1, CoM_dot_y_ocp[i_random], ".", color=OCP_color)
    CoM_dot_y_socp[i_random] = CoM_y_dot_fcn(q_socp[:, -1, i_random], qdot_socp[:, -1, i_random])
    axs[1].plot(0.5 + np.random.random(1) * 0.1, CoM_dot_y_socp[i_random], ".", color=SOCP_color)
    CoM_dot_y_socp_plus[i_random] = CoM_dot_8_fcn(q_socp_plus[:, -1, i_random], qdot_socp_plus[:, -1, i_random])
    axs[1].plot(1 + np.random.random(1) * 0.1, CoM_dot_y_socp_plus[i_random], ".", color=SOCP_plus_color)
    BodyVelocity_ocp[i_random] = BodyVelocity_fcn(
        q_ocp_integrated[:, -1, i_random], qdot_ocp_integrated[:, -1, i_random]
    )
    axs[2].plot(0 + np.random.random(1) * 0.1, BodyVelocity_ocp[i_random], ".", color=OCP_color)
    BodyVelocity_socp[i_random] = BodyVelocity_fcn(q_socp[:, -1, i_random], qdot_socp[:, -1, i_random])
    axs[2].plot(0.5 + np.random.random(1) * 0.1, BodyVelocity_socp[i_random], ".", color=SOCP_color)
    BodyVelocity_socp_plus[i_random] = BodyVelocity_8_fcn(q_socp_plus[:, -1, i_random], qdot_socp_plus[:, -1, i_random])
    axs[2].plot(1 + np.random.random(1) * 0.1, BodyVelocity_socp_plus[i_random], ".", color=SOCP_plus_color)

axs[0].set_title("CoM y")
axs[1].set_title("CoM y dot")
axs[2].set_title("Body velocity")


def box_plot(position, data, axs, color):
    axs.plot(np.array([position - 0.1, position - 0.1]), np.array([np.min(data), np.max(data)]), color=color)
    axs.plot(np.array([position - 0.15, position - 0.05]), np.array([np.min(data), np.min(data)]), color=color)
    axs.plot(np.array([position - 0.15, position - 0.05]), np.array([np.max(data), np.max(data)]), color=color)
    axs.plot(position - 0.1, np.mean(data), "s", color=color)
    axs.add_patch(
        Rectangle((position - 0.15, np.mean(data) - np.std(data)), 0.1, 2 * np.std(data), color=color, alpha=0.3)
    )


box_plot(0, CoM_y_ocp, axs[0], OCP_color)
box_plot(0.5, CoM_y_socp, axs[0], SOCP_color)
box_plot(1, CoM_y_socp_plus, axs[0], SOCP_plus_color)

box_plot(0, CoM_dot_y_ocp, axs[1], OCP_color)
box_plot(0.5, CoM_dot_y_socp, axs[1], SOCP_color)
box_plot(1, CoM_dot_y_socp_plus, axs[1], SOCP_plus_color)

box_plot(0, BodyVelocity_ocp, axs[2], OCP_color)
box_plot(0.5, BodyVelocity_socp, axs[2], SOCP_color)
box_plot(1, BodyVelocity_socp_plus, axs[2], SOCP_plus_color)

plt.savefig("graphs/landing_variability.png")
plt.show()


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
        elif i_dof > 4:
            axs[i_dof].plot(time_vector, q_socp[i_dof - 1, :, i_random], color=SOCP_color, linewidth=0.5)
    for i_random in range(nb_random):
        axs[i_dof].plot(time_vector, q_socp_plus[i_dof, :, i_random], color=SOCP_plus_color, linewidth=0.5)
plt.savefig("graphs/kinematics.png")
plt.show()


plot_comparison_reintegration(q_ocp, q_mean_socp, q_mean_socp_plus,
                                  q_all_socp, q_all_socp_plus,
                                  q_ocp_reintegrated, q_socp_reintegrated, q_socp_plus_reintegrated,
                                  time_vector_ocp, time_vector_socp, time_vector_socp_plus,
                                  OCP_color, SOCP_color, SOCP_plus_color,
                                  nb_random, nb_reintegrations)
