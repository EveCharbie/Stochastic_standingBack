import pickle

import bioviz
import casadi as cas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from DMS_deterministic import prepare_ocp
from DMS_SOCP import prepare_socp
from DMS_SOCP_VARIABLE import prepare_socp_VARIABLE
from DMS_SOCP_FEEDFORWARD import prepare_socp_FEEDFORWARD
from DMS_SOCP_VARIABLE_FEEDFORWARD import prepare_socp_VARIABLE_FEEDFORWARD
from bioptim import (
    StochasticBioModel,
    Solution,
    Shooting,
    SolutionIntegrator,
    InitialGuessList,
    InterpolationType,
)
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


def noisy_integrate_ocp(
        n_shooting,
        nb_random,
        nb_reintegrations,
        q_roots_ocp,
        q_joints_ocp,
        motor_noise_magnitude,
        tau_joints_ocp,
        time_vector_ocp,
        ocp,
        forward_dynamics_func,
):

    q_ocp = np.vstack((q_roots_ocp, q_joints_ocp))

    def ocp_dynamics(q_roots, q_joints, qdot_roots, qdot_joints, tau_joints, motor_noise,
                      nlp, nb_random, forward_dynamics_func):

        n_root = 3
        n_joints = 4

        ddq_roots = None
        ddq_joints = None
        for i in range(nb_random):
            q_this_time = np.hstack((
                q_roots[i * n_root: (i + 1) * n_root], q_joints[i * n_joints: (i + 1) * n_joints]
            ))
            qdot_this_time = np.hstack((
                qdot_roots[i * n_root: (i + 1) * n_root], qdot_joints[i * n_joints: (i + 1) * n_joints]
            ))
            tau_this_time = tau_joints[:]

            # Joint friction
            tau_this_time -= nlp.model.friction_coefficients @ qdot_this_time[n_root:]

            # Motor noise
            tau_this_time += motor_noise[:, i]

            ddq = forward_dynamics_func(q_this_time, qdot_this_time, tau_this_time)
            ddq_roots = np.vstack((ddq_roots, ddq[:n_root])) if ddq_roots is not None else ddq[:n_root]
            ddq_joints = np.vstack((ddq_joints, ddq[n_root:])) if ddq_joints is not None else ddq[n_root:]

        return qdot_roots, qdot_joints, ddq_roots.reshape(-1, ), ddq_joints.reshape(-1, )

    def integrate(time, q_roots, q_joints, qdot_roots, qdot_joints, tau_joints, motor_noise,
                  nlp, nb_random, forward_dynamics_func):
        dt = time / n_shooting
        h = dt / 5
        q_roots_integrated = np.zeros((n_root * nb_random, n_shooting + 1))
        q_joints_integrated = np.zeros((n_joints * nb_random, n_shooting + 1))
        qdot_roots_integrated = np.zeros((n_root * nb_random, n_shooting + 1))
        qdot_joints_integrated = np.zeros((n_joints * nb_random, n_shooting + 1))
        for i_random in range(nb_random):
            q_roots_integrated[n_root*i_random:n_root*(i_random+1), 0] = q_roots[:, i_random]
            q_joints_integrated[n_joints*i_random:n_joints*(i_random+1), 0] = q_joints[:, i_random]
            qdot_roots_integrated[n_root*i_random:n_root*(i_random+1), 0] = qdot_roots[:, i_random]
            qdot_joints_integrated[n_joints*i_random:n_joints*(i_random+1), 0] = qdot_joints[:, i_random]
        for i_shooting in range(n_shooting):
            q_roots_this_time = q_roots_integrated[:, i_shooting]
            q_joints_this_time = q_joints_integrated[:, i_shooting]
            qdot_roots_this_time = qdot_roots_integrated[:, i_shooting]
            qdot_joints_this_time = qdot_joints_integrated[:, i_shooting]
            tau_joints_this_time = tau_joints[:, i_shooting]
            motor_noise_this_time = motor_noise[:, :, i_shooting]
            for i_step in range(5):
                q_roots_dot1, q_joints_dot1, qdot_roots_dot1, qdot_joints_dot1 = ocp_dynamics(q_roots_this_time,
                                                                                               q_joints_this_time,
                                                                                               qdot_roots_this_time,
                                                                                               qdot_joints_this_time,
                                                                                               tau_joints_this_time,
                                                                                               motor_noise_this_time,
                                                                                               nlp,
                                                                                               nb_random,
                                                                                               forward_dynamics_func)
                q_roots_dot2, q_joints_dot2, qdot_roots_dot2, qdot_joints_dot2 = ocp_dynamics(
                    q_roots_this_time + h / 2 * q_roots_dot1, q_joints_this_time + h / 2 * q_joints_dot1,
                    qdot_roots_this_time + h / 2 * qdot_roots_dot1,
                    qdot_joints_this_time + h / 2 * qdot_joints_dot1,
                    tau_joints_this_time,
                    motor_noise_this_time,
                    nlp,
                    nb_random,
                    forward_dynamics_func)
                q_roots_dot3, q_joints_dot3, qdot_roots_dot3, qdot_joints_dot3 = ocp_dynamics(
                    q_roots_this_time + h / 2 * q_roots_dot2, q_joints_this_time + h / 2 * q_joints_dot2,
                    qdot_roots_this_time + h / 2 * qdot_roots_dot2,
                    qdot_joints_this_time + h / 2 * qdot_joints_dot2,
                    tau_joints_this_time,
                    motor_noise_this_time,
                    nlp,
                    nb_random,
                    forward_dynamics_func)
                q_roots_dot4, q_joints_dot4, qdot_roots_dot4, qdot_joints_dot4 = ocp_dynamics(
                    q_roots_this_time + h * q_roots_dot3, q_joints_this_time + h * q_joints_dot3,
                    qdot_roots_this_time + h * qdot_roots_dot3, qdot_joints_this_time + h * qdot_joints_dot3,
                    tau_joints_this_time,
                    motor_noise_this_time,
                    nlp,
                    nb_random,
                    forward_dynamics_func)
                q_roots_this_time = q_roots_this_time + h / 6 * (
                        q_roots_dot1 + 2 * q_roots_dot2 + 2 * q_roots_dot3 + q_roots_dot4)
                q_joints_this_time = q_joints_this_time + h / 6 * (
                        q_joints_dot1 + 2 * q_joints_dot2 + 2 * q_joints_dot3 + q_joints_dot4)
                qdot_roots_this_time = qdot_roots_this_time + h / 6 * (
                        qdot_roots_dot1 + 2 * qdot_roots_dot2 + 2 * qdot_roots_dot3 + qdot_roots_dot4)
                qdot_joints_this_time = qdot_joints_this_time + h / 6 * (
                        qdot_joints_dot1 + 2 * qdot_joints_dot2 + 2 * qdot_joints_dot3 + qdot_joints_dot4)
            q_roots_integrated[:, i_shooting + 1] = q_roots_this_time
            q_joints_integrated[:, i_shooting + 1] = q_joints_this_time
            qdot_roots_integrated[:, i_shooting + 1] = qdot_roots_this_time
            qdot_joints_integrated[:, i_shooting + 1] = qdot_joints_this_time
        return q_roots_integrated, q_joints_integrated, qdot_roots_integrated, qdot_joints_integrated


    n_q = ocp.nlp[0].model.nb_q
    n_root = ocp.nlp[0].model.nb_root
    n_joints = n_q - n_root

    q_all_ocp = np.zeros((n_q * (nb_random + 1), n_shooting + 1))
    for i_shooting in range(n_shooting + 1):
        for i_random in range(nb_random):
            q_all_ocp[i_random * n_q : (i_random + 1) * n_q, i_shooting] = np.reshape(
                q_ocp[:, i_shooting], (-1,)
            )
        q_all_ocp[(i_random + 1) * n_q : (i_random + 2) * n_q, i_shooting] = np.reshape(
            q_ocp[:, i_shooting], (-1,)
        )

    q_roots_integrated_ocp = np.zeros((n_root, n_shooting + 1, nb_random * nb_reintegrations))
    q_joints_integrated_ocp = np.zeros((n_joints, n_shooting + 1, nb_random * nb_reintegrations))
    qdot_roots_integrated_ocp = np.zeros((n_root, n_shooting + 1, nb_random * nb_reintegrations))
    qdot_joints_integrated_ocp = np.zeros((n_joints, n_shooting + 1, nb_random * nb_reintegrations))
    joint_frictions_ocp = np.zeros((n_joints, n_shooting, nb_random * nb_reintegrations))
    motor_noises_ocp = np.zeros((n_joints, n_shooting, nb_random * nb_reintegrations))
    for i_reintegration in range(nb_reintegrations):

        # Prepare the noises
        np.random.seed(i_reintegration)
        # the last node deos not need motor and sensory noise
        motor_noise_numerical_this_time = np.zeros((n_joints, nb_random, n_shooting + 1))
        for i_random in range(nb_random):
            for i_shooting in range(n_shooting):
                motor_noise_numerical_this_time[:, i_random, i_shooting] = np.random.normal(
                    loc=np.zeros(motor_noise_magnitude.shape[0]),
                    scale=np.reshape(np.array(motor_noise_magnitude), (n_joints,)),
                    size=n_joints,
                )

        initial_cov = np.eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P
        states_init = np.array(
            [-0.0346, 0.1207, 0.2255, 0.0, 3.1, -0.1787, 0.0,  # q
             0, 2, 2.5 * np.pi, 0, 0, 0, 0]  # qdot
        )
        noised_states = np.random.multivariate_normal(
            states_init, initial_cov, nb_random
        ).T

        # initial variability
        q_roots_init_this_time = noised_states[:n_root, :]
        q_joints_init_this_time = noised_states[n_root: n_q, :]
        qdot_roots_init_this_time = noised_states[n_q: n_q + n_root, :]
        qdot_joints_init_this_time = noised_states[n_q + n_root:, :]

        q_roots_integrated, q_joints_integrated, qdot_roots_integrated, qdot_joints_integrated = integrate(
            time_vector_ocp[-1], q_roots_init_this_time, q_joints_init_this_time, qdot_roots_init_this_time,
            qdot_joints_init_this_time,
            tau_joints_ocp,
            motor_noise_numerical_this_time,
            ocp.nlp[0],
            nb_random,
            forward_dynamics_func)

        for i_random in range(nb_random):
            for i_dof in range(n_joints):
                if i_dof < n_root:
                    q_roots_integrated_ocp[i_dof, :, i_reintegration * nb_random + i_random] = q_roots_integrated[i_dof + n_root*i_random, :]
                    qdot_roots_integrated_ocp[i_dof, :, i_reintegration * nb_random + i_random] = qdot_roots_integrated[i_dof + n_root*i_random, :]
                q_joints_integrated_ocp[i_dof, :, i_reintegration * nb_random + i_random] = q_joints_integrated[i_dof + n_joints*i_random, :]
                qdot_joints_integrated_ocp[i_dof, :, i_reintegration * nb_random + i_random] = qdot_joints_integrated[i_dof + n_joints*i_random, :]

        # if i_reintegration == 0:
        #     plt.figure()
        #     for i_shooting in range(5):
        #         for i_random in range(nb_random):
        #             plt.plot(np.ones((3, ))*i_shooting, q_roots_integrated_ocp[:, i_shooting, i_random], '.b')
        #             plt.plot(np.ones((4, ))*i_shooting, q_joints_integrated_ocp[:, i_shooting, i_random], '.b')
        #             plt.scatter(np.ones((7, ))*i_shooting, q_ocp[:, i_shooting], color='m')
        #     plt.savefig("tempo_ocp_0.png")
        #     # plt.show()

        for i_random in range(nb_random):
            for i_shooting in range(n_shooting):
                if i_shooting < n_shooting:
                    # Joint friction
                    joint_frictions_ocp[
                        :,
                        i_shooting,
                        i_reintegration * nb_random
                        + (n_q - 3) * i_random : i_reintegration * nb_random
                        + (n_q - 3) * (i_random + 1),
                    ] = (
                        ocp.nlp[0].model.friction_coefficients
                        @ qdot_joints_integrated_ocp[:, i_shooting, i_reintegration * nb_random + i_random]
                    )

                    # Motor noise
                    motor_noises_ocp[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = np.reshape(motor_noise_numerical_this_time[:, i_random, i_shooting], (-1, ))

    q_ocp_integrated = np.vstack(
        (q_roots_integrated_ocp, q_joints_integrated_ocp)
    )
    qdot_ocp_integrated = np.vstack(
        (qdot_roots_integrated_ocp, qdot_joints_integrated_ocp)
    )

    for i_random in range(nb_random):
        q_all_ocp[
            n_q * (i_random + 1) : n_q * (i_random + 2), :
        ] = q_ocp_integrated[:, :, i_random]

    return (
        q_ocp_integrated,
        qdot_ocp_integrated,
        q_all_ocp,
        joint_frictions_ocp,
        motor_noises_ocp,
    )



def noisy_integrate_socp(
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
):
    def socp_dynamics(q_roots, q_joints, qdot_roots, qdot_joints, tau_joints, k, ref, motor_noise, sensory_noise,
                      nlp, DMS_sensory_reference_func, forward_dynamics_func):

        k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

        n_root = 3
        n_joints = 4

        ddq_roots = None
        ddq_joints = None
        for i in range(nlp.model.nb_random):
            q_this_time = np.hstack((
                q_roots[i * n_root: (i + 1) * n_root], q_joints[i * n_joints: (i + 1) * n_joints]
            ))
            qdot_this_time = np.hstack((
                qdot_roots[i * n_root: (i + 1) * n_root], qdot_joints[i * n_joints: (i + 1) * n_joints]
            ))
            tau_this_time = tau_joints[:]

            # Joint friction
            tau_this_time -= nlp.model.friction_coefficients @ qdot_this_time[n_root:]

            # Motor noise
            tau_this_time += motor_noise[:, i]

            # Feedback
            tau_this_time += k_matrix @ (
                    ref - DMS_sensory_reference_func(q_this_time, qdot_this_time) + sensory_noise[:, i]
            )

            ddq = forward_dynamics_func(q_this_time, qdot_this_time, tau_this_time)
            ddq_roots = np.vstack((ddq_roots, ddq[:n_root])) if ddq_roots is not None else ddq[:n_root]
            ddq_joints = np.vstack((ddq_joints, ddq[n_root:])) if ddq_joints is not None else ddq[n_root:]

        return qdot_roots, qdot_joints, ddq_roots.reshape(-1, ), ddq_joints.reshape(-1, )

    def integrate(time, q_roots, q_joints, qdot_roots, qdot_joints, tau_joints, k, ref, motor_noise, sensory_noise,
                  nlp, DMS_sensory_reference_func, forward_dynamics_func):
        dt = time / n_shooting
        h = dt / 5
        q_roots_integrated = np.zeros((3 * nb_random, n_shooting + 1))
        q_joints_integrated = np.zeros((4 * nb_random, n_shooting + 1))
        qdot_roots_integrated = np.zeros((3 * nb_random, n_shooting + 1))
        qdot_joints_integrated = np.zeros((4 * nb_random, n_shooting + 1))
        for i_random in range(nb_random):
            q_roots_integrated[3*i_random:3*(i_random+1), 0] = q_roots[:, i_random]
            q_joints_integrated[4*i_random:4*(i_random+1), 0] = q_joints[:, i_random]
            qdot_roots_integrated[3*i_random:3*(i_random+1), 0] = qdot_roots[:, i_random]
            qdot_joints_integrated[4*i_random:4*(i_random+1), 0] = qdot_joints[:, i_random]
        for i_shooting in range(n_shooting):
            q_roots_this_time = q_roots_integrated[:, i_shooting]
            q_joints_this_time = q_joints_integrated[:, i_shooting]
            qdot_roots_this_time = qdot_roots_integrated[:, i_shooting]
            qdot_joints_this_time = qdot_joints_integrated[:, i_shooting]
            tau_joints_this_time = tau_joints[:, i_shooting]
            k_this_time = k[:, i_shooting]
            ref_this_time = ref[:, i_shooting]
            motor_noise_this_time = motor_noise[:, :, i_shooting]
            sensory_noise_this_time = sensory_noise[:, :, i_shooting]
            for i_step in range(5):
                q_roots_dot1, q_joints_dot1, qdot_roots_dot1, qdot_joints_dot1 = socp_dynamics(q_roots_this_time,
                                                                                               q_joints_this_time,
                                                                                               qdot_roots_this_time,
                                                                                               qdot_joints_this_time,
                                                                                               tau_joints_this_time,
                                                                                               k_this_time,
                                                                                               ref_this_time,
                                                                                               motor_noise_this_time,
                                                                                               sensory_noise_this_time,
                                                                                               nlp,
                                                                                               DMS_sensory_reference_func,
                                                                                               forward_dynamics_func)
                q_roots_dot2, q_joints_dot2, qdot_roots_dot2, qdot_joints_dot2 = socp_dynamics(
                    q_roots_this_time + h / 2 * q_roots_dot1, q_joints_this_time + h / 2 * q_joints_dot1,
                    qdot_roots_this_time + h / 2 * qdot_roots_dot1,
                    qdot_joints_this_time + h / 2 * qdot_joints_dot1, tau_joints_this_time, k_this_time,
                    ref_this_time, motor_noise_this_time, sensory_noise_this_time, nlp, DMS_sensory_reference_func,
                    forward_dynamics_func)
                q_roots_dot3, q_joints_dot3, qdot_roots_dot3, qdot_joints_dot3 = socp_dynamics(
                    q_roots_this_time + h / 2 * q_roots_dot2, q_joints_this_time + h / 2 * q_joints_dot2,
                    qdot_roots_this_time + h / 2 * qdot_roots_dot2,
                    qdot_joints_this_time + h / 2 * qdot_joints_dot2, tau_joints_this_time, k_this_time,
                    ref_this_time, motor_noise_this_time, sensory_noise_this_time, nlp, DMS_sensory_reference_func,
                    forward_dynamics_func)
                q_roots_dot4, q_joints_dot4, qdot_roots_dot4, qdot_joints_dot4 = socp_dynamics(
                    q_roots_this_time + h * q_roots_dot3, q_joints_this_time + h * q_joints_dot3,
                    qdot_roots_this_time + h * qdot_roots_dot3, qdot_joints_this_time + h * qdot_joints_dot3,
                    tau_joints_this_time, k_this_time, ref_this_time, motor_noise_this_time,
                    sensory_noise_this_time, nlp, DMS_sensory_reference_func, forward_dynamics_func)
                q_roots_this_time = q_roots_this_time + h / 6 * (
                        q_roots_dot1 + 2 * q_roots_dot2 + 2 * q_roots_dot3 + q_roots_dot4)
                q_joints_this_time = q_joints_this_time + h / 6 * (
                        q_joints_dot1 + 2 * q_joints_dot2 + 2 * q_joints_dot3 + q_joints_dot4)
                qdot_roots_this_time = qdot_roots_this_time + h / 6 * (
                        qdot_roots_dot1 + 2 * qdot_roots_dot2 + 2 * qdot_roots_dot3 + qdot_roots_dot4)
                qdot_joints_this_time = qdot_joints_this_time + h / 6 * (
                        qdot_joints_dot1 + 2 * qdot_joints_dot2 + 2 * qdot_joints_dot3 + qdot_joints_dot4)
            q_roots_integrated[:, i_shooting + 1] = q_roots_this_time
            q_joints_integrated[:, i_shooting + 1] = q_joints_this_time
            qdot_roots_integrated[:, i_shooting + 1] = qdot_roots_this_time
            qdot_joints_integrated[:, i_shooting + 1] = qdot_joints_this_time
        return q_roots_integrated, q_joints_integrated, qdot_roots_integrated, qdot_joints_integrated


    n_q = socp.nlp[0].model.nb_q
    n_root = socp.nlp[0].model.nb_root
    n_joints = n_q - n_root

    q_all_socp = np.zeros((n_q * (nb_random + 1), n_shooting + 1))
    for i_shooting in range(n_shooting + 1):
        for i_random in range(nb_random):
            q_all_socp[i_random * n_q : (i_random + 1) * n_q, i_shooting] = np.reshape(
                q_socp[:, i_shooting, i_random], (-1,)
            )
        q_all_socp[(i_random + 1) * n_q : (i_random + 2) * n_q, i_shooting] = np.reshape(
            q_mean_socp[:, i_shooting], (-1,)
        )

    q_roots_integrated_socp = np.zeros((3, n_shooting + 1, nb_random * nb_reintegrations))
    q_joints_integrated_socp = np.zeros((n_q - 3, n_shooting + 1, nb_random * nb_reintegrations))
    qdot_roots_integrated_socp = np.zeros((3, n_shooting + 1, nb_random * nb_reintegrations))
    qdot_joints_integrated_socp = np.zeros((n_q - 3, n_shooting + 1, nb_random * nb_reintegrations))
    joint_frictions_socp = np.zeros((n_q - 3, n_shooting, nb_random * nb_reintegrations))
    motor_noises_socp = np.zeros((n_q - 3, n_shooting, nb_random * nb_reintegrations))
    feedbacks_socp = np.zeros((n_q - 3, n_shooting, nb_random * nb_reintegrations))
    for i_reintegration in range(nb_reintegrations):

        # Prepare the noises
        np.random.seed(i_reintegration)
        # the last node deos not need motor and sensory noise
        motor_noise_numerical_this_time = np.zeros((n_joints, nb_random, n_shooting + 1))
        sensory_noise_numerical_this_time = np.zeros((2 * (n_joints + 1), nb_random, n_shooting + 1))
        for i_random in range(nb_random):
            for i_shooting in range(n_shooting):
                motor_noise_numerical_this_time[:, i_random, i_shooting] = np.random.normal(
                    loc=np.zeros(motor_noise_magnitude.shape[0]),
                    scale=np.reshape(np.array(motor_noise_magnitude), (n_joints,)),
                    size=n_joints,
                )
                sensory_noise_numerical_this_time[:, i_random, i_shooting] = np.random.normal(
                    loc=np.zeros(sensory_noise_magnitude.shape[0]),
                    scale=np.reshape(np.array(sensory_noise_magnitude), (2 * (n_joints + 1),)),
                    size=2 * (n_joints + 1),
                )

        initial_cov = np.eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P
        states_init = np.array(
            [-0.0346, 0.1207, 0.2255, 0.0, 3.1, -0.1787, 0.0,  # q
             0, 2, 2.5 * np.pi, 0, 0, 0, 0]  # qdot
        )
        noised_states = np.random.multivariate_normal(
            states_init, initial_cov, nb_random
        ).T

        # initial variability
        q_roots_init_this_time = noised_states[:n_root, :]
        q_joints_init_this_time = noised_states[n_root: n_q, :]
        qdot_roots_init_this_time = noised_states[n_q: n_q + n_root, :]
        qdot_joints_init_this_time = noised_states[n_q + n_root:, :]

        q_roots_integrated, q_joints_integrated, qdot_roots_integrated, qdot_joints_integrated = integrate(
            time_vector_socp[-1], q_roots_init_this_time, q_joints_init_this_time, qdot_roots_init_this_time,
            qdot_joints_init_this_time, tau_joints_socp,
            k_socp, ref_socp, motor_noise_numerical_this_time, sensory_noise_numerical_this_time,
            socp.nlp[0],
            DMS_sensory_reference_func, forward_dynamics_func)

        for i_random in range(nb_random):
            for i_dof in range(4):
                if i_dof < 3:
                    q_roots_integrated_socp[i_dof, :, i_reintegration * nb_random + i_random] = q_roots_integrated[i_dof + 3*i_random, :]
                    qdot_roots_integrated_socp[i_dof, :, i_reintegration * nb_random + i_random] = qdot_roots_integrated[i_dof + 3*i_random, :]
                q_joints_integrated_socp[i_dof, :, i_reintegration * nb_random + i_random] = q_joints_integrated[i_dof + 4*i_random, :]
                qdot_joints_integrated_socp[i_dof, :, i_reintegration * nb_random + i_random] = qdot_joints_integrated[i_dof + 4*i_random, :]

        # if i_reintegration == 0:
        #     plt.figure()
        #     for i_shooting in range(5):
        #         for i_random in range(nb_random):
        #             plt.plot(np.ones((3, ))*i_shooting, q_roots_integrated_socp[:, i_shooting, i_random], '.b')
        #             plt.plot(np.ones((4, ))*i_shooting, q_joints_integrated_socp[:, i_shooting, i_random], '.b')
        #             plt.scatter(np.ones((7, ))*i_shooting, q_socp[:, i_shooting, i_random], color='m')
        #     plt.savefig("tempo_socp_0.png")
        #     # plt.show()

        for i_random in range(nb_random):
            for i_shooting in range(n_shooting):
                if i_shooting < n_shooting:
                    # Joint friction
                    joint_frictions_socp[
                        :,
                        i_shooting,
                        i_reintegration * nb_random
                        + (n_q - 3) * i_random : i_reintegration * nb_random
                        + (n_q - 3) * (i_random + 1),
                    ] = (
                        socp.nlp[0].model.friction_coefficients
                        @ qdot_joints_integrated_socp[:, i_shooting, i_reintegration * nb_random + i_random]
                    )

                    # Motor noise
                    motor_noises_socp[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = np.reshape(motor_noise_numerical_this_time[:, i_random, i_shooting], (-1, ))

                    # Feedback
                    k_fb_matrix = StochasticBioModel.reshape_to_matrix(
                        k_socp[:, i_shooting], socp.nlp[0].model.matrix_shape_k
                    )
                    feedbacks_socp[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = np.reshape(
                        k_fb_matrix
                        @ (
                            ref_socp[:, i_shooting]
                            - DMS_sensory_reference_func(
                                np.hstack(
                                    (
                                        q_roots_integrated_socp[:, i_shooting, i_reintegration * nb_random + i_random],
                                        q_joints_integrated_socp[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                                np.hstack(
                                    (
                                        qdot_roots_integrated_socp[:, i_shooting, i_reintegration * nb_random + i_random],
                                        qdot_joints_integrated_socp[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                            )
                            + sensory_noise_numerical_this_time[:, i_random, i_shooting]
                        ),
                        (-1, ),
                    )

    q_socp_integrated = np.vstack(
        (q_roots_integrated_socp, q_joints_integrated_socp)
    )
    qdot_socp_integrated = np.vstack(
        (qdot_roots_integrated_socp, qdot_joints_integrated_socp)
    )

    return (
        q_socp_integrated,
        qdot_socp_integrated,
        q_all_socp,
        joint_frictions_socp,
        motor_noises_socp,
        feedbacks_socp,
    )


def noisy_integrate_socp_variable(
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
):

    def socp_variable_dynamics(q_roots, q_joints, qdot_roots, qdot_joints, tau_joints, k, ref_fb, motor_noise, sensory_noise,
                      nlp, DMS_fb_noised_sensory_input_VARIABLE_func, forward_dynamics_func):

        k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

        n_root = 3
        n_joints = 4

        ddq_roots = None
        ddq_joints = None
        for i in range(nlp.model.nb_random):
            q_this_time = np.hstack((
                q_roots[i * n_root: (i + 1) * n_root], q_joints[i * n_joints: (i + 1) * n_joints]
            ))
            qdot_this_time = np.hstack((
                qdot_roots[i * n_root: (i + 1) * n_root], qdot_joints[i * n_joints: (i + 1) * n_joints]
            ))
            tau_this_time = tau_joints[:]

            # Joint friction
            tau_this_time -= nlp.model.friction_coefficients @ qdot_this_time[n_root:]

            # Motor noise
            motor_noise_computed = motor_acuity(motor_noise[:, i], tau_joints)
            tau_this_time += motor_noise_computed

            # Feedback
            tau_this_time += k_matrix @ (
                    ref_fb - DMS_fb_noised_sensory_input_VARIABLE_func(q_this_time, qdot_this_time, sensory_noise[:, i])
            )

            ddq = forward_dynamics_func(q_this_time, qdot_this_time, tau_this_time)
            ddq_roots = np.vstack((ddq_roots, ddq[:n_root])) if ddq_roots is not None else ddq[:n_root]
            ddq_joints = np.vstack((ddq_joints, ddq[n_root:])) if ddq_joints is not None else ddq[n_root:]

        return qdot_roots, qdot_joints, ddq_roots.reshape(-1, ), ddq_joints.reshape(-1, )

    def integrate(tf, q_roots, q_joints, qdot_roots, qdot_joints, tau_joints, k, ref, motor_noise, sensory_noise,
                  nlp, DMS_fb_noised_sensory_input_VARIABLE_func, forward_dynamics_func):
        dt = tf / n_shooting
        h = dt / 5
        q_roots_integrated = np.zeros((n_root * nb_random, n_shooting + 1))
        q_joints_integrated = np.zeros((n_joints * nb_random, n_shooting + 1))
        qdot_roots_integrated = np.zeros((n_root * nb_random, n_shooting + 1))
        qdot_joints_integrated = np.zeros((n_joints * nb_random, n_shooting + 1))
        for i_random in range(nb_random):
            q_roots_integrated[n_root*i_random:n_root*(i_random+1), 0] = q_roots[:, i_random]
            q_joints_integrated[n_joints*i_random:n_joints*(i_random+1), 0] = q_joints[:, i_random]
            qdot_roots_integrated[n_root*i_random:n_root*(i_random+1), 0] = qdot_roots[:, i_random]
            qdot_joints_integrated[n_joints*i_random:n_joints*(i_random+1), 0] = qdot_joints[:, i_random]
        for i_shooting in range(n_shooting):
            q_roots_this_time = q_roots_integrated[:, i_shooting]
            q_joints_this_time = q_joints_integrated[:, i_shooting]
            qdot_roots_this_time = qdot_roots_integrated[:, i_shooting]
            qdot_joints_this_time = qdot_joints_integrated[:, i_shooting]
            tau_joints_this_time = tau_joints[:, i_shooting]
            k_this_time = k[:, i_shooting]
            ref_this_time = ref[:, i_shooting]
            motor_noise_this_time = motor_noise[:, :, i_shooting]
            sensory_noise_this_time = sensory_noise[:, :, i_shooting]
            for i_step in range(5):
                q_roots_dot1, q_joints_dot1, qdot_roots_dot1, qdot_joints_dot1 = socp_variable_dynamics(q_roots_this_time,
                                                                                               q_joints_this_time,
                                                                                               qdot_roots_this_time,
                                                                                               qdot_joints_this_time,
                                                                                               tau_joints_this_time,
                                                                                               k_this_time,
                                                                                               ref_this_time,
                                                                                               motor_noise_this_time,
                                                                                               sensory_noise_this_time,
                                                                                               nlp,
                                                                                               DMS_fb_noised_sensory_input_VARIABLE_func,
                                                                                               forward_dynamics_func)
                q_roots_dot2, q_joints_dot2, qdot_roots_dot2, qdot_joints_dot2 = socp_variable_dynamics(
                    q_roots_this_time + h / 2 * q_roots_dot1,
                    q_joints_this_time + h / 2 * q_joints_dot1,
                    qdot_roots_this_time + h / 2 * qdot_roots_dot1,
                    qdot_joints_this_time + h / 2 * qdot_joints_dot1,
                    tau_joints_this_time,
                    k_this_time,
                    ref_this_time,
                    motor_noise_this_time,
                    sensory_noise_this_time,
                    nlp,
                    DMS_fb_noised_sensory_input_VARIABLE_func,
                    forward_dynamics_func)
                q_roots_dot3, q_joints_dot3, qdot_roots_dot3, qdot_joints_dot3 = socp_variable_dynamics(
                    q_roots_this_time + h / 2 * q_roots_dot2,
                    q_joints_this_time + h / 2 * q_joints_dot2,
                    qdot_roots_this_time + h / 2 * qdot_roots_dot2,
                    qdot_joints_this_time + h / 2 * qdot_joints_dot2,
                    tau_joints_this_time,
                    k_this_time,
                    ref_this_time,
                    motor_noise_this_time,
                    sensory_noise_this_time,
                    nlp,
                    DMS_fb_noised_sensory_input_VARIABLE_func,
                    forward_dynamics_func)
                q_roots_dot4, q_joints_dot4, qdot_roots_dot4, qdot_joints_dot4 = socp_variable_dynamics(
                    q_roots_this_time + h * q_roots_dot3,
                    q_joints_this_time + h * q_joints_dot3,
                    qdot_roots_this_time + h * qdot_roots_dot3,
                    qdot_joints_this_time + h * qdot_joints_dot3,
                    tau_joints_this_time,
                    k_this_time,
                    ref_this_time,
                    motor_noise_this_time,
                    sensory_noise_this_time,
                    nlp,
                    DMS_fb_noised_sensory_input_VARIABLE_func,
                    forward_dynamics_func)
                q_roots_this_time = q_roots_this_time + h / 6 * (
                        q_roots_dot1 + 2 * q_roots_dot2 + 2 * q_roots_dot3 + q_roots_dot4)
                q_joints_this_time = q_joints_this_time + h / 6 * (
                        q_joints_dot1 + 2 * q_joints_dot2 + 2 * q_joints_dot3 + q_joints_dot4)
                qdot_roots_this_time = qdot_roots_this_time + h / 6 * (
                        qdot_roots_dot1 + 2 * qdot_roots_dot2 + 2 * qdot_roots_dot3 + qdot_roots_dot4)
                qdot_joints_this_time = qdot_joints_this_time + h / 6 * (
                        qdot_joints_dot1 + 2 * qdot_joints_dot2 + 2 * qdot_joints_dot3 + qdot_joints_dot4)
            q_roots_integrated[:, i_shooting + 1] = q_roots_this_time
            q_joints_integrated[:, i_shooting + 1] = q_joints_this_time
            qdot_roots_integrated[:, i_shooting + 1] = qdot_roots_this_time
            qdot_joints_integrated[:, i_shooting + 1] = qdot_joints_this_time
        return q_roots_integrated, q_joints_integrated, qdot_roots_integrated, qdot_joints_integrated

    n_q = socp_variable.nlp[0].model.nb_q
    n_root = socp_variable.nlp[0].model.nb_root
    n_joints = n_q - n_root

    q_all_socp_variable = np.zeros((n_q * (nb_random + 1), n_shooting + 1))
    for i_shooting in range(n_shooting + 1):
        for i_random in range(nb_random):
            q_all_socp_variable[i_random * n_q : (i_random + 1) * n_q, i_shooting] = np.reshape(
                q_socp_variable[:, i_shooting, i_random], (-1,)
            )
        q_all_socp_variable[(i_random + 1) * n_q : (i_random + 2) * n_q, i_shooting] = np.reshape(
            q_mean_socp_variable[:, i_shooting], (-1,)
        )

    q_roots_integrated_socp_variable = np.zeros((3, n_shooting + 1, nb_random * nb_reintegrations))
    q_joints_integrated_socp_variable = np.zeros((n_q - 3, n_shooting + 1, nb_random * nb_reintegrations))
    qdot_roots_integrated_socp_variable = np.zeros((3, n_shooting + 1, nb_random * nb_reintegrations))
    qdot_joints_integrated_socp_variable = np.zeros((n_q - 3, n_shooting + 1, nb_random * nb_reintegrations))
    joint_frictions_socp_variable = np.zeros((n_q - 3, n_shooting, nb_random * nb_reintegrations))
    motor_noises_socp_variable = np.zeros((n_q - 3, n_shooting, nb_random * nb_reintegrations))
    feedbacks_socp_variable = np.zeros((n_q - 3, n_shooting, nb_random * nb_reintegrations))
    for i_reintegration in range(nb_reintegrations):

        # Prepare the noises
        np.random.seed(i_reintegration)
        motor_noise_numerical = np.zeros((n_joints, nb_random, n_shooting + 1))
        sensory_noise_numerical = np.zeros((2 * (n_joints + 1), nb_random, n_shooting + 1))
        for i_random in range(nb_random):
            for i_shooting in range(n_shooting):
                motor_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                    loc=np.zeros(motor_noise_magnitude.shape[0]),
                    scale=np.reshape(np.array(motor_noise_magnitude), (n_joints,)),
                    size=n_joints,
                )
                sensory_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                    loc=np.zeros(sensory_noise_magnitude.shape[0]),
                    scale=np.reshape(np.array(sensory_noise_magnitude), (2 * (n_joints + 1),)),
                    size=2 * (n_joints + 1),
                )

        initial_cov = np.eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P
        states_init = np.random.multivariate_normal(
            np.array([-0.0346, 0.1207, 0.2255, 0.0, 3.1, -0.1787, 0, 0, 2, 2.5 * np.pi, 0, 0, 0, 0]), initial_cov, nb_random
        ).T

        # initial variability
        q_roots_init_this_time = states_init[:n_root, :]
        q_joints_init_this_time = states_init[n_root: n_q, :]
        qdot_roots_init_this_time = states_init[n_q: n_q + n_root, :]
        qdot_joints_init_this_time = states_init[n_q + n_root:, :]

        q_roots_integrated, q_joints_integrated, qdot_roots_integrated, qdot_joints_integrated = integrate(
            time_vector_socp_variable[-1],
            q_roots_init_this_time,
            q_joints_init_this_time,
            qdot_roots_init_this_time,
            qdot_joints_init_this_time,
            tau_joints_socp_variable,
            k_socp_variable,
            ref_socp_variable,
            motor_noise_numerical,
            sensory_noise_numerical,
            socp_variable.nlp[0],
            DMS_fb_noised_sensory_input_VARIABLE_func,
            forward_dynamics_func)

        for i_random in range(nb_random):
            for i_dof in range(n_joints):
                if i_dof < 3:
                    q_roots_integrated_socp_variable[i_dof, :, i_reintegration * nb_random + i_random] = q_roots_integrated[i_dof + n_root*i_random, :]
                    qdot_roots_integrated_socp_variable[i_dof, :, i_reintegration * nb_random + i_random] = qdot_roots_integrated[i_dof + n_root*i_random, :]
                q_joints_integrated_socp_variable[i_dof, :, i_reintegration * nb_random + i_random] = q_joints_integrated[i_dof + n_joints*i_random, :]
                qdot_joints_integrated_socp_variable[i_dof, :, i_reintegration * nb_random + i_random] = qdot_joints_integrated[i_dof + n_joints*i_random, :]

        # if i_reintegration == 0:
        #     plt.figure()
        #     for i_shooting in range(5):
        #         for i_random in range(nb_random):
        #             plt.plot(np.ones((n_root, ))*i_shooting, q_roots_integrated_socp_variable[:, i_shooting, i_random], '.b')
        #             plt.plot(np.ones((n_joints, ))*i_shooting, q_joints_integrated_socp_variable[:, i_shooting, i_random], '.b')
        #             plt.scatter(np.ones((n_root + n_joints, ))*i_shooting, q_socp_variable[:, i_shooting, i_random], color='m')
        #     plt.savefig("tempo_socp_variable_0.png")
        #     # plt.show()

        for i_random in range(nb_random):
            for i_shooting in range(n_shooting):
                if i_shooting < n_shooting:

                    k_matrix = StochasticBioModel.reshape_to_matrix(
                        k_socp_variable[:, i_shooting], socp_variable.nlp[0].model.matrix_shape_k
                    )

                    # Joint friction
                    joint_frictions_socp_variable[
                        :,
                        i_shooting,
                        i_reintegration * nb_random
                        + n_joints * i_random : i_reintegration * nb_random
                        + n_joints * (i_random + 1),
                    ] = (
                        socp_variable.nlp[0].model.friction_coefficients
                        @ qdot_joints_integrated_socp_variable[:, i_shooting, i_reintegration * nb_random + i_random]
                    )

                    # Motor noise
                    motor_noise_computed = motor_acuity(motor_noise_numerical[:, i_random, i_shooting], tau_joints_socp_variable[:, i_shooting])
                    motor_noise_computed[1] = 0  # No noise on the eyes
                    motor_noises_socp_variable[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = np.array(motor_noise_computed).reshape(-1, )

                    # Feedback
                    feedbacks_socp_variable[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = np.reshape(
                        k_matrix
                        @ (
                            ref_socp_variable[:, i_shooting]
                            - DMS_fb_noised_sensory_input_VARIABLE_func(
                                np.hstack(
                                    (
                                        q_roots_integrated_socp_variable[:, i_shooting, i_reintegration * nb_random + i_random],
                                        q_joints_integrated_socp_variable[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                                np.hstack(
                                    (
                                        qdot_roots_integrated_socp_variable[:, i_shooting, i_reintegration * nb_random + i_random],
                                        qdot_joints_integrated_socp_variable[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                                sensory_noise_numerical[:, i_random, i_shooting]
                            )
                        ),
                        (-1, ),
                    )


    q_socp_variable_integrated = np.vstack(
        (q_roots_integrated_socp_variable, q_joints_integrated_socp_variable)
    )
    qdot_socp_variable_integrated = np.vstack(
        (qdot_roots_integrated_socp_variable, qdot_joints_integrated_socp_variable)
    )

    return (
        q_socp_variable_integrated,
        qdot_socp_variable_integrated,
        q_all_socp_variable,
        joint_frictions_socp_variable,
        motor_noises_socp_variable,
        feedbacks_socp_variable,
    )



def noisy_integrate_socp_feedforward(
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
):

    def socp_feedforward_dynamics(q_roots, q_joints, qdot_roots, qdot_joints, tau_joints, k, ref_fb, ref_ff, motor_noise, sensory_noise,
                      current_time, tf, nlp, DMS_sensory_reference_no_eyes_func, DMS_ff_sensory_input_func, forward_dynamics_func):

        k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)
        k_matrix_fb = k_matrix[:, :-1]
        k_matrix_ff = k_matrix[:, -1]

        n_root = 3
        n_joints = 5

        ddq_roots = None
        ddq_joints = None
        for i in range(nlp.model.nb_random):
            q_this_time = np.hstack((
                q_roots[i * n_root: (i + 1) * n_root], q_joints[i * n_joints: (i + 1) * n_joints]
            ))
            qdot_this_time = np.hstack((
                qdot_roots[i * n_root: (i + 1) * n_root], qdot_joints[i * n_joints: (i + 1) * n_joints]
            ))
            tau_this_time = tau_joints[:]

            # Joint friction
            tau_this_time -= nlp.model.friction_coefficients @ qdot_this_time[n_root:]

            # Motor noise
            tau_this_time += motor_noise[:, i]

            # Feedback
            tau_this_time += k_matrix_fb @ (
                    ref_fb - DMS_sensory_reference_no_eyes_func(q_this_time, qdot_this_time) + sensory_noise[:-1, i]
            )

            # Feedforwards
            tau_this_time += k_matrix_ff @ (
                    ref_ff - DMS_ff_sensory_input_func(tf, current_time, q_this_time, qdot_this_time) + sensory_noise[-1, i]
            )

            ddq = forward_dynamics_func(q_this_time, qdot_this_time, tau_this_time)
            ddq_roots = np.vstack((ddq_roots, ddq[:n_root])) if ddq_roots is not None else ddq[:n_root]
            ddq_joints = np.vstack((ddq_joints, ddq[n_root:])) if ddq_joints is not None else ddq[n_root:]

        return qdot_roots, qdot_joints, ddq_roots.reshape(-1, ), ddq_joints.reshape(-1, )

    def integrate(tf, q_roots, q_joints, qdot_roots, qdot_joints, tau_joints, k, ref_fb, ref_ff, motor_noise, sensory_noise,
                  nlp, DMS_sensory_reference_no_eyes_func, DMS_ff_sensory_input_func, forward_dynamics_func):
        dt = tf / n_shooting
        h = dt / 5
        q_roots_integrated = np.zeros((n_root * nb_random, n_shooting + 1))
        q_joints_integrated = np.zeros((n_joints * nb_random, n_shooting + 1))
        qdot_roots_integrated = np.zeros((n_root * nb_random, n_shooting + 1))
        qdot_joints_integrated = np.zeros((n_joints * nb_random, n_shooting + 1))
        for i_random in range(nb_random):
            q_roots_integrated[n_root*i_random:n_root*(i_random+1), 0] = q_roots[:, i_random]
            q_joints_integrated[n_joints*i_random:n_joints*(i_random+1), 0] = q_joints[:, i_random]
            qdot_roots_integrated[n_root*i_random:n_root*(i_random+1), 0] = qdot_roots[:, i_random]
            qdot_joints_integrated[n_joints*i_random:n_joints*(i_random+1), 0] = qdot_joints[:, i_random]
        for i_shooting in range(n_shooting):
            q_roots_this_time = q_roots_integrated[:, i_shooting]
            q_joints_this_time = q_joints_integrated[:, i_shooting]
            qdot_roots_this_time = qdot_roots_integrated[:, i_shooting]
            qdot_joints_this_time = qdot_joints_integrated[:, i_shooting]
            tau_joints_this_time = tau_joints[:, i_shooting]
            k_this_time = k[:, i_shooting]
            ref_fb_this_time = ref_fb[:, i_shooting]
            ref_ff_this_time = ref_ff
            motor_noise_this_time = motor_noise[:, :, i_shooting]
            sensory_noise_this_time = sensory_noise[:, :, i_shooting]
            current_time = dt*i_shooting
            for i_step in range(5):
                q_roots_dot1, q_joints_dot1, qdot_roots_dot1, qdot_joints_dot1 = socp_feedforward_dynamics(q_roots_this_time,
                                                                                               q_joints_this_time,
                                                                                               qdot_roots_this_time,
                                                                                               qdot_joints_this_time,
                                                                                               tau_joints_this_time,
                                                                                               k_this_time,
                                                                                               ref_fb_this_time,
                                                                                               ref_ff_this_time,
                                                                                               motor_noise_this_time,
                                                                                               sensory_noise_this_time,
                                                                                               current_time,
                                                                                               tf,
                                                                                               nlp,
                                                                                               DMS_sensory_reference_no_eyes_func,
                                                                                               DMS_ff_sensory_input_func,
                                                                                               forward_dynamics_func)
                q_roots_dot2, q_joints_dot2, qdot_roots_dot2, qdot_joints_dot2 = socp_feedforward_dynamics(
                    q_roots_this_time + h / 2 * q_roots_dot1,
                    q_joints_this_time + h / 2 * q_joints_dot1,
                    qdot_roots_this_time + h / 2 * qdot_roots_dot1,
                    qdot_joints_this_time + h / 2 * qdot_joints_dot1,
                    tau_joints_this_time,
                    k_this_time,
                    ref_fb_this_time,
                    ref_ff_this_time,
                    motor_noise_this_time,
                    sensory_noise_this_time,
                    current_time + h / 2,
                    tf,
                    nlp,
                    DMS_sensory_reference_no_eyes_func,
                    DMS_ff_sensory_input_func,
                    forward_dynamics_func)
                q_roots_dot3, q_joints_dot3, qdot_roots_dot3, qdot_joints_dot3 = socp_feedforward_dynamics(
                    q_roots_this_time + h / 2 * q_roots_dot2,
                    q_joints_this_time + h / 2 * q_joints_dot2,
                    qdot_roots_this_time + h / 2 * qdot_roots_dot2,
                    qdot_joints_this_time + h / 2 * qdot_joints_dot2,
                    tau_joints_this_time,
                    k_this_time,
                    ref_fb_this_time,
                    ref_ff_this_time,
                    motor_noise_this_time,
                    sensory_noise_this_time,
                    current_time + h / 2,
                    tf,
                    nlp,
                    DMS_sensory_reference_no_eyes_func,
                    DMS_ff_sensory_input_func,
                    forward_dynamics_func)
                q_roots_dot4, q_joints_dot4, qdot_roots_dot4, qdot_joints_dot4 = socp_feedforward_dynamics(
                    q_roots_this_time + h * q_roots_dot3,
                    q_joints_this_time + h * q_joints_dot3,
                    qdot_roots_this_time + h * qdot_roots_dot3,
                    qdot_joints_this_time + h * qdot_joints_dot3,
                    tau_joints_this_time,
                    k_this_time,
                    ref_fb_this_time,
                    ref_ff_this_time,
                    motor_noise_this_time,
                    sensory_noise_this_time,
                    current_time + h,
                    tf,
                    nlp,
                    DMS_sensory_reference_no_eyes_func,
                    DMS_ff_sensory_input_func,
                    forward_dynamics_func)
                q_roots_this_time = q_roots_this_time + h / 6 * (
                        q_roots_dot1 + 2 * q_roots_dot2 + 2 * q_roots_dot3 + q_roots_dot4)
                q_joints_this_time = q_joints_this_time + h / 6 * (
                        q_joints_dot1 + 2 * q_joints_dot2 + 2 * q_joints_dot3 + q_joints_dot4)
                qdot_roots_this_time = qdot_roots_this_time + h / 6 * (
                        qdot_roots_dot1 + 2 * qdot_roots_dot2 + 2 * qdot_roots_dot3 + qdot_roots_dot4)
                qdot_joints_this_time = qdot_joints_this_time + h / 6 * (
                        qdot_joints_dot1 + 2 * qdot_joints_dot2 + 2 * qdot_joints_dot3 + qdot_joints_dot4)
                current_time += h

            q_roots_integrated[:, i_shooting + 1] = q_roots_this_time
            q_joints_integrated[:, i_shooting + 1] = q_joints_this_time
            qdot_roots_integrated[:, i_shooting + 1] = qdot_roots_this_time
            qdot_joints_integrated[:, i_shooting + 1] = qdot_joints_this_time
        return q_roots_integrated, q_joints_integrated, qdot_roots_integrated, qdot_joints_integrated


    n_q = socp_feedforward.nlp[0].model.nb_q
    n_root = socp_feedforward.nlp[0].model.nb_root
    n_joints = n_q - n_root

    q_all_socp_feedforward = np.zeros((n_q * (nb_random + 1), n_shooting + 1))
    for i_shooting in range(n_shooting + 1):
        for i_random in range(nb_random):
            q_all_socp_feedforward[i_random * n_q : (i_random + 1) * n_q, i_shooting] = np.reshape(
                q_socp_feedforward[:, i_shooting, i_random], (-1,)
            )
        q_all_socp_feedforward[(i_random + 1) * n_q : (i_random + 2) * n_q, i_shooting] = np.reshape(
            q_mean_socp_feedforward[:, i_shooting], (-1,)
        )

    q_roots_integrated_socp_feedforward = np.zeros((n_root, n_shooting + 1, nb_random * nb_reintegrations))
    q_joints_integrated_socp_feedforward = np.zeros((n_joints, n_shooting + 1, nb_random * nb_reintegrations))
    qdot_roots_integrated_socp_feedforward = np.zeros((n_root, n_shooting + 1, nb_random * nb_reintegrations))
    qdot_joints_integrated_socp_feedforward = np.zeros((n_joints, n_shooting + 1, nb_random * nb_reintegrations))
    joint_frictions_socp_feedforward = np.zeros((n_joints, n_shooting, nb_random * nb_reintegrations))
    motor_noises_socp_feedforward = np.zeros((n_joints, n_shooting, nb_random * nb_reintegrations))
    feedbacks_socp_feedforward = np.zeros((n_joints, n_shooting, nb_random * nb_reintegrations))
    feedforwards_socp_feedforward = np.zeros((n_joints, n_shooting, nb_random * nb_reintegrations))
    for i_reintegration in range(nb_reintegrations):

        # Prepare the noises
        np.random.seed(i_reintegration)
        # the last node deos not need motor and sensory noise
        motor_noise_numerical = np.zeros((n_joints, nb_random, n_shooting + 1))
        sensory_noise_numerical = np.zeros((2 * n_joints + 1, nb_random, n_shooting + 1))
        for i_random in range(nb_random):
            for i_shooting in range(n_shooting):
                motor_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                    loc=np.zeros(motor_noise_magnitude.shape[0]),
                    scale=np.reshape(np.array(motor_noise_magnitude), (n_joints,)),
                    size=n_joints,
                )
                sensory_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                    loc=np.zeros(sensory_noise_magnitude.shape[0]),
                    scale=np.reshape(np.array(sensory_noise_magnitude), (2 * n_joints + 1,)),
                    size=2 * n_joints + 1,
                )
        motor_noise_numerical[1, :, :] = 0  # No noise on the eyes

        initial_cov = np.eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P
        states_init = np.random.multivariate_normal(
            np.array([-0.0346, 0.1207, 0.2255, 0.0, 0.0045, 3.1, -0.1787, 0, 0, 2, 2.5 * np.pi, 0, 0, 0, 0, 0]), initial_cov, nb_random
        ).T

        # initial variability
        q_roots_init_this_time = states_init[:n_root, :]
        q_joints_init_this_time = states_init[n_root:n_q, :]
        qdot_roots_init_this_time = states_init[n_q: n_q + n_root, :]
        qdot_joints_init_this_time = states_init[n_q + n_root:, :]

        q_roots_integrated, q_joints_integrated, qdot_roots_integrated, qdot_joints_integrated = integrate(
            time_vector_socp_feedforward[-1],
            q_roots_init_this_time,
            q_joints_init_this_time,
            qdot_roots_init_this_time,
            qdot_joints_init_this_time,
            tau_joints_socp_feedforward,
            k_socp_feedforward,
            ref_fb_socp_feedforward,
            ref_ff_socp_feedforward,
            motor_noise_numerical,
            sensory_noise_numerical,
            socp_feedforward.nlp[0],
            DMS_sensory_reference_no_eyes_func,
            DMS_ff_sensory_input_func,
            forward_dynamics_func)

        for i_random in range(nb_random):
            for i_dof in range(n_joints):
                if i_dof < n_root:
                    q_roots_integrated_socp_feedforward[i_dof, :, i_reintegration * nb_random + i_random] = q_roots_integrated[i_dof + n_root*i_random, :]
                    qdot_roots_integrated_socp_feedforward[i_dof, :, i_reintegration * nb_random + i_random] = qdot_roots_integrated[i_dof + n_root*i_random, :]
                q_joints_integrated_socp_feedforward[i_dof, :, i_reintegration * nb_random + i_random] = q_joints_integrated[i_dof + n_joints*i_random, :]
                qdot_joints_integrated_socp_feedforward[i_dof, :, i_reintegration * nb_random + i_random] = qdot_joints_integrated[i_dof + n_joints*i_random, :]

        # if i_reintegration == 0:
        #     plt.figure()
        #     for i_shooting in range(n_joints):
        #         for i_random in range(nb_random):
        #             plt.plot(np.ones((n_root + n_joints,)) * i_shooting, q_socp_feedforward[:, i_shooting, i_random], 'om')
        #             plt.plot(np.ones((n_root, ))*i_shooting, q_roots_integrated_socp_feedforward[:, i_shooting, i_random], '.b')
        #             plt.plot(np.ones((n_joints, ))*i_shooting, q_joints_integrated_socp_feedforward[:, i_shooting, i_random], '.b')
        #     plt.savefig("tempo_socp_feedforward_0.png")
        #     # plt.show()

        tf = time_vector_socp_feedforward[-1]
        dt = tf / n_shooting
        for i_random in range(nb_random):
            for i_shooting in range(n_shooting):
                if i_shooting < n_shooting:

                    k_matrix = StochasticBioModel.reshape_to_matrix(
                        k_socp_feedforward[:, i_shooting], socp_feedforward.nlp[0].model.matrix_shape_k
                    )

                    # Joint friction
                    joint_frictions_socp_feedforward[
                        :,
                        i_shooting,
                        i_reintegration * nb_random
                        + n_joints * i_random : i_reintegration * nb_random
                        + n_joints * (i_random + 1),
                    ] = (
                        socp_feedforward.nlp[0].model.friction_coefficients
                        @ qdot_joints_integrated_socp_feedforward[:, i_shooting, i_reintegration * nb_random + i_random]
                    )

                    # Motor noise
                    motor_noises_socp_feedforward[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = motor_noise_numerical[:, i_random, i_shooting]

                    # Feedback
                    k_fb_matrix = k_matrix[:, :-1]
                    feedbacks_socp_feedforward[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = np.reshape(
                        k_fb_matrix
                        @ (
                            ref_fb_socp_feedforward[:, i_shooting]
                            - DMS_sensory_reference_no_eyes_func(
                                np.hstack(
                                    (
                                        q_roots_integrated_socp_feedforward[:, i_shooting, i_reintegration * nb_random + i_random],
                                        q_joints_integrated_socp_feedforward[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                                np.hstack(
                                    (
                                        qdot_roots_integrated_socp_feedforward[:, i_shooting, i_reintegration * nb_random + i_random],
                                        qdot_joints_integrated_socp_feedforward[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                            ) + sensory_noise_numerical[:-1, i_random, i_shooting]
                        ),
                        (-1, ),
                    )

                    # Feedforward
                    # TODO: Not exactly true, since this is the value at the node, not during the integration
                    k_ff_matrix = k_matrix[:, -1]
                    feedforwards_socp_feedforward[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = np.reshape(
                        k_ff_matrix
                        @ (
                            ref_ff_socp_feedforward
                            - DMS_ff_sensory_input_func(
                                time_vector_socp_feedforward[-1],
                                dt * i_shooting,
                                np.hstack(
                                    (
                                        q_roots_integrated_socp_feedforward[:, i_shooting, i_reintegration * nb_random + i_random],
                                        q_joints_integrated_socp_feedforward[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                                np.hstack(
                                    (
                                        qdot_roots_integrated_socp_feedforward[:, i_shooting, i_reintegration * nb_random + i_random],
                                        qdot_joints_integrated_socp_feedforward[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                            ) + sensory_noise_numerical[-1, i_random, i_shooting]
                        ),
                        (-1, ),
                    )

    q_socp_feedforward_integrated = np.vstack(
        (q_roots_integrated_socp_feedforward, q_joints_integrated_socp_feedforward)
    )
    qdot_socp_feedforward_integrated = np.vstack(
        (qdot_roots_integrated_socp_feedforward, qdot_joints_integrated_socp_feedforward)
    )

    return (
        q_socp_feedforward_integrated,
        qdot_socp_feedforward_integrated,
        q_all_socp_feedforward,
        joint_frictions_socp_feedforward,
        motor_noises_socp_feedforward,
        feedbacks_socp_feedforward,
        feedforwards_socp_feedforward,
    )


def noisy_integrate_socp_plus(
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
):

    def socp_plus_dynamics(q_roots, q_joints, qdot_roots, qdot_joints, tau_joints, k, ref_fb, ref_ff, motor_noise, sensory_noise,
                      current_time, tf, nlp, DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func, DMS_ff_noised_sensory_input_func, forward_dynamics_func):

        k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)
        k_matrix_fb = k_matrix[:, :-1]
        k_matrix_ff = k_matrix[:, -1]

        n_root = 3
        n_joints = 5

        ddq_roots = None
        ddq_joints = None
        for i in range(nlp.model.nb_random):
            q_this_time = np.hstack((
                q_roots[i * n_root: (i + 1) * n_root], q_joints[i * n_joints: (i + 1) * n_joints]
            ))
            qdot_this_time = np.hstack((
                qdot_roots[i * n_root: (i + 1) * n_root], qdot_joints[i * n_joints: (i + 1) * n_joints]
            ))
            tau_this_time = tau_joints[:]

            # Joint friction
            tau_this_time -= nlp.model.friction_coefficients @ qdot_this_time[n_root:]

            # Motor noise
            motor_noise_computed = motor_acuity(motor_noise[:, i], tau_joints)
            motor_noise_computed[1] = 0  # No noise on the eyes
            tau_this_time += motor_noise_computed

            # Feedback
            tau_this_time += k_matrix_fb @ (
                    ref_fb - DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func(q_this_time, qdot_this_time, sensory_noise[:, i])
            )

            # Feedforwards
            tau_this_time += k_matrix_ff @ (
                    ref_ff - DMS_ff_noised_sensory_input_func(tf, current_time, q_this_time, qdot_this_time, sensory_noise[-1, i])
            )

            ddq = forward_dynamics_func(q_this_time, qdot_this_time, tau_this_time)
            ddq_roots = np.vstack((ddq_roots, ddq[:n_root])) if ddq_roots is not None else ddq[:n_root]
            ddq_joints = np.vstack((ddq_joints, ddq[n_root:])) if ddq_joints is not None else ddq[n_root:]

        return qdot_roots, qdot_joints, ddq_roots.reshape(-1, ), ddq_joints.reshape(-1, )

    def integrate(tf, q_roots, q_joints, qdot_roots, qdot_joints, tau_joints, k, ref_fb, ref_ff, motor_noise, sensory_noise,
                  nlp, DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func, DMS_ff_noised_sensory_input_func, forward_dynamics_func):
        n_steps = 5
        dt = tf / n_shooting
        h = dt / n_steps
        q_roots_integrated = np.zeros((n_root * nb_random, n_shooting + 1))
        q_joints_integrated = np.zeros((n_joints * nb_random, n_shooting + 1))
        qdot_roots_integrated = np.zeros((n_root * nb_random, n_shooting + 1))
        qdot_joints_integrated = np.zeros((n_joints * nb_random, n_shooting + 1))
        for i_random in range(nb_random):
            q_roots_integrated[n_root*i_random:n_root*(i_random+1), 0] = q_roots[:, i_random]
            q_joints_integrated[n_joints*i_random:n_joints*(i_random+1), 0] = q_joints[:, i_random]
            qdot_roots_integrated[n_root*i_random:n_root*(i_random+1), 0] = qdot_roots[:, i_random]
            qdot_joints_integrated[n_joints*i_random:n_joints*(i_random+1), 0] = qdot_joints[:, i_random]
        for i_shooting in range(n_shooting):
            q_roots_this_time = q_roots_integrated[:, i_shooting].copy()
            q_joints_this_time = q_joints_integrated[:, i_shooting].copy()
            qdot_roots_this_time = qdot_roots_integrated[:, i_shooting].copy()
            qdot_joints_this_time = qdot_joints_integrated[:, i_shooting].copy()
            tau_joints_this_time = tau_joints[:, i_shooting]
            k_this_time = k[:, i_shooting]
            ref_fb_this_time = ref_fb[:, i_shooting]
            ref_ff_this_time = ref_ff
            motor_noise_this_time = motor_noise[:, :, i_shooting]
            sensory_noise_this_time = sensory_noise[:, :, i_shooting]
            current_time = dt*i_shooting
            for i_step in range(n_steps):
                q_roots_dot1, q_joints_dot1, qdot_roots_dot1, qdot_joints_dot1 = socp_plus_dynamics(q_roots_this_time,
                                                                                               q_joints_this_time,
                                                                                               qdot_roots_this_time,
                                                                                               qdot_joints_this_time,
                                                                                               tau_joints_this_time,
                                                                                               k_this_time,
                                                                                               ref_fb_this_time,
                                                                                               ref_ff_this_time,
                                                                                               motor_noise_this_time,
                                                                                               sensory_noise_this_time,
                                                                                               current_time,
                                                                                               tf,
                                                                                               nlp,
                                                                                               DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func,
                                                                                               DMS_ff_noised_sensory_input_func,
                                                                                               forward_dynamics_func)
                q_roots_dot2, q_joints_dot2, qdot_roots_dot2, qdot_joints_dot2 = socp_plus_dynamics(
                    q_roots_this_time + h / 2 * q_roots_dot1,
                    q_joints_this_time + h / 2 * q_joints_dot1,
                    qdot_roots_this_time + h / 2 * qdot_roots_dot1,
                    qdot_joints_this_time + h / 2 * qdot_joints_dot1,
                    tau_joints_this_time,
                    k_this_time,
                    ref_fb_this_time,
                    ref_ff_this_time,
                    motor_noise_this_time,
                    sensory_noise_this_time,
                    current_time + h / 2,
                    tf,
                    nlp,
                    DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func,
                    DMS_ff_noised_sensory_input_func,
                    forward_dynamics_func)
                q_roots_dot3, q_joints_dot3, qdot_roots_dot3, qdot_joints_dot3 = socp_plus_dynamics(
                    q_roots_this_time + h / 2 * q_roots_dot2,
                    q_joints_this_time + h / 2 * q_joints_dot2,
                    qdot_roots_this_time + h / 2 * qdot_roots_dot2,
                    qdot_joints_this_time + h / 2 * qdot_joints_dot2,
                    tau_joints_this_time,
                    k_this_time,
                    ref_fb_this_time,
                    ref_ff_this_time,
                    motor_noise_this_time,
                    sensory_noise_this_time,
                    current_time + h / 2,
                    tf,
                    nlp,
                    DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func,
                    DMS_ff_noised_sensory_input_func,
                    forward_dynamics_func)
                q_roots_dot4, q_joints_dot4, qdot_roots_dot4, qdot_joints_dot4 = socp_plus_dynamics(
                    q_roots_this_time + h * q_roots_dot3,
                    q_joints_this_time + h * q_joints_dot3,
                    qdot_roots_this_time + h * qdot_roots_dot3,
                    qdot_joints_this_time + h * qdot_joints_dot3,
                    tau_joints_this_time,
                    k_this_time,
                    ref_fb_this_time,
                    ref_ff_this_time,
                    motor_noise_this_time,
                    sensory_noise_this_time,
                    current_time + h,
                    tf,
                    nlp,
                    DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func,
                    DMS_ff_noised_sensory_input_func,
                    forward_dynamics_func)
                q_roots_this_time = q_roots_this_time + h / 6 * (
                        q_roots_dot1 + 2 * q_roots_dot2 + 2 * q_roots_dot3 + q_roots_dot4)
                q_joints_this_time = q_joints_this_time + h / 6 * (
                        q_joints_dot1 + 2 * q_joints_dot2 + 2 * q_joints_dot3 + q_joints_dot4)
                qdot_roots_this_time = qdot_roots_this_time + h / 6 * (
                        qdot_roots_dot1 + 2 * qdot_roots_dot2 + 2 * qdot_roots_dot3 + qdot_roots_dot4)
                qdot_joints_this_time = qdot_joints_this_time + h / 6 * (
                        qdot_joints_dot1 + 2 * qdot_joints_dot2 + 2 * qdot_joints_dot3 + qdot_joints_dot4)
                current_time += h

            q_roots_integrated[:, i_shooting + 1] = q_roots_this_time
            q_joints_integrated[:, i_shooting + 1] = q_joints_this_time
            qdot_roots_integrated[:, i_shooting + 1] = qdot_roots_this_time
            qdot_joints_integrated[:, i_shooting + 1] = qdot_joints_this_time
        return q_roots_integrated, q_joints_integrated, qdot_roots_integrated, qdot_joints_integrated


    n_q = socp_plus.nlp[0].model.nb_q
    n_root = socp_plus.nlp[0].model.nb_root
    n_joints = n_q - n_root

    q_all_socp_plus = np.zeros((n_q * (nb_random + 1), n_shooting + 1))
    for i_shooting in range(n_shooting + 1):
        for i_random in range(nb_random):
            q_all_socp_plus[i_random * n_q : (i_random + 1) * n_q, i_shooting] = np.reshape(
                q_socp_plus[:, i_shooting, i_random], (-1,)
            )
        q_all_socp_plus[(i_random + 1) * n_q : (i_random + 2) * n_q, i_shooting] = np.reshape(
            q_mean_socp_plus[:, i_shooting], (-1,)
        )

    q_roots_integrated_socp_plus = np.zeros((n_root, n_shooting + 1, nb_random * nb_reintegrations))
    q_joints_integrated_socp_plus = np.zeros((n_joints, n_shooting + 1, nb_random * nb_reintegrations))
    qdot_roots_integrated_socp_plus = np.zeros((n_root, n_shooting + 1, nb_random * nb_reintegrations))
    qdot_joints_integrated_socp_plus = np.zeros((n_joints, n_shooting + 1, nb_random * nb_reintegrations))
    joint_frictions_socp_plus = np.zeros((n_joints, n_shooting, nb_random * nb_reintegrations))
    motor_noises_socp_plus = np.zeros((n_joints, n_shooting, nb_random * nb_reintegrations))
    feedbacks_socp_plus = np.zeros((n_joints, n_shooting, nb_random * nb_reintegrations))
    feedforwards_socp_plus = np.zeros((n_joints, n_shooting, nb_random * nb_reintegrations))
    for i_reintegration in range(nb_reintegrations):

        # Prepare the noises
        np.random.seed(i_reintegration)
        # the last node deos not need motor and sensory noise
        motor_noise_numerical = np.zeros((n_joints, nb_random, n_shooting + 1))
        sensory_noise_numerical = np.zeros((2 * n_joints + 1, nb_random, n_shooting + 1))
        for i_random in range(nb_random):
            for i_shooting in range(n_shooting):
                motor_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                    loc=np.zeros(motor_noise_magnitude.shape[0]),
                    scale=np.reshape(np.array(motor_noise_magnitude), (n_joints,)),
                    size=n_joints,
                )
                sensory_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                    loc=np.zeros(sensory_noise_magnitude.shape[0]),
                    scale=np.reshape(np.array(sensory_noise_magnitude), (2 * n_joints + 1,)),
                    size=2 * n_joints + 1,
                )
        motor_noise_numerical[1, :, :] = 0  # No noise on the eyes

        initial_cov = np.eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P
        states_init = np.random.multivariate_normal(
            np.array([-0.0346, 0.1207, 0.2255, 0.0, 0.0045, 3.1, -0.1787, 0, 0, 2, 2.5 * np.pi, 0, 0, 0, 0, 0]), initial_cov, nb_random
        ).T

        # initial variability
        q_roots_init_this_time = states_init[:n_root, :]
        q_joints_init_this_time = states_init[n_root:n_q, :]
        qdot_roots_init_this_time = states_init[n_q:n_q + n_root, :]
        qdot_joints_init_this_time = states_init[n_q + n_root:, :]

        q_roots_integrated, q_joints_integrated, qdot_roots_integrated, qdot_joints_integrated = integrate(
            time_vector_socp_plus[-1],
            q_roots_init_this_time,
            q_joints_init_this_time,
            qdot_roots_init_this_time,
            qdot_joints_init_this_time,
            tau_joints_socp_plus,
            k_socp_plus,
            ref_fb_socp_plus,
            ref_ff_socp_plus,
            motor_noise_numerical,
            sensory_noise_numerical,
            socp_plus.nlp[0],
            DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func,
            DMS_ff_noised_sensory_input_func,
            forward_dynamics_func)

        for i_random in range(nb_random):
            for i_dof in range(n_joints):
                if i_dof < n_root:
                    q_roots_integrated_socp_plus[i_dof, :, i_reintegration * nb_random + i_random] = q_roots_integrated[i_dof + n_root*i_random, :]
                    qdot_roots_integrated_socp_plus[i_dof, :, i_reintegration * nb_random + i_random] = qdot_roots_integrated[i_dof + n_root*i_random, :]
                q_joints_integrated_socp_plus[i_dof, :, i_reintegration * nb_random + i_random] = q_joints_integrated[i_dof + n_joints*i_random, :]
                qdot_joints_integrated_socp_plus[i_dof, :, i_reintegration * nb_random + i_random] = qdot_joints_integrated[i_dof + n_joints*i_random, :]

        # if i_reintegration == 0:
        #     plt.figure()
        #     for i_shooting in range(n_joints):
        #         for i_random in range(nb_random):
        #             plt.plot(np.ones((n_root + n_joints, ))*i_shooting, q_socp_plus[:, i_shooting, i_random], 'om')
        #             plt.plot(np.ones((n_root, ))*i_shooting, q_roots_integrated_socp_plus[:, i_shooting, i_random], '.b')
        #             plt.plot(np.ones((n_joints, ))*i_shooting, q_joints_integrated_socp_plus[:, i_shooting, i_random], '.b')
        #     plt.savefig("tempo_socp_plus_0.png")
        #     # plt.show()

        tf = time_vector_socp_plus[-1]
        dt = tf / n_shooting
        for i_random in range(nb_random):
            for i_shooting in range(n_shooting):
                if i_shooting < n_shooting:

                    k_matrix = StochasticBioModel.reshape_to_matrix(
                        k_socp_plus[:, i_shooting], socp_plus.nlp[0].model.matrix_shape_k
                    )

                    # Joint friction
                    joint_frictions_socp_plus[
                        :,
                        i_shooting,
                        i_reintegration * nb_random
                        + n_joints * i_random : i_reintegration * nb_random
                        + n_joints * (i_random + 1),
                    ] = (
                        socp_plus.nlp[0].model.friction_coefficients
                        @ qdot_joints_integrated_socp_plus[:, i_shooting, i_reintegration * nb_random + i_random]
                    )

                    # Motor noise
                    motor_noise_computed = motor_acuity(motor_noise_numerical[:, i_random, i_shooting], tau_joints_socp_plus[:, i_shooting])
                    motor_noise_computed[1] = 0  # No noise on the eyes
                    motor_noises_socp_plus[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = np.array(motor_noise_computed).reshape(-1, )

                    # Feedback
                    k_fb_matrix = k_matrix[:, :-1]
                    feedbacks_socp_plus[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = np.reshape(
                        k_fb_matrix
                        @ (
                            ref_fb_socp_plus[:, i_shooting]
                            - DMS_fb_noised_sensory_input_VARIABLE_no_eyes_func(
                                np.hstack(
                                    (
                                        q_roots_integrated_socp_plus[:, i_shooting, i_reintegration * nb_random + i_random],
                                        q_joints_integrated_socp_plus[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                                np.hstack(
                                    (
                                        qdot_roots_integrated_socp_plus[:, i_shooting, i_reintegration * nb_random + i_random],
                                        qdot_joints_integrated_socp_plus[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                                sensory_noise_numerical[:, i_random, i_shooting]
                            )
                        ),
                        (-1, ),
                    )

                    # Feedforward
                    # TODO: Not exactly true, since this is the value at the node, not during the integration
                    k_ff_matrix = k_matrix[:, -1]
                    feedforwards_socp_plus[
                        :,
                        i_shooting,
                        i_reintegration * nb_random + i_random,
                    ] = np.reshape(
                        k_ff_matrix
                        @ (
                            ref_ff_socp_plus
                            - DMS_ff_noised_sensory_input_func(
                                time_vector_socp_plus[-1],
                                dt * i_shooting,
                                np.hstack(
                                    (
                                        q_roots_integrated_socp_plus[:, i_shooting, i_reintegration * nb_random + i_random],
                                        q_joints_integrated_socp_plus[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                                np.hstack(
                                    (
                                        qdot_roots_integrated_socp_plus[:, i_shooting, i_reintegration * nb_random + i_random],
                                        qdot_joints_integrated_socp_plus[:, i_shooting, i_reintegration * nb_random + i_random],
                                    )
                                ),
                                sensory_noise_numerical[-1, i_random, i_shooting]
                            )
                        ),
                        (-1, ),
                    )

    q_socp_plus_integrated = np.vstack(
        (q_roots_integrated_socp_plus, q_joints_integrated_socp_plus)
    )
    qdot_socp_plus_integrated = np.vstack(
        (qdot_roots_integrated_socp_plus, qdot_joints_integrated_socp_plus)
    )

    return (
        q_socp_plus_integrated,
        qdot_socp_plus_integrated,
        q_all_socp_plus,
        joint_frictions_socp_plus,
        motor_noises_socp_plus,
        feedbacks_socp_plus,
        feedforwards_socp_plus,
    )