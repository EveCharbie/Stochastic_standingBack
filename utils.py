import numpy as np
import casadi as cas
import biorbd_casadi as biorbd

import sys

sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    PenaltyController,
    DynamicsFunctions,
    NonLinearProgram,
    StochasticBioModel,
)


def CoM_over_toes(controller: PenaltyController) -> cas.MX:
    q_roots = controller.states["q_roots"].cx_start
    q_joints = controller.states["q_joints"].cx_start
    q = cas.vertcat(q_roots, q_joints)
    CoM_pos = controller.model.center_of_mass(q)
    CoM_pos_y = CoM_pos[1]
    marker_pos = controller.model.markers(q)[4]
    marker_pos_y = marker_pos[1]
    return marker_pos_y - CoM_pos_y


def SOCP_sensory_reference(
    time: cas.MX | cas.SX,
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    algebraic_states: cas.MX | cas.SX,
    nlp: NonLinearProgram,
):
    """
    This functions returns the sensory reference for the feedback gains.
    The feedback is vestibular (position and velocity of the head linked to the pelvis)
    and proprioceptive (position and velocity of the joints).
    """
    q_roots = states[nlp.states["q_roots"].index]
    q_joints = states[nlp.states["q_joints"].index]
    qdot_roots = states[nlp.states["qdot_roots"].index]
    qdot_joints = states[nlp.states["qdot_joints"].index]
    vestibular_and_joints_feedback = cas.vertcat(
        q_joints, qdot_joints, cas.reshape(q_roots[2], (1, -1)), cas.reshape(qdot_roots[2], (1, -1))
    )
    return vestibular_and_joints_feedback


def reach_landing_position_consistantly(controller: PenaltyController) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    This is a multi-node constraint because the covariance matrix depends on all the precedent nodes, but it only
    applies at the END node.
    """
    n_q = controller.model.nb_q
    n_root = controller.model.nb_root
    n_joints = n_q - n_root
    Q_root = cas.MX.sym("q_root", n_root)
    Q_joints = cas.MX.sym("q_joints", n_joints)
    Qdot_root = cas.MX.sym("qdot_root", n_root)
    Qdot_joints = cas.MX.sym("qdot_joints", n_joints)

    cov_sym = cas.MX.sym("cov", controller.model.matrix_shape_cov[0] * controller.model.matrix_shape_cov[1])
    cov_matrix = StochasticBioModel.reshape_to_matrix(cov_sym, controller.model.matrix_shape_cov)

    # What should we use as a reference?
    CoM_pos = controller.model.center_of_mass(cas.vertcat(Q_root, Q_joints))[:2]
    CoM_vel = controller.model.center_of_mass_velocity(
        cas.vertcat(Q_root, Q_joints), cas.vertcat(Qdot_root, Qdot_joints)
    )[:2]
    CoM_ang_vel = controller.model.body_rotation_rate(
        cas.vertcat(Q_root, Q_joints), cas.vertcat(Qdot_root, Qdot_joints)
    )[0]

    jac_CoM_q = cas.jacobian(CoM_pos, cas.vertcat(Q_root, Q_joints))
    jac_CoM_qdot = cas.jacobian(CoM_vel, cas.vertcat(Q_root, Q_joints, Qdot_root, Qdot_joints))
    jac_CoM_ang_vel = cas.jacobian(CoM_ang_vel, cas.vertcat(Q_root, Q_joints, Qdot_root, Qdot_joints))

    P_matrix_q = cov_matrix[:n_q, :n_q]
    P_matrix_qdot = cov_matrix[:, :]

    pos_constraint = jac_CoM_q @ P_matrix_q @ jac_CoM_q.T
    vel_constraint = jac_CoM_qdot @ P_matrix_qdot @ jac_CoM_qdot.T
    rot_constraint = jac_CoM_ang_vel @ P_matrix_qdot @ jac_CoM_ang_vel.T

    out = cas.vertcat(
        pos_constraint[0, 0], pos_constraint[1, 1], vel_constraint[0, 0], vel_constraint[1, 1], rot_constraint[0, 0]
    )

    fun = cas.Function("reach_target_consistantly", [Q_root, Q_joints, Qdot_root, Qdot_joints, cov_sym], [out])
    val = fun(
        controller.states["q_roots"].cx_start,
        controller.states["q_joints"].cx_start,
        controller.states["qdot_roots"].cx_start,
        controller.states["qdot_joints"].cx_start,
        controller.algebraic_states["cov"].cx_start,
    )
    # Since the stochastic variables are defined with ns+1, the cx_start actually refers to the last node (when using node=Node.END)

    return val


def compute_SOCP_torques_from_noise_and_feedback(
    nlp, time, states, controls, parameters, algebraic_states, sensory_noise, motor_noise
):
    tau_nominal = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)

    ref = DynamicsFunctions.get(nlp.algebraic_states["ref"], algebraic_states)
    k = DynamicsFunctions.get(nlp.algebraic_states["k"], algebraic_states)
    k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

    sensory_input = nlp.model.sensory_reference(time, states, controls, parameters, algebraic_states, nlp)
    tau_fb = k_matrix @ ((sensory_input - ref) + sensory_noise)

    tau_motor_noise = motor_noise

    tau_joints = tau_nominal + tau_fb + tau_motor_noise

    return tau_joints


def gaussian_function(x, sigma=1, mhu=0, offset=0, scaling_factor=1, flip=False):
    """
    Gaussian function
    mhu: mean
    sigma: standard deviation
    flip: if True, the gaussian is flipped vertically
    """
    sign = 1 if not flip else -1
    flip_offset = 0
    if flip:
        flip_offset = scaling_factor / sigma * cas.sqrt(2 * np.pi)
    return (
        scaling_factor * sign * 1 / (sigma * cas.sqrt(2 * np.pi)) * cas.exp(-0.5 * ((x - mhu) / sigma) ** 2)
        + flip_offset
        + offset
    )


def smooth_square_function(x, a, width, center=0, offset=0, scaling_factor=0):
    minimum = scaling_factor / cas.sqrt(a**2 + 1)
    b = 1 / (width / np.pi)
    h = offset + minimum
    k = width * center / scaling_factor + np.pi / 2
    return scaling_factor * cas.sin(b * x - k) / cas.sqrt(a**2 + cas.sin(b * x - k) ** 2) + h


def motor_acuity(motor_noise, tau_nominal):
    adjusted_motor_noise = gaussian_function(
        x=tau_nominal, sigma=100, offset=motor_noise, scaling_factor=1000, flip=True
    )
    return adjusted_motor_noise


def fb_noised_sensory_input(model, q_roots, q_joints, qdot_roots, qdot_joints, sensory_noise):
    n_joints = model.nb_q - model.nb_root
    q = cas.vertcat(q_roots, q_joints)
    qdot = cas.vertcat(qdot_roots, qdot_joints)

    sensory_input = SOCP_VARIABLE_FEEDFORWARD_sensory_input_function(model, q_roots, q_joints, qdot_roots, qdot_joints, 0, 0)
    proprioceptive_feedback = sensory_input[: 2 * n_joints]
    vestibular_feedback = sensory_input[2 * n_joints : -1]

    proprioceptive_noise = cas.MX.ones(2 * n_joints, 1) * sensory_noise[: 2 * n_joints]
    noised_propriceptive_feedback = proprioceptive_feedback + proprioceptive_noise

    head_idx = model.segment_index("Head")
    vestibular_noise = cas.MX.zeros(2, 1)
    head_velocity = model.segment_angular_velocity(q, qdot, head_idx)[0]
    for i in range(2):
        vestibular_noise[i] = gaussian_function(
            x=head_velocity,
            sigma=10,
            offset=sensory_noise[2 * (model.nb_q - model.nb_root) + i],
            scaling_factor=10,
            flip=True,
        )
    noised_vestibular_feedback = vestibular_feedback + vestibular_noise

    return cas.vertcat(noised_propriceptive_feedback, noised_vestibular_feedback)


def ff_noised_sensory_input(model, tf, time, q_roots, q_joints, qdot_roots, qdot_joints, sensory_noise):
    def visual_noise(model, q, qdot, sensory_noise):
        floor_normal_vector = cas.MX.zeros(3, 1)
        floor_normal_vector[2] = 1
        eyes_vect_start = model.marker(q, model.marker_index("eyes_vect_start"))
        eyes_vect_end = model.marker(q, model.marker_index("eyes_vect_end"))
        gaze_vector = eyes_vect_end - eyes_vect_start
        angle = cas.acos(
            cas.dot(gaze_vector, floor_normal_vector) / (cas.norm_fro(gaze_vector) * cas.norm_fro(floor_normal_vector))
        )
        # if the athlete is looking upward, consider he does not see the floor
        angle_to_consider = cas.if_else(gaze_vector[2] > 0, np.pi / 2, angle)
        noise_on_where_you_look = smooth_square_function(
            x=angle_to_consider,
            a=0.1,
            width=np.pi / 2,
            offset=sensory_noise[2 * (model.nb_q - model.nb_root) + 2],
            scaling_factor=sensory_noise[2 * (model.nb_q - model.nb_root) + 2],
        )

        head_velocity = model.segment_angular_velocity(q, qdot, model.segment_index("Head"))[0]
        vestibular_noise = gaussian_function(
            x=head_velocity,
            sigma=10,
            offset=sensory_noise[2 * (model.nb_q - model.nb_root) + 1],
            scaling_factor=10,
            flip=True,
        )

        return noise_on_where_you_look + vestibular_noise

    q = cas.vertcat(q_roots, q_joints)
    qdot = cas.vertcat(qdot_roots, qdot_joints)

    time_to_contact = tf - time
    time_to_contact_noise = visual_noise(model, q, qdot, sensory_noise)
    noised_time_to_contact = time_to_contact + time_to_contact_noise

    somersault_velocity = model.body_rotation_rate(q, qdot)[0]
    head_angular_velocity = model.segment_angular_velocity(q, qdot, model.segment_index("Head"))[0]
    somersault_velocity_noise = gaussian_function(
        x=head_angular_velocity,
        sigma=10,
        offset=sensory_noise[2 * (model.nb_q - model.nb_root) + 1],
        scaling_factor=10,
        flip=True,
    )
    noised_somersault_velocity = somersault_velocity + somersault_velocity_noise

    curent_somersault_angle = q_roots[2]
    curent_somersault_angle_noise = gaussian_function(
        x=head_angular_velocity,
        sigma=10,
        offset=sensory_noise[2 * (model.nb_q - model.nb_root) + 1],
        scaling_factor=10,
        flip=True,
    )
    noised_curent_somersault_angle = curent_somersault_angle + curent_somersault_angle_noise

    return noised_curent_somersault_angle + noised_somersault_velocity * noised_time_to_contact


def SOCP_VARIABLE_FEEDFORWARD_compute_torques_from_noise_and_feedback(
    nlp, time, states, controls, parameters, algebraic_states, motor_noise, sensory_noise
):
    n_q = nlp.model.nb_q
    n_root = nlp.model.nb_root
    n_joints = n_q - n_root

    tf = nlp.tf_mx
    q_roots = DynamicsFunctions.get(nlp.states["q_roots"], states)
    q_joints = DynamicsFunctions.get(nlp.states["q_joints"], states)
    qdot_roots = DynamicsFunctions.get(nlp.states["qdot_roots"], states)
    qdot_joints = DynamicsFunctions.get(nlp.states["qdot_joints"], states)
    tau_nominal = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)

    fb_ref = DynamicsFunctions.get(nlp.algebraic_states["ref"], algebraic_states)[: 2 * n_joints + 2]
    ff_ref = DynamicsFunctions.get(nlp.algebraic_states["ref"], algebraic_states)[2 * n_joints + 2]

    k = DynamicsFunctions.get(nlp.algebraic_states["k"], algebraic_states)
    k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

    k_fb = k_matrix[:, : 2 * n_joints + 2]
    k_ff = k_matrix[:, 2 * n_joints + 2 :]

    tau_fb = k_fb @ (
        fb_noised_sensory_input(nlp.model, q_roots, q_joints, qdot_roots, qdot_joints, sensory_noise) - fb_ref
    )
    tau_ff = k_ff @ (
        ff_noised_sensory_input(nlp.model, tf, time, q_roots, q_joints, qdot_roots, qdot_joints, sensory_noise) - ff_ref
    )
    tau_motor_noise = motor_acuity(motor_noise, tau_nominal)

    tau = tau_nominal + tau_fb + tau_ff + tau_motor_noise

    return tau


def SOCP_VARIABLE_compute_torques_from_noise_and_feedback(
    nlp, time, states, controls, parameters, algebraic_states, motor_noise, sensory_noise
):
    n_q = nlp.model.nb_q
    n_root = nlp.model.nb_root
    n_joints = n_q - n_root

    q_roots = DynamicsFunctions.get(nlp.states["q_roots"], states)
    q_joints = DynamicsFunctions.get(nlp.states["q_joints"], states)
    qdot_roots = DynamicsFunctions.get(nlp.states["qdot_roots"], states)
    qdot_joints = DynamicsFunctions.get(nlp.states["qdot_joints"], states)
    tau_nominal = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)

    fb_ref = DynamicsFunctions.get(nlp.algebraic_states["ref"], algebraic_states)[: 2 * n_joints + 2]

    k = DynamicsFunctions.get(nlp.algebraic_states["k"], algebraic_states)
    k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

    tau_fb = k_matrix @ (
        fb_noised_sensory_input(nlp.model, q_roots, q_joints, qdot_roots, qdot_joints, sensory_noise) - fb_ref
    )
    tau_motor_noise = motor_acuity(motor_noise, tau_nominal)

    tau = tau_nominal + tau_fb + tau_motor_noise

    return tau


def SOCP_FEEDFORWARD_compute_torques_from_noise_and_feedback(
    nlp, time, states, controls, parameters, algebraic_states, motor_noise, sensory_noise
):
    n_q = nlp.model.nb_q
    n_root = nlp.model.nb_root
    n_joints = n_q - n_root

    tau_nominal = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)

    fb_ref = DynamicsFunctions.get(nlp.algebraic_states["ref"], algebraic_states)[: 2 * n_joints + 2]
    ff_ref = DynamicsFunctions.get(nlp.algebraic_states["ref"], algebraic_states)[2 * n_joints + 2]

    k = DynamicsFunctions.get(nlp.algebraic_states["k"], algebraic_states)
    k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

    k_fb = k_matrix[:, : 2 * n_joints + 2]
    k_ff = k_matrix[:, 2 * n_joints + 2 :]

    sensory_input = nlp.model.sensory_reference(time, states, controls, parameters, algebraic_states, nlp)
    fb_sensory_input = sensory_input[: 2 * n_joints + 2]
    ff_sensory_input = sensory_input[2 * n_joints + 2]

    tau_fb = k_fb @ ((fb_sensory_input - fb_ref) + sensory_noise[: 2 * n_joints + 2])
    tau_ff = k_ff @ ((ff_sensory_input - ff_ref) + sensory_noise[2 * n_joints + 2])

    tau = tau_nominal + tau_fb + tau_ff + motor_noise

    return tau


def SOCP_VARIABLE_FEEDFORWARD_sensory_input_function(model, q_roots, q_joints, qdot_roots, qdot_joints, tf, time):
    q = cas.vertcat(q_roots, q_joints)
    qdot = cas.vertcat(qdot_roots, qdot_joints)
    proprioceptive_feedback = cas.vertcat(q_joints, qdot_joints)
    head_idx = model.segment_index("Head")
    head_orientation = model.segment_orientation(q, head_idx)
    head_velocity = model.segment_angular_velocity(q, qdot, head_idx)
    vestibular_feedback = cas.vertcat(head_orientation[0], head_velocity[0])

    q = cas.vertcat(q_roots, q_joints)
    qdot = cas.vertcat(qdot_roots, qdot_joints)
    time_to_contact = tf - time
    somersault_velocity = model.body_rotation_rate(q, qdot)[0]
    curent_somersault_angle = q_roots[2]
    visual_feedforward = curent_somersault_angle + somersault_velocity * time_to_contact

    return cas.vertcat(proprioceptive_feedback, vestibular_feedback, visual_feedforward)


def SOCP_VARIABLE_FEEDFORWARD_sensory_reference(
    time: cas.MX | cas.SX,
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    algebraic_states: cas.MX | cas.SX,
    nlp: NonLinearProgram,
):
    """
    This functions returns the sensory reference for the feedback gains.
    The feedback is vestibular (position and velocity of the head linked to the pelvis)
    and proprioceptive (position and velocity of the joints).
    """
    q_roots = states[nlp.states["q_roots"].index]
    q_joints = states[nlp.states["q_joints"].index]
    qdot_roots = states[nlp.states["qdot_roots"].index]
    qdot_joints = states[nlp.states["qdot_joints"].index]
    tf_mx = nlp.tf_mx
    return SOCP_VARIABLE_FEEDFORWARD_sensory_input_function(nlp.model, q_roots, q_joints, qdot_roots, qdot_joints, tf_mx, time)
