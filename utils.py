import sys

import casadi as cas
import numpy as np

sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    PenaltyController,
    StochasticBioModel,
)


def CoM_over_toes(controller: PenaltyController) -> cas.MX:
    q_roots = controller.states["q_roots"].cx_start
    q_joints = controller.states["q_joints"].cx_start
    q = cas.vertcat(q_roots, q_joints)
    CoM_pos = controller.model.center_of_mass(q)
    CoM_pos_y = CoM_pos[1]
    marker_index = controller.model.marker_index("Foot_Toe")
    marker_pos = controller.model.markers(q)[marker_index]
    marker_pos_y = marker_pos[1]
    return marker_pos_y - CoM_pos_y


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
        flip_offset = sigma / scaling_factor * cas.sqrt(2 * np.pi)
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


# ------------------- DMS ------------------- #


def minimize_nominal_and_feedback_efforts(
    controller: PenaltyController, motor_noise_numerical, sensory_noise_numerical
) -> cas.MX:
    nb_root = controller.model.nb_root
    nb_q = controller.model.nb_q
    nb_joints = nb_q - nb_root

    q_roots = controller.states["q_roots"].mx
    q_joints = controller.states["q_joints"].mx
    qdot_roots = controller.states["qdot_roots"].mx
    qdot_joints = controller.states["qdot_joints"].mx
    tau_joints = controller.controls["tau_joints"].mx
    k = controller.controls["k"].mx
    k_matrix = StochasticBioModel.reshape_to_matrix(k, controller.model.matrix_shape_k)
    ref = controller.controls["ref"].mx
    motor_noise = None
    sensory_noise = None
    for i in range(controller.model.nb_random):
        if motor_noise is None:
            motor_noise = controller.numerical_timeseries[f"motor_noise_numerical_{i}"].mx
            sensory_noise = controller.numerical_timeseries[f"sensory_noise_numerical_{i}"].mx
        else:
            motor_noise = cas.horzcat(motor_noise, controller.numerical_timeseries[f"motor_noise_numerical_{i}"].mx)
            sensory_noise = cas.horzcat(sensory_noise, controller.numerical_timeseries[f"sensory_noise_numerical_{i}"].mx)

    all_tau = 0
    for i in range(controller.model.nb_random):
        q_this_time = cas.vertcat(
            q_roots[i * nb_root : (i + 1) * nb_root], q_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        qdot_this_time = cas.vertcat(
            qdot_roots[i * nb_root : (i + 1) * nb_root], qdot_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        tau_this_time = tau_joints[:]

        # Joint friction
        tau_this_time += controller.model.friction_coefficients @ qdot_this_time[nb_root:]

        # Motor noise
        tau_this_time += motor_noise_numerical[:, i]

        # Feedback
        tau_this_time += k_matrix @ (
            ref
            - DMS_sensory_reference(controller.model, nb_root, q_this_time, qdot_this_time)
            + sensory_noise_numerical[:, i]
        )
        all_tau += cas.sum1(tau_this_time**2)

    all_tau_cx = controller.mx_to_cx(
        "all_tau",
        all_tau,
        controller.states["q_roots"],
        controller.states["q_joints"],
        controller.states["qdot_roots"],
        controller.states["qdot_joints"],
        controller.controls["tau_joints"],
        controller.controls["k"],
        controller.controls["ref"],
    )

    return all_tau_cx


def always_reach_landing_position(controller: PenaltyController) -> cas.MX:
    """
    Encourage the model to land consistently in approximately the same position and orientation.
    """

    nb_root = controller.model.nb_root
    nb_joints = controller.model.nb_q - nb_root

    q_roots = controller.states["q_roots"].mx
    q_joints = controller.states["q_joints"].mx
    qdot_roots = controller.states["qdot_roots"].mx
    qdot_joints = controller.states["qdot_joints"].mx

    CoM_pos = cas.MX()
    CoM_vel = cas.MX()
    CoM_ang_vel = cas.MX()
    mean_CoM_pos = 0
    mean_CoM_vel = 0
    mean_CoM_ang_vel = 0
    for i in range(controller.model.nb_random):
        q_this_time = cas.vertcat(
            q_roots[i * nb_root : (i + 1) * nb_root], q_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        qdot_this_time = cas.vertcat(
            qdot_roots[i * nb_root : (i + 1) * nb_root], qdot_joints[i * nb_joints : (i + 1) * nb_joints]
        )

        CoM_pos = cas.vertcat(CoM_pos, controller.model.center_of_mass(q_this_time)[1])
        CoM_vel = cas.vertcat(CoM_vel, controller.model.center_of_mass_velocity(q_this_time, qdot_this_time)[1])
        CoM_ang_vel = cas.vertcat(CoM_ang_vel, controller.model.body_rotation_rate(q_this_time, qdot_this_time)[0])

        mean_CoM_pos += 1 / controller.model.nb_random * controller.model.center_of_mass(q_this_time)[1]
        mean_CoM_vel += (
            1 / controller.model.nb_random * controller.model.center_of_mass_velocity(q_this_time, qdot_this_time)[1]
        )
        mean_CoM_ang_vel += (
            1 / controller.model.nb_random * controller.model.body_rotation_rate(q_this_time, qdot_this_time)[0]
        )

    out = (
        cas.sum1((CoM_pos - mean_CoM_pos) ** 2)
        + cas.sum1((CoM_vel - mean_CoM_vel) ** 2)
        + cas.sum1((CoM_ang_vel - mean_CoM_ang_vel) ** 2)
    )

    val = controller.mx_to_cx(
        "reach_target_consistantly",
        out,
        controller.states["q_roots"],
        controller.states["q_joints"],
        controller.states["qdot_roots"],
        controller.states["qdot_joints"],
    )

    return val


def DMS_sensory_reference(model, nb_roots, q_this_time, qdot_this_time):

    proprioceptive_feedback = cas.vertcat(q_this_time[nb_roots:], qdot_this_time[nb_roots:])
    pelvis_orientation = q_this_time[2]
    somersault_velocity = model.body_rotation_rate(q_this_time, qdot_this_time)[0]
    # head_idx = model.segment_index("Head")
    # head_orientation = model.segment_orientation(q_this_time, head_idx)
    # head_velocity = model.segment_angular_velocity(q_this_time, qdot_this_time, head_idx)
    # vestibular_feedback = cas.vertcat(head_orientation[0], head_velocity[0])

    return cas.vertcat(proprioceptive_feedback, pelvis_orientation, somersault_velocity)


def DMS_sensory_reference_no_eyes(model, nb_roots, q_this_time, qdot_this_time):

    proprioceptive_feedback = cas.vertcat(
        q_this_time[nb_roots], q_this_time[nb_roots + 2 :], qdot_this_time[nb_roots], qdot_this_time[nb_roots + 2 :]
    )
    pelvis_orientation = q_this_time[2]
    somersault_velocity = model.body_rotation_rate(q_this_time, qdot_this_time)[0]
    # head_idx = model.segment_index("Head")
    # head_orientation = model.segment_orientation(q_this_time, head_idx)
    # head_velocity = model.segment_angular_velocity(q_this_time, qdot_this_time, head_idx)
    # vestibular_feedback = cas.vertcat(head_orientation[0], head_velocity[0])

    return cas.vertcat(proprioceptive_feedback, pelvis_orientation, somersault_velocity)


def DMS_fb_noised_sensory_input_VARIABLE_no_eyes(model, q_roots, q_joints, qdot_roots, qdot_joints, sensory_noise):
    nb_roots = model.nb_root
    nb_joints = model.nb_q - nb_roots
    q = cas.vertcat(q_roots, q_joints)
    qdot = cas.vertcat(qdot_roots, qdot_joints)

    sensory_input = model.sensory_reference(model, nb_roots, q, qdot)

    proprioceptive_feedback = sensory_input[: 2 * (nb_joints - 1)]
    vestibular_feedback = sensory_input[2 * (nb_joints - 1) :]

    proprioceptive_noise = cas.MX.ones(2 * (nb_joints - 1), 1) * sensory_noise[: 2 * (nb_joints - 1)]
    noised_propriceptive_feedback = proprioceptive_feedback + proprioceptive_noise

    vestibular_noise = cas.MX.zeros(2, 1)
    head_idx = model.segment_index("Head")
    head_velocity = model.segment_angular_velocity(q, qdot, head_idx)[0]
    # head_velocity = qdot_roots[2] + qdot_joints[0]  # pelvis + head rotations
    for i in range(2):
        vestibular_noise[i] = gaussian_function(
            x=head_velocity,
            sigma=10,
            offset=sensory_noise[2 * (nb_joints - 1) + i],
            scaling_factor=10,
            flip=True,
        )
    noised_vestibular_feedback = vestibular_feedback + vestibular_noise

    return cas.vertcat(noised_propriceptive_feedback, noised_vestibular_feedback)


def toe_marker_on_floor(controller: PenaltyController) -> cas.MX:

    nb_root = controller.model.nb_root
    nb_joints = controller.model.nb_q - nb_root

    q_roots = controller.states["q_roots"].mx
    q_joints = controller.states["q_joints"].mx

    toe_idx = controller.model.marker_index("Foot_Toe")
    toe_marker_height = controller.cx()
    for i in range(controller.model.nb_random):
        q_this_time = cas.vertcat(
            q_roots[i * nb_root : (i + 1) * nb_root], q_joints[i * nb_joints : (i + 1) * nb_joints]
        )

        toe_marker_height = cas.vertcat(
            toe_marker_height, 1 / controller.model.nb_random * controller.model.marker(q_this_time, toe_idx)[2]
        )

    mean_height = cas.sum1(toe_marker_height)

    mean_height_cx = controller.mx_to_cx(
        "mean_height", mean_height, controller.states["q_roots"], controller.states["q_joints"]
    )

    return mean_height_cx


def ref_equals_mean_sensory(controller: PenaltyController) -> cas.MX:

    nb_root = controller.model.nb_root
    nb_joints = controller.model.nb_q - nb_root

    q_roots = controller.states["q_roots"].mx
    q_joints = controller.states["q_joints"].mx
    qdot_roots = controller.states["qdot_roots"].mx
    qdot_joints = controller.states["qdot_joints"].mx
    ref = controller.controls["ref"].mx

    ref_measured = cas.MX.zeros(ref.shape[0])
    for i in range(controller.model.nb_random):
        q_this_time = cas.vertcat(
            q_roots[i * nb_root : (i + 1) * nb_root], q_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        qdot_this_time = cas.vertcat(
            qdot_roots[i * nb_root : (i + 1) * nb_root], qdot_joints[i * nb_joints : (i + 1) * nb_joints]
        )

        ref_this_time = controller.model.sensory_reference(controller.model, nb_root, q_this_time, qdot_this_time)
        ref_measured += 1 / controller.model.nb_random * ref_this_time

    mean_ref_cx = controller.mx_to_cx(
        "mean_ref",
        ref - ref_measured,  # Difference between the reference and the mean sensory input
        controller.states["q_roots"],
        controller.states["q_joints"],
        controller.states["qdot_roots"],
        controller.states["qdot_joints"],
        controller.controls["ref"],
    )
    return mean_ref_cx


def DMS_CoM_over_toes(controller: PenaltyController) -> cas.MX:

    nb_root = controller.model.nb_root
    nb_joints = controller.model.nb_q - nb_root

    q_roots = controller.states["q_roots"].mx
    q_joints = controller.states["q_joints"].mx

    toe_idx = controller.model.marker_index("Foot_Toe")
    CoM_pos = cas.MX()
    marker_pos = cas.MX()
    for i in range(controller.model.nb_random):
        q_this_time = cas.vertcat(
            q_roots[i * nb_root : (i + 1) * nb_root], q_joints[i * nb_joints : (i + 1) * nb_joints]
        )

        CoM_pos = cas.vertcat(CoM_pos, 1 / controller.model.nb_random * controller.model.center_of_mass(q_this_time)[1])
        marker_pos = cas.vertcat(
            marker_pos, 1 / controller.model.nb_random * controller.model.marker(q_this_time, toe_idx)[1]
        )

    mean_distance_cx = controller.mx_to_cx(
        "mean_distance",
        marker_pos - CoM_pos,  # Difference between the CoM and the toe marker
        controller.states["q_roots"],
        controller.states["q_joints"],
    )

    return mean_distance_cx


def minimize_nominal_and_feedback_efforts_VARIABLE(controller: PenaltyController) -> cas.MX:
    nb_root = controller.model.nb_root
    nb_q = controller.model.nb_q
    nb_joints = nb_q - nb_root

    q_roots = controller.states["q_roots"].mx
    q_joints = controller.states["q_joints"].mx
    qdot_roots = controller.states["qdot_roots"].mx
    qdot_joints = controller.states["qdot_joints"].mx
    tau_joints = controller.controls["tau_joints"].mx
    k = controller.controls["k"].mx
    k_matrix = StochasticBioModel.reshape_to_matrix(k, controller.model.matrix_shape_k)
    fb_ref = controller.controls["ref"].mx
    motor_noise = None
    sensory_noise = None
    for i in range(controller.model.nb_random):
        if motor_noise is None:
            motor_noise = controller.numerical_timeseries[f"motor_noise_numerical_{i}"].mx
            sensory_noise = controller.numerical_timeseries[f"sensory_noise_numerical_{i}"].mx
        else:
            motor_noise = cas.horzcat(motor_noise, controller.numerical_timeseries[f"motor_noise_numerical_{i}"].mx)
            sensory_noise = cas.horzcat(sensory_noise, controller.numerical_timeseries[f"sensory_noise_numerical_{i}"].mx)

    all_tau = 0
    for i in range(controller.model.nb_random):
        q_this_time = cas.vertcat(
            q_roots[i * nb_root : (i + 1) * nb_root], q_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        qdot_this_time = cas.vertcat(
            qdot_roots[i * nb_root : (i + 1) * nb_root], qdot_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        tau_this_time = tau_joints[:]

        # Joint friction
        tau_this_time += controller.model.friction_coefficients @ qdot_this_time[nb_root:]

        # Motor noise
        tau_this_time += motor_acuity(motor_noise[:, i], tau_joints) - tau_joints[:]

        # Feedback
        tau_this_time += k_matrix @ (
            fb_ref
            - DMS_fb_noised_sensory_input_VARIABLE(
                controller.model,
                q_this_time[:nb_root],
                q_this_time[nb_root:],
                qdot_this_time[:nb_root],
                qdot_this_time[nb_root:],
                sensory_noise[:, i],
            )
        )
        all_tau += cas.sum1(tau_this_time**2)

    all_tau_cx = controller.mx_to_cx(
        "all_tau",
        all_tau,
        controller.states["q_roots"],
        controller.states["q_joints"],
        controller.states["qdot_roots"],
        controller.states["qdot_joints"],
        controller.controls["tau_joints"],
        controller.controls["k"],
        controller.controls["ref"],
        controller.numerical_timeseries,
    )

    return all_tau_cx


def DMS_fb_noised_sensory_input_VARIABLE(model, q_roots, q_joints, qdot_roots, qdot_joints, sensory_noise):
    nb_roots = model.nb_root
    nb_joints = model.nb_q - nb_roots
    q = cas.vertcat(q_roots, q_joints)
    qdot = cas.vertcat(qdot_roots, qdot_joints)

    sensory_input = model.sensory_reference(model, nb_roots, q, qdot)

    proprioceptive_feedback = sensory_input[: 2 * nb_joints]
    vestibular_feedback = sensory_input[2 * nb_joints :]

    proprioceptive_noise = cas.MX.ones(2 * nb_joints, 1) * sensory_noise[: 2 * nb_joints]
    noised_propriceptive_feedback = proprioceptive_feedback + proprioceptive_noise

    vestibular_noise = cas.MX.zeros(2, 1)
    head_idx = model.segment_index("Head")
    head_velocity = model.segment_angular_velocity(q, qdot, head_idx)[0]
    # head_velocity = qdot_roots[2] + qdot_joints[0]  # pelvis + head rotations
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


def DMS_ff_sensory_input(model, tf, time, q_this_time, qdot_this_time):

    time_to_contact = tf - time
    # somersault_velocity = model.body_rotation_rate(q_this_time, qdot_this_time)[0]
    somersault_velocity = qdot_this_time[2]
    curent_somersault_angle = q_this_time[2]
    visual_feedforward = curent_somersault_angle + somersault_velocity * time_to_contact

    return visual_feedforward


def minimize_nominal_and_feedback_efforts_FEEDFORWARD(controller: PenaltyController) -> cas.MX:
    nb_root = controller.model.nb_root
    nb_q = controller.model.nb_q
    nb_joints = nb_q - nb_root

    q_roots = controller.states["q_roots"].mx
    q_joints = controller.states["q_joints"].mx
    qdot_roots = controller.states["qdot_roots"].mx
    qdot_joints = controller.states["qdot_joints"].mx
    tau_joints = controller.controls["tau_joints"].mx
    k = controller.controls["k"].mx
    k_matrix = StochasticBioModel.reshape_to_matrix(k, controller.model.matrix_shape_k)
    k_matrix_fb = k_matrix[:, : controller.model.n_feedbacks]
    k_matrix_ff = k_matrix[:, controller.model.n_feedbacks :]
    fb_ref = controller.controls["ref"].mx
    ff_ref = controller.parameters["final_somersault"].mx
    motor_noise = None
    sensory_noise = None
    for i in range(controller.model.nb_random):
        if motor_noise is None:
            motor_noise = controller.numerical_timeseries[f"motor_noise_numerical_{i}"].mx
            sensory_noise = controller.numerical_timeseries[f"sensory_noise_numerical_{i}"].mx
        else:
            motor_noise = cas.horzcat(motor_noise, controller.numerical_timeseries[f"motor_noise_numerical_{i}"].mx)
            sensory_noise = cas.horzcat(sensory_noise, controller.numerical_timeseries[f"sensory_noise_numerical_{i}"].mx)

    all_tau = 0
    for i in range(controller.model.nb_random):
        q_this_time = cas.vertcat(
            q_roots[i * nb_root : (i + 1) * nb_root], q_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        qdot_this_time = cas.vertcat(
            qdot_roots[i * nb_root : (i + 1) * nb_root], qdot_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        tau_this_time = tau_joints[:]

        # Joint friction
        tau_this_time += controller.model.friction_coefficients @ qdot_this_time[nb_root:]

        # Motor noise
        tau_this_time += motor_noise[:, i]

        # Feedback
        tau_this_time += k_matrix_fb @ (
            fb_ref
            - DMS_sensory_reference_no_eyes(controller.model, nb_root, q_this_time, qdot_this_time)
            + sensory_noise[: controller.model.n_feedbacks, i]
        )

        # Feedforward
        tau_this_time += k_matrix_ff @ (
            ff_ref
            - DMS_ff_sensory_input(controller.model, controller.tf.mx, controller.time.mx, q_this_time, qdot_this_time)
            + sensory_noise[controller.model.n_feedbacks :, i]
        )

        all_tau += cas.sum1(tau_this_time**2)

    all_tau_cx = controller.mx_to_cx(
        "all_tau",
        all_tau,
        controller.states["q_roots"],
        controller.states["q_joints"],
        controller.states["qdot_roots"],
        controller.states["qdot_joints"],
        controller.controls["tau_joints"],
        controller.controls["k"],
        controller.controls["ref"],
        controller.parameters["final_somersault"],
        controller.time,
        controller.dt,
        controller.numerical_timeseries,
    )

    return all_tau_cx


def DMS_ff_noised_sensory_input(model, tf, time, q_this_time, qdot_this_time, sensory_noise):

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
            offset=sensory_noise,
            scaling_factor=sensory_noise,
        )

        head_velocity = model.segment_angular_velocity(q, qdot, model.segment_index("Head"))[0]
        vestibular_noise = gaussian_function(
            x=head_velocity,
            sigma=10,
            offset=sensory_noise,
            scaling_factor=10,
            flip=True,
        )

        return noise_on_where_you_look + vestibular_noise

    time_to_contact = tf - time
    time_to_contact_noise = visual_noise(model, q_this_time, qdot_this_time, sensory_noise)
    noised_time_to_contact = time_to_contact + time_to_contact_noise

    somersault_velocity = model.body_rotation_rate(q_this_time, qdot_this_time)[0]
    head_angular_velocity = model.segment_angular_velocity(q_this_time, qdot_this_time, model.segment_index("Head"))[0]
    somersault_velocity_noise = gaussian_function(
        x=head_angular_velocity,
        sigma=10,
        offset=sensory_noise,
        scaling_factor=10,
        flip=True,
    )
    noised_somersault_velocity = somersault_velocity + somersault_velocity_noise

    curent_somersault_angle = q_this_time[2]
    curent_somersault_angle_noise = gaussian_function(
        x=head_angular_velocity,
        sigma=10,
        offset=sensory_noise,
        scaling_factor=10,
        flip=True,
    )
    noised_curent_somersault_angle = curent_somersault_angle + curent_somersault_angle_noise

    return noised_curent_somersault_angle + noised_somersault_velocity * noised_time_to_contact


def minimize_nominal_and_feedback_efforts_VARIABLE_FEEDFORWARD(controller: PenaltyController) -> cas.MX:
    nb_root = controller.model.nb_root
    nb_q = controller.model.nb_q
    nb_joints = nb_q - nb_root

    q_roots = controller.states["q_roots"].mx
    q_joints = controller.states["q_joints"].mx
    qdot_roots = controller.states["qdot_roots"].mx
    qdot_joints = controller.states["qdot_joints"].mx
    tau_joints = controller.controls["tau_joints"].mx
    k = controller.controls["k"].mx
    k_matrix = StochasticBioModel.reshape_to_matrix(k, controller.model.matrix_shape_k)
    k_matrix_fb = k_matrix[:, : controller.model.n_feedbacks]
    k_matrix_ff = k_matrix[:, controller.model.n_feedbacks :]
    fb_ref = controller.controls["ref"].mx
    ff_ref = controller.parameters["final_somersault"].mx
    motor_noise = None
    sensory_noise = None
    for i in range(controller.model.nb_random):
        if motor_noise is None:
            motor_noise = controller.numerical_timeseries[f"motor_noise_numerical_{i}"].mx
            sensory_noise = controller.numerical_timeseries[f"sensory_noise_numerical_{i}"].mx
        else:
            motor_noise = cas.horzcat(motor_noise, controller.numerical_timeseries[f"motor_noise_numerical_{i}"].mx)
            sensory_noise = cas.horzcat(sensory_noise, controller.numerical_timeseries[f"sensory_noise_numerical_{i}"].mx)

    all_tau = 0
    for i in range(controller.model.nb_random):
        q_this_time = cas.vertcat(
            q_roots[i * nb_root : (i + 1) * nb_root], q_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        qdot_this_time = cas.vertcat(
            qdot_roots[i * nb_root : (i + 1) * nb_root], qdot_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        tau_this_time = tau_joints[:]

        # Joint friction
        tau_this_time += controller.model.friction_coefficients @ qdot_this_time[nb_root:]

        # Motor noise
        tau_this_time += motor_acuity(motor_noise[:, i], tau_joints) - tau_joints[:]

        # Feedback
        tau_this_time += k_matrix_fb @ (
            fb_ref
            - DMS_fb_noised_sensory_input_VARIABLE_no_eyes(
                controller.model,
                q_this_time[:nb_root],
                q_this_time[nb_root:],
                qdot_this_time[:nb_root],
                qdot_this_time[nb_root:],
                sensory_noise[: controller.model.n_feedbacks, i],
            )
        )

        # Feedforward
        tau_this_time += k_matrix_ff @ (
            ff_ref
            - DMS_ff_noised_sensory_input(
                controller.model,
                controller.tf.mx,
                controller.time.mx,
                q_this_time,
                qdot_this_time,
                sensory_noise[controller.model.n_feedbacks :, i],
            )
        )

        all_tau += cas.sum1(tau_this_time**2)

    all_tau_cx = controller.mx_to_cx(
        "all_tau",
        all_tau,
        controller.states["q_roots"],
        controller.states["q_joints"],
        controller.states["qdot_roots"],
        controller.states["qdot_joints"],
        controller.controls["tau_joints"],
        controller.controls["k"],
        controller.controls["ref"],
        controller.parameters["final_somersault"],
        controller.time,
        controller.dt,
        controller.numerical_timeseries,
    )

    return all_tau_cx
