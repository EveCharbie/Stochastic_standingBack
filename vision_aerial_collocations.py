"""
...
"""

import platform

import pickle
import biorbd_casadi as biorbd
import matplotlib.pyplot as plt
import casadi as cas
import numpy as np
import scipy
from IPython import embed

from utils import CoM_over_toes

import sys

sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    OptimalControlProgram,
    StochasticOptimalControlProgram,
    InitialGuessList,
    ObjectiveFcn,
    Solver,
    StochasticBiorbdModel,
    StochasticBioModel,
    ObjectiveList,
    NonLinearProgram,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    DynamicsList,
    BoundsList,
    InterpolationType,
    PenaltyController,
    Node,
    ConstraintList,
    ConstraintFcn,
    MultinodeConstraintList,
    MultinodeObjectiveList,
    BiMappingList,
    DynamicsFcn,
    Axis,
    OdeSolver,
    SocpType,
    CostType,
    VariableScalingList,
    ControlType,
    PhaseDynamics,
)

from seg3_aerial_collocations import reach_landing_position_consistantly
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

    sensory_input = sensory_input_function(model, q_roots, q_joints, qdot_roots, qdot_joints, 0, 0)
    proprioceptive_feedback = sensory_input[: 2 * n_joints]
    vestibular_feedback = sensory_input[2 * n_joints :-1]

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


def compute_torques_from_noise_and_feedback(
    nlp, time, states, controls, parameters, stochastic_variables, sensory_noise, motor_noise
):
    n_q = nlp.model.nb_q
    n_root = nlp.model.nb_root
    n_joints = n_q - n_root

    tf = parameters[0]
    q_roots = DynamicsFunctions.get(nlp.states["q_roots"], states)
    q_joints = DynamicsFunctions.get(nlp.states["q_joints"], states)
    qdot_roots = DynamicsFunctions.get(nlp.states["qdot_roots"], states)
    qdot_joints = DynamicsFunctions.get(nlp.states["qdot_joints"], states)
    tau_nominal = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)

    fb_ref = DynamicsFunctions.get(nlp.stochastic_variables["ref"], stochastic_variables)[: 2 * n_joints + 2]
    ff_ref = DynamicsFunctions.get(nlp.stochastic_variables["ref"], stochastic_variables)[2 * n_joints + 2]

    k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
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


def sensory_input_function(model, q_roots, q_joints, qdot_roots, qdot_joints, tf, time):
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


def sensory_reference(
    time: cas.MX | cas.SX,
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    stochastic_variables: cas.MX | cas.SX,
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
    tf = nlp.tf
    return sensory_input_function(nlp.model, q_roots, q_joints, qdot_roots, qdot_joints, tf, time)


def prepare_socp_vision(
    biorbd_model_path: str,
    polynomial_degree: int,
    time_last: float,
    n_shooting: int,
    motor_noise_magnitude: cas.DM,
    sensory_noise_magnitude: cas.DM,
    q_roots_last: np.ndarray = None,
    q_joints_last: np.ndarray = None,
    qdot_roots_last: np.ndarray = None,
    qdot_joints_last: np.ndarray = None,
    tau_joints_last: np.ndarray = None,
    k_last: np.ndarray = None,
    ref_last: np.ndarray = None,
    m_last: np.ndarray = None,
    cov_last: np.ndarray = None,
) -> StochasticOptimalControlProgram:
    """
    Sensory inputs:
    - proprioceptive: joint angles and velocities (5+5)
    - vestibular: head orientation and angular velocity (1+1)
    - visual: vision (1)
    """

    biorbd_model = biorbd.Model(biorbd_model_path)
    n_q = biorbd_model.nbQ()
    n_root = biorbd_model.nbRoot()
    n_joints = n_q - n_root
    friction_coefficients = cas.DM.zeros(n_joints, n_joints)
    for i in range(n_joints):
        friction_coefficients[i, i] = 0.1

    initial_cov = cas.DM_eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P

    auto_initialization = False if k_last is not None else True
    problem_type = SocpType.COLLOCATION(
        polynomial_degree=polynomial_degree,
        method="legendre",
        auto_initialization=auto_initialization,
        initial_cov=initial_cov,
    )

    bio_model = StochasticBiorbdModel(
        biorbd_model_path,
        sensory_noise_magnitude=sensory_noise_magnitude,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_reference=sensory_reference,
        compute_torques_from_noise_and_feedback=compute_torques_from_noise_and_feedback,
        n_references=2 * n_joints + 2 + 1,
        n_feedbacks=2 * n_joints + 2,  # The last one is a feedforward
        n_noised_states=n_q * 2,
        n_noised_controls=n_joints,
        n_collocation_points=polynomial_degree + 1,
        friction_coefficients=friction_coefficients,
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="tau", weight=0.01,
    #                         quadratic=True)
    objective_functions.add(
        ObjectiveFcn.Lagrange.STOCHASTIC_MINIMIZE_EXPECTED_FEEDBACK_EFFORTS,
        node=Node.ALL_SHOOTING,
        weight=1e3 / 2,
        quadratic=True,
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, min_bound=0.1, max_bound=1)
    if np.sum(sensory_noise_magnitude) == 0:
        objective_functions.add(
            ObjectiveFcn.Lagrange.STOCHASTIC_MINIMIZE_VARIABLE, key="k", weight=0.01, quadratic=True
        )

    objective_functions.add(
        reach_landing_position_consistantly, custom_type=ObjectiveFcn.Mayer, node=Node.END, weight=1e3, quadratic=True
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_MARKERS, marker_index=2, axes=Axis.Z, node=Node.END)
    constraints.add(CoM_over_toes, node=Node.END)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.STOCHASTIC_TORQUE_DRIVEN_FREE_FLOATING_BASE,
        problem_type=problem_type,
        with_cholesky=False,
        with_friction=True,
    )

    pose_at_first_node = np.array(
        [-0.0346, 0.1207, 0.2255, 0.0, 0.0045, 3.1, -0.1787, 0.0]
    )  # Initial position approx from bioviz
    pose_at_last_node = np.array(
        [-0.0346, 0.1207, 5.8292, -0.1801, -0.2954, 0.5377, 0.8506, -0.6856]
    )  # Final position approx from bioviz


    x_bounds = BoundsList()
    q_roots_min = bio_model.bounds_from_ranges("q_roots").min
    q_roots_max = bio_model.bounds_from_ranges("q_roots").max
    q_joints_min = bio_model.bounds_from_ranges("q_joints").min
    q_joints_max = bio_model.bounds_from_ranges("q_joints").max
    qdot_roots_min = bio_model.bounds_from_ranges("qdot_roots").min
    qdot_roots_max = bio_model.bounds_from_ranges("qdot_roots").max
    qdot_joints_min = bio_model.bounds_from_ranges("qdot_joints").min
    qdot_joints_max = bio_model.bounds_from_ranges("qdot_joints").max

    q_roots_min[:, 0] = pose_at_first_node[:n_root]
    q_roots_max[:, 0] = pose_at_first_node[:n_root]
    q_joints_min[:, 0] = pose_at_first_node[n_root:]
    q_joints_max[:, 0] = pose_at_first_node[n_root:]
    q_roots_min[2, 2] = pose_at_last_node[2] - 0.5
    q_roots_max[2, 2] = pose_at_last_node[2] + 0.5
    qdot_roots_min[:, 0] = 0
    qdot_roots_max[:, 0] = 0
    qdot_joints_min[:, 0] = 0
    qdot_joints_max[:, 0] = 0
    qdot_roots_min[1, 0] = 2
    qdot_roots_max[1, 0] = 2
    qdot_roots_min[2, 0] = 2.5 * np.pi
    qdot_roots_max[2, 0] = 2.5 * np.pi

    x_bounds.add(
        "q_roots",
        min_bound=q_roots_min,
        max_bound=q_roots_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    x_bounds.add(
        "q_joints",
        min_bound=q_joints_min,
        max_bound=q_joints_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    x_bounds.add(
        "qdot_roots",
        min_bound=qdot_roots_min,
        max_bound=qdot_roots_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    x_bounds.add(
        "qdot_joints",
        min_bound=qdot_joints_min,
        max_bound=qdot_joints_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    u_bounds = BoundsList()
    tau_min = np.ones((n_q - n_root, 3)) * -500
    tau_max = np.ones((n_q - n_root, 3)) * 500
    u_bounds.add(
        "tau_joints",
        min_bound=tau_min,
        max_bound=tau_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    # Initial guesses
    x_init = InitialGuessList()
    if q_roots_last is None:
        x_init.add(
            "q_roots",
            initial_guess=np.vstack((pose_at_first_node[:n_root], pose_at_last_node[:n_root])).T,
            interpolation=InterpolationType.LINEAR,
        )
        x_init.add(
            "q_joints",
            initial_guess=np.vstack((pose_at_first_node[n_root:], pose_at_last_node[n_root:])).T,
            interpolation=InterpolationType.LINEAR,
        )
        x_init.add("qdot_roots", initial_guess=[0.01] * n_root, interpolation=InterpolationType.CONSTANT)
        x_init.add("qdot_joints", initial_guess=[0.01] * n_joints, interpolation=InterpolationType.CONSTANT)
    else:
        x_init.add("q_roots", initial_guess=q_roots_last, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("q_joints", initial_guess=q_joints_last, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("qdot_roots", initial_guess=qdot_roots_last, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("qdot_joints", initial_guess=qdot_joints_last, interpolation=InterpolationType.ALL_POINTS)

    u_init = InitialGuessList()
    if tau_joints_last is None:
        u_init.add("tau_joints", initial_guess=[0.01] * n_joints, interpolation=InterpolationType.CONSTANT)
    else:
        u_init.add("tau_joints", initial_guess=tau_joints_last[:, :-1], interpolation=InterpolationType.EACH_FRAME)

    n_ref = 2 * n_joints + 2 + 1  # ref(13)
    n_k = n_joints * n_ref  # K(3x13)
    n_m = (2 * n_q) ** 2 * (polynomial_degree + 1)  # M(16x16x4)
    n_stochastic = n_k + n_ref + n_m
    n_cov = (2 * n_q) ** 2  # Cov(16x16)
    n_stochastic += n_cov

    s_init = InitialGuessList()
    s_bounds = BoundsList()

    if k_last is not None:
        s_init.add("k", initial_guess=k_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("k", min_bound=[-500] * n_k, max_bound=[500] * n_k, interpolation=InterpolationType.CONSTANT)

    ref_min = cas.vertcat(
        x_bounds["q_joints"].min,
        x_bounds["qdot_joints"].min,
        x_bounds["q_roots"].min[2, :].reshape(1, 3) - 1,
        x_bounds["qdot_roots"].min[2, :].reshape(1, 3) - 10,
        np.ones((1, 3)) * pose_at_last_node[2],
    )
    ref_max = cas.vertcat(
        x_bounds["q_joints"].max,
        x_bounds["qdot_joints"].max,
        x_bounds["q_roots"].max[2, :].reshape(1, 3) + 1,
        x_bounds["qdot_roots"].max[2, :].reshape(1, 3) + 10,
        np.ones((1, 3)) * pose_at_last_node[2],
    )

    if ref_last is not None:
        s_init.add("ref", initial_guess=ref_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add(
        "ref",
        min_bound=ref_min,
        max_bound=ref_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    if m_last is not None:
        s_init.add("m", initial_guess=m_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("m", min_bound=[-50] * n_m, max_bound=[50] * n_m, interpolation=InterpolationType.CONSTANT)

    cov_min = np.ones((n_cov, 3)) * -500
    cov_max = np.ones((n_cov, 3)) * 500
    cov_min[:, 0] = np.reshape(StochasticBioModel.reshape_to_vector(initial_cov), (-1, ))
    cov_max[:, 0] = np.reshape(StochasticBioModel.reshape_to_vector(initial_cov), (-1, ))
    if cov_last is not None:
        s_init.add("cov", initial_guess=cov_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add(
        "cov",
        min_bound=cov_min,
        max_bound=cov_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    return StochasticOptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        time_last,
        x_init=x_init,
        u_init=u_init,
        s_init=s_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        s_bounds=s_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        n_threads=1,
        problem_type=problem_type,
    )


def main():
    model_name = "Model2D_7Dof_0C_3M"
    biorbd_model_path = f"models/{model_name}_vision.bioMod"
    biorbd_model_path_with_mesh = f"models/{model_name}_vision_with_mesh.bioMod"

    save_path = f"results/{model_name}_aerial_vision_socp_collocations.pkl"

    dt = 0.05
    final_time = 0.8
    n_shooting = int(final_time / dt)

    # TODO: How do we choose the values?
    motor_noise_std = 0.05
    wPq_std = 0.001
    wPqdot_std = 0.003

    motor_noise_magnitude = cas.DM(
        np.array(
            [
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
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
                wPq_std**2 / dt,
                wPqdot_std**2 / dt,  # Proprioceptive velocity
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPq_std**2 / dt,  # Vestibular position
                wPq_std**2 / dt,  # Vestibular velocity
                wPq_std**2 / dt,  # Visual
            ]
        )
    )

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma97")
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_maximum_iterations(1000)
    solver.set_hessian_approximation("limited-memory")  # Mandatory, otherwise RAM explodes!
    solver._nlp_scaling_method = "none"

    socp = prepare_socp_vision(
        biorbd_model_path, final_time, n_shooting, motor_noise_magnitude, sensory_noise_magnitude
    )
    sol = socp.solve(solver)

    q_roots_sol = sol.states["q_roots"]
    q_joints_sol = sol.states["q_joints"]
    qdot_roots_sol = sol.states["qdot_roots"]
    qdot_joints_sol = sol.states["qdot_joints"]
    tau_joints_sol = sol.controls["tau_joints"]
    time_sol = sol.parameters["time"][0][0]
    data = {
        "q_roots_sol": q_roots_sol,
        "q_joints_sol": q_joints_sol,
        "qdot_roots_sol": qdot_roots_sol,
        "qdot_joints_sol": qdot_joints_sol,
        "tau_joints_sol": tau_joints_sol,
        "time_sol": time_sol,
    }

    if sol.status != 0:
        save_path = save_path.replace(".pkl", "_DVG.pkl")
    else:
        save_path = save_path.replace(".pkl", "_CVG.pkl")

    with open(save_path, "wb") as file:
        pickle.dump(data, file)

    import bioviz

    b = bioviz.Viz(biorbd_model_path_with_mesh)
    b.load_movement(q_opt)
    b.exec()


if __name__ == "__main__":
    main()
