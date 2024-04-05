"""
...
"""

import sys

import biorbd_casadi as biorbd
import casadi as cas
import numpy as np

from utils import (
    DMS_CoM_over_toes,
    always_reach_landing_position,
    minimize_nominal_and_feedback_efforts_VARIABLE,
    toe_marker_on_floor,
    ref_equals_mean_sensory,
    motor_acuity,
    DMS_fb_noised_sensory_input_VARIABLE,
    DMS_sensory_reference,
)

sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    OptimalControlProgram,
    InitialGuessList,
    ObjectiveFcn,
    StochasticBiorbdModel,
    StochasticBioModel,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InterpolationType,
    Node,
    ConstraintList,
    PhaseDynamics,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    OdeSolver,
)


def custom_dynamics(
    time,
    states,
    controls,
    parameters,
    algebraic_states,
    numerical_timeseries,
    nlp,
) -> DynamicsEvaluation:

    nb_root = nlp.model.nb_root
    nb_q = nlp.model.nb_q
    nb_joints = nb_q - nb_root
    nb_random = nlp.model.nb_random

    q_roots = DynamicsFunctions.get(nlp.states["q_roots"], states)
    q_joints = DynamicsFunctions.get(nlp.states["q_joints"], states)
    qdot_roots = DynamicsFunctions.get(nlp.states["qdot_roots"], states)
    qdot_joints = DynamicsFunctions.get(nlp.states["qdot_joints"], states)
    tau_joints = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)
    k = DynamicsFunctions.get(nlp.controls["k"], controls)
    k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)
    fb_ref = DynamicsFunctions.get(nlp.controls["ref"], controls)[: 2 * nb_joints + 2]
    motor_noise = None
    sensory_noise = None
    for i in range(nb_random):
        if motor_noise == None:
            motor_noise = DynamicsFunctions.get(
                nlp.numerical_timeseries[f"motor_noise_numerical_{i}"], numerical_timeseries
            )
            sensory_noise = DynamicsFunctions.get(
                nlp.numerical_timeseries[f"sensory_noise_numerical_{i}"], numerical_timeseries
            )
        else:
            motor_noise = cas.horzcat(
                motor_noise,
                DynamicsFunctions.get(nlp.numerical_timeseries[f"motor_noise_numerical_{i}"], numerical_timeseries),
            )
            sensory_noise = cas.horzcat(
                sensory_noise,
                DynamicsFunctions.get(nlp.numerical_timeseries[f"sensory_noise_numerical_{i}"], numerical_timeseries),
            )

    dq = cas.vertcat(qdot_roots, qdot_joints)
    dxdt = cas.MX(nlp.states.shape, 1)
    dxdt[: nb_q * nb_random] = dq
    ddq_roots = cas.MX()
    ddq_joints = cas.MX()
    for i in range(nlp.model.nb_random):
        q_this_time = cas.vertcat(
            q_roots[i * nb_root : (i + 1) * nb_root], q_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        qdot_this_time = cas.vertcat(
            qdot_roots[i * nb_root : (i + 1) * nb_root], qdot_joints[i * nb_joints : (i + 1) * nb_joints]
        )
        tau_this_time = tau_joints[:]

        # Joint friction
        tau_this_time += nlp.model.friction_coefficients @ qdot_this_time[nb_root:]

        # Motor noise
        tau_this_time += motor_acuity(motor_noise[:, i], tau_joints)

        # Feedback
        tau_this_time += k_matrix @ (
            fb_ref
            - DMS_fb_noised_sensory_input_VARIABLE(
                nlp.model,
                q_this_time[:nb_root],
                q_this_time[nb_root:],
                qdot_this_time[:nb_root],
                qdot_this_time[nb_root:],
                sensory_noise[:, i],
            )
        )
        tau_this_time = cas.vertcat(cas.MX.zeros(nb_root), tau_this_time)

        ddq = nlp.model.forward_dynamics(q_this_time, qdot_this_time, tau_this_time)
        ddq_roots = cas.vertcat(ddq_roots, ddq[:nb_root])
        ddq_joints = cas.vertcat(ddq_joints, ddq[nb_root:])

    dxdt[nb_q * nb_random : (nb_q + nb_root) * nb_random] = ddq_roots
    dxdt[(nb_q + nb_root) * nb_random :] = ddq_joints

    return DynamicsEvaluation(dxdt=dxdt, defects=None)


def custom_configure(ocp, nlp):

    nb_root = nlp.model.nb_root
    nb_q = nlp.model.nb_q
    nb_random = nlp.model.nb_random

    # States
    name_q_roots = []
    for j in range(nb_random):
        for i in range(nb_root):
            name_q_roots += [f"{i}_{j}"]
    ConfigureProblem.configure_new_variable(
        "q_roots",
        name_q_roots,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_states_dot=False,
    )

    name_q_joints = []
    for j in range(nb_random):
        for i in range(nb_root, nb_q):
            name_q_joints += [f"{i}_{j}"]
    ConfigureProblem.configure_new_variable(
        "q_joints",
        name_q_joints,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_states_dot=False,
    )

    ConfigureProblem.configure_new_variable(
        "qdot_roots",
        name_q_roots,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_states_dot=True,
    )

    ConfigureProblem.configure_new_variable(
        "qdot_joints",
        name_q_joints,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_states_dot=True,
    )

    ConfigureProblem.configure_new_variable(
        "qddot_roots",
        name_q_roots,
        ocp,
        nlp,
        as_states=False,
        as_controls=False,
        as_states_dot=True,
    )

    ConfigureProblem.configure_new_variable(
        "qddot_joints",
        name_q_joints,
        ocp,
        nlp,
        as_states=False,
        as_controls=False,
        as_states_dot=True,
    )

    # Controls
    name_tau_joints = [nlp.model.name_dof[i] for i in range(nb_root, nb_q)]
    ConfigureProblem.configure_new_variable(
        "tau_joints",
        name_tau_joints,
        ocp,
        nlp,
        as_states=False,
        as_controls=True,
        as_states_dot=False,
    )

    name_k = []
    control_names = [f"control_{i}" for i in range(nlp.model.n_noised_controls)]
    ref_names = [f"feedback_{i}" for i in range(nlp.model.n_references)]
    for name_1 in control_names:
        for name_2 in ref_names:
            name_k += [name_1 + "_&_" + name_2]
    ConfigureProblem.configure_new_variable(
        "k",
        name_k,
        ocp,
        nlp,
        as_states=False,
        as_controls=True,
        as_states_dot=False,
        as_algebraic_states=False,
    )
    ConfigureProblem.configure_new_variable(
        "ref",
        ref_names,
        ocp,
        nlp,
        as_states=False,
        as_controls=True,
        as_states_dot=False,
        as_algebraic_states=False,
    )

    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)


def prepare_socp_VARIABLE(
    biorbd_model_path: str,
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
    nb_random: int = 30,
):
    """
    Sensory inputs:
    - proprioceptive: joint angles and velocities (4+4)
    - vestibular: head orientation and angular velocity (1+1)
    """

    biorbd_model = biorbd.Model(biorbd_model_path)
    n_q = biorbd_model.nbQ()
    n_root = biorbd_model.nbRoot()
    n_joints = n_q - n_root
    friction_coefficients = cas.DM.zeros(n_joints, n_joints)
    for i in range(n_joints):
        friction_coefficients[i, i] = 0.1

    bio_model = StochasticBiorbdModel(
        biorbd_model_path,
        sensory_noise_magnitude=sensory_noise_magnitude,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_reference=DMS_sensory_reference,
        compute_torques_from_noise_and_feedback=None,
        n_references=2 * (n_joints + 1),
        n_feedbacks=2 * (n_joints + 1),
        n_noised_states=n_q * 2,
        n_noised_controls=n_joints,
        friction_coefficients=friction_coefficients,
    )
    bio_model.nb_random = nb_random

    # Prepare the noises
    np.random.seed(0)
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

    # Objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        minimize_nominal_and_feedback_efforts_VARIABLE,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        weight=1,
        quadratic=False,  # Already squared in the function
    )
    objective_functions.add(
        minimize_nominal_and_feedback_efforts_VARIABLE,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        weight=1,
        quadratic=True,
        derivative=True,
    )
    objective_functions.add(
        always_reach_landing_position,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        weight=10000,
        quadratic=False,  # Already squared in the function
    )

    # Regularization
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, min_bound=0.1, max_bound=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="k", weight=1e-5, quadratic=True)

    # Constraints
    constraints = ConstraintList()
    constraints.add(toe_marker_on_floor, node=Node.END)
    constraints.add(DMS_CoM_over_toes, node=Node.END)
    constraints.add(ref_equals_mean_sensory, node=Node.ALL_SHOOTING)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        custom_configure,
        dynamic_function=custom_dynamics,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries={
            "motor_noise_numerical": motor_noise_numerical,
            "sensory_noise_numerical": sensory_noise_numerical,
        },
    )

    pose_at_first_node = np.array(
        [-0.0346, 0.1207, 0.2255, 0.0, 3.1, -0.1787, 0.0]
    )  # Initial position approx from bioviz
    pose_at_last_node = np.array(
        [-0.0346, 0.1207, 5.8292, -0.1801, 0.5377, 0.8506, -0.6856]
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
    for i in range(1, nb_random):
        q_roots_min = np.vstack((q_roots_min, bio_model.bounds_from_ranges("q_roots").min))
        q_roots_max = np.vstack((q_roots_max, bio_model.bounds_from_ranges("q_roots").max))
        q_joints_min = np.vstack((q_joints_min, bio_model.bounds_from_ranges("q_joints").min))
        q_joints_max = np.vstack((q_joints_max, bio_model.bounds_from_ranges("q_joints").max))
        qdot_roots_min = np.vstack((qdot_roots_min, bio_model.bounds_from_ranges("qdot_roots").min))
        qdot_roots_max = np.vstack((qdot_roots_max, bio_model.bounds_from_ranges("qdot_roots").max))
        qdot_joints_min = np.vstack((qdot_joints_min, bio_model.bounds_from_ranges("qdot_joints").min))
        qdot_joints_max = np.vstack((qdot_joints_max, bio_model.bounds_from_ranges("qdot_joints").max))

    # initial variability
    initial_cov = np.eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P
    noised_states = np.random.multivariate_normal(
        np.hstack((pose_at_first_node, np.array([0, 2, 2.1 * np.pi, 0, 0, 0, 0]))), initial_cov, nb_random
    ).T

    for i in range(nb_random):
        q_roots_min[i * n_root : (i + 1) * n_root, 0] = noised_states[:n_root, i]
        q_roots_max[i * n_root : (i + 1) * n_root, 0] = noised_states[:n_root, i]
        q_joints_min[i * n_joints : (i + 1) * n_joints, 0] = noised_states[n_root:n_q, i]
        q_joints_max[i * n_joints : (i + 1) * n_joints, 0] = noised_states[n_root:n_q, i]
        qdot_roots_min[i * n_root : (i + 1) * n_root, 0] = noised_states[n_q : n_q + n_root, i]
        qdot_roots_max[i * n_root : (i + 1) * n_root, 0] = noised_states[n_q : n_q + n_root, i]
        qdot_joints_min[i * n_joints : (i + 1) * n_joints, 0] = noised_states[n_q + n_root :, i]
        qdot_joints_max[i * n_joints : (i + 1) * n_joints, 0] = noised_states[n_q + n_root :, i]
        q_roots_min[i * n_root + 2, 2] = pose_at_last_node[2] - 0.5
        q_roots_max[i * n_root + 2, 2] = pose_at_last_node[2] + 0.5

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
    tau_min = np.ones((n_joints, 3)) * -500
    tau_max = np.ones((n_joints, 3)) * 500
    u_bounds.add(
        "tau_joints",
        min_bound=tau_min,
        max_bound=tau_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    # Initial guesses
    x_init = InitialGuessList()
    if q_roots_last is None:
        q_roots_init = np.vstack((pose_at_first_node[:n_root], pose_at_last_node[:n_root]))
        q_joints_init = np.vstack((pose_at_first_node[n_root:], pose_at_last_node[n_root:]))
        for i in range(1, nb_random):
            q_roots_init = np.hstack(
                (q_roots_init, np.vstack((pose_at_first_node[:n_root], pose_at_last_node[:n_root])))
            )
            q_joints_init = np.hstack(
                (q_joints_init, np.vstack((pose_at_first_node[n_root:], pose_at_last_node[n_root:])))
            )
        x_init.add(
            "q_roots",
            initial_guess=q_roots_init,
            interpolation=InterpolationType.LINEAR,
        )
        x_init.add(
            "q_joints",
            initial_guess=q_joints_init,
            interpolation=InterpolationType.LINEAR,
        )
        x_init.add("qdot_roots", initial_guess=[0.01] * n_root * nb_random, interpolation=InterpolationType.CONSTANT)
        x_init.add("qdot_joints", initial_guess=[0.01] * n_joints * nb_random, interpolation=InterpolationType.CONSTANT)
    else:
        q_roots_init = q_roots_last
        q_joints_init = q_joints_last
        qdot_roots_init = qdot_roots_last
        qdot_joints_init = qdot_joints_last
        for i in range(1, nb_random):
            q_roots_init = np.vstack((q_roots_init, q_roots_last))
            q_joints_init = np.vstack((q_joints_init, q_joints_last))
            qdot_roots_init = np.vstack((qdot_roots_init, qdot_roots_last))
            qdot_joints_init = np.vstack((qdot_joints_init, qdot_joints_last))
        x_init.add("q_roots", initial_guess=q_roots_init, interpolation=InterpolationType.EACH_FRAME)
        x_init.add("q_joints", initial_guess=q_joints_init, interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot_roots", initial_guess=qdot_roots_init, interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot_joints", initial_guess=qdot_joints_init, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    if tau_joints_last is None:
        u_init.add("tau_joints", initial_guess=[0.01] * n_joints, interpolation=InterpolationType.CONSTANT)
    else:
        u_init.add("tau_joints", initial_guess=tau_joints_last, interpolation=InterpolationType.EACH_FRAME)

    # The stochastic variables will be put in the controls for simplicity
    n_ref = 2 * (n_joints + 1)  # ref(8)
    n_k = n_joints * n_ref  # K(3x8)

    if k_last is not None:
        u_init.add("k", initial_guess=k_last, interpolation=InterpolationType.EACH_FRAME)
    else:
        u_init.add("k", initial_guess=[0.01] * n_k, interpolation=InterpolationType.CONSTANT)

    u_bounds.add("k", min_bound=[-50] * n_k, max_bound=[50] * n_k, interpolation=InterpolationType.CONSTANT)

    ref_min = [-1000] * n_ref
    ref_max = [1000] * n_ref

    q_sym = cas.MX.sym("q", n_q, 1)
    qdot_sym = cas.MX.sym("qdot", n_q, 1)
    ref_fun = cas.Function(
        "ref_func", [q_sym, qdot_sym], [bio_model.sensory_reference(bio_model, n_root, q_sym, qdot_sym)]
    )

    if ref_last is not None:
        u_init.add("ref", initial_guess=ref_last, interpolation=InterpolationType.EACH_FRAME)
    else:
        ref_init = np.zeros((n_ref, n_shooting + 1))
        for i in range(n_shooting):
            q_roots_this_time = q_roots_init[:n_root, i].T
            q_joints_this_time = q_joints_init[:n_joints, i].T
            qdot_roots_this_time = qdot_roots_init[:n_root, i].T
            qdot_joints_this_time = qdot_joints_init[:n_joints, i].T
            for j in range(1, nb_random):
                q_roots_this_time = np.vstack((q_roots_this_time, q_roots_init[j * n_root : (j + 1) * n_root, i].T))
                q_joints_this_time = np.vstack(
                    (q_joints_this_time, q_joints_init[j * n_joints : (j + 1) * n_joints, i].T)
                )
                qdot_roots_this_time = np.vstack(
                    (qdot_roots_this_time, qdot_roots_init[j * n_root : (j + 1) * n_root, i].T)
                )
                qdot_joints_this_time = np.vstack(
                    (qdot_joints_this_time, q_joints_init[j * n_joints : (j + 1) * n_joints, i].T)
                )
            q_mean = np.hstack((np.mean(q_roots_this_time, axis=0), np.mean(q_joints_this_time, axis=0)))
            qdot_mean = np.hstack((np.mean(qdot_roots_this_time, axis=0), np.mean(qdot_joints_this_time, axis=0)))
            ref_init[:, i] = np.reshape(ref_fun(q_mean, qdot_mean), (n_ref,))

    u_bounds.add(
        "ref",
        min_bound=ref_min,
        max_bound=ref_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        time_last,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=OdeSolver.RK4(),
        n_threads=32,
    )
    return motor_noise_numerical, sensory_noise_numerical, ocp
