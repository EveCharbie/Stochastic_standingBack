"""
...
"""

import pickle
import biorbd_casadi as biorbd
import casadi as cas
import numpy as np


from utils import (
    CoM_over_toes,
    SOCP_sensory_reference,
    reach_landing_position_consistantly,
    compute_SOCP_torques_from_noise_and_feedback,
)

import sys

sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    StochasticOptimalControlProgram,
    InitialGuessList,
    ObjectiveFcn,
    Solver,
    StochasticBiorbdModel,
    StochasticBioModel,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InterpolationType,
    Node,
    ConstraintList,
    ConstraintFcn,
    DynamicsFcn,
    Axis,
    SocpType,
)


def prepare_socp(
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
        sensory_reference=SOCP_sensory_reference,
        compute_torques_from_noise_and_feedback=compute_SOCP_torques_from_noise_and_feedback,
        n_references=2 * (n_joints + 1),
        n_feedbacks=2 * (n_joints + 1),
        n_noised_states=n_q * 2,
        n_noised_controls=n_joints,
        n_collocation_points=polynomial_degree + 1,
        friction_coefficients=friction_coefficients,
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="tau_joints", weight=0.01, quadratic=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.STOCHASTIC_MINIMIZE_EXPECTED_FEEDBACK_EFFORTS,
        node=Node.ALL_SHOOTING,
        weight=1e3 / 2,
        quadratic=True,
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, min_bound=0.1, max_bound=1)
    if np.sum(sensory_noise_magnitude) == 0:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_ALGEBRAIC_STATES, key="k", weight=0.01, quadratic=True)

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
        u_init.add("tau_joints", initial_guess=tau_joints_last, interpolation=InterpolationType.EACH_FRAME)

    n_ref = 2 * (n_joints + 1)  # ref(8)
    n_k = n_joints * n_ref  # K(3x8)
    n_m = (2 * n_q) ** 2 * (polynomial_degree + 1)  # M(12x12x4)
    n_stochastic = n_k + n_ref + n_m
    n_cov = (2 * n_q) ** 2  # Cov(12x12)
    n_stochastic += n_cov

    a_init = InitialGuessList()
    a_bounds = BoundsList()

    if k_last is not None:
        a_init.add("k", initial_guess=k_last, interpolation=InterpolationType.EACH_FRAME)
    a_bounds.add("k", min_bound=[-500] * n_k, max_bound=[500] * n_k, interpolation=InterpolationType.CONSTANT)

    ref_min = cas.vertcat(
        x_bounds["q_joints"].min,
        x_bounds["qdot_joints"].min,
        x_bounds["q_roots"].min[2, :].reshape(1, 3),
        x_bounds["qdot_roots"].min[2, :].reshape(1, 3),
    )
    ref_max = cas.vertcat(
        x_bounds["q_joints"].max,
        x_bounds["qdot_joints"].max,
        x_bounds["q_roots"].max[2, :].reshape(1, 3),
        x_bounds["qdot_roots"].max[2, :].reshape(1, 3),
    )

    if ref_last is not None:
        a_init.add("ref", initial_guess=ref_last, interpolation=InterpolationType.EACH_FRAME)
    a_bounds.add(
        "ref",
        min_bound=ref_min,
        max_bound=ref_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    if m_last is not None:
        a_init.add("m", initial_guess=m_last, interpolation=InterpolationType.EACH_FRAME)
    a_bounds.add("m", min_bound=[-50] * n_m, max_bound=[50] * n_m, interpolation=InterpolationType.CONSTANT)

    cov_min = np.ones((n_cov, 3)) * -500
    cov_max = np.ones((n_cov, 3)) * 500
    cov_min[:, 0] = np.reshape(StochasticBioModel.reshape_to_vector(initial_cov), (-1,))
    cov_max[:, 0] = np.reshape(StochasticBioModel.reshape_to_vector(initial_cov), (-1,))
    if cov_last is not None:
        a_init.add("cov", initial_guess=cov_last, interpolation=InterpolationType.EACH_FRAME)
    a_bounds.add(
        "cov",
        min_bound=cov_min,
        max_bound=cov_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    # x_scaling = VariableScalingList()
    # x_scaling.add("q_roots", scaling=[1] * n_root)
    # x_scaling.add("q_joints", scaling=[1] * n_joints)
    # x_scaling.add("qdot_roots", scaling=[10] * n_root)
    # x_scaling.add("qdot_joints", scaling=[10] * n_joints)
    #
    # u_scaling = VariableScalingList()
    # u_scaling.add("tau_joints", scaling=[10] * n_joints)
    #
    # a_scaling = VariableScalingList()
    # a_scaling.add("k", scaling=[10] * n_k)
    # a_scaling.add("ref", scaling=[10] * n_ref)
    # a_scaling.add("m", scaling=[1] * n_m)
    # a_scaling.add("cov", scaling=[1e-3] * n_cov)

    return StochasticOptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        time_last,
        x_init=x_init,
        u_init=u_init,
        a_init=a_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        a_bounds=a_bounds,
        # x_scaling=x_scaling,
        # u_scaling=u_scaling,
        # a_scaling=a_scaling,
        objective_functions=objective_functions,
        constraints=constraints,
        n_threads=32,
        problem_type=problem_type,
    )


def main():
    model_name = "Model2D_6Dof_0C_3M"
    biorbd_model_path = f"models/{model_name}.bioMod"
    biorbd_model_path_with_mesh = f"models/{model_name}_with_mesh.bioMod"

    save_path = f"results/{model_name}_aerial_socp_collocations.pkl"

    n_q = 7
    n_root = 3

    dt = 0.05
    final_time = 0.8
    n_shooting = int(final_time / dt)

    # TODO: How do we choose the values?
    motor_noise_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024

    motor_noise_magnitude = cas.DM(
        np.array([motor_noise_std**2 / dt for _ in range(n_q - n_root)])
    )  # All DoFs except root
    sensory_noise_magnitude = cas.DM(
        cas.vertcat(
            np.array([wPq_std**2 / dt for _ in range(n_q - n_root + 1)]),
            np.array([wPqdot_std**2 / dt for _ in range(n_q - n_root + 1)]),
        )
    )  # since the head is fixed to the pelvis, the vestibular feedback is in the states ref

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

    socp = prepare_socp(biorbd_model_path, final_time, n_shooting, motor_noise_magnitude, sensory_noise_magnitude)
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
