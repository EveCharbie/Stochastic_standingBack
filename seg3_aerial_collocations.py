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
)

def sensory_reference(
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
    vestibular_and_joints_feedback = cas.vertcat(q_roots[2], q_joints, qdot_roots[2], qdot_joints)
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
    CoM_vel = controller.model.center_of_mass_velocity(cas.vertcat(Q_root, Q_joints), cas.vertcat(Qdot_root, Qdot_joints))[:2]
    CoM_ang_vel = controller.model.body_rotation_rate(cas.vertcat(Q_root, Q_joints), cas.vertcat(Qdot_root, Qdot_joints))[0]

    jac_CoM_q = cas.jacobian(CoM_pos, cas.vertcat(Q_root, Q_joints))
    jac_CoM_qdot = cas.jacobian(CoM_vel, cas.vertcat(Q_root, Q_joints, Qdot_root, Qdot_joints))
    jac_CoM_ang_vel = cas.jacobian(CoM_ang_vel, cas.vertcat(Q_root, Q_joints, Qdot_root, Qdot_joints))

    P_matrix_q = cov_matrix[:n_q, :n_q]
    P_matrix_qdot = cov_matrix[:, :]

    pos_constraint = jac_CoM_q @ P_matrix_q @ jac_CoM_q.T
    vel_constraint = jac_CoM_qdot @ P_matrix_qdot @ jac_CoM_qdot.T
    rot_constraint = jac_CoM_ang_vel @ P_matrix_qdot @ jac_CoM_ang_vel.T

    out = cas.vertcat(pos_constraint[0, 0], pos_constraint[1, 1], vel_constraint[0, 0], vel_constraint[1, 1], rot_constraint[0, 0])

    fun = cas.Function("reach_target_consistantly", [Q_root, Q_joints, Qdot_root, Qdot_joints, cov_sym], [out])
    val = fun(controller.states["q_roots"].cx_start,
              controller.states["q_joints"].cx_start,
              controller.states["qdot_roots"].cx_start,
              controller.states["qdot_joints"].cx_start,
              controller.stochastic_variables["cov"].cx_start
              )
    # Since the stochastic variables are defined with ns+1, the cx_start actually refers to the last node (when using node=Node.END)

    return val

def CoM_over_toes(controller: PenaltyController) -> cas.MX:
    q_roots = controller.states["q_roots"].cx_start
    q_joints = controller.states["q_joints"].cx_start
    q = cas.vertcat(q_roots, q_joints)
    CoM_pos = controller.model.center_of_mass(q)
    CoM_pos_y = CoM_pos[1]
    marker_pos = controller.model.markers(q)[4]
    marker_pos_y = marker_pos[1]
    return marker_pos_y - CoM_pos_y


def prepare_socp(
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
    m_last: np.ndarray = None,
    cov_last: np.ndarray = None,
) -> StochasticOptimalControlProgram:
    """
    ...
    """

    polynomial_degree = 3

    problem_type = SocpType.COLLOCATION(polynomial_degree=polynomial_degree, method="legendre")

    n_q = 6
    n_root = 3
    n_joints = n_q - n_root
    friction_coefficients = cas.DM.zeros(n_root, n_root)
    for i in range(n_joints):
        friction_coefficients[i, i] = 0.1

    bio_model = StochasticBiorbdModel(
        biorbd_model_path,
        sensory_noise_magnitude=sensory_noise_magnitude,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_reference=sensory_reference,
        n_references=8,
        n_noised_states=n_q*2,
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
        node=Node.ALL,
        weight=1e3 / 2,
        quadratic=True,
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, min_bound=0.1, max_bound=1)
    # if np.sum(sensory_noise_magnitude) == 0:
    #     objective_functions.add(ObjectiveFcn.Lagrange.STOCHASTIC_MINIMIZE_VARIABLE, key="k", weight=0.01, quadratic=True)

    objective_functions.add(reach_landing_position_consistantly,
                    custom_type=ObjectiveFcn.Mayer,
                    node=Node.END,
                    weight=1e3,
                    quadratic=True)

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
    )

    pose_at_first_node = np.array([-0.0422, 0.0892, 0.2386, 0.0, -0.1878, 0.0])  # Initial position approx from bioviz
    pose_at_last_node = np.array([-0.0422, 0.0892, 5.7904, 0.0, 0.5036, 0.0])  # Final position approx from bioviz

    x_bounds = BoundsList()
    x_bounds["q_roots"] = bio_model.bounds_from_ranges("q_roots")
    x_bounds["q_joints"] = bio_model.bounds_from_ranges("q_joints")
    x_bounds["q_roots"].min[:, 0] = pose_at_first_node[:n_root]
    x_bounds["q_roots"].max[:, 0] = pose_at_first_node[:n_root]
    x_bounds["q_joints"].min[:, 0] = pose_at_first_node[n_root:]
    x_bounds["q_joints"].max[:, 0] = pose_at_first_node[n_root:]
    x_bounds["q_roots"].min[2, 2] = 2*np.pi - 0.2
    x_bounds["q_roots"].max[2, 2] = 2*np.pi + 0.2
    x_bounds["qdot_roots"] = bio_model.bounds_from_ranges("qdot_roots")
    x_bounds["qdot_joints"] = bio_model.bounds_from_ranges("qdot_joints")
    x_bounds["qdot_roots"].min[:, 0] = 0
    x_bounds["qdot_roots"].max[:, 0] = 0
    x_bounds["qdot_joints"].min[:, 0] = 0
    x_bounds["qdot_joints"].max[:, 0] = 0
    x_bounds["qdot_roots"].min[1, 0] = 2
    x_bounds["qdot_roots"].max[1, 0] = 2
    x_bounds["qdot_roots"].min[2, 0] = 2.5 * np.pi
    x_bounds["qdot_roots"].max[2, 0] = 2.5 * np.pi

    u_bounds = BoundsList()
    tau_min = np.ones((n_q-n_root, 3)) * -500
    tau_max = np.ones((n_q-n_root, 3)) * 500
    tau_min[:, 0] = 0
    tau_max[:, 0] = 0
    u_bounds.add("tau_joints", min_bound=tau_min, max_bound=tau_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    # Initial guesses
    x_init = InitialGuessList()
    if q_roots_last is None:
        x_init.add("q_roots", initial_guess=np.vstack((pose_at_first_node[:n_root], pose_at_last_node[:n_root])).T, interpolation=InterpolationType.LINEAR)
        x_init.add("q_joints", initial_guess=np.vstack((pose_at_first_node[n_root:], pose_at_last_node[n_root:])).T, interpolation=InterpolationType.LINEAR)
        x_init.add("qdot_roots", initial_guess=[0.01]*n_root, interpolation=InterpolationType.CONSTANT)
        x_init.add("qdot_joints", initial_guess=[0.01]*n_joints, interpolation=InterpolationType.CONSTANT)
    else:
        x_init.add("q_roots", initial_guess=q_roots_last, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("q_joints", initial_guess=q_joints_last, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("qdot_roots", initial_guess=qdot_roots_last, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("qdot_joints", initial_guess=qdot_joints_last, interpolation=InterpolationType.ALL_POINTS)

    u_init = InitialGuessList()
    if tau_last is None:
        u_init.add("tau_joints", initial_guess=[0.01] * n_joints, interpolation=InterpolationType.CONSTANT)
    else:
        u_init.add("tau_joints", initial_guess=tau_joints_last, interpolation=InterpolationType.ALL_POINTS)

    n_ref = 2*(n_joints+1)  # ref(8)
    n_k = n_joints * n_ref  # K(3x8)
    n_m = (2*n_q)**2 * (polynomial_degree+1)  # M(12x12x4)
    n_stochastic = n_k + n_ref + n_m
    n_cov = (2 * n_q) ** 2  # Cov(12x12)
    n_stochastic += n_cov

    s_init = InitialGuessList()
    s_bounds = BoundsList()


    # TODO: intiialize !!!!


    if k_last is None:
        k_last = np.ones((n_k, n_shooting + 1)) * 0.01
    s_init.add("k", initial_guess=k_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("k", min_bound=[-500]*n_k, max_bound=[500]*n_k, interpolation=InterpolationType.CONSTANT)

    ref_min = cas.horzcat(x_bounds["q_roots"].min[2, :], x_bounds["q_joints"].min, x_bounds["qdot_roots"].min[2, :], x_bounds["qdot_joints"].min).T
    ref_max = cas.horzcat(x_bounds["q_roots"].max[2, :], x_bounds["q_joints"].max, x_bounds["qdot_roots"].max[2, :], x_bounds["qdot_joints"].max).T

    if ref_last is None:
        # ref_last = get_ref_init(q_last, qdot_last)
        ref_last = np.ones((n_ref, n_shooting + 1)) * 0.01
    s_init.add("ref", initial_guess=ref_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("ref", min_bound=ref_min, max_bound=ref_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    if m_last is None:
        # m_last = get_m_init(biorbd_model_path, n_joints, n_stochastic, n_shooting, time_last, polynomial_degree, q_last, qdot_last, tau_last, k_last, ref_last, motor_noise_magnitude, sensory_noise_magnitude)
        m_last = np.ones((n_m, n_shooting + 1)) * 0.01
    s_init.add("m", initial_guess=m_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("m", min_bound=[-50]*n_m, max_bound=[50]*n_m, interpolation=InterpolationType.CONSTANT)

    P_0 = cas.DM_eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P
    cov_min = np.ones((n_cov, 3)) * -500
    cov_max = np.ones((n_cov, 3)) * 500
    cov_vector = np.zeros((n_cov, 1))
    for i in range(2*n_q):
        for j in range(2*n_q):
            cov_vector[i*n_q+j] = P_0[i, j]
    if cov_last is None:
        cov_last = np.repeat(cov_vector, n_shooting + 1, axis=1)
    cov_min[:, 0] = cov_vector.reshape((-1, ))
    cov_max[:, 0] = cov_vector.reshape((-1, ))

    s_init.add("cov", initial_guess=cov_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("cov", min_bound=cov_min, max_bound=cov_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

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
        # u_scaling=u_scaling,
        # s_scaling=s_scaling,
        objective_functions=objective_functions,
        constraints=constraints,
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        n_threads=1,
        problem_type=problem_type,
    )


def main():

    model_name = "Model2D_6Dof_0C_3M"
    biorbd_model_path = f"models/{model_name}.bioMod"
    biorbd_model_path_with_mesh = f"models/{model_name}_with_mesh.bioMod"

    save_path = f"results/{model_name}_aerial_socp_collocations.pkl"


    n_q = 6
    n_root = 3

    dt = 0.05
    final_time = 0.8
    n_shooting = int(final_time / dt)

    # TODO: How do we choose the values?
    motor_noise_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024

    motor_noise_magnitude = cas.DM(
        np.array([motor_noise_std ** 2 / dt for _ in range(n_q - n_root)]))  # All DoFs except root
    sensory_noise_magnitude = cas.DM(cas.vertcat(
        np.array([wPq_std ** 2 / dt for _ in range(n_q - n_root + 1)]),
        np.array([wPqdot_std ** 2 / dt for _ in range(n_q - n_root + 1)])
    ))  # since the head is fixed to the pelvis, the vestibular feedback is in the states ref

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_linear_solver('ma97')
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_maximum_iterations(1000)
    solver.set_hessian_approximation('limited-memory')  # Mandatory, otherwise RAM explodes!
    solver._nlp_scaling_method = "none"

    socp = prepare_socp(biorbd_model_path,
                        final_time,
                        n_shooting,
                        motor_noise_magnitude,
                        sensory_noise_magnitude)
    sol = socp.solve(solver)

    q_roots_sol = sol.states["q_roots"]
    q_joints_sol = sol.states["q_joints"]
    qdot_roots_sol = sol.states["qdot_roots"]
    qdot_joints_sol = sol.states["qdot_joints"]
    tau_joints_sol = sol.controls["tau_joints"]
    time_sol = sol.parameters["time"][0][0]
    data = {"q_roots_sol": q_roots_sol,
            "q_joints_sol": q_joints_sol,
            "qdot_roots_sol": qdot_roots_sol,
            "qdot_joints_sol": qdot_joints_sol,
            "tau_joints_sol": tau_joints_sol,
            "time_sol": time_sol}

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


