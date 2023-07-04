"""
...
"""

import platform

import pickle
import biorbd_casadi as biorbd
import matplotlib.pyplot as plt
import casadi as cas
import numpy as np
from IPython import embed
import scipy

import sys
sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    OptimalControlProgram,
    Bounds,
    InitialGuess,
    ObjectiveFcn,
    Solver,
    BiorbdModel,
    ObjectiveList,
    NonLinearProgram,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    DynamicsList,
    BoundsList,
    InterpolationType,
    OcpType,
    PenaltyController,
    Node,
    ConstraintList,
    ConstraintFcn,
    MultinodeConstraintList,
    MultinodeObjectiveList,
    BiMappingList,
)

def get_excitation_with_feedback(K, EE, EE_ref, wS):
    return K @ ((EE - EE_ref) + wS)

def stochastic_forward_dynamics(
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    stochastic_variables: cas.MX | cas.SX,
    nlp: NonLinearProgram,
    wM,
    wS,
    with_gains=True,
) -> DynamicsEvaluation:

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    tau_fb = tau
    if with_gains:
        ee_ref = DynamicsFunctions.get(nlp.stochastic_variables["ee_ref"], stochastic_variables)
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        n_controls = nlp.controls.cx_start.shape[0]
        n_q = nlp.states['q'].cx.shape[0]
        n_qdot = nlp.states['qdot'].cx.shape[0]

        K_matrix = cas.MX(n_controls, n_q+n_qdot)
        for s0 in range(n_controls):
            for s1 in range(n_q+n_qdot):
                K_matrix[s0, s1] = k[s0*(n_q+n_qdot) + s1]

        ee = cas.vertcat(q[2], qdot[2])

        tau_fb += get_excitation_with_feedback(K_matrix, ee, ee_ref, wS)

    torques_computed = tau_fb + wM
    dq_computed = qdot  ### Do not use "DynamicsFunctions.compute_qdot(nlp, q, qdot)" it introduces errors!!

    friction = np.zeros((4, 4))
    friction[3, 3] = 0.05
    # np.array([[0.05, 0.025], [0.025, 0.05]])

    mass_matrix = nlp.model.mass_matrix(q)
    nleffects = nlp.model.nleffects(q, qdot)

    dqdot_computed = cas.inv(mass_matrix) @ (torques_computed - nleffects - friction @ qdot)

    return DynamicsEvaluation(dxdt=cas.vertcat(dq_computed, dqdot_computed), defects=None)

def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram, wM, wS):

    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    ConfigureProblem.configure_tau(ocp, nlp, False, True)

    # Stochastic variables
    ConfigureProblem.configure_k(ocp, nlp, n_controls=1, n_feedbacks=3)  # Pelvis position & velocity + actuated q angles
    ConfigureProblem.configure_ee_ref(ocp, nlp, n_references=2)  # Pelvis position & velocity
    ConfigureProblem.configure_m(ocp, nlp)
    ConfigureProblem.configure_cov(ocp, nlp)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, stochastic_forward_dynamics, wM=wM, wS=wS, with_gains=False, expand=False)

def minimize_uncertainty(controllers: list[PenaltyController], key: str) -> cas.MX:
    """
    Minimize the uncertainty (covariance matrix) of the states.
    """
    dt = controllers[0].tf / controllers[0].ns
    out = 0
    for i, ctrl in enumerate(controllers):
        P_matrix = ctrl.restore_matrix_from_vector(ctrl.update_values, ctrl.states.cx.shape[0],
                                                         ctrl.states.cx.shape[0], Node.START, "cov")
        P_partial = P_matrix[ctrl.states[key].index, ctrl.states[key].index]
        out += cas.trace(P_partial) * dt
    return out

def pelvis_equals_pelvis_ref(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    ee_ref = controller.stochastic_variables["ee_ref"].cx_start
    ee = cas.vertcat(q[2], qdot[2])
    return ee - ee_ref

def get_p_mat(nlp, node_index):

    nlp.states.node_index = node_index - 1
    nlp.controls.node_index = node_index - 1
    nlp.stochastic_variables.node_index = node_index - 1
    nlp.update_values.node_index = node_index - 1

    nx = nlp.states.cx_start.shape[0]
    M_matrix = nlp.restore_matrix_from_vector(nlp.stochastic_variables, nx, nx, Node.START, "m")

    dt = nlp.tf / nlp.ns
    n_tau = nlp.controls['tau'].cx_start.shape[0]
    n_q = nlp.states['q'].cx_start.shape[0]
    n_qdot = nlp.states['qdot'].cx_start.shape[0]
    wM = cas.MX.sym("wM", n_tau)
    wP = cas.MX.sym("wP", n_q)
    wPdot = cas.MX.sym("wPdot", n_qdot)
    sigma_w = cas.vertcat(wP, wPdot, wM) * cas.MX_eye(n_tau + n_q + n_qdot)
    cov_sym = cas.MX.sym("cov", nlp.update_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = nlp.restore_matrix_from_vector(cov_sym_dict, nx, nx, Node.START, "cov")

    dx = stochastic_forward_dynamics(nlp.states.cx_start, nlp.controls.cx_start,
                                     nlp.parameters, nlp.stochastic_variables.cx_start,
                                     nlp, wM, wP, wPdot, with_gains=True)

    ddx_dwM = cas.jacobian(dx.dxdt, cas.vertcat(wP, wPdot, wM))
    dg_dw = - ddx_dwM * dt
    ddx_dx = cas.jacobian(dx.dxdt, nlp.states.cx_start)
    dg_dx = - (ddx_dx * dt / 2 + cas.MX_eye(ddx_dx.shape[0]))

    p_next = M_matrix @ (dg_dx @ cov_matrix @ dg_dx.T + dg_dw @ sigma_w @ dg_dw.T) @ M_matrix.T
    func_eval = cas.Function("p_next", [nlp.states.cx_start, nlp.controls.cx_start,
                                          nlp.parameters, nlp.stochastic_variables.cx_start, cov_sym,
                                          wM, wP, wPdot], [p_next])(nlp.states.cx_start,
                                                                          nlp.controls.cx_start,
                                                                          nlp.parameters,
                                                                          nlp.stochastic_variables.cx_start,
                                                                          nlp.update_values.cx_start,  # Should be the right shape to work
                                                                          wM_numerical,
                                                                          wS_numerical)
    p_vector = nlp.restore_vector_from_matrix(func_eval)
    return p_vector

def reach_target_consistantly(controllers: list[PenaltyController]) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    This is a multi-node constraint because the covariance matrix depends on all the precedent nodes, but it only
    applies at the END node.
    """

    Q = cas.MX.sym("q_sym", controllers[-1].states["q"].cx_start.shape[0])
    Qdot = cas.MX.sym("qdot_sym", controllers[-1].states["qdot"].cx_start.shape[0])
    cov_sym = cas.MX.sym("cov", controllers[-1].update_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = controllers[-1].restore_matrix_from_vector(cov_sym_dict, controllers[-1].states.cx_start.shape[0], controllers[-1].states.cx_start.shape[0], Node.START, "cov")

    CoM_pos = get_CoM(controllers[-1].model, Q)
    CoM_vel = get_CoMdot(controllers[-1].model, Q, Qdot)

    jac_marker_q = cas.jacobian(CoM_pos, Q)
    jac_marker_qdot = cas.jacobian(CoM_vel, cas.vertcat(Q, Qdot))

    P_matrix_q = cov_matrix[:2, :2]
    P_matrix_qdot = cov_matrix[:4, :4]

    pos_constraint = jac_marker_q @ P_matrix_q @ jac_marker_q.T
    vel_constraint = jac_marker_qdot @ P_matrix_qdot @ jac_marker_qdot.T

    out = cas.vertcat(pos_constraint[0, 0], pos_constraint[1, 1], vel_constraint[0, 0], vel_constraint[1, 1])

    fun = cas.Function("reach_target_consistantly", [Q, Qdot, cov_sym], [out])
    val = fun(controllers[-1].states["q"].cx_start, controllers[-1].states["qdot"].cx_start, controllers[-1].update_values.cx_start)
    # Since the stochastic variables are defined with ns+1, the cx_start actually refers to the last node (when using node=Node.END)

    return val

def get_CoM(model, q):
    return model.center_of_mass(q)

def get_CoMdot(model, q, qdot):
    return model.center_of_mass_dot(q, qdot)

def get_ee(model, q, qdot):
    return cas.vertcat(get_CoM(model, q), get_CoMdot(model, q, qdot))

def expected_feedback_effort(controllers: list[PenaltyController]) -> cas.MX:
    """
    ...
    """
    # Constants TODO: remove fom here
    # TODO: How do we choose?
    global wM_numerical, wPq_numerical, wPqdot_numerical

    dt = controllers[0].tf / controllers[0].ns
    sensory_noise = cas.vertcat(wPq_numerical, wPqdot_numerical)
    sensory_noise_matrix = sensory_noise * cas.MX_eye(4)

    # create the casadi function to be evaluated
    # Get the symbolic variables
    pelvis_ref = controllers[0].stochastic_variables["ee_ref"].cx_start
    cov_sym = cas.MX.sym("cov", controllers[0].update_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = controllers[0].restore_matrix_from_vector(cov_sym_dict, controllers[0].states.cx_start.shape[0], controllers[0].states.cx_start.shape[0], Node.START, "cov")
    K_matrix = controllers[0].restore_matrix_from_vector(controllers[0].stochastic_variables,
                                               controllers[0].states["muscles"].cx.shape[0],
                                               controllers[0].states["q"].cx.shape[0] + controllers[0].states["qdot"].cx.shape[0],
                                               Node.START,
                                               "k")

    # Compute the expected effort
    trace_k_sensor_k = cas.trace(K_matrix @ sensory_noise_matrix @ K_matrix.T)
    pelvis_pos_velo = cas.vertcat(controllers[0].states["q"].cx_start[2], controllers[0].states["qot"].cx_start[2])
    e_fb = K_matrix @ ((pelvis_pos_velo - pelvis_ref) + sensory_noise)
    jac_e_fb_x = cas.jacobian(e_fb, controllers[0].states.cx_start)
    trace_jac_p_jack = cas.trace(jac_e_fb_x @ cov_matrix @ jac_e_fb_x.T)
    expectedEffort_fb_mx = trace_jac_p_jack + trace_k_sensor_k
    func = cas.Function('f_expectedEffort_fb',
                                       [controllers[0].states.cx_start, controllers[0].stochastic_variables.cx_start, cov_sym],
                                       [expectedEffort_fb_mx])

    f_expectedEffort_fb = 0
    for i, ctrl in enumerate(controllers):
        P_vector = ctrl.update_values.cx_start
        out = func(ctrl.states.cx_start, ctrl.stochastic_variables.cx_start, P_vector)
        f_expectedEffort_fb += out * dt

    return f_expectedEffort_fb


def zero_acceleration(controller: PenaltyController, wM: np.ndarray, wS: np.ndarray) -> cas.MX:
    dx = stochastic_forward_dynamics(controller.states.cx_start, controller.controls.cx_start,
                                     controller.parameters.cx_start, controller.stochastic_variables.cx_start,
                                     controller.get_nlp, wM, wS)
    return dx.dxdt[2:4]

def CoM_over_ankle(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    CoM_pos = get_CoM(controller.model, q)
    CoM_pos_y = CoM_pos[1]
    marker_pos = controller.model.marker_pos(q)[0]
    marker_pos_y = marker_pos[1]
    return marker_pos_y - CoM_pos_y

def leuven_trapezoidal(controllers: list[PenaltyController]) -> cas.MX:

    wM = np.zeros((2, 1))
    wS = np.zeros((4, 1))
    dt = controllers[0].tf / controllers[0].ns

    dX_i = stochastic_forward_dynamics(controllers[0].states.cx_start, controllers[0].controls.cx_start,
                                        controllers[0].parameters.cx_start, controllers[0].stochastic_variables.cx_start,
                                        controllers[0].get_nlp, wM, wS, with_gains=False).dxdt
    dX_i_plus = stochastic_forward_dynamics(controllers[1].states.cx_start, controllers[1].controls.cx_start,
                                        controllers[1].parameters.cx_start, controllers[1].stochastic_variables.cx_start,
                                        controllers[1].get_nlp, wM, wS, with_gains=False).dxdt

    out = controllers[1].states.cx_start - (controllers[0].states.cx_start + (dX_i + dX_i_plus) / 2 * dt)

    return out * 1e3

def prepare_socp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    variable_mappings = BiMappingList()
    variable_mappings.add("tau", to_second=[None, None, None, 0], to_first=[0])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="tau", weight=1e3/2, quadratic=True)

    multinode_objectives = MultinodeObjectiveList()
    multinode_objectives.add(minimize_uncertainty,
                                nodes_phase=[0 for _ in range(n_shooting)],
                                nodes=[i for i in range(n_shooting)],
                                key="muscles",
                                weight=1e3 / 2,
                                quadratic=False)
    multinode_objectives.add(expected_feedback_effort,
                             nodes_phase=[0 for _ in range(n_shooting)],
                             nodes=[i for i in range(n_shooting)],
                             weight=1e3 / 2,
                             quadratic=False)

    # Constraints
    constraints = ConstraintList()
    constraints.add(pelvis_equals_pelvis_ref, node=Node.ALL_SHOOTING)
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", node=Node.START, target=np.array([0, 0, -0.5, 0.5]))
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.START, target=np.array([0, 0, 0, 0]))
    # constraints.add(zero_acceleration, node=Node.START, wM=np.zeros((2, 1)), wS=np.zeros((4, 1)))
    constraints.add(CoM_over_ankle, node=Node.PENULTIMATE)
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.PENULTIMATE, target=np.array([0, 0, 0, 0]))
    # constraints.add(zero_acceleration, node=Node.PENULTIMATE, wM=np.zeros((2, 1)), wS=np.zeros((4, 1)))  # Not possible sice the control on the last node is NaN
    constraints.add(ConstraintFcn.TRACK_CONTROL, key="tau", node=Node.ALL_SHOOTING, min_bound=-100, max_bound=100)
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", node=Node.ALL, min_bound=-5*np.pi, max_bound=5*np.pi)

    multinode_constraints = MultinodeConstraintList()
    multinode_constraints.add(reach_target_consistantly,
                              nodes_phase=[0 for _ in range(n_shooting)],
                              nodes=[i for i in range(n_shooting)],
                              min_bound=np.array([-cas.inf, -cas.inf, -cas.inf, -cas.inf]),
                              max_bound=np.array([0.004**2, 0.05**2]))
    for i in range(n_shooting-1):
        multinode_constraints.add(leuven_trapezoidal,
                                  nodes_phase=[0, 0],
                                  nodes=[i, i+1])

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(configure_stochastic_optimal_control_problem, dynamic_function=stochastic_forward_dynamics, wM=np.zeros((2, 1)), wS=np.zeros((4, 1)), expand=False)

    n_tau = 1
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_states = n_q + n_qdot

    states_min = np.ones((n_states, n_shooting+1)) * -cas.inf
    states_max = np.ones((n_states, n_shooting+1)) * cas.inf
    # states_min[bio_model.nb_q + bio_model.nb_qdot:, :] = 0.001  # Activations
    # states_max[bio_model.nb_q + bio_model.nb_qdot:, :] = 1  # Activations
    # states_min[0:4, 0] = [shoulder_pos_initial, elbow_pos_initial, 0, 0]  # Initial position + velocities
    # states_max[0:4, 0] = [shoulder_pos_initial, elbow_pos_initial, 0, 0]  # Initial position + velocities
    # states_min[0:2, 1:] = 0
    # states_max[0:2, 1:] = 180  # should be np.pi, but there is a mistake in the original code
    # states_min[2:4, n_shooting-1] = 0
    # states_max[2:4, n_shooting-1] = 0


    x_bounds = BoundsList()
    x_bounds.add(bounds=Bounds(states_min, states_max, interpolation=InterpolationType.EACH_FRAME))

    u_bounds = BoundsList()
    controls_min = np.ones((n_tau, 3)) * -cas.inf
    controls_max = np.ones((n_tau, 3)) * cas.inf
    # controls_min = np.ones((n_muscles, 3)) * 0.001
    # controls_max = np.ones((n_muscles, 3)) * 1
    u_bounds.add(bounds=Bounds(controls_min, controls_max))

    # Initial guesses
    states_init = np.zeros((n_states, n_shooting + 1))
    states_init[2, :-1] = np.linspace(-0.5, 0, n_shooting)
    states_init[2, -1] = 0
    states_init[3, :-1] = np.linspace(0.5, 0, n_shooting)
    states_init[3, -1] = 0
    states_init[n_q + n_qdot:, :] = 0.01
    x_init = InitialGuess(states_init, interpolation=InterpolationType.EACH_FRAME)

    controls_init = np.ones((n_tau, n_shooting)) * 1
    u_init = InitialGuess(controls_init, interpolation=InterpolationType.EACH_FRAME)

    # TODO: This should probably be done automatically, not defined by the user
    n_stochastic = n_tau*(n_q + n_qdot) + n_q+n_qdot + n_states*n_states  # K(1x4) + ee_ref(2x1) + M(4x4)
    stochastic_init = np.zeros((n_stochastic, n_shooting + 1))

    curent_index = 0
    stochastic_init[:n_tau * (n_q + n_qdot), :] = 0.01  # K
    curent_index += n_tau * (n_q + n_qdot)
    stochastic_init[curent_index: curent_index + n_q + n_qdot, :] = 0.01  # ee_ref
    # stochastic_init[curent_index : curent_index + n_q+n_qdot, 0] = np.array([ee_initial_position[0], ee_initial_position[1], 0, 0])  # ee_ref
    # stochastic_init[curent_index : curent_index + n_q+n_qdot, 1] = np.array([ee_final_position[0], ee_final_position[1], 0, 0])
    curent_index += n_q + n_qdot
    stochastic_init[curent_index: curent_index + n_states * n_states, :] = 0.01  # M

    s_init = InitialGuess(stochastic_init, interpolation=InterpolationType.EACH_FRAME)

    s_bounds = BoundsList()
    stochastic_min = np.ones((n_stochastic, 3)) * -cas.inf
    stochastic_max = np.ones((n_stochastic, 3)) * cas.inf
    s_bounds.add(bounds=Bounds(stochastic_min, stochastic_max))
    # TODO: we should probably change the name stochastic_variables -> helper_variables ?

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        s_init=s_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        s_bounds=s_bounds,
        objective_functions=objective_functions,
        multinode_objectives=multinode_objectives,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        variable_mappings=variable_mappings,
        ode_solver=None,
        skip_continuity=True,
        n_threads=1,
        assume_phase_dynamics=False,
        problem_type=OcpType.SOCP_EXPLICIT,  # TODO: seems weird for me to do StochasticOPtim... (comme mhe)
        update_value_function=get_p_mat,
    )

def main():

    global wM_std, wPq_std, wPqdot_std
    RUN_OPTIM_FLAG = True  # False

    biorbd_model_path = "2segments_1(2)contact.bioMod"

    # import bioviz
    # b = bioviz.Viz(biorbd_model_path)
    # b.exec()

    # --- Prepare the ocp --- #
    dt = 0.01
    final_time = 0.4
    n_shooting = int(final_time/dt) + 1
    final_time += dt

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_linear_solver('mumps')
    # solver.set_linear_solver('ma57')
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_maximum_iterations(10000)
    solver.set_hessian_approximation('limited-memory')

    socp = prepare_socp(biorbd_model_path=biorbd_model_path,
                        final_time=final_time,
                        n_shooting=n_shooting)

    if RUN_OPTIM_FLAG:
        sol_socp = socp.solve(solver)

        q_sol = sol_socp.states["q"]
        qdot_sol = sol_socp.states["qdot"]
        activations_sol = sol_socp.states["muscles"]
        excitations_sol = sol_socp.controls["muscles"]
        k_sol = sol_socp.stochastic_variables["k"]
        ee_ref_sol = sol_socp.stochastic_variables["ee_ref"]
        m_sol = sol_socp.stochastic_variables["m"]
        # cov_sol = sol_socp.update_values["cov"]
        stochastic_variables_sol = np.vstack((k_sol, ee_ref_sol, m_sol))  # , cov_sol))
        data = {"q_sol": q_sol,
                "qdot_sol": qdot_sol,
                "activations_sol": activations_sol,
                "excitations_sol": excitations_sol,
                "k_sol": k_sol,
                "ee_ref_sol": ee_ref_sol,
                "m_sol": m_sol,
                # "cov_sol": cov_sol,
                "stochastic_variables_sol": stochastic_variables_sol}

        # --- Save the results --- #
        with open(f"{biorbd_model_path}_torque_driven_socp.pkl", "wb") as file:
            pickle.dump(data, file)
    else:
        with open(f"{biorbd_model_path}_torque_driven_socp.pkl", "rb") as file:
            data = pickle.load(file)
        q_sol = data["q_sol"]
        qdot_sol = data["qdot_sol"]
        activations_sol = data["activations_sol"]
        excitations_sol = data["excitations_sol"]
        k_sol = data["k_sol"]
        ee_ref_sol = data["ee_ref_sol"]
        m_sol = data["m_sol"]
        # cov_sol = data["cov_sol"]
        stochastic_variables_sol = np.vstack((k_sol, ee_ref_sol, m_sol))
        # # Save .mat files
        # scipy.io.savemat(f"leuvenarm_muscle_driven_socp.mat",
        #                     {"q_sol": q_sol,
        #                         "qdot_sol": qdot_sol,
        #                         "activations_sol": activations_sol,
        #                         "excitations_sol": excitations_sol,
        #                         "k_sol": k_sol,
        #                         "ee_ref_sol": ee_ref_sol,
        #                         "m_sol": m_sol,
        #                         "stochastic_variables_sol": stochastic_variables_sol})


    embed()
    import bioviz
    b = bioviz.Viz(model_path=biorbd_model_path)
    b.load_movement(q_sol[:, :-1])
    b.exec()

    # --- Plot the results --- #
    embed()
    states = socp.nlp[0].states.cx_start
    controls = socp.nlp[0].controls.cx_start
    parameters = socp.nlp[0].parameters.cx_start
    stochastic_variables = socp.nlp[0].stochastic_variables.cx_start
    nlp = socp.nlp[0]
    wM_sym = cas.MX.sym('wM', 2, 1)
    wS_sym = cas.MX.sym('wS', 2, 1)
    out = stochastic_forward_dynamics(states, controls, parameters, stochastic_variables, nlp, wM_sym, wS_sym with_gains=True)
    dyn_fun = cas.Function("dyn_fun", [states, controls, parameters, stochastic_variables, wM_sym, wS_sym], [out.dxdt])

    fig, axs = plt.subplots(3, 2)
    n_simulations = 30
    q_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
    qdot_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
    mus_activation_simulated = np.zeros((n_simulations, 6, n_shooting + 1))
    hand_pos_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
    hand_vel_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
    for i_simulation in range(n_simulations):
        wM = np.random.normal(0, wM_std, (2, n_shooting + 1))
        wPq = np.random.normal(0, wPq_std, (2, n_shooting + 1))
        wPqdot = np.random.normal(0, wPqdot_std, (2, n_shooting + 1))
        q_simulated[i_simulation, :, 0] = q_sol[:, 0]
        qdot_simulated[i_simulation, :, 0] = qdot_sol[:, 0]
        mus_activation_simulated[i_simulation, :, 0] = activations_sol[:, 0]
        for i_node in range(n_shooting):
            x_prev = cas.vertcat(q_simulated[i_simulation, :, i_node], qdot_simulated[i_simulation, :, i_node], mus_activation_simulated[i_simulation, :, i_node])
            hand_pos_simulated[i_simulation, :, i_node] = np.reshape(hand_pos_fcn(x_prev[:2])[:2], (2,))
            hand_vel_simulated[i_simulation, :, i_node] = np.reshape(hand_vel_fcn(x_prev[:2], x_prev[2:4])[:2], (2,))
            u = excitations_sol[:, i_node]
            s = stochastic_variables_sol[:, i_node]
            k1 = dyn_fun(x_prev, u, [], s, wM[:, i_node], wPq[:, i_node], wPqdot[:, i_node])
            x_next = x_prev + dt * dyn_fun(x_prev + dt / 2 * k1, u, [], s, wM[:, i_node], wPq[:, i_node], wPqdot[:, i_node])
            q_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[:2], (2, ))
            qdot_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[2:4], (2, ))
            mus_activation_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[4:], (6, ))
        hand_pos_simulated[i_simulation, :, i_node + 1] = np.reshape(hand_pos_fcn(x_next[:2])[:2], (2,))
        hand_vel_simulated[i_simulation, :, i_node + 1] = np.reshape(hand_vel_fcn(x_next[:2], x_next[2:4])[:2], (2, ))
        axs[0, 0].plot(hand_pos_simulated[i_simulation, 0, :], hand_pos_simulated[i_simulation, 1, :], color="tab:red")
        axs[1, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_simulated[i_simulation, 0, :], color="k")
        axs[2, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_simulated[i_simulation, 1, :], color="k")
        axs[0, 1].plot(hand_vel_simulated[i_simulation, 0, :], hand_vel_simulated[i_simulation, 1, :], color="tab:red")
        axs[1, 1].plot(np.linspace(0, final_time, n_shooting + 1), qdot_simulated[i_simulation, 0, :], color="k")
        axs[2, 1].plot(np.linspace(0, final_time, n_shooting + 1), qdot_simulated[i_simulation, 1, :], color="k")
    hand_pos_without_noise = np.zeros((2, n_shooting + 1))
    for i_node in range(n_shooting + 1):
        hand_pos_without_noise[:, i_node] = np.reshape(hand_pos_fcn(q_sol[:, i_node])[:2], (2,))
    axs[0, 0].plot(hand_pos_without_noise[0, :], hand_pos_without_noise[1, :], color="k")
    axs[0, 0].plot(ee_initial_position[0], ee_initial_position[1], color="tab:green", marker="o")
    axs[0, 0].plot(ee_final_position[0], ee_final_position[1], color="tab:red", marker="o")
    axs[0, 0].set_xlabel("X [m]")
    axs[0, 0].set_ylabel("Y [m]")
    axs[0, 0].set_title("Hand position simulated")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Shoulder angle [rad]")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Elbow angle [rad]")
    axs[0, 1].set_xlabel("X velocity [m/s]")
    axs[0, 1].set_ylabel("Y velocity [m/s]")
    axs[0, 1].set_title("Hand velocity simulated")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Shoulder velocity [rad/s]")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Elbow velocity [rad/s]")
    axs[0, 0].axis("equal")
    plt.tight_layout()
    plt.savefig("simulated_results.png", dpi=300)
    plt.show()

    # TODO: integrate to see the error they commit with the trapezoidal


# --- Define constants to specify the model  --- #
# (To simplify the problem, relationships are used instead of "real" muscles)
dM_coefficients = np.array([[0, 0, 0.0100, 0.0300, -0.0110, 1.9000],
                            [0, 0, 0.0100, -0.0190, 0, 0.0100],
                            [0.0400, -0.0080, 1.9000, 0, 0, 0.0100],
                            [-0.0420, 0, 0.0100, 0, 0, 0.0100],
                            [0.0300, -0.0110, 1.9000, 0.0320, -0.0100, 1.9000],
                            [-0.0390, 0, 0.0100, -0.0220, 0, 0.0100]])

LMT_coefficients = np.array([[1.1000, -5.206336195535022],
                             [0.8000, -7.538918356984516],
                             [1.2000, -3.938098437958920],
                             [0.7000, -3.031522725559912],
                             [1.1000, -2.522778221157014],
                             [0.8500, -1.826368199415192]])
vMtilde_max = np.ones((6, 1)) * 10
Fiso = np.array([572.4000, 445.2000, 699.6000, 381.6000, 159.0000, 318.0000])
Faparam = np.array([0.814483478343008, 1.055033428970575, 0.162384573599574, 0.063303448465465, 0.433004984392647, 0.716775413397760, -0.029947116970696, 0.200356847296188])
Fvparam = np.array([-0.318323436899127, -8.149156043475250, -0.374121508647863, 0.885644059915004])
Fpparam = np.array([-0.995172050006169, 53.598150033144236])
muscleDampingCoefficient = np.ones((6, 1)) * 0.01
tau_coef = 0.1500

m1 = 1.4
m2 = 1
l1 = 0.3
l2 = 0.33
lc1 = 0.11
lc2 = 0.16
I1 = 0.025
I2 = 0.045

# TODO: How do we choose the values?
wM_std = 0.05
wPq_std = 3e-4
wPqdot_std = 0.0024

dt = 0.01
wM_numerical = np.array([wM_std ** 2 / dt, wM_std ** 2 / dt])
wPq_numerical = np.array([wPq_std ** 2 / dt, wPq_std ** 2 / dt])
wPqdot_numerical = np.array([wPqdot_std ** 2 / dt, wPqdot_std ** 2 / dt])
wS_numerical = np.concatenate((wPq_numerical, wPqdot_numerical))

if __name__ == "__main__":
    main()

# TODO: Check expected feedback effort
