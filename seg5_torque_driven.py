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

from bioptim import (
    OptimalControlProgram,
    StochasticOptimalControlProgram,
    InitialGuessList,
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
    OcpType,
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
    with_gains,
) -> DynamicsEvaluation:

    biorbd_model = biorbd.Model("models/Model2D_7Dof_1C_3M.bioMod")
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    n_q = biorbd_model.nbQ()
    n_root = biorbd_model.nbRoot()

    tau_fb = tau[n_root:]
    if with_gains:
        ee_ref = DynamicsFunctions.get(nlp.stochastic_variables["ee_ref"], stochastic_variables)
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        K_matrix = cas.MX((n_q-n_root) * 2, n_q-n_root)
        for s0 in range(2*(n_q-n_root)):
            for s1 in range(n_q-n_root):
                K_matrix[s0, s1] = k[s0*1 + s1]
        K_matrix = K_matrix.T

        ee = cas.vertcat(q[n_root:], qdot[n_root:])

        tau_fb += get_excitation_with_feedback(K_matrix, ee, ee_ref, wS) + wM

    dq_computed = qdot

    friction = cas.MX.zeros(n_q, n_q)
    for i in range(n_root, n_q):
        friction[i, i] = 0.05

    # dqdot_computed = biorbd_model.ForwardDynamics(q, qdot, tau_fb + friction @ qdot).to_mx()
    tau_full = cas.vertcat(cas.MX.zeros(n_root), tau_fb)
    dqdot_computed = biorbd_model.ForwardDynamicsConstraintsDirect(q, qdot, tau_full + friction @ qdot).to_mx()

    return DynamicsEvaluation(dxdt=cas.vertcat(dq_computed, dqdot_computed), defects=None)


def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram, wM, wS):

    n_q = ocp.nlp[0].model.nb_q
    n_root = ocp.nlp[0].model.nb_root

    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    ConfigureProblem.configure_tau(ocp, nlp, False, True)

    # Stochastic variables
    ConfigureProblem.configure_stochastic_k(ocp, nlp, n_noised_controls=n_q-n_root, n_feedbacks=2*(n_q-n_root))  # Actuated states + vestibular eventually
    ConfigureProblem.configure_stochastic_ee_ref(ocp, nlp, n_references=2*(n_q-n_root))  # Hip position & velocity + vestibular eventually
    ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=2*n_q)
    mat_p_init = cas.DM_eye(2*n_q) * np.hstack((np.ones((n_q, )) * 1e-4, np.ones((n_q, )) * 1e-7))  # P
    ConfigureProblem.configure_stochastic_cov(ocp, nlp, n_noised_states=2*n_q, initial_matrix=mat_p_init)
    ConfigureProblem.configure_dynamics_function(ocp, nlp,
                                                 dyn_func=nlp.dynamics_type.dynamic_function,
                                                 wM=wM, wS=wS, with_gains=False, expand=False)


def states_equals_ref_kinematics(controller: PenaltyController) -> cas.MX:
    n_root = controller.model.nb_root
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    ee_ref = controller.stochastic_variables["ee_ref"].cx_start
    ee = cas.vertcat(q[n_root:], qdot[n_root:])  # Non-root states
    return ee - ee_ref

def pelvis_equals_pelvis_ref(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    ee_ref = controller.stochastic_variables["ee_ref"].cx_start
    ee = cas.vertcat(q[2], qdot[2])  # Pelvis rotation index
    return ee - ee_ref

def get_p_mat(nlp, node_index, wM_magnitude, wS_magnitude):

    dt = nlp.tf / nlp.ns

    nlp.states.node_index = node_index - 1
    nlp.controls.node_index = node_index - 1
    nlp.stochastic_variables.node_index = node_index - 1
    nlp.integrated_values.node_index = node_index - 1

    nx = nlp.states.cx_start.shape[0]
    n_tau = nlp.controls['tau'].cx_start.shape[0]
    n_q = nlp.states['q'].cx_start.shape[0]
    n_root = nlp.model.nb_root

    M_matrix = nlp.stochastic_variables["m"].reshape_to_matrix(nlp.stochastic_variables, nx, nx, Node.START, "m")

    wM = cas.MX.sym("wM", n_tau)
    wS = cas.MX.sym("wS", 2*(n_q-n_root))
    sigma_w = cas.vertcat(wS, wM) * cas.MX_eye(cas.vertcat(wS, wM).shape[0])
    cov_sym = cas.MX.sym("cov", nlp.integrated_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = nlp.integrated_values["cov"].reshape_to_matrix(cov_sym_dict, nx, nx, Node.START, "cov")

    dx = stochastic_forward_dynamics(nlp.states.cx_start, nlp.controls.cx_start,
                                     nlp.parameters, nlp.stochastic_variables.cx_start,
                                     nlp, wM, wS, with_gains=True)

    ddx_dwM = cas.jacobian(dx.dxdt, cas.vertcat(wS, wM))
    dg_dw = - ddx_dwM * dt
    ddx_dx = cas.jacobian(dx.dxdt, nlp.states.cx_start)
    dg_dx = - (ddx_dx * dt / 2 + cas.MX_eye(ddx_dx.shape[0]))

    p_next = M_matrix @ (dg_dx @ cov_matrix @ dg_dx.T + dg_dw @ sigma_w @ dg_dw.T) @ M_matrix.T

    func_eval = cas.Function("p_next", [nlp.states.cx_start, nlp.controls.cx_start,
                                          nlp.parameters, nlp.stochastic_variables.cx_start, cov_sym,
                                          wM, wS], [p_next])(nlp.states.cx_start,
                                                                          nlp.controls.cx_start,
                                                                          nlp.parameters,
                                                                          nlp.stochastic_variables.cx_start,
                                                                          nlp.integrated_values["cov"].cx_start,  # Should be the right shape to work
                                                                          wM_magnitude,
                                                                          wS_magnitude)
    p_vector = nlp.integrated_values["cov"].reshape_to_vector(func_eval)
    return p_vector

def reach_standing_position_consistantly(controllers: list[PenaltyController]) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    This is a multi-node constraint because the covariance matrix depends on all the precedent nodes, but it only
    applies at the END node.
    """
    n_q = controllers[-1].model.nb_q
    n_root = controllers[-1].model.nb_root
    n_joints = controllers[-1].model.nb_q - n_root
    Q_root = cas.MX.sym("q_root", n_root)
    Q_joints = cas.MX.sym("q_joints", n_joints)
    Qdot_root = cas.MX.sym("qdot_root", n_root)
    Qdot_joints = cas.MX.sym("qdot_joints", n_joints)
    n_states = n_q * 2

    cov_sym = cas.MX.sym("cov", controllers[-1].integrated_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = controllers[-1].integrated_values["cov"].reshape_to_matrix(cov_sym_dict, n_states, n_states, Node.START, "cov")

    CoM_pos = get_CoM(controllers[-1].model, cas.vertcat(Q_root, Q_joints))[1]
    CoM_vel = get_CoMdot(controllers[-1].model, cas.vertcat(Q_root, Q_joints), cas.vertcat(Qdot_root, Qdot_joints))[1]

    jac_CoM_q = cas.jacobian(CoM_pos, cas.vertcat(Q_root, Q_joints))
    jac_CoM_qdot = cas.jacobian(CoM_vel, cas.vertcat(Q_root, Q_joints, Qdot_root, Qdot_joints))

    P_matrix_q = cov_matrix[:n_q, :n_q]
    P_matrix_qdot = cov_matrix[:n_q*2, :n_q*2]

    pos_constraint = jac_CoM_q @ P_matrix_q @ jac_CoM_q.T
    vel_constraint = jac_CoM_qdot @ P_matrix_qdot @ jac_CoM_qdot.T

    out = cas.vertcat(pos_constraint, vel_constraint)

    fun = cas.Function("reach_target_consistantly", [Q_root, Q_joints, Qdot_root, Qdot_joints, cov_sym], [out])
    val = fun(controllers[-1].states["q"].cx_start[:n_root],
              controllers[-1].states["q"].cx_start[n_root:n_root+n_joints],
              controllers[-1].states["qdot"].cx_start[:n_root],
              controllers[-1].states["qdot"].cx_start[n_root:n_root+n_joints],
              controllers[-1].integrated_values.cx_start)
    # Since the stochastic variables are defined with ns+1, the cx_start actually refers to the last node (when using node=Node.END)

    return val

def get_CoM(model, q):
    return model.center_of_mass(q)

def get_CoMdot(model, q, qdot):
    return model.center_of_mass_velocity(q, qdot)

def get_CoM_and_CoMdot(model, q, qdot):
    return cas.vertcat(get_CoM(model, q), get_CoMdot(model, q, qdot))

def expected_feedback_effort(controllers: list[PenaltyController], wS_magnitude: cas.DM) -> cas.MX:
    """
    ...
    """

    n_q = controllers[0].model.nb_q
    n_root = controllers[0].model.nb_root
    n_joints = n_q - n_root
    n_states = n_q*2

    dt = controllers[0].tf / controllers[0].ns
    sensory_noise_matrix = wS_magnitude * cas.MX_eye(wS_magnitude.shape[0])

    states_ref = controllers[0].stochastic_variables["ee_ref"].cx_start
    cov_sym = cas.MX.sym("cov", controllers[0].integrated_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = controllers[0].integrated_values.reshape_to_matrix(cov_sym_dict, n_states, n_states, Node.START, "cov")

    k = controllers[0].stochastic_variables["k"].cx_start
    K_matrix = cas.MX(2*(n_q-n_root), n_q-n_root)
    for s0 in range(2*(n_q-n_root)):
        for s1 in range(n_q-n_root):
            K_matrix[s0, s1] = k[s0 * (n_q-n_root) + s1]
    K_matrix = K_matrix.T

    # Compute the expected effort
    trace_k_sensor_k = cas.trace(K_matrix @ sensory_noise_matrix @ K_matrix.T)
    # pelvis_pos_velo = cas.vertcat(controllers[0].states["q"].cx_start[2], controllers[0].states["qot"].cx_start[2])  # Pelvis pos + velo should be used as a feedback for all the joints eventually
    states_pos_velo = cas.vertcat(controllers[0].states["q"].cx_start[n_root:], controllers[0].states["qdot"].cx_start[n_root:])
    T_fb = K_matrix @ ((states_pos_velo - states_ref) + wS_magnitude)
    jac_T_fb_x = cas.jacobian(T_fb, controllers[0].states.cx_start)
    trace_jac_p_jack = cas.trace(jac_T_fb_x @ cov_matrix @ jac_T_fb_x.T)
    expectedEffort_fb_mx = trace_jac_p_jack + trace_k_sensor_k
    func = cas.Function('f_expectedEffort_fb',
                                       [controllers[0].states.cx_start, controllers[0].stochastic_variables.cx_start, cov_sym],
                                       [expectedEffort_fb_mx])

    f_expectedEffort_fb = 0
    for i, ctrl in enumerate(controllers):
        P_vector = ctrl.integrated_values.cx_start
        out = func(ctrl.states.cx_start, ctrl.stochastic_variables.cx_start, P_vector)
        f_expectedEffort_fb += out * dt

    return f_expectedEffort_fb


def zero_acceleration(controller: PenaltyController, wM: np.ndarray, wS: np.ndarray) -> cas.MX:
    dx = stochastic_forward_dynamics(controller.states.cx_start, controller.controls.cx_start,
                                     controller.parameters.cx_start, controller.stochastic_variables.cx_start,
                                     controller.get_nlp, wM, wS, with_gains=False)
    return dx.dxdt[controller.states_dot.index("qddot")]

def CoM_over_ankle(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    CoM_pos = get_CoM(controller.model, q)
    CoM_pos_y = CoM_pos[1]
    marker_pos = controller.model.markers(q)[0]
    marker_pos_y = marker_pos[1]
    return marker_pos_y - CoM_pos_y

def leuven_trapezoidal(controllers: list[PenaltyController]) -> cas.MX:

    n_q = controllers[0].model.nb_q
    n_root = controllers[0].model.nb_root
    n_joints = n_q - n_root
    wM = np.zeros((n_joints, 1))
    wS = np.zeros((2*(n_q-n_root), 1))
    dt = controllers[0].tf / controllers[0].ns

    dX_i = stochastic_forward_dynamics(controllers[0].states.cx_start, controllers[0].controls.cx_start,
                                        controllers[0].parameters.cx_start, controllers[0].stochastic_variables.cx_start,
                                        controllers[0].get_nlp, wM, wS, with_gains=False).dxdt
    dX_i_plus = stochastic_forward_dynamics(controllers[1].states.cx_start, controllers[1].controls.cx_start,
                                        controllers[1].parameters.cx_start, controllers[1].stochastic_variables.cx_start,
                                        controllers[1].get_nlp, wM, wS, with_gains=False).dxdt

    out = controllers[1].states.cx_start - (controllers[0].states.cx_start + (dX_i + dX_i_plus) / 2 * dt)

    return out * 1e3

def leuven_trapezoidal_deterministic(controllers: list[PenaltyController]) -> cas.MX:

    dt = controllers[0].tf / controllers[0].ns

    dX_i = controllers[0].dynamics(controllers[0].states.cx_start, controllers[0].controls.cx_start,
                                        controllers[0].parameters.cx_start, controllers[0].stochastic_variables.cx_start)
    dX_i_plus = controllers[1].dynamics(controllers[1].states.cx_start, controllers[1].controls.cx_start,
                                        controllers[1].parameters.cx_start, controllers[1].stochastic_variables.cx_start)

    out = controllers[1].states.cx_start - (controllers[0].states.cx_start + (dX_i + dX_i_plus) / 2 * dt)

    return out * 1e3

def custom_contact_force_constraint(controller: PenaltyController, contact_index:int) -> cas.MX:
    """
    # TODO: ask Pariterre/Ipuch why configure_contact does not work.
    """

    n_q =controller.model.nb_q
    n_root = controller.model.nb_root

    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    tau = controller.controls["tau"].cx_start

    friction = cas.MX.zeros(n_q, n_q)
    for i in range(n_root, n_q):
        friction[i, i] = 0.05

    tau_full = cas.vertcat(cas.MX.zeros(n_root), tau) + friction @ qdot

    out = controller.model.contact_forces_from_constrained_forward_dynamics(q, qdot, tau_full)[contact_index]

    return out


def prepare_socp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    wM_magnitude: cas.DM,
    wS_magnitude: cas.DM,
    DETERMINISTIC: bool = False,
) -> StochasticOptimalControlProgram:
    """
    ...
    """

    bio_model = BiorbdModel(biorbd_model_path)

    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_tau = bio_model.nb_tau
    n_states = n_q + n_qdot
    n_root = bio_model.nb_root
    n_trans = 2  # 2D model

    variable_mappings = BiMappingList()
    variable_mappings.add("tau", to_second=[None, None, None, 0, 1, 2, 3], to_first=[3, 4, 5, 6])

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="tau", weight=1e3/2,
    #                         quadratic=True, phase=0)  # Do I really need this one? (expected_feedback_effort does it)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-10000, quadratic=False,
                            axes=Axis.Z, phase=0, expand=False)  # Temporary while in 1 phase ?
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.1, min_bound=0.1, max_bound=0.3, phase=0)

    multinode_objectives = MultinodeObjectiveList()
    if not DETERMINISTIC:
        multinode_objectives.add(expected_feedback_effort,
                                 nodes_phase=[0 for _ in range(n_shooting)],
                                 nodes=[i for i in range(n_shooting)],
                                 weight=1e3 / 2,
                                 quadratic=False,
                                 wS_magnitude=wS_magnitude,
                                 expand=False)

    # Constraints
    constraints = ConstraintList()
    if not DETERMINISTIC:
        constraints.add(states_equals_ref_kinematics, node=Node.ALL_SHOOTING, expand=False)

    constraints.add(CoM_over_ankle, node=Node.END, phase=0, expand=False)
    # constraints.add(
    #     ConstraintFcn.TRACK_CONTACT_FORCES,
    #     min_bound=5,
    #     max_bound=np.inf,
    #     node=Node.ALL_SHOOTING,
    #     contact_index=0,
    #     phase=0
    # )
    constraints.add(custom_contact_force_constraint, node=Node.ALL_SHOOTING, contact_index=1, min_bound=0.5,
                    max_bound=np.inf, phase=0, expand=False)

    multinode_constraints = MultinodeConstraintList()
    if not DETERMINISTIC:
        multinode_constraints.add(reach_standing_position_consistantly,
                                  nodes_phase=[0 for _ in range(n_shooting)],
                                  nodes=[i for i in range(n_shooting)],
                                  min_bound=np.array([-cas.inf, -cas.inf]),
                                  max_bound=np.array([0.004**2, 0.05**2]),
                                  expand=False)  # constrain only the CoM in Y (don't give a **** about CoM height)
        for i in range(n_shooting-1):
            multinode_constraints.add(leuven_trapezoidal,
                                      nodes_phase=[0, 0],
                                      nodes=[i, i+1],
                                      expand=False)
    else:
        for i in range(n_shooting - 1):
            multinode_constraints.add(leuven_trapezoidal_deterministic,
                                      nodes_phase=[0, 0],
                                      nodes=[i, i + 1])

    # Dynamics
    dynamics = DynamicsList()
    if not DETERMINISTIC:
        dynamics.add(configure_stochastic_optimal_control_problem,
                     dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, wM, wS,
                                             with_gains: stochastic_forward_dynamics(states, controls, parameters,
                                                                                 stochastic_variables, nlp, wM, wS,
                                                                                 with_gains=with_gains),
                     wM=np.zeros((2, 1)), wS=np.zeros((4, 1)), expand=False)
    else:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)


    pose_at_first_node = np.array([
        -0.19,
        -0.43,
        -1.01,
        0.0044735684524460015,
        2.5999996919248574,
        -2.299999479653955,
        0.6999990764981876,
    ])  # Initial position approx from Anais bioviz

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"].min[:, 0] = pose_at_first_node  # - 0.3
    x_bounds["q"].max[:, 0] = pose_at_first_node  # + 0.3
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"].min[:, 0] = 0
    x_bounds["qdot"].max[:, 0] = 0

    u_bounds = BoundsList()
    tau_min = np.ones((n_tau-n_root, 3)) * -1000
    tau_max = np.ones((n_tau-n_root, 3)) * 1000
    tau_min[:, 0] = 0
    tau_max[:, 0] = 0
    u_bounds.add("tau", min_bound=tau_min, max_bound=tau_max, phase=0)

    # Initial guesses
    q_init = np.zeros((n_q, n_shooting + 1))
    # q_init[2, :-1] = np.linspace(-1.0471, 0, n_shooting)
    # q_init[2, -1] = 0
    # q_init[3, :-1] = np.linspace(1.4861, 0, n_shooting)
    # q_init[3, -1] = 0

    qdot_init = np.ones((n_qdot, n_shooting + 1))

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", initial_guess=qdot_init, interpolation=InterpolationType.EACH_FRAME, phase=0)

    controls_init = np.ones((n_q-n_root, n_shooting))
    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=controls_init, interpolation=InterpolationType.EACH_FRAME, phase=0)

    n_k = (n_q-n_root) * (n_q-n_root + n_qdot-n_root)  # K(4x8)
    n_ee_ref = n_q-n_root + n_qdot-n_root  # ee_ref(4x1)
    n_m = n_states**2  # M(7x7)
    n_stochastic = n_k + n_ee_ref + n_m

    s_init = InitialGuessList()
    s_bounds = BoundsList()
    integrated_value_functions = None

    if not DETERMINISTIC:
        k_init = np.ones((n_k, n_shooting + 1)) * 0.01
        k_min = np.ones((n_k, n_shooting + 1)) * -500
        k_max = np.ones((n_k, n_shooting + 1)) * 500

        s_init.add("k", initial_guess=k_init, interpolation=InterpolationType.EACH_FRAME)
        s_bounds.add("k", min_bound=k_min, max_bound=k_max, interpolation=InterpolationType.EACH_FRAME)

        ee_ref_init = np.ones((n_ee_ref, n_shooting + 1)) * 0.01
        ee_ref_min = np.ones((n_ee_ref, n_shooting + 1)) * -500
        ee_ref_max = np.ones((n_ee_ref, n_shooting + 1)) * 500

        ee_ref_min[0, :] = -np.pi/8
        ee_ref_max[0, :] = np.pi/2-np.pi/8
        ee_ref_min[1, :] = -np.pi/2
        ee_ref_max[1, :] = np.pi/2
        ee_ref_min[2, :] = -10*np.pi
        ee_ref_max[2, :] = 10*np.pi
        ee_ref_min[3, :] = -10*np.pi
        ee_ref_max[3, :] = 10*np.pi

        s_init.add("ee_ref", initial_guess=ee_ref_init, interpolation=InterpolationType.EACH_FRAME)
        s_bounds.add("ee_ref", min_bound=ee_ref_min, max_bound=ee_ref_max, interpolation=InterpolationType.EACH_FRAME)

        m_init = np.ones((n_m, n_shooting + 1)) * 0.01
        m_min = np.ones((n_m, n_shooting + 1)) * -500
        m_max = np.ones((n_m, n_shooting + 1)) * 500

        s_init.add("m", initial_guess=m_init, interpolation=InterpolationType.EACH_FRAME)
        s_bounds.add("m", min_bound=m_min, max_bound=m_max, interpolation=InterpolationType.EACH_FRAME)

        integrated_value_functions = {"cov": lambda nlp, node_index: get_p_mat(nlp, node_index, wM_magnitude=wM_magnitude, wS_magnitude=wS_magnitude)}

    if not DETERMINISTIC:
        return StochasticOptimalControlProgram(
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
            problem_type=OcpType.SOCP_EXPLICIT(wM_magnitude, wS_magnitude),
            integrated_value_functions=integrated_value_functions,
        )
    else:
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
            # ode_solver=OdeSolver.RK8(n_integration_steps=10),
            ode_solver=None,
            skip_continuity=True,
            n_threads=1,
            assume_phase_dynamics=False,
            integrated_value_functions=integrated_value_functions,
        )

def main():

    RUN_OPTIM_FLAG = True  # False
    DETERMINISTIC = False #  True

    biorbd_model_path = "models/Model2D_7Dof_1C_3M.bioMod"
    n_q = 7
    n_root = 3

    if DETERMINISTIC:
        save_path = f"results/{biorbd_model_path[7:-7]}_torque_driven_1phase_ocp.pkl"
    else:
        save_path = f"results/{biorbd_model_path[7:-7]}_torque_driven_1phase_socp.pkl"

    # import bioviz
    # b = bioviz.Viz(biorbd_model_path)
    # b.exec()

    # --- Prepare the ocp --- #
    dt = 0.03
    final_time = 0.3
    n_shooting = int(final_time/dt)  # There is no U on the last node (I do not hack it here)
    final_time += dt

    # TODO: How do we choose the values?
    wM_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024

    # TODO: Add vestibular feedback
    wM_magnitude = cas.DM(np.array([wM_std ** 2 / dt for _ in range(n_q-n_root)]))  # All DoFs except root
    wPq_magnitude = cas.DM(np.array([wPq_std ** 2 / dt for _ in range(n_q-n_root)]))  # All DoFs except root
    wPqdot_magnitude = cas.DM(np.array([wPqdot_std ** 2 / dt for _ in range(n_q-n_root)]))  # All DoFs except root
    wS_magnitude = cas.vertcat(wPq_magnitude, wPqdot_magnitude)

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    # solver.set_linear_solver('mumps')
    solver.set_linear_solver('ma57')
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_maximum_iterations(10000)
    solver._nlp_scaling_method = "none"
    solver.set_hessian_approximation('limited-memory')
    # solver.set_nlp_scaling_method = "none"  # Does not work

    socp = prepare_socp(biorbd_model_path=biorbd_model_path,
                        final_time=final_time,
                        n_shooting=n_shooting,
                        wM_magnitude=wM_magnitude,
                        wS_magnitude=wS_magnitude,
                        DETERMINISTIC=DETERMINISTIC)

    if RUN_OPTIM_FLAG:
        sol_socp = socp.solve(solver)

        q_sol = sol_socp.states["q"]
        qdot_sol = sol_socp.states["qdot"]
        tau_sol = sol_socp.controls["tau"]
        data = {"q_sol": q_sol,
                "qdot_sol": qdot_sol,
                "tau_sol": tau_sol}

        # --- Save the results --- #
        with open(save_path, "wb") as file:
            pickle.dump(data, file)
    else:
        with open(save_path, "rb") as file:
            data = pickle.load(file)
        q_sol = data["q_sol"]
        qdot_sol = data["qdot_sol"]
        tau_sol = data["tau_sol"]
        k_sol = data["k_sol"]
        ee_ref_sol = data["ee_ref_sol"]
        m_sol = data["m_sol"]
        cov_sol = data["cov_sol"]
        stochastic_variables_sol = data["stochastic_variables_sol"]

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
    n_q = 5
    n_root = 3
    n_tau = n_q - n_root
    wM_sym = cas.MX.sym('wM', n_tau, 1)
    wS_sym = cas.MX.sym('wS', n_tau*2, 1)
    out = stochastic_forward_dynamics(states, controls, parameters, stochastic_variables, nlp, wM_sym, wS_sym, with_gains=True)
    dyn_fun = cas.Function("dyn_fun", [states, controls, parameters, stochastic_variables, wM_sym, wS_sym], [out.dxdt])

    fig, axs = plt.subplots(3, 2)
    n_simulations = 30
    q_simulated = np.zeros((n_simulations, n_q, n_shooting + 1))
    qdot_simulated = np.zeros((n_simulations, n_q, n_shooting + 1))
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

if __name__ == "__main__":
    main()

# TODO: Check expected feedback effort