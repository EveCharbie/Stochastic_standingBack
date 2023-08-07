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
    SocpType,
    CostType,
    VariableScalingList,
    ControlType,
)

def get_excitation_with_feedback(K, EE, ref, sensory_noise):
    return K @ ((EE - ref) + sensory_noise)

def stochastic_forward_dynamics(
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    stochastic_variables: cas.MX | cas.SX,
    nlp: NonLinearProgram,
    motor_noise,
    sensory_noise,
    with_gains,
) -> DynamicsEvaluation:

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    n_q = nlp.model.nb_q
    n_root = nlp.model.nb_root

    tau_fb = tau[n_root:]
    if with_gains:
        ref = DynamicsFunctions.get(nlp.stochastic_variables["ref"], stochastic_variables)
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        K_matrix = cas.MX((n_q-n_root) * 2, n_q-n_root)
        for s0 in range(2*(n_q-n_root)):
            for s1 in range(n_q-n_root):
                K_matrix[s0, s1] = k[s0*1 + s1]
        K_matrix = K_matrix.T

        ee = cas.vertcat(q[n_root:], qdot[n_root:])

        tau_fb += get_excitation_with_feedback(K_matrix, ee, ref, sensory_noise) + motor_noise

    friction = cas.MX.zeros(n_q, n_q)
    for i in range(n_root, n_q):
        friction[i, i] = 0.05

    tau_full = cas.vertcat(cas.MX.zeros(n_root), tau_fb)
    dqdot_computed = nlp.model.constrained_forward_dynamics(q, qdot, tau_full + friction @ qdot)

    return DynamicsEvaluation(dxdt=cas.vertcat(qdot, dqdot_computed))


def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram, motor_noise, sensory_noise, with_cholesky):

    n_q = ocp.nlp[0].model.nb_q
    n_root = ocp.nlp[0].model.nb_root
    nu = n_q - n_root

    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    ConfigureProblem.configure_tau(ocp, nlp, False, True)

    # Stochastic variables
    ConfigureProblem.configure_stochastic_k(ocp, nlp, n_noised_controls=nu, n_feedbacks=2*nu)  # Actuated states + vestibular eventually
    ConfigureProblem.configure_stochastic_ref(ocp, nlp, n_references=2*nu)  # Hip position & velocity + vestibular eventually
    ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=2*nu, n_constraints=3)
    if with_cholesky:
        ConfigureProblem.configure_stochastic_cholesky_cov(ocp, nlp, n_noised_states=2*nu)
    else:
        ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=2*nu)
    ConfigureProblem.configure_dynamics_function(ocp, nlp,
                                                 # dyn_func=nlp.dynamics_type.dynamic_function,
                                                 dyn_func=lambda states, controls, parameters, stochastic_variables,
                                                                 nlp, motor_noise,
                                                                 sensory_noise: nlp.dynamics_type.dynamic_function(
                                                     states, controls, parameters, stochastic_variables, nlp,
                                                     motor_noise, sensory_noise, with_gains=False
                                                 ),
                                                 motor_noise=motor_noise, sensory_noise=sensory_noise)
    ConfigureProblem.configure_stochastic_dynamics_function(
        ocp,
        nlp,
        noised_dyn_func=lambda states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise: nlp.dynamics_type.dynamic_function(
            states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise, with_gains=True
        ),
    )


def states_equals_ref_kinematics(controller: PenaltyController) -> cas.MX:
    n_root = controller.model.nb_root
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    ref = controller.stochastic_variables["ref"].cx_start
    ee = cas.vertcat(q[n_root:], qdot[n_root:])  # Non-root states
    return ee - ref

def pelvis_equals_pelvis_ref(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    ref = controller.stochastic_variables["ref"].cx_start
    ee = cas.vertcat(q[2], qdot[2])  # Pelvis rotation index
    return ee - ref

def reach_standing_position_consistantly(controller: PenaltyController) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    This is a multi-node constraint because the covariance matrix depends on all the precedent nodes, but it only
    applies at the END node.
    """
    n_q = controller.model.nb_q
    n_root = controller.model.nb_root
    n_joints = controller.model.nb_q - n_root
    Q_root = cas.MX.sym("q_root", n_root)
    Q_joints = cas.MX.sym("q_joints", n_joints)
    Qdot_root = cas.MX.sym("qdot_root", n_root)
    Qdot_joints = cas.MX.sym("qdot_joints", n_joints)
    n_states = n_q * 2

    if "cholesky_cov" in controller.stochastic_variables.keys():
        cov_sym = cas.MX.sym("cov", controller.stochastic_variables["cholesky_cov"].cx_start.shape[0])
        cov_sym_dict = {"cholesky_cov": cov_sym}
        cov_sym_dict["cholesky_cov"].cx_start = cov_sym
        l_cov_matrix = (
            controller
            .stochastic_variables["cholesky_cov"]
            .reshape_to_cholesky_matrix(
                cov_sym_dict,
                2 * n_joints,
                Node.START,
                "cholesky_cov",
            )
        )
        cov_matrix = l_cov_matrix @ l_cov_matrix.T
    else:
        cov_sym = cas.MX.sym("cov", controller.stochastic_variables.cx_start.shape[0])
        cov_sym_dict = {"cov": cov_sym}
        cov_sym_dict["cov"].cx_start = cov_sym
        cov_matrix = controller.stochastic_variables["cov"].reshape_to_matrix(cov_sym_dict, 2*n_joints, 2*n_joints, Node.START, "cov")

    CoM_pos = get_CoM(controller.model, cas.vertcat(Q_root, Q_joints))[1]
    CoM_vel = get_CoMdot(controller.model, cas.vertcat(Q_root, Q_joints), cas.vertcat(Qdot_root, Qdot_joints))[1]

    jac_CoM_q = cas.jacobian(CoM_pos, cas.vertcat(Q_joints))
    jac_CoM_qdot = cas.jacobian(CoM_vel, cas.vertcat(Q_joints, Qdot_joints))

    P_matrix_q = cov_matrix[:n_joints, :n_joints]
    P_matrix_qdot = cov_matrix[:, :]

    pos_constraint = jac_CoM_q @ P_matrix_q @ jac_CoM_q.T
    vel_constraint = jac_CoM_qdot @ P_matrix_qdot @ jac_CoM_qdot.T

    out = cas.vertcat(pos_constraint, vel_constraint)

    fun = cas.Function("reach_target_consistantly", [Q_root, Q_joints, Qdot_root, Qdot_joints, cov_sym], [out])
    val = fun(controller.states["q"].cx_start[:n_root],
              controller.states["q"].cx_start[n_root:n_root+n_joints],
              controller.states["qdot"].cx_start[:n_root],
              controller.states["qdot"].cx_start[n_root:n_root+n_joints],
              controller.stochastic_variables["cholesky_cov"].cx_start
                  if "cholesky_cov" in controller.stochastic_variables.keys()
                  else controller.stochastic_variables["cov"].cx_start
              )
    # Since the stochastic variables are defined with ns+1, the cx_start actually refers to the last node (when using node=Node.END)

    return val

def get_CoM(model, q):
    return model.center_of_mass(q)

def get_CoMdot(model, q, qdot):
    return model.center_of_mass_velocity(q, qdot)

def get_CoM_and_CoMdot(model, q, qdot):
    return cas.vertcat(get_CoM(model, q), get_CoMdot(model, q, qdot))

def expected_feedback_effort(controller: PenaltyController, sensory_noise_magnitude: cas.DM) -> cas.MX:
    """
    ...
    """

    n_q = controller.model.nb_q
    n_root = controller.model.nb_root
    nu = n_q - n_root
    n_stochastic = controller.stochastic_variables.shape


    stochastic_sym = cas.MX.sym("stochastic_sym", n_stochastic, 1)
    sensory_noise_matrix = sensory_noise_magnitude * cas.MX_eye(sensory_noise_magnitude.shape[0])

    stochastic_sym_dict = {
        key: stochastic_sym[controller.stochastic_variables[key].index]
        for key in controller.stochastic_variables.keys()
    }
    for key in controller.stochastic_variables.keys():
        stochastic_sym_dict[key].cx_start = stochastic_sym_dict[key]

    if "cholesky_cov" in stochastic_sym_dict.keys():
        l_cov_matrix = controller.stochastic_variables["cholesky_cov"].reshape_to_cholesky_matrix(
            stochastic_sym_dict,
            2*nu,
            Node.START,
            "cholesky_cov",
        )
        cov_matrix = l_cov_matrix @ l_cov_matrix.T
    else:
        cov_matrix = controller.stochastic_variables["cov"].reshape_to_matrix(
            stochastic_sym_dict,
            2*nu,
            2*nu,
            Node.START,
            "cov",
        )

    states_ref = stochastic_sym_dict["ref"].cx_start

    k = stochastic_sym_dict["k"].cx_start
    K_matrix = cas.MX(2*(n_q-n_root), n_q-n_root)
    for s0 in range(2*(n_q-n_root)):
        for s1 in range(n_q-n_root):
            K_matrix[s0, s1] = k[s0 * (n_q-n_root) + s1]
    K_matrix = K_matrix.T

    q_joints = cas.MX.sym("q_joints", nu, 1)
    qdot_joints = cas.MX.sym("qdot_joints", nu, 1)

    # Compute the expected effort
    trace_k_sensor_k = cas.trace(K_matrix @ sensory_noise_matrix @ K_matrix.T)
    # pelvis_pos_velo = cas.vertcat(controllers[0].states["q"].cx_start[2], controllers[0].states["qot"].cx_start[2])  # Pelvis pos + velo should be used as a feedback for all the joints eventually
    states_pos_velo = cas.vertcat(q_joints, qdot_joints)

    T_fb = K_matrix @ ((states_pos_velo - states_ref) + sensory_noise_magnitude)
    jac_T_fb_x = cas.jacobian(T_fb, cas.vertcat(q_joints, qdot_joints))
    trace_jac_p_jack = cas.trace(jac_T_fb_x @ cov_matrix @ jac_T_fb_x.T)
    expectedEffort_fb_mx = trace_jac_p_jack + trace_k_sensor_k
    func = cas.Function('f_expectedEffort_fb',
                                       [q_joints,
                                        qdot_joints,
                                        stochastic_sym],
                                       [expectedEffort_fb_mx])

    out = func(controller.states["q"].cx_start[n_root:],
               controller.states["qdot"].cx_start[n_root:],
               controller.stochastic_variables.cx_start)
    return out


def zero_acceleration(controller: PenaltyController, motor_noise: np.ndarray, sensory_noise: np.ndarray) -> cas.MX:
    dx = stochastic_forward_dynamics(controller.states.cx_start, controller.controls.cx_start,
                                     controller.parameters.cx_start, controller.stochastic_variables.cx_start,
                                     controller.get_nlp, motor_noise, sensory_noise, with_gains=False)
    return dx.dxdt[controller.states_dot.index("qddot")]

def CoM_over_ankle(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    CoM_pos = get_CoM(controller.model, q)
    CoM_pos_y = CoM_pos[1]
    marker_pos = controller.model.markers(q)[0]
    marker_pos_y = marker_pos[1]
    return marker_pos_y - CoM_pos_y


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
    motor_noise_magnitude: cas.DM,
    wPq_magnitude: cas.DM,
    wPqdot_magnitude: cas.DM,
    q_deterministic: np.ndarray,
    qdot_deterministic: np.ndarray,
    tau_deterministic: np.ndarray,
    with_cholesky: bool = False,
) -> StochasticOptimalControlProgram:
    """
    ...
    """

    sensory_noise_magnitude = cas.vertcat(wPq_magnitude, wPqdot_magnitude)

    bio_model = BiorbdModel(biorbd_model_path)

    n_q = bio_model.nb_q
    n_root = bio_model.nb_root
    nu = n_q - n_root

    variable_mappings = BiMappingList()
    variable_mappings.add("tau", to_second=[None, None, None, 0, 1, 2, 3], to_first=[3, 4, 5, 6])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-10000, quadratic=False,
                            axes=Axis.Z, phase=0)  # Temporary while in 1 phase ?
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-100, quadratic=False,
                            axes=Axis.Z, phase=0)  # Temporary while in 1 phase ? ### was not there before
    # objective_functions.add(reach_standing_position_consistantly,  ### was there before
    #                 custom_type=ObjectiveFcn.Mayer,
    #                 node=Node.PENULTIMATE,
    #                 weight=1e3,
    #                 phase=0,
    #                 quadratic=True)# constrain only the CoM in Y (don't give a **** about CoM height)
    # objective_functions.add(expected_feedback_effort,
    #                         custom_type=ObjectiveFcn.Lagrange,
    #                         node=Node.ALL_SHOOTING,
    #                         weight=10,
    #                         quadratic=True,
    #                         sensory_noise_magnitude=sensory_noise_magnitude,
    #                         phase=0)

    # Constraints
    constraints = ConstraintList()
    # constraints.add(states_equals_ref_kinematics, node=Node.ALL_SHOOTING) ### was there before
    # constraints.add(CoM_over_ankle, node=Node.END, phase=0)  ### was there before
    # constraints.add(custom_contact_force_constraint, node=Node.ALL_SHOOTING, contact_index=1, min_bound=0.1,
    #                 max_bound=np.inf, phase=0)
    # constraints.add(reach_standing_position_consistantly,
    #                           node=Node.PENULTIMATE,
    #                           min_bound=np.array([-cas.inf, -cas.inf]),
    #                           max_bound=np.array([0.004**2, 0.05**2]))  # constrain only the CoM in Y (don't give a **** about CoM height)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(configure_stochastic_optimal_control_problem,
                 dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise,
                                         with_gains: stochastic_forward_dynamics(states, controls, parameters,
                                                                             stochastic_variables, nlp, motor_noise, sensory_noise,
                                                                             with_gains=with_gains),
                 motor_noise=np.zeros((2, 1)), sensory_noise=np.zeros((4, 1)), with_cholesky=with_cholesky, expand=False)


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
    tau_min = np.ones((nu, 3)) * -500
    tau_max = np.ones((nu, 3)) * 500
    # tau_min[:, 0] = 0
    # tau_max[:, 0] = 0
    u_bounds.add("tau", min_bound=tau_min, max_bound=tau_max, phase=0)

    # Initial guesses
    q_init = q_deterministic
    qdot_init = qdot_deterministic

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", initial_guess=qdot_init, interpolation=InterpolationType.EACH_FRAME, phase=0)

    controls_init = tau_deterministic[:, :-1]
    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=controls_init, interpolation=InterpolationType.EACH_FRAME, phase=0)

    n_k = nu * (nu + nu)  # K(4x8)
    n_ref = nu + nu  # ref(8x1)
    n_m = (2*nu)**2 * 3  # M(8x8x3)
    n_stochastic = n_k + n_ref + n_m
    if not with_cholesky:
        n_cov = (2 * nu) ** 2  # Cov(8x8)
        n_stochastic += n_cov  # + cov(4, 4)
        n_cholesky_cov = 0
    else:
        n_cov = 0
        n_cholesky_cov = 0
        for i in range(nu):
            for j in range(i + 1):
                n_cholesky_cov += 1
        n_stochastic += n_cholesky_cov  # + cholesky_cov(10)

    s_init = InitialGuessList()
    s_bounds = BoundsList()

    k_init = np.ones((n_k, n_shooting + 1)) * 0.01
    k_min = np.ones((n_k, n_shooting + 1)) * -500
    k_max = np.ones((n_k, n_shooting + 1)) * 500

    s_init.add("k", initial_guess=k_init, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("k", min_bound=k_min, max_bound=k_max, interpolation=InterpolationType.EACH_FRAME)

    ref_init = np.ones((n_ref, n_shooting + 1)) * 0.01
    ref_min = np.ones((n_ref, n_shooting + 1)) * -500
    ref_max = np.ones((n_ref, n_shooting + 1)) * 500

    ref_min[0, :] = -np.pi/8
    ref_max[0, :] = np.pi/2-np.pi/8
    ref_min[1, :] = -np.pi/2
    ref_max[1, :] = np.pi/2
    ref_min[2, :] = -10*np.pi
    ref_max[2, :] = 10*np.pi
    ref_min[3, :] = -10*np.pi
    ref_max[3, :] = 10*np.pi

    s_init.add("ref", initial_guess=ref_init, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("ref", min_bound=ref_min, max_bound=ref_max, interpolation=InterpolationType.EACH_FRAME)

    m_init = np.ones((n_m, n_shooting + 1)) * 0.01
    m_min = np.ones((n_m, n_shooting + 1)) * -50
    m_max = np.ones((n_m, n_shooting + 1)) * 50

    s_init.add("m", initial_guess=m_init, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("m", min_bound=m_min, max_bound=m_max, interpolation=InterpolationType.EACH_FRAME)

    if with_cholesky:
        cov_init = np.ones((n_cholesky_cov, n_shooting + 1)) * 0.01
        cov_min = np.ones((n_cholesky_cov, n_shooting + 1)) * -500
        cov_max = np.ones((n_cholesky_cov, n_shooting + 1)) * 500
        P_0 = cas.DM_eye(2*nu) * np.hstack((np.ones((nu, )) * 1e-4, np.ones((nu, )) * 1e-7))  # P
        idx = 0
        for i in range(nu):
            for j in range(nu):
                cov_init[idx, 0] = P_0[i, j]
                cov_min[idx, 0] = P_0[i, j]
                cov_max[idx, 0] = P_0[i, j]
        s_init.add(
            "cholesky_cov",
            initial_guess=cov_init,
            interpolation=InterpolationType.EACH_FRAME,
        )
        s_bounds.add(
            "cholesky_cov",
            min_bound=cov_min,
            max_bound=cov_max,
            interpolation=InterpolationType.EACH_FRAME,
        )
    else:
        cov_init = np.ones((n_cov, n_shooting + 1)) * 0.01
        cov_min = np.ones((n_cov, n_shooting + 1)) * -500
        cov_max = np.ones((n_cov, n_shooting + 1)) * 500
        P_0 = cas.DM_eye(2*nu) * np.hstack((np.ones((nu, )) * 1e-4, np.ones((nu, )) * 1e-7))  # P
        cov_vector = np.zeros((n_cov, ))
        for i in range(nu):
            for j in range(nu):
                cov_vector[i*nu+j] = P_0[i, j]
        cov_min[:, 0] = cov_vector
        cov_max[:, 0] = cov_vector

        s_init.add("cov", initial_guess=cov_init, interpolation=InterpolationType.EACH_FRAME)
        s_bounds.add("cov", min_bound=cov_min, max_bound=cov_max, interpolation=InterpolationType.EACH_FRAME)

    # # Vaiables scaling
    # u_scaling = VariableScalingList()
    # u_scaling["tau"] = [10] * nu
    #
    # s_scaling = VariableScalingList()
    # s_scaling["k"] = [100] * n_k
    # s_scaling["ref"] = [1] * n_ref
    # s_scaling["m"] = [1] * n_m
    # if not with_cholesky:
    #     s_scaling["cholesky_cov"] = [0.01] * n_cholesky_cov # should be 0.01 for q, and 0.05 for qdot
    # else:
    #     s_scaling["cov"] = [0.01] * n_cov

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
        # u_scaling=u_scaling,
        # s_scaling=s_scaling,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=variable_mappings,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, method="legendre"),
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        n_threads=1,
        assume_phase_dynamics=False,
        problem_type=SocpType.SOCP_COLLOCATION(motor_noise_magnitude, sensory_noise_magnitude),
    )

def main():

    RUN_OPTIM_FLAG = True  # False
    with_cholesky = False

    biorbd_model_path = "models/Model2D_7Dof_1C_3M.bioMod"
    n_q = 7
    n_root = 3

    save_path = f"results/{biorbd_model_path[7:-7]}_torque_driven_1phase_socp_implicit.pkl"

    # import bioviz
    # b = bioviz.Viz(biorbd_model_path)
    # b.exec()

    # Load the deterministic solution to warm-start
    path_to_results = "/home/charbie/Documents/Programmation/Stochastic_standingBack/results/Model2D_7Dof_1C_3M_torque_driven_1phase_ocp.pkl"
    with open(path_to_results, 'rb') as file:
        data = pickle.load(file)
        q_deterministic = data['q_sol']
        qdot_deterministic = data['qdot_sol']
        tau_deterministic = data['tau_sol']

    # --- Prepare the ocp --- #
    dt = 0.01
    final_time = 0.5
    n_shooting = int(final_time/dt)  # There is no U on the last node (I do not hack it here)

    # TODO: How do we choose the values?
    motor_noise_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024

    # TODO: Add vestibular feedback
    motor_noise_magnitude = cas.DM(np.array([motor_noise_std ** 2 / dt for _ in range(n_q-n_root)]))  # All DoFs except root
    wPq_magnitude = cas.DM(np.array([wPq_std ** 2 / dt for _ in range(n_q-n_root)]))  # All DoFs except root
    wPqdot_magnitude = cas.DM(np.array([wPqdot_std ** 2 / dt for _ in range(n_q-n_root)]))  # All DoFs except root

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    # solver.set_linear_solver('mumps')
    solver.set_linear_solver('ma57')
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_maximum_iterations(0) # 1000
    solver.set_hessian_approximation('limited-memory')  # Mandatory, otherwise RAM explodes!
    solver._nlp_scaling_method = "none"

    socp = prepare_socp(biorbd_model_path=biorbd_model_path,
                        final_time=final_time,
                        n_shooting=n_shooting,
                        motor_noise_magnitude=motor_noise_magnitude,
                        wPq_magnitude=wPq_magnitude,
                        wPqdot_magnitude=wPqdot_magnitude,
                        q_deterministic=q_deterministic,
                        qdot_deterministic=qdot_deterministic,
                        tau_deterministic=tau_deterministic,
                        with_cholesky=with_cholesky)
    # socp.add_plot_penalty(CostType.ALL)
    # socp.check_conditioning()

    if RUN_OPTIM_FLAG:
        sol_socp = socp.solve(solver)
        sol_socp.graphs()

        q_sol = sol_socp.states["q"]
        qdot_sol = sol_socp.states["qdot"]
        tau_sol = sol_socp.controls["tau"]
        k_sol = sol_socp.stochastic_variables["k"]
        ref_sol = sol_socp.stochastic_variables["ref"]
        m_sol = sol_socp.stochastic_variables["m"]
        cov_sol = sol_socp.stochastic_variables["cov"]
        a_sol = sol_socp.stochastic_variables["a"]
        c_sol = sol_socp.stochastic_variables["c"]
        data = {"q_sol": q_sol,
                "qdot_sol": qdot_sol,
                "tau_sol": tau_sol,
                "k_sol": k_sol,
                "ref_sol": ref_sol,
                "m_sol": m_sol,
                "cov_sol": cov_sol,
                "a_sol": a_sol,
                "c_sol": c_sol}

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
        ref_sol = data["ref_sol"]
        m_sol = data["m_sol"]
        cov_sol = data["cov_sol"]
        stochastic_variables_sol = data["stochastic_variables_sol"]

    import bioviz
    b = bioviz.Viz(model_path=biorbd_model_path)
    b.load_movement(q_sol[:, :-1])
    b.exec()

if __name__ == "__main__":
    main()

# TODO: Check expected feedback effort