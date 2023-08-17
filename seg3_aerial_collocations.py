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

def stochastic_forward_dynamics_numerical(states, controls, stochastic_variables, model, motor_noise, sensory_noise, with_gains):

    q = states[:model.nbQ()]
    qdot = states[model.nbQ():]
    tau_fb = controls

    n_q = model.nbQ()
    n_root = model.nbRoot()
    n_joints = n_q-n_root
    n_k = n_joints*2*n_joints
    n_ref = 2*n_joints

    if with_gains:
        ref = stochastic_variables[n_k: n_k + n_ref]
        k = stochastic_variables[:n_k]

        k_matrix = cas.MX(n_ref, n_joints)
        for s0 in range(n_ref):
            for s1 in range(n_joints):
                k_matrix[s0, s1] = k[s0 * n_joints + s1]
        K_matrix = k_matrix.T
        # ee = cas.vertcat(q[2], qdot[2])
        ee = cas.horzcat(q, qdot)

        tau_fb[n_root:] += get_excitation_with_feedback(K_matrix, ee, ref, sensory_noise) + motor_noise

    friction = cas.MX.zeros(n_q, n_q)
    for i in range(n_root, n_q):
        friction[i, i] = 0.1

    tau_full = cas.vertcat(cas.MX.zeros(n_root), tau_fb)
    dqdot_computed = model.ForwardDynamics(q, qdot, tau_full + friction @ qdot).to_mx()

    return cas.vertcat(qdot, dqdot_computed)

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
    n_joints = n_q-n_root

    tau_fb = tau[:]
    if with_gains:
        ref = DynamicsFunctions.get(nlp.stochastic_variables["ref"], stochastic_variables)
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        K_matrix = nlp.stochastic_variables["k"].reshape_sym_to_matrix(k, n_joints, 2)

        # ee = cas.vertcat(q[2], qdot[2])
        ee = cas.vertcat(q, qdot)

        tau_fb[n_root:] += get_excitation_with_feedback(K_matrix, ee, ref, sensory_noise) + motor_noise

    friction = cas.MX.zeros(n_q, n_q)
    for i in range(n_root, n_q):
        friction[i, i] = 0.1

    dqdot_computed = nlp.model.forward_dynamics(q, qdot, tau_fb + friction @ qdot)

    return DynamicsEvaluation(dxdt=cas.vertcat(qdot, dqdot_computed))


def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram, motor_noise, sensory_noise, with_cholesky):

    n_q = ocp.nlp[0].model.nb_q
    n_root = ocp.nlp[0].model.nb_root
    n_joints = n_q - n_root

    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    ConfigureProblem.configure_tau(ocp, nlp, False, True)

    # Stochastic variables
    n_ref = 2 * n_jonts
    ConfigureProblem.configure_stochastic_k(ocp, nlp, n_noised_controls=n_joints, n_feedbacks=2)  # Only vestibular
    ConfigureProblem.configure_stochastic_ref(ocp, nlp, n_references=n_ref)
    ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=2*n_joints, n_collocation_points=3+1)
    if with_cholesky:
        ConfigureProblem.configure_stochastic_cholesky_cov(ocp, nlp, n_noised_states=2*n_joints)
    else:
        ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=2*n_joints)
    ConfigureProblem.configure_dynamics_function(ocp, nlp,
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


def ref_equals_mean_values(controller: PenaltyController) -> cas.MX:
    """
    Returns the pelvis position and velocity sice there is no thoracic joint it is the same as semi-circular canal.
    """
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    ref = controller.stochastic_variables["ref"].cx_start
    # ee = cas.vertcat(q[2], qdot[2])
    ee = cas.vertcat(q, qdot)
    return ee - ref

def reach_landing_position_consistantly(controller: PenaltyController) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    This is a multi-node constraint because the covariance matrix depends on all the precedent nodes, but it only
    applies at the END node.
    """
    n_root = controller.model.nb_root
    n_joints = controller.model.nb_q - n_root
    Q_root = cas.MX.sym("q_root", n_root)
    Q_joints = cas.MX.sym("q_joints", n_joints)
    Qdot_root = cas.MX.sym("qdot_root", n_root)
    Qdot_joints = cas.MX.sym("qdot_joints", n_joints)

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

    # What should we use as a reference?
    CoM_pos = get_CoM(controller.model, cas.vertcat(Q_root, Q_joints))[:2]
    CoM_vel = get_CoMdot(controller.model, cas.vertcat(Q_root, Q_joints), cas.vertcat(Qdot_root, Qdot_joints))[:2]
    CoM_ang_vel = get_CoM_ang_velo(controller.model, cas.vertcat(Q_root, Q_joints), cas.vertcat(Qdot_root, Qdot_joints))[0]

    jac_CoM_q = cas.jacobian(CoM_pos, cas.vertcat(Q_joints))
    jac_CoM_qdot = cas.jacobian(CoM_vel, cas.vertcat(Q_joints, Qdot_joints))
    jac_CoM_ang_vel = cas.jacobian(CoM_ang_vel, cas.vertcat(Q_joints, Qdot_joints))

    P_matrix_q = cov_matrix[:n_joints, :n_joints]
    P_matrix_qdot = cov_matrix[:, :]

    pos_constraint = jac_CoM_q @ P_matrix_q @ jac_CoM_q.T
    vel_constraint = jac_CoM_qdot @ P_matrix_qdot @ jac_CoM_qdot.T
    rot_constraint = jac_CoM_ang_vel @ P_matrix_qdot @ jac_CoM_ang_vel.T

    out = cas.vertcat(pos_constraint[0, 0], pos_constraint[1, 1], vel_constraint[0, 0], vel_constraint[1, 1], rot_constraint[0, 0])

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

def get_CoM_ang_velo(model, q, qdot):
    return model.center_of_mass_rotation(q, qdot)

def get_CoM_and_CoMdot(model, q, qdot):
    return cas.vertcat(get_CoM(model, q), get_CoMdot(model, q, qdot))

def expected_feedback_effort(controller: PenaltyController, sensory_noise_magnitude: cas.DM) -> cas.MX:
    """
    ...
    """

    n_q = controller.model.nb_q
    n_root = controller.model.nb_root
    n_joints = n_q - n_root
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
            2*n_joints,
            Node.START,
            "cholesky_cov",
        )
        cov_matrix = l_cov_matrix @ l_cov_matrix.T
    else:
        cov_matrix = controller.stochastic_variables["cov"].reshape_to_matrix(
            stochastic_sym_dict,
            2*n_joints,
            2*n_joints,
            Node.START,
            "cov",
        )

    ref = stochastic_sym_dict["ref"].cx_start

    K_matrix = controller.stochastic_variables["k"].reshape_sym_to_matrix(stochastic_sym_dict["k"], n_joints, 2)

    q_joints = cas.MX.sym("q_joints", n_joints, 1)
    qdot_joints = cas.MX.sym("qdot_joints", n_joints, 1)

    # Compute the expected effort
    trace_k_sensor_k = cas.trace(K_matrix @ sensory_noise_matrix @ K_matrix.T)
    pelvis_pos_velo = cas.vertcat(controllers[0].states["q"].cx_start[2], controllers[0].states["qot"].cx_start[2])

    T_fb = K_matrix @ ((pelvis_pos_velo - ref) + sensory_noise_magnitude)
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

def get_ref_init(q_last, qdot_last):
    # ref_last = cas.horzcat(q_last[2, :], qdot_last[2, :])
    ref_last = cas.horzcat(q_last, qdot_last)
    ref_last = ref_last[:, 0::4]
    return ref_last

def integrator(model, polynomial_degree, n_shooting, duration, states, controls, stochastic_variables, motor_noise, sensory_noise):

    h = duration / n_shooting
    method = "legendre"

    # Coefficients of the collocation equation
    _c = cas.MX.zeros((polynomial_degree + 1, polynomial_degree + 1))

    # Coefficients of the continuity equation
    _d = cas.MX.zeros(polynomial_degree + 1)

    # Choose collocation points
    step_time = [0] + cas.collocation_points(polynomial_degree, method)

    # Dimensionless time inside one control interval
    time_control_interval = cas.MX.sym("time_control_interval")

    # For all collocation points
    for j in range(polynomial_degree + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        _l = 1
        for r in range(polynomial_degree + 1):
            if r != j:
                _l *= (time_control_interval - step_time[r]) / (step_time[j] - step_time[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        if method == "radau":
            _d[j] = 1 if j == polynomial_degree else 0
        else:
            lfcn = cas.Function("lfcn", [time_control_interval], [_l])
            _d[j] = lfcn(1.0)

        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        _l = 1
        for r in range(polynomial_degree + 1):
            if r != j:
                _l *= (time_control_interval - step_time[r]) / (step_time[j] - step_time[r])

        # Evaluate the time derivative of the polynomial at all collocation points to get
        # the coefficients of the continuity equation
        tfcn = cas.Function("tfcn", [time_control_interval], [cas.tangent(_l, time_control_interval)])
        for r in range(polynomial_degree + 1):
            _c[j, r] = tfcn(step_time[r])

    # Total number of variables for one finite element
    states_end = _d[0] * states[:, 0]
    defects = []
    for j in range(1, polynomial_degree + 1):
        # Expression for the state derivative at the collocation point
        xp_j = 0
        for r in range(polynomial_degree + 1):
            xp_j += _c[r, j] * states[:, r]

        f_j = stochastic_forward_dynamics_numerical(
            states=states[:, j],
            controls=controls,  # Piecewise constant control
            stochastic_variables=stochastic_variables,
            model=model,
            motor_noise=motor_noise,
            sensory_noise=sensory_noise,
            with_gains=True,
        )

        defects.append(h * f_j - xp_j)

        # Add contribution to the end state
        states_end += _d[j] * states[:, j]

    # Concatenate constraints
    defects = cas.vertcat(*defects)

    return states_end, defects


def get_m_init(model_path,
               n_joints,
               n_stochastic,
               n_shooting,
               duration,
               polynomial_degree,
               q_last,
               qdot_last,
               tau_last,
               k_last,
               ref_last,
               motor_noise_magnitude,
               sensory_noise_magnitude):
    """
    M = -dF_dz @ inv(dG_dz)
    """

    model = biorbd.Model(model_path)

    nb_root = model.nbRoot()
    nu = model.nbQ() - nb_root
    non_root_index_continuity = []
    non_root_index_defects = []
    for i in range(2):
        for j in range(polynomial_degree+1):
            non_root_index_defects += list(
                range(
                    (nb_root + nu) * (i * (polynomial_degree+1) + j) + nb_root,
                    (nb_root + nu) * (i * (polynomial_degree+1) + j) + nb_root + nu,
                )
            )
        non_root_index_continuity += list(
            range((nb_root + nu) * i + nb_root, (nb_root + nu) * i + nb_root + nu)
        )

    x_q_root = cas.MX.sym("x_q_root", nb_root, 1)
    x_q_joints = cas.MX.sym("x_q_joints", nu, 1)
    x_qdot_root = cas.MX.sym("x_qdot_root", nb_root, 1)
    x_qdot_joints = cas.MX.sym("x_qdot_joints", nu, 1)
    z_q_root = cas.MX.sym("z_q_root", nb_root, polynomial_degree)
    z_q_joints = cas.MX.sym("z_q_joints", nu, polynomial_degree)
    z_qdot_root = cas.MX.sym("z_qdot_root", nb_root, polynomial_degree)
    z_qdot_joints = cas.MX.sym("z_qdot_joints", nu, polynomial_degree)
    controls_sym = cas.MX.sym("controls", nu, 1)
    stochastic_variables_sym = cas.MX.sym("stochastic_variables", n_stochastic, 1)

    states_full = cas.vertcat(
        cas.horzcat(x_q_root, z_q_root),
        cas.horzcat(x_q_joints, z_q_joints),
        cas.horzcat(x_qdot_root, z_qdot_root),
        cas.horzcat(x_qdot_joints, z_qdot_joints),
    )

    states_end, defects = integrator(model, polynomial_degree, n_shooting, duration, states_full, controls_sym, stochastic_variables_sym, motor_noise_magnitude, sensory_noise_magnitude)
    initial_polynomial_evaluation = cas.vertcat(x_q_root, x_q_joints, x_qdot_root, x_qdot_joints)
    defects = cas.vertcat(initial_polynomial_evaluation, defects)[non_root_index_defects]
    states_end = states_end[non_root_index_continuity]

    df_dz = cas.horzcat(
        cas.jacobian(states_end, x_q_joints),
        cas.jacobian(states_end, z_q_joints),
        cas.jacobian(states_end, x_qdot_joints),
        cas.jacobian(states_end, z_qdot_joints),
    )

    dg_dz = cas.horzcat(
        cas.jacobian(defects, x_q_joints),
        cas.jacobian(defects, z_q_joints),
        cas.jacobian(defects, x_qdot_joints),
        cas.jacobian(defects, z_qdot_joints),
    )

    df_dz_fun = cas.Function(
        "df_dz",
        [
            x_q_root,
            x_q_joints,
            x_qdot_root,
            x_qdot_joints,
            z_q_root,
            z_q_joints,
            z_qdot_root,
            z_qdot_joints,
            controls_sym,
            stochastic_variables_sym,
        ],
        [df_dz],
    )
    dg_dz_fun = cas.Function(
        "dg_dz",
        [
            x_q_root,
            x_q_joints,
            x_qdot_root,
            x_qdot_joints,
            z_q_root,
            z_q_joints,
            z_qdot_root,
            z_qdot_joints,
            controls_sym,
            stochastic_variables_sym,
        ],
        [dg_dz],
    )


    m_last = np.zeros((2 * n_joints * 2 * n_joints * (polynomial_degree+1), n_shooting + 1))
    for i in range(n_shooting):
        index_this_time = [i * polynomial_degree + j for j in range(4)]
        df_dz_evaluated = df_dz_fun(
            q_last[:nb_root, index_this_time[0]],
            q_last[nb_root: nb_root + nu, index_this_time[0]],
            qdot_last[:nb_root, index_this_time[0]],
            qdot_last[nb_root: nb_root + nu, index_this_time[0]],
            q_last[:nb_root, index_this_time[1:]],
            q_last[nb_root: nb_root + nu, index_this_time[1:]],
            qdot_last[:nb_root, index_this_time[1:]],
            qdot_last[nb_root: nb_root + nu, index_this_time[1:]],
            tau_last[:, i],
            np.vstack((k_last[:, i].reshape((-1, 1)),
                       ref_last[:, i].reshape((-1, 1)),
                       np.zeros((2 * n_joints * 2 * n_joints * (polynomial_degree+1), 1)),  # M
                       np.zeros((2 * n_joints * 2 * n_joints, 1)))),  # cov
        )
        dg_dz_evaluated = dg_dz_fun(
            q_last[:nb_root, index_this_time[0]],
            q_last[nb_root: nb_root + nu, index_this_time[0]],
            qdot_last[:nb_root, index_this_time[0]],
            qdot_last[nb_root: nb_root + nu, index_this_time[0]],
            q_last[:nb_root, index_this_time[1:]],
            q_last[nb_root: nb_root + nu, index_this_time[1:]],
            qdot_last[:nb_root, index_this_time[1:]],
            qdot_last[nb_root: nb_root + nu, index_this_time[1:]],
            tau_last[:, i],
            np.vstack((k_last[:, i].reshape((-1, 1)),
                       ref_last[:, i].reshape((-1, 1)),
                       np.zeros((2 * n_joints * 2 * n_joints * (polynomial_degree+1), 1)),
                       np.zeros((2 * n_joints * 2 * n_joints, 1)))),
        )

        m_this_time = -df_dz_evaluated @ np.linalg.inv(dg_dz_evaluated)
        shape_0 = m_this_time.shape[0]
        shape_1 = m_this_time.shape[1]
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                m_last[shape_0 * s1 + s0, i] = m_this_time[s0, s1]

    m_last[:, -1] = m_last[:, -2]
    return m_last

def get_cov_init(model_path,
                 n_shooting,
                 polynomial_degree,
                 duration,
                 q_last,
                 qdot_last,
                 tau_last,
                 k_last,
                 ref_last,
                 m_last,
                 cov_init,
                 motor_noise_magnitude,
                 sensory_noise_magnitude):
    """
    P_k+1 = M_k @ (dG_dx @ P_k @ dG_dx.T + dG_dw @ sigma_w @ dG_dw.T) @ M_k.T
    """

    model = biorbd.Model(model_path)

    nb_root = model.nbRoot()
    nu = model.nbQ() - nb_root
    non_root_index_continuity = []
    non_root_index_defects = []
    for i in range(2):
        for j in range(polynomial_degree+1):
            non_root_index_defects += list(
                range(
                    (nb_root + nu) * (i * (polynomial_degree+1) + j) + nb_root,
                    (nb_root + nu) * (i * (polynomial_degree+1) + j) + nb_root + nu,
                )
            )
        non_root_index_continuity += list(
            range((nb_root + nu) * i + nb_root, (nb_root + nu) * i + nb_root + nu)
        )

    x_q_root = cas.MX.sym("x_q_root", nb_root, 1)
    x_q_joints = cas.MX.sym("x_q_joints", nu, 1)
    x_qdot_root = cas.MX.sym("x_qdot_root", nb_root, 1)
    x_qdot_joints = cas.MX.sym("x_qdot_joints", nu, 1)
    z_q_root = cas.MX.sym("z_q_root", nb_root, polynomial_degree)
    z_q_joints = cas.MX.sym("z_q_joints", nu, polynomial_degree)
    z_qdot_root = cas.MX.sym("z_qdot_root", nb_root, polynomial_degree)
    z_qdot_joints = cas.MX.sym("z_qdot_joints", nu, polynomial_degree)
    controls_sym = cas.MX.sym("controls", nu, 1)
    stochastic_variables_sym = cas.MX.sym("stochastic_variables", n_stochastic, 1)
    motor_noise_sym = cas.MX.sym("motor_noise", nu, 1)
    sensory_noise_sym = cas.MX.sym("sensory_noise", 2*nu, 1)

    states_full = cas.vertcat(
        cas.horzcat(x_q_root, z_q_root),
        cas.horzcat(x_q_joints, z_q_joints),
        cas.horzcat(x_qdot_root, z_qdot_root),
        cas.horzcat(x_qdot_joints, z_qdot_joints),
    )

    sigma_w = cas.vertcat(sensory_noise, motor_noise) * cas.MX_eye(cas.vertcat(sensory_noise, motor_noise).shape[0])

    states_end, defects = integrator(model, polynomial_degree, n_shooting, duration, states_full, controls_sym,
                                     stochastic_variables_sym, motor_noise_sym, sensory_noise_sym)
    initial_polynomial_evaluation = cas.vertcat(x_q_root, x_q_joints, x_qdot_root, x_qdot_joints)
    defects = cas.vertcat(initial_polynomial_evaluation, defects)[non_root_index_defects]

    dg_dx = cas.horzcat(
        cas.jacobian(defects, x_q_joints),
        cas.jacobian(defects, x_qdot_joints),
    )

    dg_dw = cas.horzcat(
        cas.jacobian(defects, sensory_noise),
        cas.jacobian(defects, motor_noise),
    )

    dg_dx_fun = cas.Function(
        "dg_dx",
        [
            x_q_root,
            x_q_joints,
            x_qdot_root,
            x_qdot_joints,
            z_q_root,
            z_q_joints,
            z_qdot_root,
            z_qdot_joints,
            controls_sym,
            stochastic_variables_sym,
            motor_noise_sym,
            sensory_noise_sym,
        ],
        [dg_dx],
    )
    dg_dw_fun = cas.Function(
        "dg_dw",
        [
            x_q_root,
            x_q_joints,
            x_qdot_root,
            x_qdot_joints,
            z_q_root,
            z_q_joints,
            z_qdot_root,
            z_qdot_joints,
            controls_sym,
            stochastic_variables_sym,
            motor_noise_sym,
            sensory_noise_sym,
        ],
        [dg_dw],
    )


    cov_last = np.zeros((2 * n_joints * 2 * n_joints, n_shooting + 1))
    cov_last[:, 0] = cov_init
    for i in range(n_shooting):
        index_this_time = [i * polynomial_degree + j for j in range(4)]
        dg_dx_evaluated = dg_dx_fun(
            q_last[:nb_root, index_this_time[0]],
            q_last[nb_root: nb_root + nu, index_this_time[0]],
            qdot_last[:nb_root, index_this_time[0]],
            qdot_last[nb_root: nb_root + nu, index_this_time[0]],
            q_last[:nb_root, index_this_time[1:]],
            q_last[nb_root: nb_root + nu, index_this_time[1:]],
            qdot_last[:nb_root, index_this_time[1:]],
            qdot_last[nb_root: nb_root + nu, index_this_time[1:]],
            tau_last[:, i],
            np.vstack((k_last[:, i].reshape((-1, 1)),
                       ref_last[:, i].reshape((-1, 1)),
                       m_last[:, i].reshape((-1, 1)),
                       np.zeros((2 * n_joints * 2 * n_joints, 1)))),  # cov
            motor_noise_magnitude,
            sensory_noise_magnitude,
        )
        dg_dw_evaluated = dg_dw_fun(
            q_last[:nb_root, index_this_time[0]],
            q_last[nb_root: nb_root + nu, index_this_time[0]],
            qdot_last[:nb_root, index_this_time[0]],
            qdot_last[nb_root: nb_root + nu, index_this_time[0]],
            q_last[:nb_root, index_this_time[1:]],
            q_last[nb_root: nb_root + nu, index_this_time[1:]],
            qdot_last[:nb_root, index_this_time[1:]],
            qdot_last[nb_root: nb_root + nu, index_this_time[1:]],
            tau_last[:, i],
            np.vstack((k_last[:, i].reshape((-1, 1)),
                       ref_last[:, i].reshape((-1, 1)),
                       m_last[:, i].reshape((-1, 1)),
                       np.zeros((2 * n_joints * 2 * n_joints, 1)))),
            motor_noise_magnitude,
            sensory_noise_magnitude,
        )

        m_matrix = MX(2*n_joints, 2*n_joints*(polynomial_degree+1))
        for s0 in range(2*n_joints*(polynomial_degree+1)):
            for s1 in range(2*n_joints):
                m_matrix[s1, s0] = m_last[s0 * 2*n_joints + s1, i]

        cov_matrix = MX(2*n_joints, 2*n_joints)
        for s0 in range(2*n_joints):
            for s1 in range(2*n_joints):
                m_matrix[s1, s0] = cov_last[s0 * 2*n_joints + s1, i]

        cov_this_time = m_matrix @ (dg_dx_evaluated @ cov_matrix @ dg_dx_evaluated.T + dg_dw_evaluated @ sigma_w @ dg_dw_evaluated.T) @ m_matrix.T
        shape_0 = m_this_time.shape[0]
        shape_1 = m_this_time.shape[1]
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                cov_last[shape_0 * s1 + s0, i+1] = cov_this_time[s0, s1]

    return cov_last

def prepare_socp(
    biorbd_model_path: str,
    time_last: float,
    n_shooting: int,
    motor_noise_magnitude: cas.DM,
    sensory_noise_magnitude: cas.DM,
    q_last: np.ndarray,
    qdot_last: np.ndarray,
    tau_last: np.ndarray,
    k_last: np.ndarray = None,
    ref_last: np.ndarray = None,
    m_last: np.ndarray = None,
    cov_last: np.ndarray = None,
    with_cholesky: bool = False,
) -> StochasticOptimalControlProgram:
    """
    ...
    """

    bio_model = BiorbdModel(biorbd_model_path)
    polynomial_degree = 3

    n_q = bio_model.nb_q
    n_root = bio_model.nb_root
    n_joints = n_q - n_root

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="tau", weight=0.01,
                            quadratic=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, min_bound=0.1, max_bound=1)
    if np.sum(sensory_noise_magnitude) == 0:
        objective_functions.add(ObjectiveFcn.Lagrange.STOCHASTIC_MINIMIZE_VARIABLE, key="k", weight=0.01, quadratic=True)

    # objective_functions.add(reach_landing_position_consistantly,
    #                 custom_type=ObjectiveFcn.Mayer,
    #                 node=Node.END,
    #                 weight=1e3,
    #                 quadratic=True)
    # objective_functions.add(expected_feedback_effort,
    #                         custom_type=ObjectiveFcn.Lagrange,
    #                         node=Node.ALL_SHOOTING,
    #                         weight=10,
    #                         quadratic=True,
    #                         sensory_noise_magnitude=sensory_noise_magnitude)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_MARKERS, marker_index=2, axes=Axis.Z, node=Node.END)
    constraints.add(CoM_over_ankle, node=Node.END)
    constraints.add(ConstraintFcn.TRACK_CONTROL, key="tau", index=[0, 1, 2], node=Node.ALL)
    constraints.add(ref_equals_mean_values, node=Node.ALL)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(configure_stochastic_optimal_control_problem,
                 dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise,
                                         with_gains: stochastic_forward_dynamics(states, controls, parameters,
                                                                             stochastic_variables, nlp, motor_noise, sensory_noise,
                                                                             with_gains=with_gains),
                 motor_noise=np.zeros((2, 1)), sensory_noise=np.zeros((4, 1)), with_cholesky=with_cholesky, expand=False)


    pose_at_first_node = np.array([-0.0422, 0.0892, 0.2386, 0.0, -0.1878, 0.0])  # Initial position approx from bioviz

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"].min[:, 0] = pose_at_first_node
    x_bounds["q"].max[:, 0] = pose_at_first_node
    x_bounds["q"].min[2, 2] = 2*np.pi - 0.2
    x_bounds["q"].max[2, 2] = 2*np.pi + 0.2
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"].min[:, 0] = 0
    x_bounds["qdot"].max[:, 0] = 0
    x_bounds["qdot"].min[1, 0] = 2
    x_bounds["qdot"].max[1, 0] = 2
    x_bounds["qdot"].min[2, 0] = 2.5 * np.pi
    x_bounds["qdot"].max[2, 0] = 2.5 * np.pi

    u_bounds = BoundsList()
    tau_min = np.ones((n_q, 3)) * -500
    tau_max = np.ones((n_q, 3)) * 500
    tau_min[:, 0] = 0
    tau_max[:, 0] = 0
    u_bounds.add("tau", min_bound=tau_min, max_bound=tau_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    # Initial guesses
    x_init = InitialGuessList()
    x_init.add("q", initial_guess=q_last, interpolation=InterpolationType.ALL_POINTS)
    x_init.add("qdot", initial_guess=qdot_last, interpolation=InterpolationType.ALL_POINTS)

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=tau_last, interpolation=InterpolationType.ALL_POINTS)

    n_k = n_joints * 2 * n_joints  # K(3x6)
    # n_ref = 2  # ref(2)
    n_ref = 2*n_joints  # ref(6)
    n_m = (2*n_joints)**2 * 4  # M(6x6x3)
    n_stochastic = n_k + n_ref + n_m
    if not with_cholesky:
        n_cov = (2 * n_joints) ** 2  # Cov(6x6)
        n_stochastic += n_cov
        n_cholesky_cov = 0
    else:
        n_cov = 0
        n_cholesky_cov = 0
        for i in range(n_joints):
            for j in range(i + 1):
                n_cholesky_cov += 1
        n_stochastic += n_cholesky_cov  # + cholesky_cov(21)

    s_init = InitialGuessList()
    s_bounds = BoundsList()

    if k_last is None:
        k_last = np.ones((n_k, n_shooting + 1)) * 0.01
    s_init.add("k", initial_guess=k_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("k", min_bound=[-500]*n_k, max_bound=[500]*n_k, interpolation=InterpolationType.CONSTANT)

    ref_min = x_bounds["q"].min
    ref_max = x_bounds["q"].max

    if ref_last is None:
        ref_last = get_ref_init(q_last, qdot_last)
    s_init.add("ref", initial_guess=ref_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("ref", min_bound=ref_min, max_bound=ref_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    if m_last is None:
        m_last = get_m_init(biorbd_model_path, n_joints, n_stochastic, n_shooting, time_last, polynomial_degree, q_last, qdot_last, tau_last, k_last, ref_last, motor_noise_magnitude, sensory_noise_magnitude)
        # m_last = np.ones((n_m, n_shooting + 1)) * 0.01
    s_init.add("m", initial_guess=m_last, interpolation=InterpolationType.EACH_FRAME)
    s_bounds.add("m", min_bound=[-50]*n_m, max_bound=[50]*n_m, interpolation=InterpolationType.CONSTANT)

    if with_cholesky:
        # cov_init = get_cov_init()
        cov_min = np.ones((n_cholesky_cov, 3)) * -500
        cov_max = np.ones((n_cholesky_cov, 3)) * 500
        P_0 = cas.DM_eye(2*n_joints) * np.hstack((np.ones((n_joints, )) * 1e-4, np.ones((n_joints, )) * 1e-7))  # P
        idx = 0
        cov_init = np.zeros((n_cholesky_cov, 1))
        for i in range(2*n_joints):
            for j in range(2*n_joints):
                cov_init[idx, 0] = P_0[i, j]
                cov_min[idx, 0] = P_0[i, j]
                cov_max[idx, 0] = P_0[i, j]
                idx += 1
        if cov_last is None:
            cov_last = np.repeat(cov_init, n_shooting + 1, axis=1)

        s_init.add(
            "cholesky_cov",
            initial_guess=cov_last,
            interpolation=InterpolationType.EACH_FRAME,
        )
        s_bounds.add(
            "cholesky_cov",
            min_bound=cov_min,
            max_bound=cov_max,
            interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        )
    else:
        P_0 = cas.DM_eye(2 * n_joints) * np.hstack((np.ones((n_joints,)) * 1e-4, np.ones((n_joints,)) * 1e-7))  # P
        if cov_last is None:
            cov_last = get_cov_init(model_path,
                 n_shooting,
                 polynomial_degree,
                 duration,
                 q_last,
                 qdot_last,
                 tau_last,
                 k_last,
                 ref_last,
                 m_last,
                 P_0,
                 motor_noise_magnitude,
                 sensory_noise_magnitude)
        cov_min = np.ones((n_cov, 3)) * -500
        cov_max = np.ones((n_cov, 3)) * 500
        cov_vector = np.zeros((n_cov, 1))
        for i in range(2*n_joints):
            for j in range(2*n_joints):
                cov_vector[i*n_joints+j] = P_0[i, j]
        if cov_last is None:
            cov_last = np.repeat(cov_vector, n_shooting + 1, axis=1)
        cov_min[:, 0] = cov_vector.reshape((-1, ))
        cov_max[:, 0] = cov_vector.reshape((-1, ))

        s_init.add("cov", initial_guess=cov_last, interpolation=InterpolationType.EACH_FRAME)
        s_bounds.add("cov", min_bound=cov_min, max_bound=cov_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    # # Vaiables scaling
    # u_scaling = VariableScalingList()
    # u_scaling["tau"] = [1/10] * n_joints
    #
    # s_scaling = VariableScalingList()
    # s_scaling["k"] = [1/100] * n_k
    # s_scaling["ref"] = [1] * n_ref
    # s_scaling["m"] = [1] * n_m
    # if not with_cholesky:
    #     s_scaling["cholesky_cov"] = [1/0.01] * n_cholesky_cov # should be 0.01 for q, and 0.05 for qdot
    # else:
    #     s_scaling["cov"] = [1/0.01] * n_cov

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
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, method="legendre"),
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        n_threads=1,
        assume_phase_dynamics=False,
        problem_type=SocpType.SOCP_COLLOCATION(motor_noise_magnitude, sensory_noise_magnitude),
    )

