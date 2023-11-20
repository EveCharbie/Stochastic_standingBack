
import numpy as np
import casadi as cas
import biorbd_casadi as biorbd

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

def CoM_over_toes(controller: PenaltyController) -> cas.MX:
    q_roots = controller.states["q_roots"].cx_start
    q_joints = controller.states["q_joints"].cx_start
    q = cas.vertcat(q_roots, q_joints)
    CoM_pos = controller.model.center_of_mass(q)
    CoM_pos_y = CoM_pos[1]
    marker_pos = controller.model.markers(q)[4]
    marker_pos_y = marker_pos[1]
    return marker_pos_y - CoM_pos_y


def get_ref_init(q_roots_last, q_joints_last, qdot_roots_last, qdot_joints_last, polynomial_degree):
    ref_last = cas.vertcat(q_roots_last[2, :].reshape(1, 65), q_joints_last, qdot_roots_last[2, :].reshape(1, 65), qdot_joints_last)
    ref_last = ref_last[:, 0::(polynomial_degree+1)]
    return ref_last


def get_excitation_with_feedback(K, EE, ref, sensory_noise):
    return K @ ((EE - ref) + sensory_noise)

def stochastic_forward_dynamics_numerical(states, controls, stochastic_variables, model, motor_noise, sensory_noise, with_gains):

    friction = model.friction_coefficients

    model_path = model.path
    model_biorbd = biorbd.Model(model_path)

    n_q = model_biorbd.nbQ()
    n_root = model_biorbd.nbRoot()
    n_joints = n_q-n_root
    n_k = n_joints*2*(n_joints+1)
    n_ref = 2*(1+n_joints)

    q = states[:n_q]
    qdot = states[n_q:]
    tau_fb = controls[:]

    if with_gains:
        k = stochastic_variables[:n_k]
        ref = stochastic_variables[n_k: n_k + n_ref]
        K_matrix = StochasticBioModel.reshape_to_matrix(k, model.matrix_shape_k)
        ee = cas.vertcat(q[2:], qdot[2:])

        tau_fb += get_excitation_with_feedback(K_matrix, ee, ref, sensory_noise) + motor_noise

    tau = tau_fb + friction @ qdot[n_root:]
    tau = cas.vertcat(cas.MX.zeros(n_root, 1), tau)

    dqdot_computed = model_biorbd.ForwardDynamics(q, qdot, tau).to_mx()

    return cas.vertcat(qdot, dqdot_computed)


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


def get_m_init(model,
               n_joints,
               n_stochastic,
               n_shooting,
               duration,
               polynomial_degree,
               q_roots_last,
               q_joints_last,
               qdot_roots_last,
               qdot_joints_last,
               tau_joints_last,
               k_last,
               ref_last,
               motor_noise_magnitude,
               sensory_noise_magnitude):
    """
    M = -dF_dz @ inv(dG_dz)
    """

    n_q = model.model.nbQ()
    nb_root = model.model.nbRoot()
    n_joints = n_q - nb_root

    x_q_root = cas.MX.sym("x_q_root", nb_root, 1)
    x_q_joints = cas.MX.sym("x_q_joints", n_joints, 1)
    x_qdot_root = cas.MX.sym("x_qdot_root", nb_root, 1)
    x_qdot_joints = cas.MX.sym("x_qdot_joints", n_joints, 1)
    z_q_root = cas.MX.sym("z_q_root", nb_root, polynomial_degree)
    z_q_joints = cas.MX.sym("z_q_joints", n_joints, polynomial_degree)
    z_qdot_root = cas.MX.sym("z_qdot_root", nb_root, polynomial_degree)
    z_qdot_joints = cas.MX.sym("z_qdot_joints", n_joints, polynomial_degree)
    controls_sym = cas.MX.sym("controls", n_joints, 1)
    stochastic_variables_sym = cas.MX.sym("stochastic_variables", n_stochastic, 1)

    states_full = cas.vertcat(
        cas.horzcat(x_q_root, z_q_root),
        cas.horzcat(x_q_joints, z_q_joints),
        cas.horzcat(x_qdot_root, z_qdot_root),
        cas.horzcat(x_qdot_joints, z_qdot_joints),
    )

    states_end, defects = integrator(model, polynomial_degree, n_shooting, duration, states_full, controls_sym, stochastic_variables_sym, motor_noise_magnitude, sensory_noise_magnitude)
    initial_polynomial_evaluation = cas.vertcat(x_q_root, x_q_joints, x_qdot_root, x_qdot_joints)
    defects = cas.vertcat(initial_polynomial_evaluation, defects)

    df_dz = cas.horzcat(
        cas.jacobian(states_end, x_q_root),
        cas.jacobian(states_end, x_q_joints),
        cas.jacobian(states_end, z_q_root),
        cas.jacobian(states_end, z_q_joints),
        cas.jacobian(states_end, x_qdot_root),
        cas.jacobian(states_end, x_qdot_joints),
        cas.jacobian(states_end, z_qdot_root),
        cas.jacobian(states_end, z_qdot_joints),
    )

    dg_dz = cas.horzcat(
        cas.jacobian(defects, x_q_root),
        cas.jacobian(defects, x_q_joints),
        cas.jacobian(defects, z_q_root),
        cas.jacobian(defects, z_q_joints),
        cas.jacobian(defects, x_qdot_root),
        cas.jacobian(defects, x_qdot_joints),
        cas.jacobian(defects, z_qdot_root),
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


    m_last = np.zeros((2 * n_q * 2 * n_q * (polynomial_degree+1), n_shooting + 1))
    for i in range(n_shooting):
        index_this_time = [i * polynomial_degree + j for j in range(4)]
        df_dz_evaluated = df_dz_fun(
            q_roots_last[:, index_this_time[0]],
            q_joints_last[:, index_this_time[0]],
            qdot_roots_last[:, index_this_time[0]],
            qdot_joints_last[:, index_this_time[0]],
            q_roots_last[:, index_this_time[1:]],
            q_joints_last[:, index_this_time[1:]],
            qdot_roots_last[:, index_this_time[1:]],
            qdot_joints_last[:, index_this_time[1:]],
            tau_joints_last[:, i],
            np.vstack((k_last[:, i].reshape((-1, 1)),
                       ref_last[:, i].reshape((-1, 1)),
                       np.zeros((2 * n_q * 2 * n_q * (polynomial_degree+1), 1)),  # M
                       np.zeros((2 * n_q * 2 * n_q, 1)))),  # cov
        )
        dg_dz_evaluated = dg_dz_fun(
            q_roots_last[:, index_this_time[0]],
            q_joints_last[:, index_this_time[0]],
            qdot_roots_last[:, index_this_time[0]],
            qdot_joints_last[:, index_this_time[0]],
            q_roots_last[:, index_this_time[1:]],
            q_joints_last[:, index_this_time[1:]],
            qdot_roots_last[:, index_this_time[1:]],
            qdot_joints_last[:, index_this_time[1:]],
            tau_joints_last[:, i],
            np.vstack((k_last[:, i].reshape((-1, 1)),
                       ref_last[:, i].reshape((-1, 1)),
                       np.zeros((2 * n_q * 2 * n_q * (polynomial_degree+1), 1)),
                       np.zeros((2 * n_q * 2 * n_q, 1)))),
        )

        m_this_time = -df_dz_evaluated @ np.linalg.inv(dg_dz_evaluated)
        shape_0 = m_this_time.shape[0]
        shape_1 = m_this_time.shape[1]
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                m_last[shape_0 * s1 + s0, i] = m_this_time[s0, s1]

    m_last[:, -1] = m_last[:, -2]
    return m_last

def get_cov_init(model,
                 polynomial_degree,
                 n_shooting,
                 n_stochastic,
                 duration,
                 q_roots_last,
                 q_joints_last,
                 qdot_roots_last,
                 qdot_joints_last,
                 tau_joints_last,
                 k_last,
                 ref_last,
                 m_last,
                 cov_init,
                 motor_noise_magnitude,
                 sensory_noise_magnitude):
    """
    P_k+1 = M_k @ (dG_dx @ P_k @ dG_dx.T + dG_dw @ sigma_w @ dG_dw.T) @ M_k.T
    """

    n_q = model.model.nbQ()
    nb_root = model.model.nbRoot()
    n_joints = n_q - nb_root

    x_q_root = cas.MX.sym("x_q_roots", nb_root, 1)
    x_q_joints = cas.MX.sym("x_q_joints", n_joints, 1)
    x_qdot_root = cas.MX.sym("x_qdot_roots", nb_root, 1)
    x_qdot_joints = cas.MX.sym("x_qdot_joints", n_joints, 1)
    z_q_root = cas.MX.sym("z_q_roots", nb_root, polynomial_degree)
    z_q_joints = cas.MX.sym("z_q_joints", n_joints, polynomial_degree)
    z_qdot_root = cas.MX.sym("z_qdot_roots", nb_root, polynomial_degree)
    z_qdot_joints = cas.MX.sym("z_qdot_joints", n_joints, polynomial_degree)
    controls_sym = cas.MX.sym("controls", n_joints, 1)
    stochastic_variables_sym = cas.MX.sym("stochastic_variables", n_stochastic, 1)
    motor_noise_sym = cas.MX.sym("motor_noise", n_joints, 1)
    sensory_noise_sym = cas.MX.sym("sensory_noise", 2*(n_joints+1), 1)

    states_full = cas.vertcat(
        cas.horzcat(x_q_root, z_q_root),
        cas.horzcat(x_q_joints, z_q_joints),
        cas.horzcat(x_qdot_root, z_qdot_root),
        cas.horzcat(x_qdot_joints, z_qdot_joints),
    )

    states_end, defects = integrator(model, polynomial_degree, n_shooting, duration, states_full, controls_sym,
                                     stochastic_variables_sym, motor_noise_sym, sensory_noise_sym)
    initial_polynomial_evaluation = cas.vertcat(x_q_root, x_q_joints, x_qdot_root, x_qdot_joints)
    defects = cas.vertcat(initial_polynomial_evaluation, defects)

    dg_dw = cas.horzcat(
        cas.jacobian(defects, sensory_noise_sym),
        cas.jacobian(defects, motor_noise_sym),
    )

    dg_dx = cas.jacobian(defects, cas.vertcat(x_q_root, x_q_joints, x_qdot_root, x_qdot_joints))

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

    sigma_w_dm = cas.vertcat(sensory_noise_magnitude, motor_noise_magnitude) * cas.DM_eye(
        cas.vertcat(sensory_noise_magnitude, motor_noise_magnitude).shape[0])
    cov_last = np.zeros((2 * n_q * 2 * n_q, n_shooting + 1))
    cov_last[:, 0] = cov_init[:, 0]
    for i in range(n_shooting):
        index_this_time = [i * polynomial_degree + j for j in range(4)]
        dg_dx_evaluated = dg_dx_fun(
            q_roots_last[:, index_this_time[0]],
            q_joints_last[:, index_this_time[0]],
            qdot_roots_last[:, index_this_time[0]],
            qdot_joints_last[:, index_this_time[0]],
            q_roots_last[:, index_this_time[1:]],
            q_joints_last[:, index_this_time[1:]],
            qdot_roots_last[:, index_this_time[1:]],
            qdot_joints_last[:, index_this_time[1:]],
            tau_joints_last[:, i],
            np.vstack((k_last[:, i].reshape((-1, 1)),
                       ref_last[:, i].reshape((-1, 1)),
                       m_last[:, i].reshape((-1, 1)),
                       np.zeros((2 * n_q * 2 * n_q, 1)))),  # cov
            motor_noise_magnitude,
            sensory_noise_magnitude,
        )
        dg_dw_evaluated = dg_dw_fun(
            q_roots_last[:, index_this_time[0]],
            q_joints_last[:, index_this_time[0]],
            qdot_roots_last[:, index_this_time[0]],
            qdot_joints_last[:, index_this_time[0]],
            q_roots_last[:, index_this_time[1:]],
            q_joints_last[:, index_this_time[1:]],
            qdot_roots_last[:, index_this_time[1:]],
            qdot_joints_last[:, index_this_time[1:]],
            tau_joints_last[:, i],
            np.vstack((k_last[:, i].reshape((-1, 1)),
                       ref_last[:, i].reshape((-1, 1)),
                       m_last[:, i].reshape((-1, 1)),
                       np.zeros((2 * n_q * 2 * n_q, 1)))),
            motor_noise_magnitude,
            sensory_noise_magnitude,
        )

        m_matrix = np.zeros((2*n_q, 2*n_q*(polynomial_degree+1)))
        for s0 in range(2*n_q*(polynomial_degree+1)):
            for s1 in range(2*n_q):
                m_matrix[s1, s0] = m_last[s0 * 2*n_q + s1, i]

        cov_matrix = np.zeros((2*n_q, 2*n_q))
        for s0 in range(2*n_q):
            for s1 in range(2*n_q):
                m_matrix[s1, s0] = cov_last[s0 * 2*n_q + s1, i]

        cov_this_time = (
                m_matrix @ (dg_dx_evaluated @ cov_matrix @ dg_dx_evaluated.T + dg_dw_evaluated @ sigma_w_dm @ dg_dw_evaluated.T) @ m_matrix.T)
        for s0 in range(2*n_q):
            for s1 in range(2*n_q):
                cov_last[2*n_q * s1 + s0, i+1] = cov_this_time[s0, s1]

    return cov_last

