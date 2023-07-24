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
)

def get_CoM(model, q):
    return model.center_of_mass(q)

def CoM_over_ankle(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    CoM_pos = get_CoM(controller.model, q)
    CoM_pos_y = CoM_pos[1]
    marker_pos = controller.model.markers(q)[0]
    marker_pos_y = marker_pos[1]
    return marker_pos_y - CoM_pos_y

def Custom_track_markers(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    markers = controller.model.markers(q)
    markers_pos = markers[0]
    return markers_pos[0] ** 2 + markers_pos[1] ** 2 + markers_pos[2] ** 2

def leuven_trapezoidal_deterministic(controllers: list[PenaltyController]) -> cas.MX:

    dt = controllers[0].tf / controllers[0].ns

    dX_i = controllers[0].dynamics(controllers[0].states.cx_start, controllers[0].controls.cx_start,
                                        controllers[0].parameters.cx_start, controllers[0].stochastic_variables.cx_start)
    dX_i_plus = controllers[1].dynamics(controllers[1].states.cx_start, controllers[1].controls.cx_start,
                                        controllers[1].parameters.cx_start, controllers[1].stochastic_variables.cx_start)

    out = controllers[1].states.cx_start - (controllers[0].states.cx_start + (dX_i + dX_i_plus) / 2 * dt)

    return out * 1e3

def prepare_ocp(
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
                            axes=Axis.Z, phase=0)  # Temporary while in 1 phase ?
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-100, quadratic=False,
                            axes=Axis.Z, phase=0)  # Temporary while in 1 phase ?
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.1, max_bound=0.3, phase=0)

    # Constraints
    constraints = ConstraintList()
    # constraints.add(CoM_over_ankle, node=Node.PENULTIMATE, phase=0)
    # constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.START, index=2, min_bound=np.array([0, 0, 0]),
    #                 max_bound=np.array([0, 0, 0]), phase=0)
    # constraints.add(Custom_track_markers, node=Node.START, index=0, min_bound=0,
    #                 max_bound=0, phase=0)
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.1,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=0,
        phase=0
    )

    multinode_constraints = MultinodeConstraintList()
    for i in range(n_shooting - 1):
        multinode_constraints.add(leuven_trapezoidal_deterministic,
                                  nodes_phase=[0, 0],
                                  nodes=[i, i + 1])

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=0)

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
    x_bounds["q"].min[:, 0] = pose_at_first_node - 0.05
    x_bounds["q"].max[:, 0] = pose_at_first_node + 0.05
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"].min[:, 0] = 0
    x_bounds["qdot"].max[:, 0] = 0

    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[-1000] * (n_tau-n_root), max_bound=[1000] * (n_tau-n_root), phase=0)

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

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        variable_mappings=variable_mappings,
        # ode_solver=OdeSolver.RK8(n_integration_steps=10),
        ode_solver=None,
        skip_continuity=True,
        n_threads=1,
        assume_phase_dynamics=False,
    )

def main():

    biorbd_model_path = "models/Model2D_7Dof_1C_3M.bioMod"
    save_path = f"results/{biorbd_model_path[8:-7]}_torque_driven_1phase_ocp.pkl"

    # import bioviz
    # b = bioviz.Viz(biorbd_model_path)
    # b.exec()

    # --- Prepare the ocp --- #
    dt = 0.01
    final_time = 0.3
    n_shooting = int(final_time/dt) + 1
    final_time += dt

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    # # solver.set_linear_solver('mumps')
    # solver.set_linear_solver('ma57')
    # solver.set_tol(1e-3)
    # solver.set_dual_inf_tol(3e-4)
    # solver.set_constr_viol_tol(1e-7)
    # solver.set_maximum_iterations(10000)
    solver.set_hessian_approximation('limited-memory')

    ocp = prepare_ocp(biorbd_model_path=biorbd_model_path,
                        final_time=final_time,
                        n_shooting=n_shooting)

    sol_ocp = ocp.solve(solver)

    q_sol = sol_ocp.states["q"]
    qdot_sol = sol_ocp.states["qdot"]
    tau_sol = sol_ocp.controls["tau"]
    data = {"q_sol": q_sol,
            "qdot_sol": qdot_sol,
            "tau_sol": tau_sol}

    # --- Save the results --- #
    with open(save_path, "wb") as file:
        pickle.dump(data, file)

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
