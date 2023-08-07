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

# def leuven_trapezoidal_deterministic(controllers: list[PenaltyController]) -> cas.MX:
#
#     dt = controllers[0].tf / controllers[0].ns
#
#     dX_i = controllers[0].dynamics(controllers[0].states.cx_start, controllers[0].controls.cx_start,
#                                         controllers[0].parameters.cx_start, controllers[0].stochastic_variables.cx_start)
#     dX_i_plus = controllers[1].dynamics(controllers[1].states.cx_start, controllers[1].controls.cx_start,
#                                         controllers[1].parameters.cx_start, controllers[1].stochastic_variables.cx_start)
#
#     out = controllers[1].states.cx_start - (controllers[0].states.cx_start + (dX_i + dX_i_plus) / 2 * dt)
#
#     return out * 1e3

def casadi_mean(elements: cas.MX) -> cas.MX:
    return cas.sum1(elements) / elements.shape[0]

def casadi_std_squared(elements: cas.MX) -> cas.MX:
    mean = casadi_mean(elements)
    return cas.sum1((elements - mean) ** 2) / elements.shape[0]

def try_to_reach_standing_position_consistantly(controllers: list[PenaltyController], motor_noise_magnitude: cas.DM, nb_random: int) -> cas.MX:

    nu = controllers[0].controls.shape
    nq = controllers[0].states['q'].shape

    CoM_positions = cas.MX.zeros(nb_random)
    CoM_velocities = cas.MX.zeros(nb_random)
    pelvis_rots = cas.MX.zeros(nb_random)
    pelvis_velocities = cas.MX.zeros(nb_random)
    noise = np.random.normal(loc=0, scale=motor_noise_magnitude, size=(nu, nb_random, controllers[0].ns))
    for j in range(nb_random):
        states_integrated = controllers[0].states.cx_start
        for i, ctrl in enumerate(controllers[:-1]):
            controls = ctrl.controls.cx_start + noise[:, j, i]
            new_states = ctrl._nlp.dynamics[i](states_integrated[:, -1],
                                       controls,
                                       ctrl.parameters.cx_start,
                                       ctrl.stochastic_variables.cx_start)[0]  # select "xf"
            states_integrated = cas.horzcat(states_integrated,
                                            new_states)
        CoM_positions[j] = controllers[-1].model.center_of_mass(states_integrated[:nq, -1])[1]
        CoM_velocities[j] = controllers[-1].model.center_of_mass_velocity(states_integrated[:nq, -1], states_integrated[nq:, -1])[1]
        pelvis_rots[j] = states_integrated[2, -1]
        pelvis_velocities[j] = states_integrated[nq+2, -1]
    std_standing_position = casadi_std_squared(CoM_positions) + casadi_std_squared(CoM_velocities) + casadi_std_squared(pelvis_rots) + casadi_std_squared(pelvis_velocities)

    return std_standing_position

def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    motor_noise_magnitude: float,
    weight: int,
    nb_random: int
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
    n_root = bio_model.nb_root

    variable_mappings = BiMappingList()
    variable_mappings.add("tau", to_second=[None, None, None, 0, 1, 2, 3], to_first=[3, 4, 5, 6])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-10000, quadratic=False,
                            axes=Axis.Z, phase=0)  # Temporary while in 1 phase ?
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-100, quadratic=False,
                            axes=Axis.Z, phase=0)  # Temporary while in 1 phase ?
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="tau", weight=0.01,
    #                         quadratic=True, phase=0)  # Minimize efforts (instead of expected efforts)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.1,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=0
    )

    # multinode_constraints = MultinodeConstraintList()
    # for i in range(n_shooting - 1):
    #     multinode_constraints.add(leuven_trapezoidal_deterministic,
    #                               nodes_phase=[0, 0],
    #                               nodes=[i, i + 1])

    multinode_objectives = MultinodeObjectiveList()
    multinode_objectives.add(try_to_reach_standing_position_consistantly,
                            nodes_phase=[0 for _ in range(n_shooting)],
                            nodes=[i for i in range(n_shooting)],
                            motor_noise_magnitude=motor_noise_magnitude,
                            nb_random=nb_random,
                            phase=0, weight=weight, quadratic=True)  # objective only on the CoM and CoMdot in Y (don't give a **** about CoM height)

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

    pose_at_last_node = np.array([
        -0.1136,
        0.2089,
        0.0545,
        2.931,
        0.0932,
        -0.0433,
        -0.7
    ])

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"].min[:, 0] = pose_at_first_node - 0.05
    x_bounds["q"].max[:, 0] = pose_at_first_node + 0.05
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"].min[:, 0] = 0
    x_bounds["qdot"].max[:, 0] = 0

    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[-500] * (n_tau-n_root), max_bound=[500] * (n_tau-n_root), phase=0)

    # Initial guesses
    q_init = np.zeros((n_q, n_shooting + 1))
    for dof in range(n_q):
        q_init[dof, :] = np.linspace(pose_at_first_node[dof], pose_at_last_node[dof], n_shooting + 1)
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
        # multinode_constraints=multinode_constraints,
        multinode_objectives=multinode_objectives,
        variable_mappings=variable_mappings,
        # ode_solver=None,
        # skip_continuity=True,
        n_threads=1,
        assume_phase_dynamics=False,
    )

def main():

	# TODO: Try to add a local (MHE style) objective which would prevent divergence at each node with a lighter weight.
    biorbd_model_path = "models/Model2D_7Dof_1C_3M.bioMod"
    motor_noise_magnitude = 2.5
    weight = 10
    nb_random = 30

    save_path = f"results/{biorbd_model_path[7:-7]}_torque_driven_1phase_simulated_noise{motor_noise_magnitude}_weight{weight}_random{nb_random}.pkl"

    # import bioviz
    # b = bioviz.Viz(biorbd_model_path)
    # b.exec()

    # --- Prepare the ocp --- #
    dt = 0.01
    final_time = 0.5
    n_shooting = int(final_time/dt) + 1
    final_time += dt

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    # # solver.set_linear_solver('mumps')
    solver.set_linear_solver('ma57')
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_maximum_iterations(10000)
    solver.set_hessian_approximation('limited-memory')

    ocp = prepare_ocp(biorbd_model_path=biorbd_model_path,
                      final_time=final_time,
                      n_shooting=n_shooting,
                      motor_noise_magnitude=motor_noise_magnitude,
                      weight=weight,
                      nb_random=nb_random
                      )

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

if __name__ == "__main__":
    main()
