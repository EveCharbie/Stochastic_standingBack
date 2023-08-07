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
    ControlType,
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
    #                         quadratic=True, phase=0)
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
        contact_index=1,
        phase=0
    )

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

    controls_init = np.ones((n_q-n_root, n_shooting+1))
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
        variable_mappings=variable_mappings,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, method="legendre"),
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        n_threads=1,
        assume_phase_dynamics=False,
    )

def main():

    biorbd_model_path = "models/Model2D_7Dof_1C_3M.bioMod"
    save_path = f"results/{biorbd_model_path[7:-7]}_torque_driven_1phase_ocp.pkl"

    # import bioviz
    # b = bioviz.Viz(biorbd_model_path)
    # b.exec()

    # --- Prepare the ocp --- #
    dt = 0.01
    final_time = 0.5
    n_shooting = int(final_time/dt)

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

if __name__ == "__main__":
    main()

