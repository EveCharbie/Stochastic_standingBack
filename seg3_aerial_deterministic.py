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

from utils import CoM_over_toes

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


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver,
) -> OptimalControlProgram:
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_q = biorbd_model.nbQ()
    n_root = biorbd_model.nbRoot()
    n_joints = n_q - n_root
    friction_coefficients = cas.DM.zeros(n_joints, n_joints)
    for i in range(n_joints):
        friction_coefficients[i, i] = 0.1

    bio_model = BiorbdModel(biorbd_model_path, friction_coefficients=friction_coefficients)

    n_q = bio_model.nb_q
    n_root = bio_model.nb_root

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau_joints", node=Node.ALL_SHOOTING, weight=0.01, quadratic=True
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, min_bound=0.1, max_bound=1)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_MARKERS, marker_index=2, axes=Axis.Z, node=Node.END)
    constraints.add(CoM_over_toes, node=Node.END)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_FREE_FLOATING_BASE)

    pose_at_first_node = np.array(
        [-0.0346, 0.1207, 0.2255, 0.0, 3.1, -0.1787, 0.0]
    )  # Initial position approx from bioviz
    pose_at_last_node = np.array(
        [-0.0346, 0.1207, 5.8292, -0.1801, 0.5377, 0.8506, -0.6856]
    )  # Final position approx from bioviz

    x_bounds = BoundsList()
    q_roots_min = bio_model.bounds_from_ranges("q").min[:n_root, :]
    q_roots_max = bio_model.bounds_from_ranges("q").max[:n_root, :]
    q_joints_min = bio_model.bounds_from_ranges("q").min[n_root:, :]
    q_joints_max = bio_model.bounds_from_ranges("q").max[n_root:, :]
    qdot_roots_min = bio_model.bounds_from_ranges("qdot").min[:n_root, :]
    qdot_roots_max = bio_model.bounds_from_ranges("qdot").max[:n_root, :]
    qdot_joints_min = bio_model.bounds_from_ranges("qdot").min[n_root:, :]
    qdot_joints_max = bio_model.bounds_from_ranges("qdot").max[n_root:, :]

    q_roots_min[:, 0] = pose_at_first_node[:n_root]
    q_roots_max[:, 0] = pose_at_first_node[:n_root]
    q_joints_min[:, 0] = pose_at_first_node[n_root:]
    q_joints_max[:, 0] = pose_at_first_node[n_root:]
    q_roots_min[2, 2] = 5.8292 - 0.5
    q_roots_max[2, 2] = 5.8292 + 0.5
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
    x_init.add(
        "q_roots",
        initial_guess=np.vstack((pose_at_first_node, pose_at_last_node)).T[:n_root, :],
        interpolation=InterpolationType.LINEAR,
    )
    x_init.add(
        "q_joints",
        initial_guess=np.vstack((pose_at_first_node, pose_at_last_node)).T[n_root:, :],
        interpolation=InterpolationType.LINEAR,
    )
    x_init.add("qdot_roots", initial_guess=[0.01] * n_root, interpolation=InterpolationType.CONSTANT)
    x_init.add("qdot_joints", initial_guess=[0.01] * (n_q - n_root), interpolation=InterpolationType.CONSTANT)

    u_init = InitialGuessList()
    u_init.add("tau_joints", initial_guess=[0.01] * (n_q - n_root), interpolation=InterpolationType.CONSTANT)

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
        ode_solver=ode_solver,
        n_threads=1,
    )


if __name__ == "__main__":
    model_name = "Model2D_6Dof_0C_3M"
    biorbd_model_path = f"models/{model_name}.bioMod"
    biorbd_model_path_with_mesh = f"models/{model_name}_with_mesh.bioMod"
    save_path = f"results/{model_name}_aerial_ocp.pkl"

    dt = 0.05
    final_time = 0.8
    n_shooting = int(final_time / dt)
    ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3)

    ocp = prepare_ocp(biorbd_model_path, final_time, n_shooting, ode_solver)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma97")
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)

    q_roots_sol = sol.states["q_roots"]
    q_joints_sol = sol.states["q_joints"]
    qdot_roots_sol = sol.states["qdot_roots"]
    qdot_joints_sol = sol.states["qdot_joints"]
    tau_sol = sol.controls["tau_joints"]
    time_sol = sol.parameters["time"][0][0]
    data = {
        "q_roots": q_roots_sol,
        "q_joints": q_joints_sol,
        "qdot_roots": qdot_roots_sol,
        "qdot_joints": qdot_joints_sol,
        "tau": tau_sol,
        "time": time_sol,
    }

    if sol.status != 0:
        save_path = save_path.replace(".pkl", "_DVG.pkl")
    else:
        save_path = save_path.replace(".pkl", "_CVG.pkl")

    with open(save_path, "wb") as file:
        pickle.dump(data, file)

    import bioviz

    b = bioviz.Viz(biorbd_model_path_with_mesh)
    b.load_movement(q_sol)
    b.exec()
