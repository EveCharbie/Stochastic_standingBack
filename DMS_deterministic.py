"""
...
"""

import sys

import biorbd_casadi as biorbd
import casadi as cas
import numpy as np

from utils import CoM_over_toes

sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    OptimalControlProgram,
    InitialGuessList,
    ObjectiveFcn,
    BiorbdModel,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InterpolationType,
    Node,
    ConstraintList,
    ConstraintFcn,
    DynamicsFcn,
    Axis,
    OdeSolver,
)


def prepare_ocp(
    biorbd_model_path: str,
    time_last: float,
    n_shooting: int,
    q_roots_last: np.ndarray = None,
    q_joints_last: np.ndarray = None,
    qdot_roots_last: np.ndarray = None,
    qdot_joints_last: np.ndarray = None,
    tau_joints_last: np.ndarray = None,
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
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau_joints", node=Node.ALL_SHOOTING, weight=0.01, quadratic=True, derivative=True,
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, min_bound=0.1, max_bound=1)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_MARKERS, marker_index="Foot_Toe", axes=Axis.Z, node=Node.END)
    constraints.add(CoM_over_toes, node=Node.END)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_FREE_FLOATING_BASE, with_friction=True)

    pose_at_first_node = np.array(
        [-0.0346, 0.1207, 0.2255, 0.0, 3.1, -0.1787, 0.0]
    )  # Initial position approx from bioviz
    pose_at_last_node = np.array(
        [-0.0346, 0.1207, 5.8292, -0.1801, 0.5377, 0.8506, -0.6856]
    )  # Final position approx from bioviz

    x_bounds = BoundsList()
    q_roots_min = bio_model.bounds_from_ranges("q_roots").min
    q_roots_max = bio_model.bounds_from_ranges("q_roots").max
    q_joints_min = bio_model.bounds_from_ranges("q_joints").min
    q_joints_max = bio_model.bounds_from_ranges("q_joints").max
    qdot_roots_min = bio_model.bounds_from_ranges("qdot_roots").min
    qdot_roots_max = bio_model.bounds_from_ranges("qdot_roots").max
    qdot_joints_min = bio_model.bounds_from_ranges("qdot_joints").min
    qdot_joints_max = bio_model.bounds_from_ranges("qdot_joints").max

    q_roots_min[:, 0] = pose_at_first_node[:n_root]
    q_roots_max[:, 0] = pose_at_first_node[:n_root]
    q_joints_min[:, 0] = pose_at_first_node[n_root:]
    q_joints_max[:, 0] = pose_at_first_node[n_root:]
    q_roots_min[2, 2] = pose_at_last_node[2] - 0.5
    q_roots_max[2, 2] = pose_at_last_node[2] + 0.5
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
    if q_roots_last is None:
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
    else:
        x_init.add(
            "q_roots",
            initial_guess=q_roots_last,
            interpolation=InterpolationType.ALL_POINTS,
        )
        x_init.add(
            "q_joints",
            initial_guess=q_joints_last,
            interpolation=InterpolationType.ALL_POINTS,
        )
        x_init.add("qdot_roots", initial_guess=qdot_roots_last, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("qdot_joints", initial_guess=qdot_joints_last, interpolation=InterpolationType.ALL_POINTS)

    u_init = InitialGuessList()
    if tau_joints_last is None:
        u_init.add("tau_joints", initial_guess=[0.01] * (n_q - n_root), interpolation=InterpolationType.CONSTANT)
    else:
        u_init.add(
            "tau_joints",
            initial_guess=tau_joints_last[:, :-1],
            interpolation=InterpolationType.EACH_FRAME,
        )

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        time_last,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=OdeSolver.RK4(),
        n_threads=32,
    )
