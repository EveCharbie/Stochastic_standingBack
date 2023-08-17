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

def zero_acceleration(controller: PenaltyController, motor_noise: np.ndarray, sensory_noise: np.ndarray) -> cas.MX:
    dx = stochastic_forward_dynamics(controller.states.cx_start, controller.controls.cx_start,
                                     controller.parameters.cx_start, controller.stochastic_variables.cx_start,
                                     controller.get_nlp, motor_noise, sensory_noise, with_gains=False)
    return dx.dxdt[controller.states_dot.index("qddot")]


def CoM_over_ankle(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    CoM_pos = controller.model.center_of_mass(q)
    CoM_pos_y = CoM_pos[1]
    marker_pos = controller.model.markers(q)[2]
    marker_pos_y = marker_pos[1]
    return marker_pos_y - CoM_pos_y


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver,
) -> OptimalControlProgram:
    """
    ...
    """

    bio_model = BiorbdModel(biorbd_model_path)

    n_q = bio_model.nb_q

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=0.01,
                            quadratic=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, min_bound=0.1, max_bound=1)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_MARKERS, marker_index=2, axes=Axis.Z, node=Node.END)
    constraints.add(CoM_over_ankle, node=Node.END)
    # constraints.add(ConstraintFcn.TRACK_CONTROL, key="tau", index=[0, 1, 2], node=Node.ALL)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    pose_at_first_node = np.array([-0.0422, 0.0892, 0.2386, 0.0, -0.1878, 0.0])  # Initial position approx from bioviz
    pose_at_last_node = np.array([-0.0422, 0.0892, 5.7904, 0.0, 0.5036, 0.0])  # Final position approx from bioviz

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
    tau_min[[0, 1, 2], :] = 0
    tau_max[[0, 1, 2], :] = 0
    u_bounds.add("tau", min_bound=tau_min, max_bound=tau_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    # Initial guesses
    x_init = InitialGuessList()
    x_init.add("q", initial_guess=np.vstack((pose_at_first_node, pose_at_last_node)).T, interpolation=InterpolationType.LINEAR)
    x_init.add("qdot", initial_guess=[0.01]*n_q, interpolation=InterpolationType.CONSTANT)

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=[0.01]*n_q, interpolation=InterpolationType.CONSTANT)

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
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        n_threads=1,
        assume_phase_dynamics=False,
    )

