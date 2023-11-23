
import numpy as np
import casadi as cas
import biorbd_casadi as biorbd

import sys
sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    PenaltyController,
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
