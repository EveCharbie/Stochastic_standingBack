
import biorbd
import pickle
import casadi as cas
import numpy as np
import os

import sys

sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import Solver, OdeSolver, SolutionMerge

from seg3_aerial_deterministic import prepare_ocp
from seg3_aerial_collocations import prepare_socp
from SOCP_VARIABLE_aerial_collocations import prepare_socp_SOCP_VARIABLE
from SOCP_FEEDFORWARD_aerial_collocations import prepare_socp_SOCP_FEEDFORWARD
from SOCP_VARIABLE_FEEDFORWARD_aerial_collocations import prepare_socp_SOCP_VARIABLE_FEEDFORWARD

polynomial_degree = 3

RUN_OCP = False
RUN_SOCP = True
RUN_SOCP_VARIABLE = False
RUN_SOCP_FEEDFORWARD = False
RUN_SOCP_VARIABLE_FEEDFORWARD = False


ode_solver = OdeSolver.COLLOCATION(
    polynomial_degree=polynomial_degree,
    method="legendre",
    duplicate_starting_point=True,
)

model_name = "Model2D_7Dof_0C_3M"
biorbd_model_path = f"models/{model_name}.bioMod"
biorbd_model_path_with_mesh = f"models/{model_name}_with_mesh.bioMod"
biorbd_model_path_vision = f"models/{model_name}_vision.bioMod"
biorbd_model_path_vision_with_mesh = f"models/{model_name}_vision_with_mesh.bioMod"

n_q = 7
n_root = 3

# import bioviz
# b = bioviz.Viz(biorbd_model_path_vision_with_mesh,
#                background_color=(1, 1, 1),
#                show_local_ref_frame=False,
#                show_markers=False,
#                show_segments_center_of_mass=False,
#                show_global_center_of_mass=False,
#                show_global_ref_frame=False,
#                show_gravity_vector=False,
#                )
# b.exec()

# dt = 0.025
dt = 0.05
# final_time = 0.5
final_time = 0.8
n_shooting = int(final_time / dt)
tol = 1e-3  # 1e-3 OK

# Solver parameters
solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
solver.set_linear_solver("ma97")
solver.set_bound_frac(1e-8)
solver.set_bound_push(1e-8)
solver.set_maximum_iterations(10000)  # 32
solver.set_hessian_approximation("limited-memory")
solver._nlp_scaling_method = "none"
# solver.set_check_derivatives_for_naninf(False)  # does not raise an error, but might slow down the resolution


# --- Run the deterministic collocation --- #
save_path = f"results/{model_name}_aerial_ocp_collocations.pkl"

if RUN_OCP:
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path, time_last=final_time, n_shooting=n_shooting, ode_solver=ode_solver
    )
    ocp.add_plot_penalty()
    # ocp.add_plot_check_conditioning()

    solver.set_tol(1e-8)
    sol_ocp = ocp.solve(solver=solver)

    states = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_ocp.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol_ocp.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
        states["q_roots"],
        states["q_joints"],
        states["qdot_roots"],
        states["qdot_joints"],
    )
    tau_joints_sol = controls["tau_joints"]
    time_sol = float(sol_ocp.decision_time()[-1])

    data = {
        "q_roots_sol": q_roots_sol,
        "q_joints_sol": q_joints_sol,
        "qdot_roots_sol": qdot_roots_sol,
        "qdot_joints_sol": qdot_joints_sol,
        "tau_joints_sol": tau_joints_sol,
        "time_sol": time_sol,
    }

    if sol_ocp.status != 0:
        save_path = save_path.replace(".pkl", f"_DVG_1e-8.pkl")
    else:
        save_path = save_path.replace(".pkl", f"_CVG_1e-8.pkl")

    with open(save_path, "wb") as file:
        pickle.dump(data, file)

    # print(save_path)
    # import bioviz
    # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
    # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
    # b.exec()


# --- Run the SOCP collocation --- #
noise_factor = 1.0  # 0.05, 0.1, 0.5,

# TODO: How do we choose the values?
motor_noise_std = 0.05 * noise_factor
wPq_std = 0.001 * noise_factor
wPqdot_std = 0.003 * noise_factor

print_motor_noise_std = "{:.1e}".format(motor_noise_std)
print_wPq_std = "{:.1e}".format(wPq_std)
print_wPqdot_std = "{:.1e}".format(wPqdot_std)
print_tol = "{:.1e}".format(tol)
save_path = (
    f"results/{model_name}_aerial_socp_collocations_{print_motor_noise_std}_"
    f"{print_wPq_std}_"
    f"{print_wPqdot_std}.pkl"
)

motor_noise_magnitude = cas.DM(np.array([motor_noise_std**2 / dt for _ in range(n_q - n_root)]))  # All DoFs except root
sensory_noise_magnitude = cas.DM(
    cas.vertcat(
        np.array([wPq_std**2 / dt for _ in range(n_q - n_root + 1)]),
        np.array([wPqdot_std**2 / dt for _ in range(n_q - n_root + 1)]),
    )
)  # since the head is fixed to the pelvis, the vestibular feedback is in the states ref

if RUN_SOCP:

    path_to_results = f"results/{model_name}_aerial_ocp_collocations_CVG_1e-8.pkl"
    with open(path_to_results, "rb") as file:
        data = pickle.load(file)
        q_roots_last = data["q_roots_sol"]
        q_joints_last = data["q_joints_sol"]
        qdot_roots_last = data["qdot_roots_sol"]
        qdot_joints_last = data["qdot_joints_sol"]
        tau_joints_last = data["tau_joints_sol"]
        time_last = data["time_sol"]
        k_last = None
        ref_last = None
        m_last = None
        cov_last = None

    socp = prepare_socp(
        biorbd_model_path=biorbd_model_path,
        polynomial_degree=polynomial_degree,
        time_last=time_last,
        n_shooting=n_shooting,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        q_roots_last=q_roots_last,
        q_joints_last=q_joints_last,
        qdot_roots_last=qdot_roots_last,
        qdot_joints_last=qdot_joints_last,
        tau_joints_last=tau_joints_last,
        k_last=k_last,
        ref_last=ref_last,
        m_last=m_last,
        cov_last=cov_last,
    )
    socp.add_plot_penalty()
    # socp.add_plot_ipopt_outputs()
    # socp.check_conditioning()

    solver.set_tol(tol)
    sol_socp = socp.solve(solver)

    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol_socp.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
        states["q_roots"],
        states["q_joints"],
        states["qdot_roots"],
        states["qdot_joints"],
    )
    tau_joints_sol = controls["tau_joints"]
    k_sol, ref_sol, m_sol, cov_sol = (
        algebraic_states["k"],
        algebraic_states["ref"],
        algebraic_states["m"],
        algebraic_states["cov"],
    )
    time_sol = sol_socp.decision_time()[-1]

    data = {
        "q_roots_sol": q_roots_sol,
        "q_joints_sol": q_joints_sol,
        "qdot_roots_sol": qdot_roots_sol,
        "qdot_joints_sol": qdot_joints_sol,
        "tau_joints_sol": tau_joints_sol,
        "time_sol": time_sol,
        "k_sol": k_sol,
        "ref_sol": ref_sol,
        "m_sol": m_sol,
        "cov_sol": cov_sol,
    }

    if sol_socp.status != 0:
        save_path = save_path.replace(".pkl", f"_DVG_{print_tol}.pkl")
    else:
        save_path = save_path.replace(".pkl", f"_CVG_{print_tol}.pkl")

    # --- Save the results --- #
    with open(save_path, "wb") as file:
        pickle.dump(data, file)

    print(save_path)
    import bioviz
    b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
    b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
    b.exec()


# --- Run the SOCP+ collocation (variable noise) --- #
save_path_vision = save_path.replace(".pkl", "_VARIABLE.pkl")

if RUN_SOCP_VARIABLE:

    motor_noise_magnitude = cas.DM(
        np.array(
            [
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
            ]
        )
    )  # All DoFs except root
    sensory_noise_magnitude = cas.DM(
        np.array(
            [
                wPq_std**2 / dt,  # Proprioceptive position
                wPq_std**2 / dt,
                wPq_std**2 / dt,
                wPq_std**2 / dt,
                wPqdot_std**2 / dt,  # Proprioceptive velocity
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPq_std**2 / dt,  # Vestibular position
                wPq_std**2 / dt,  # Vestibular velocity
            ]
        )
    )

    path_to_results = f"results/Model2D_7Dof_0C_3M_aerial_ocp_collocations_CVG_1e-8.pkl"
    with open(path_to_results, "rb") as file:
        data = pickle.load(file)
        q_roots_last = data["q_roots_sol"]
        q_joints_last = data["q_joints_sol"]
        qdot_roots_last = data["qdot_roots_sol"]
        qdot_joints_last = data["qdot_joints_sol"]
        tau_joints_last = data["tau_joints_sol"]
        time_last = data["time_sol"]
        k_last = None
        ref_last = None
        m_last = None
        cov_last = None

    socp = prepare_socp_SOCP_VARIABLE(
        biorbd_model_path=biorbd_model_path,
        polynomial_degree=polynomial_degree,
        time_last=time_last,
        n_shooting=n_shooting,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        q_roots_last=q_roots_last,
        q_joints_last=q_joints_last,
        qdot_roots_last=qdot_roots_last,
        qdot_joints_last=qdot_joints_last,
        tau_joints_last=tau_joints_last,
        k_last=None,
        ref_last=None,
        m_last=None,
        cov_last=None,
    )

    socp.add_plot_penalty()
    # socp.add_plot_check_conditioning()
    sol_socp = socp.solve(solver)

    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol_socp.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
        states["q_roots"],
        states["q_joints"],
        states["qdot_roots"],
        states["qdot_joints"],
    )
    tau_joints_sol = controls["tau_joints"]
    k_sol, ref_sol, m_sol, cov_sol = (
        algebraic_states["k"],
        algebraic_states["ref"],
        algebraic_states["m"],
        algebraic_states["cov"],
    )
    time_sol = sol_socp.decision_time()[-1]

    data = {
        "q_roots_sol": q_roots_sol,
        "q_joints_sol": q_joints_sol,
        "qdot_roots_sol": qdot_roots_sol,
        "qdot_joints_sol": qdot_joints_sol,
        "tau_joints_sol": tau_joints_sol,
        "time_sol": time_sol,
        "k_sol": k_sol,
        "ref_sol": ref_sol,
        "m_sol": m_sol,
        "cov_sol": cov_sol,
    }

    if sol_socp.status != 0:
        save_path_vision = save_path_vision.replace(".pkl", f"_DVG_{print_tol}.pkl")
    else:
        save_path_vision = save_path_vision.replace(".pkl", f"_CVG_{print_tol}.pkl")

    # --- Save the results --- #
    with open(save_path_vision, "wb") as file:
        pickle.dump(data, file)

    print(save_path)
    import bioviz
    b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
    b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
    b.exec()


# --- Run the SOCP+ collocation (feedforward) --- #
save_path_vision = save_path.replace(".pkl", "_FEEDFORWARD.pkl")
n_q += 1

if RUN_SOCP_FEEDFORWARD:

    motor_noise_magnitude = cas.DM(
        np.array(
            [
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
            ]
        )
    )  # All DoFs except root
    sensory_noise_magnitude = cas.DM(
        np.array(
            [
                wPq_std**2 / dt,  # Proprioceptive position
                wPq_std**2 / dt,
                wPq_std**2 / dt,
                wPq_std**2 / dt,
                wPq_std**2 / dt,
                wPqdot_std**2 / dt,  # Proprioceptive velocity
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPq_std**2 / dt,  # Vestibular position
                wPq_std**2 / dt,  # Vestibular velocity
                wPq_std**2 / dt,  # Visual
            ]
        )
    )

    path_to_results = f"results/Model2D_7Dof_0C_3M_aerial_ocp_collocations_CVG_1e-8.pkl"
    with open(path_to_results, "rb") as file:
        data = pickle.load(file)
        q_roots_last = data["q_roots_sol"]
        q_joints_last = data["q_joints_sol"]
        qdot_roots_last = data["qdot_roots_sol"]
        qdot_joints_last = data["qdot_joints_sol"]
        tau_joints_last = data["tau_joints_sol"]
        time_last = data["time_sol"]
        k_last = None
        ref_last = None
        m_last = None
        cov_last = None

    q_joints_last = np.vstack((q_joints_last[0, :], np.zeros((1, q_joints_last.shape[1])), q_joints_last[1:, :]))
    qdot_joints_last = np.vstack(
        (qdot_joints_last[0, :], np.zeros((1, qdot_joints_last.shape[1])), qdot_joints_last[1:, :])
    )
    tau_joints_last = np.vstack(
        (tau_joints_last[0, :], np.zeros((1, tau_joints_last.shape[1])), tau_joints_last[1:, :])
    )

    socp = prepare_socp_SOCP_FEEDFORWARD(
        biorbd_model_path=biorbd_model_path_vision,
        polynomial_degree=polynomial_degree,
        time_last=time_last,
        n_shooting=n_shooting,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        q_roots_last=q_roots_last,
        q_joints_last=q_joints_last,
        qdot_roots_last=qdot_roots_last,
        qdot_joints_last=qdot_joints_last,
        tau_joints_last=tau_joints_last,
        k_last=None,
        ref_last=None,
        m_last=None,
        cov_last=None,
    )

    socp.add_plot_penalty()
    sol_socp = socp.solve(solver)

    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol_socp.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
        states["q_roots"],
        states["q_joints"],
        states["qdot_roots"],
        states["qdot_joints"],
    )
    tau_joints_sol = controls["tau_joints"]
    k_sol, ref_sol, m_sol, cov_sol = (
        algebraic_states["k"],
        algebraic_states["ref"],
        algebraic_states["m"],
        algebraic_states["cov"],
    )
    time_sol = sol_socp.decision_time()[-1]

    data = {
        "q_roots_sol": q_roots_sol,
        "q_joints_sol": q_joints_sol,
        "qdot_roots_sol": qdot_roots_sol,
        "qdot_joints_sol": qdot_joints_sol,
        "tau_joints_sol": tau_joints_sol,
        "time_sol": time_sol,
        "k_sol": k_sol,
        "ref_sol": ref_sol,
        "m_sol": m_sol,
        "cov_sol": cov_sol,
    }

    if sol_socp.status != 0:
        save_path_vision = save_path_vision.replace(".pkl", f"_DVG_{print_tol}.pkl")
    else:
        save_path_vision = save_path_vision.replace(".pkl", f"_CVG_{print_tol}.pkl")

    # --- Save the results --- #
    with open(save_path_vision, "wb") as file:
        pickle.dump(data, file)

    print(save_path)
    import bioviz
    b = bioviz.Viz(model_path=biorbd_model_path_vision_with_mesh)
    b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
    b.exec()


# --- Run the SOCP+ collocation (variable noise & feedforward) --- #
save_path_vision = save_path.replace(".pkl", "_VARIABLE_FEEDFORWARD.pkl")
n_q += 1

if RUN_SOCP_VARIABLE_FEEDFORWARD:

    motor_noise_magnitude = cas.DM(
        np.array(
            [
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
                motor_noise_std**2 / dt,
            ]
        )
    )  # All DoFs except root
    sensory_noise_magnitude = cas.DM(
        np.array(
            [
                wPq_std**2 / dt,  # Proprioceptive position
                wPq_std**2 / dt,
                wPq_std**2 / dt,
                wPq_std**2 / dt,
                wPq_std**2 / dt,
                wPqdot_std**2 / dt,  # Proprioceptive velocity
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPqdot_std**2 / dt,
                wPq_std**2 / dt,  # Vestibular position
                wPq_std**2 / dt,  # Vestibular velocity
                wPq_std**2 / dt,  # Visual
            ]
        )
    )

    path_to_results = f"results/Model2D_7Dof_0C_3M_aerial_ocp_collocations_CVG_1e-8.pkl"
    with open(path_to_results, "rb") as file:
        data = pickle.load(file)
        q_roots_last = data["q_roots_sol"]
        q_joints_last = data["q_joints_sol"]
        qdot_roots_last = data["qdot_roots_sol"]
        qdot_joints_last = data["qdot_joints_sol"]
        tau_joints_last = data["tau_joints_sol"]
        time_last = data["time_sol"]
        k_last = None
        ref_last = None
        m_last = None
        cov_last = None

    q_joints_last = np.vstack((q_joints_last[0, :], np.zeros((1, q_joints_last.shape[1])), q_joints_last[1:, :]))
    qdot_joints_last = np.vstack(
        (qdot_joints_last[0, :], np.zeros((1, qdot_joints_last.shape[1])), qdot_joints_last[1:, :])
    )
    tau_joints_last = np.vstack(
        (tau_joints_last[0, :], np.zeros((1, tau_joints_last.shape[1])), tau_joints_last[1:, :])
    )

    socp = prepare_socp_SOCP_VARIABLE_FEEDFORWARD(
        biorbd_model_path=biorbd_model_path_vision,
        polynomial_degree=polynomial_degree,
        time_last=time_last,
        n_shooting=n_shooting,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        q_roots_last=q_roots_last,
        q_joints_last=q_joints_last,
        qdot_roots_last=qdot_roots_last,
        qdot_joints_last=qdot_joints_last,
        tau_joints_last=tau_joints_last,
        k_last=None,
        ref_last=None,
        m_last=None,
        cov_last=None,
    )

    socp.add_plot_penalty()
    sol_socp = socp.solve(solver)

    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol_socp.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
        states["q_roots"],
        states["q_joints"],
        states["qdot_roots"],
        states["qdot_joints"],
    )
    tau_joints_sol = controls["tau_joints"]
    k_sol, ref_sol, m_sol, cov_sol = (
        algebraic_states["k"],
        algebraic_states["ref"],
        algebraic_states["m"],
        algebraic_states["cov"],
    )
    time_sol = sol_socp.decision_time()[-1]

    data = {
        "q_roots_sol": q_roots_sol,
        "q_joints_sol": q_joints_sol,
        "qdot_roots_sol": qdot_roots_sol,
        "qdot_joints_sol": qdot_joints_sol,
        "tau_joints_sol": tau_joints_sol,
        "time_sol": time_sol,
        "k_sol": k_sol,
        "ref_sol": ref_sol,
        "m_sol": m_sol,
        "cov_sol": cov_sol,
    }

    if sol_socp.status != 0:
        save_path_vision = save_path_vision.replace(".pkl", f"_DVG_{print_tol}.pkl")
    else:
        save_path_vision = save_path_vision.replace(".pkl", f"_CVG_{print_tol}.pkl")

    # --- Save the results --- #
    with open(save_path_vision, "wb") as file:
        pickle.dump(data, file)

    print(save_path)
    import bioviz
    b = bioviz.Viz(model_path=biorbd_model_path_vision_with_mesh)
    b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
    b.exec()
