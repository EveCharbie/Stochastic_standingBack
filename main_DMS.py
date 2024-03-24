
import biorbd
import pickle
import casadi as cas
import numpy as np
import os

import sys

sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import Solver, OdeSolver, SolutionMerge

from DMS_deterministic import prepare_ocp
from DMS_SOCP import prepare_socp
from DMS_SOCP_VARIABLE import prepare_socp_VARIABLE
from DMS_SOCP_FEEDFORWARD import prepare_socp_FEEDFORWARD
from DMS_SOCP_VARIABLE_FEEDFORWARD import prepare_socp_VARIABLE_FEEDFORWARD

RUN_OCP = False  # OK
RUN_SOCP = False  # OK 1e-3
RUN_SOCP_VARIABLE = False  # OK 1e-3
RUN_SOCP_FEEDFORWARD = True
RUN_SOCP_VARIABLE_FEEDFORWARD = False

nb_random = 15

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

dt = 0.05
final_time = 0.8
n_shooting = int(final_time / dt)
tol = 1e-8  # 1e-3 OK

# Solver parameters
solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
solver.set_linear_solver("ma97")
solver.set_bound_frac(1e-8)
solver.set_bound_push(1e-8)
solver.set_maximum_iterations(10000)  # 32
solver.set_hessian_approximation("limited-memory")
# solver._nlp_scaling_method = "none"
# solver.set_check_derivatives_for_naninf(False)  # does not raise an error, but might slow down the resolution


# --- Run the deterministic --- #
save_path = f"results/{model_name}_ocp_DMS.pkl"

if RUN_OCP:
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path, time_last=final_time, n_shooting=n_shooting
    )
    ocp.add_plot_penalty()
    # ocp.add_plot_check_conditioning()
    ocp.add_plot_ipopt_outputs()

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

    save_path = save_path.replace(".", "p")
    if sol_ocp.status != 0:
        save_path = save_path.replace("ppkl", f"_DVG_1e-8.pkl")
    else:
        save_path = save_path.replace("ppkl", f"_CVG_1e-8.pkl")

    with open(save_path, "wb") as file:
        pickle.dump(data, file)

    print(save_path)
    # import bioviz
    # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
    # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
    # b.exec()


# --- Run the SOCP --- #
motor_noise_std = 0.05
wPq_std = 0.001
wPqdot_std = 0.003

print_motor_noise_std = "{:.1e}".format(motor_noise_std)
print_wPq_std = "{:.1e}".format(wPq_std)
print_wPqdot_std = "{:.1e}".format(wPqdot_std)
print_tol = "{:.1e}".format(tol)
save_path = (
    f"results/{model_name}_socp_DMS_{print_motor_noise_std}_"
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

    path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
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

    motor_noise_numerical, sensory_noise_numerical, socp = prepare_socp(
        biorbd_model_path=biorbd_model_path,
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
        nb_random=nb_random,
    )
    socp.add_plot_penalty()
    # socp.add_plot_ipopt_outputs()
    # socp.check_conditioning()

    solver.set_tol(tol)
    sol_socp = socp.solve(solver)

    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)

    q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
        states["q_roots"],
        states["q_joints"],
        states["qdot_roots"],
        states["qdot_joints"],
    )
    tau_joints_sol, k_sol, ref_sol = controls["tau_joints"], controls["k"], controls["ref"]

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
        "motor_noise_numerical": motor_noise_numerical,
        "sensory_noise_numerical": sensory_noise_numerical,
    }

    save_path = save_path.replace(".", "p")
    if sol_socp.status != 0:
        save_path = save_path.replace("ppkl", f"_DMS_{nb_random}random_DVG_{print_tol}.pkl")
    else:
        save_path = save_path.replace("ppkl", f"_DMS_{nb_random}random_CVG_{print_tol}.pkl")

    # --- Save the results --- #
    with open(save_path, "wb") as file:
        pickle.dump(data, file)

    print(save_path)
    # import bioviz
    # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
    # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
    # b.exec()


# --- Run the SOCP+ (variable noise) --- #
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

    path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
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

    motor_noise_numerical, sensory_noise_numerical, socp = prepare_socp_VARIABLE(
        biorbd_model_path=biorbd_model_path,
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
        nb_random=nb_random,
    )

    socp.add_plot_penalty()
    # socp.add_plot_check_conditioning()
    socp.add_plot_ipopt_outputs()
    solver.set_tol(tol)
    sol_socp = socp.solve(solver)

    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)

    q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
        states["q_roots"],
        states["q_joints"],
        states["qdot_roots"],
        states["qdot_joints"],
    )
    tau_joints_sol, k_sol, ref_sol = controls["tau_joints"], controls["k"], controls["ref"]
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
        "motor_noise_numerical": motor_noise_numerical,
        "sensory_noise_numerical": sensory_noise_numerical,
    }

    save_path = save_path.replace(".", "p")
    if sol_socp.status != 0:
        save_path_vision = save_path_vision.replace("ppkl", f"_DMS_{nb_random}random_DVG_{print_tol}.pkl")
    else:
        save_path_vision = save_path_vision.replace("ppkl", f"_DMS_{nb_random}random_CVG_{print_tol}.pkl")

    # --- Save the results --- #
    with open(save_path_vision, "wb") as file:
        pickle.dump(data, file)

    print(save_path)
    # import bioviz
    # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
    # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
    # b.exec()


# --- Run the SOCP+ (feedforward) --- #
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

    path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
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

    q_joints_last = np.vstack((q_joints_last[0, :], np.zeros((1, q_joints_last.shape[1])), q_joints_last[1:, :]))
    qdot_joints_last = np.vstack(
        (qdot_joints_last[0, :], np.zeros((1, qdot_joints_last.shape[1])), qdot_joints_last[1:, :])
    )
    tau_joints_last = np.vstack(
        (tau_joints_last[0, :], np.zeros((1, tau_joints_last.shape[1])), tau_joints_last[1:, :])
    )

    motor_noise_numerical, sensory_noise_numerical, socp = prepare_socp_FEEDFORWARD(
        biorbd_model_path=biorbd_model_path_vision,
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
        nb_random=nb_random,
    )

    socp.add_plot_penalty()

    solver.set_tol(tol)
    sol_socp = socp.solve(solver)

    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)

    q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
        states["q_roots"],
        states["q_joints"],
        states["qdot_roots"],
        states["qdot_joints"],
    )
    tau_joints_sol, k_sol, ref_sol = controls["tau_joints"], controls["k"], controls["ref"]
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
        "motor_noise_numerical": motor_noise_numerical,
        "sensory_noise_numerical": sensory_noise_numerical,
    }

    save_path = save_path.replace(".", "p")
    if sol_socp.status != 0:
        save_path_vision = save_path_vision.replace("ppkl", f"_DMS_{nb_random}random_DVG_{print_tol}.pkl")
    else:
        save_path_vision = save_path_vision.replace("ppkl", f"_DMS_{nb_random}random_CVG_{print_tol}.pkl")

    # --- Save the results --- #
    with open(save_path_vision, "wb") as file:
        pickle.dump(data, file)

    print(save_path)
    # import bioviz
    # b = bioviz.Viz(model_path=biorbd_model_path_vision_with_mesh)
    # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
    # b.exec()


# --- Run the SOCP+ (variable noise & feedforward) --- #
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

    path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
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

    q_joints_last = np.vstack((q_joints_last[0, :], np.zeros((1, q_joints_last.shape[1])), q_joints_last[1:, :]))
    qdot_joints_last = np.vstack(
        (qdot_joints_last[0, :], np.zeros((1, qdot_joints_last.shape[1])), qdot_joints_last[1:, :])
    )
    tau_joints_last = np.vstack(
        (tau_joints_last[0, :], np.zeros((1, tau_joints_last.shape[1])), tau_joints_last[1:, :])
    )

    motor_noise_numerical, sensory_noise_numerical, socp = prepare_socp_VARIABLE_FEEDFORWARD(
        biorbd_model_path=biorbd_model_path_vision,
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
        nb_random=nb_random,
    )

    socp.add_plot_penalty()

    solver.set_tol(tol)
    sol_socp = socp.solve(solver)

    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)

    q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
        states["q_roots"],
        states["q_joints"],
        states["qdot_roots"],
        states["qdot_joints"],
    )
    tau_joints_sol, k_sol, ref_sol = controls["tau_joints"], controls["k"], controls["ref"]
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
        "motor_noise_numerical": motor_noise_numerical,
        "sensory_noise_numerical": sensory_noise_numerical,
    }

    save_path = save_path.replace(".", "p")
    if sol_socp.status != 0:
        save_path_vision = save_path_vision.replace("ppkl", f"_DMS_{nb_random}random_DVG_{print_tol}.pkl")
    else:
        save_path_vision = save_path_vision.replace("ppkl", f"_DMS_{nb_random}random_CVG_{print_tol}.pkl")

    # --- Save the results --- #
    with open(save_path_vision, "wb") as file:
        pickle.dump(data, file)

    print(save_path)
    # import bioviz
    # b = bioviz.Viz(model_path=biorbd_model_path_vision_with_mesh)
    # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
    # b.exec()
