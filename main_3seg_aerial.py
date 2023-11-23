import bioviz
import pickle
import casadi as cas
import numpy as np
import os

import sys

sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import Solver, OdeSolver

from seg3_aerial_deterministic import prepare_ocp
from seg3_aerial_collocations import prepare_socp
from vision_aerial_collocations import prepare_socp_vision

RUN_OCP = False
RUN_SOCP = True
RUN_VISION = False
ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3, method="legendre", duplicate_collocation_starting_point=True)

model_name = "Model2D_7Dof_0C_3M"
biorbd_model_path = f"models/{model_name}.bioMod"
biorbd_model_path_with_mesh = f"models/{model_name}_with_mesh.bioMod"
biorbd_model_path_vision = f"models/{model_name}_vision.bioMod"
biorbd_model_path_vision_with_mesh = f"models/{model_name}_vision_with_mesh.bioMod"

n_q = 7
n_root = 3

# import bioviz
# b = bioviz.Viz(biorbd_model_path)
# b.exec()

dt = 0.05
final_time = 0.8
n_shooting = int(final_time / dt)

# Solver parameters
solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
solver.set_linear_solver("ma97")
solver.set_tol(1e-6)  # 1e-3
solver.set_bound_frac(1e-8)
solver.set_bound_push(1e-8)
solver.set_maximum_iterations(3000)
solver.set_hessian_approximation("limited-memory")
# solver._nlp_scaling_method = "none"

if isinstance(ode_solver, OdeSolver.COLLOCATION):
    # --- Run the deterministic collocation --- #
    save_path = f"results/{model_name}_aerial_ocp_collocations.pkl"

    if RUN_OCP:
        ocp = prepare_ocp(
            biorbd_model_path=biorbd_model_path, final_time=final_time, n_shooting=n_shooting, ode_solver=ode_solver
        )
        sol_ocp = ocp.solve(solver=solver)

        q_roots_sol = sol_ocp.states["q_roots"]
        q_joints_sol = sol_ocp.states["q_joints"]
        qdot_roots_sol = sol_ocp.states["qdot_roots"]
        qdot_joints_sol = sol_ocp.states["qdot_joints"]
        tau_joints_sol = sol_ocp.controls["tau_joints"]
        time_sol = sol_ocp.parameters["time"][0][0]
        data = {
            "q_roots_sol": q_roots_sol,
            "q_joints_sol": q_joints_sol,
            "qdot_roots_sol": qdot_roots_sol,
            "qdot_joints_sol": qdot_joints_sol,
            "tau_joints_sol": tau_joints_sol,
            "time_sol": time_sol,
        }

        if sol_ocp.status != 0:
            save_path = save_path.replace(".pkl", "_DVG.pkl")
        else:
            save_path = save_path.replace(".pkl", "_CVG.pkl")

        with open(save_path, "wb") as file:
            pickle.dump(data, file)

        b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
        b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
        b.exec()

    # --- Run the SOCP collocation with increasing noise --- #
    noise_factors = [0, 0.05, 0.1, 0.5, 1.0]

    for i_noise, noise_factor in enumerate(noise_factors):
        # TODO: How do we choose the values?
        motor_noise_std = 0.05 * noise_factor
        wPq_std = 0.001 * noise_factor
        wPqdot_std = 0.003 * noise_factor

        save_path = (
            f"results/{model_name}_aerial_socp_collocations_{round(motor_noise_std, 6)}_"
            f"{round(wPq_std, 6)}_"
            f"{round(wPqdot_std, 6)}.pkl"
        )

        motor_noise_magnitude = cas.DM(
            np.array([motor_noise_std**2 / dt for _ in range(n_q - n_root)])
        )  # All DoFs except root
        sensory_noise_magnitude = cas.DM(
            cas.vertcat(
                np.array([wPq_std**2 / dt for _ in range(n_q - n_root + 1)]),
                np.array([wPqdot_std**2 / dt for _ in range(n_q - n_root + 1)]),
            )
        )  # since the head is fixed to the pelvis, the vestibular feedback is in the states ref

        if RUN_SOCP:
            # if save_path.replace(".pkl", "_CVG.pkl") in os.listdir("results"):
            #     print(f"Already did {save_path}!")
            #     continue

            if noise_factor == 0:
                path_to_results = f"results/{model_name}_aerial_ocp_collocations_CVG.pkl"
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

            else:
                path_to_results = (
                    f"results/{model_name}_aerial_socp_collocations_{round(0.05 * noise_factors[i_noise-1], 6)}_"
                    f"{round(3e-4 * noise_factors[i_noise-1], 6)}_"
                    f"{round(0.0024 * noise_factors[i_noise-1], 6)}_CVG.pkl"
                )

                with open(path_to_results, "rb") as file:
                    data = pickle.load(file)
                    q_roots_last = data["q_roots_sol"]
                    q_joints_last = data["q_joints_sol"]
                    qdot_roots_last = data["qdot_roots_sol"]
                    qdot_joints_last = data["qdot_joints_sol"]
                    tau_joints_last = data["tau_joints_sol"]
                    time_last = data["time_sol"]
                    k_last = data["k_sol"]
                    ref_last = data["ref_sol"]
                    m_last = data["m_sol"]
                    cov_last = data["cov_sol"]

            socp = prepare_socp(
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
                m_last=None,
                cov_last=None,
            )

            sol_socp = socp.solve(solver)

            q_roots_sol = sol_socp.states["q_roots"]
            q_joints_sol = sol_socp.states["q_joints"]
            qdot_roots_sol = sol_socp.states["qdot_roots"]
            qdot_joints_sol = sol_socp.states["qdot_joints"]
            tau_joints_sol = sol_socp.controls["tau_joints"]
            time_sol = sol_socp.parameters["time"][0][0]
            k_sol = sol_socp.stochastic_variables["k"]
            ref_sol = sol_socp.stochastic_variables["ref"]
            m_sol = sol_socp.stochastic_variables["m"]
            cov_sol = sol_socp.stochastic_variables["cov"]
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
                save_path = save_path.replace(".pkl", "_DVG.pkl")
            else:
                save_path = save_path.replace(".pkl", "_CVG.pkl")

            # --- Save the results --- #
            with open(save_path, "wb") as file:
                pickle.dump(data, file)

            b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
            b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
            b.exec()

        else:
            print("to be checked")
            # # --- Load the results --- #
            # with open(save_path, 'rb') as file:
            #     data = pickle.load(file)
            #     q_roots_sol = data['q_roots_sol']
            #     q_joints_sol = data['q_joints_sol']
            #     qdot_roots_sol = data['qdot_roots_sol']
            #     qdot_joints_sol = data['qdot_joints_sol']
            #     tau_joints_sol = data['tau_joints_sol']
            #     time_sol = data['time_sol']
            #     k_sol = data['k_sol']
            #     ref_sol = data['ref_sol']
            #     m_sol = data['m_sol']
            #     cov_sol = data['cov_sol']

            # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
            # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
            # b.exec()

        save_path_vision = save_path.replace(".pkl", "_vision.pkl")

        if RUN_VISION:
            # --- Run the vision --- #
            model_name = "Model2D_6Dof_0C_3M_vision"

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

            # if save_path.replace(".pkl", "_CVG.pkl") in os.listdir("results"):
            #     print(f"Already did {save_path}!")
            #     continue

            # with open(save_path, 'rb') as file:
            #     data = pickle.load(file)
            #     q_roots_last = data['q_roots_sol']
            #     q_joints_last = data['q_joints_sol']
            #     qdot_roots_last = data['qdot_roots_sol']
            #     qdot_joints_last = data['qdot_joints_sol']
            #     tau_joints_last = data['tau_joints_sol']
            #     time_last = data['time_sol']
            #     k_last = data['k_sol']
            #     ref_last = data['ref_sol']
            #     m_last = data['m_sol']
            #     cov_last = data['cov_sol']
            # q_joints_last = np.vstack((np.zeros((1, q_joints_last.shape[1])), q_joints_last))
            # qdot_joints_last = np.vstack((np.zeros((1, qdot_joints_last.shape[1])), qdot_joints_last))

            q_roots_last = None
            q_joints_last = None
            qdot_roots_last = None
            qdot_joints_last = None
            tau_joints_last = None
            time_last = 0.8

            socp = prepare_socp_vision(
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
                m_last=None,
                cov_last=None,
            )

            sol_socp = socp.solve(solver)

            q_roots_sol = sol_socp.states["q_roots"]
            q_joints_sol = sol_socp.states["q_joints"]
            qdot_roots_sol = sol_socp.states["qdot_roots"]
            qdot_joints_sol = sol_socp.states["qdot_joints"]
            tau_joints_sol = sol_socp.controls["tau_joints"]
            time_sol = sol_socp.parameters["time"][0][0]
            k_sol = sol_socp.stochastic_variables["k"]
            ref_sol = sol_socp.stochastic_variables["ref"]
            m_sol = sol_socp.stochastic_variables["m"]
            cov_sol = sol_socp.stochastic_variables["cov"]
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
                save_path_vision = save_path_vision.replace(".pkl", "_DVG.pkl")
            else:
                save_path_vision = save_path_vision.replace(".pkl", "_CVG.pkl")

            # --- Save the results --- #
            with open(save_path_vision, "wb") as file:
                pickle.dump(data, file)

            b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
            b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
            b.exec()

        else:
            # --- Load the results --- #
            with open(save_path_vision, "rb") as file:
                data = pickle.load(file)
                q_roots_sol = data["q_roots_sol"]
                q_joints_sol = data["q_joints_sol"]
                qdot_roots_sol = data["qdot_roots_sol"]
                qdot_joints_sol = data["qdot_joints_sol"]
                tau_joints_sol = data["tau_joints_sol"]
                time_sol = data["time_sol"]
                k_sol = data["k_sol"]
                ref_sol = data["ref_sol"]
                m_sol = data["m_sol"]
                cov_sol = data["cov_sol"]

            # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
            # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
            # b.exec()
