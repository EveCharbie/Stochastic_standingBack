import bioviz
import biorbd
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

polynomial_degree = 3

RUN_OCP = True
RUN_SOCP = True
RUN_VISION = False
ode_solver = OdeSolver.COLLOCATION(polynomial_degree=polynomial_degree,
                                   method="legendre",
                                   duplicate_collocation_starting_point=True,
                                   )

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
solver.set_tol(1e-4)  # 1e-3
solver.set_bound_frac(1e-8)
solver.set_bound_push(1e-8)
solver.set_maximum_iterations(10000)
solver.set_hessian_approximation("limited-memory")
# solver._nlp_scaling_method = "none"

if isinstance(ode_solver, OdeSolver.COLLOCATION):
    # --- Run the deterministic collocation --- #
    save_path = f"results/{model_name}_aerial_ocp_collocations.pkl"

    if RUN_OCP:
        ocp = prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            time_last=final_time,
            n_shooting=n_shooting,
            ode_solver=ode_solver
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

        # # These constraints are OK in duplicate_collocation_starting_point=False and True
        # polynomial_degree = ocp.nlp[0].ode_solver.polynomial_degree
        # time_vector = np.linspace(0, time_sol, n_shooting + 1)
        # n_cx = ocp.nlp[0].ode_solver.n_cx - 1
        # ns = ocp.nlp[0].ns
        #
        # # Constraint values
        # x_opt = cas.vertcat(q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol)
        # x_sol = np.zeros((x_opt.shape[0], n_cx, ns))
        # for i_node in range(ocp.n_shooting):
        #     x_sol[:, :, i_node] = x_opt[:, i_node * n_cx:(i_node + 1) * n_cx]
        #
        # constraint_value = ocp.nlp[0].g[0].function[0](0,
        #                                                 x_opt[:, -1],
        #                                                 tau_joints_sol[:, -1],
        #                                                 time_sol,
        #                                                 [],
        #                                                 )
        # print("Toe marker on the ground at landing: ", constraint_value)
        #
        # constraint_value = ocp.nlp[0].g[1].function[0](0,
        #                                                 x_opt[:, -1],
        #                                                 tau_joints_sol[:, -1],
        #                                                 time_sol,
        #                                                 [],
        #                                                 )
        # print("CoM over toes at landing: ", constraint_value)
        #
        # x_multi_thread = np.zeros((2*n_q*(n_cx+1), ns))
        # for i_state in range(2 * n_q):
        #     for i_node in range(ns):
        #         for i_coll in range(n_cx):
        #             x_multi_thread[i_coll * 2 * n_q + i_state, i_node] = x_sol[i_state, i_coll, i_node]
        #         if i_node < ns - 1:
        #             x_multi_thread[(i_coll + 1) * 2 * n_q + i_state, i_node] = x_sol[i_state, 0, i_node + 1]
        #         else:
        #             x_multi_thread[(i_coll + 1) * 2 * n_q + i_state, i_node] = x_opt[i_state, -1]
        #
        # # OK
        # u_multi_thread = np.zeros((tau_joints_sol.shape[0]*2, ns))
        # u_multi_thread[:tau_joints_sol.shape[0], :] = tau_joints_sol[:, :ns]
        # u_multi_thread[tau_joints_sol.shape[0]:, :] = tau_joints_sol[:, 1:ns + 1]
        # u_multi_thread[tau_joints_sol.shape[0]:, -1] = tau_joints_sol[:, -2]
        #
        # constraint_value = ocp.nlp[0].g_internal[0].function[0](time_sol,
        #                                                          x_multi_thread,
        #                                                          u_multi_thread,
        #                                                          time_sol,
        #                                                          [],
        #                                                          )
        # print("States continuity: ", constraint_value)


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
    noise_factors = [1.0]  # 0.05, 0.1, 0.5,

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

            # if noise_factor == 0:
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

            # else:
            #     path_to_results = (
            #         f"results/{model_name}_aerial_socp_collocations_{round(0.05 * noise_factors[i_noise-1], 6)}_"
            #         f"{round(3e-4 * noise_factors[i_noise-1], 6)}_"
            #         f"{round(0.0024 * noise_factors[i_noise-1], 6)}_CVG.pkl"
            #     )
            #
            #     with open(path_to_results, "rb") as file:
            #         data = pickle.load(file)
            #         q_roots_last = data["q_roots_sol"]
            #         q_joints_last = data["q_joints_sol"]
            #         qdot_roots_last = data["qdot_roots_sol"]
            #         qdot_joints_last = data["qdot_joints_sol"]
            #         tau_joints_last = data["tau_joints_sol"]
            #         time_last = data["time_sol"]
            #         k_last = data["k_sol"]
            #         ref_last = data["ref_sol"]
            #         m_last = data["m_sol"]
            #         cov_last = data["cov_sol"]

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

            # polynomial_degree = socp.nlp[0].ode_solver.polynomial_degree
            # time_vector = np.linspace(0, time_sol, n_shooting + 1)
            # n_cx = socp.nlp[0].ode_solver.n_cx - 1
            # ns = socp.nlp[0].ns
            #
            # # Constraint values
            # x_opt = cas.vertcat(q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol)
            # x_sol = np.zeros((x_opt.shape[0], n_cx, ns))
            # for i_node in range(ns):
            #     x_sol[:, :, i_node] = x_opt[:, i_node * n_cx:(i_node + 1) * n_cx]
            # s_sol = cas.vertcat(k_sol, ref_sol, m_sol, cov_sol)
            #
            # constraint_value = socp.nlp[0].g[0].function[0](0,
            #                                              x_opt[:, -1],
            #                                              tau_joints_sol[:, -1],
            #                                              time_sol,
            #                                              s_sol[:, -1],
            #                                              )
            # print("Toe marker on the ground at landing: ", constraint_value)
            #
            # constraint_value = socp.nlp[0].g[1].function[0](0,
            #                                              x_opt[:, -1],
            #                                              tau_joints_sol[:, -1],
            #                                              time_sol,
            #                                              s_sol[:, -1],
            #                                              )
            # print("CoM over toes at landing: ", constraint_value)
            #
            # for i_node in range(socp.n_shooting):
            #     constraint_value = socp.nlp[0].g[2].function[i_node](0,
            #                               x_sol[:, :, i_node].flatten(order="F"),
            #                               tau_joints_sol[:, i_node],
            #                               time_sol,
            #                               s_sol[:, i_node],
            #                               )
            #     print("Sensory input = reference: ", constraint_value)
            #
            # for i_node in range(socp.n_shooting):
            #     constraint_value = socp.nlp[0].g[3].function[i_node](0,
            #                               x_sol[:, :, i_node].flatten(order="F"),
            #                               tau_joints_sol[:, i_node],
            #                               time_sol,
            #                               s_sol[:, i_node],
            #                               )
            #     print("Constraint on M: ", constraint_value)
            #
            # x_multi_thread = np.zeros((2 * n_q * (n_cx + 1), ns))
            # for i_state in range(2 * n_q):
            #     for i_node in range(ns):
            #         for i_coll in range(n_cx):
            #             x_multi_thread[i_coll * 2 * n_q + i_state, i_node] = x_sol[i_state, i_coll, i_node]
            #         if i_node < ns - 1:
            #             x_multi_thread[(i_coll + 1) * 2 * n_q + i_state, i_node] = x_sol[i_state, 0, i_node + 1]
            #         else:
            #             x_multi_thread[(i_coll + 1) * 2 * n_q + i_state, i_node] = x_opt[i_state, -1]
            #
            # u_multi_thread = np.zeros((tau_joints_sol.shape[0]*2, ns))
            # u_multi_thread[:tau_joints_sol.shape[0], :] = tau_joints_sol[:, :ns]
            # u_multi_thread[tau_joints_sol.shape[0]:, :] = tau_joints_sol[:, 1:ns + 1]
            # u_multi_thread[tau_joints_sol.shape[0]:, -1] = tau_joints_sol[:, -2]
            #
            # s_multi_thread = np.zeros((s_sol.shape[0]*2, ns))
            # s_multi_thread[:s_sol.shape[0], :] = s_sol[:, :ns]
            # s_multi_thread[s_sol.shape[0]:, :] = s_sol[:, 1:ns + 1]
            # s_multi_thread[s_sol.shape[0]:, -1] = np.reshape(s_sol[:, -2], (-1, ))
            #
            # constraint_value = socp.nlp[0].g[4].function[0](0,
            #                           x_multi_thread,
            #                           u_multi_thread,
            #                           time_sol,
            #                           s_multi_thread,
            #                           )
            # print("Covariance continuity: ", constraint_value)  # 14x14x16
            #
            # constraint_value = socp.nlp[0].g_internal[0].function[0](time_sol,
            #                                                          x_multi_thread,
            #                                                          u_multi_thread,
            #                                                          time_sol,
            #                                                          [],
            #                                                          )
            # print("States continuity: ", constraint_value)
            #
            # for i_node in range(socp.n_shooting):
            #     constraint_value = socp.nlp[0].g_internal[1].function[i_node](0,
            #                               x_sol[:, :, i_node].flatten(order="F"),
            #                               tau_joints_sol[:, i_node],
            #                               time_sol,
            #                               s_sol[:, i_node],
            #                               )
            #     print("First collocation point equals state: ", constraint_value)


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
            n_q += 1

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

            # with open(save_path.replace(".pkl", "_CVG.pkl"), 'rb') as file:
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

            path_to_results = f"results/Model2D_7Dof_0C_3M_aerial_ocp_collocations_CVG.pkl"
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
            qdot_joints_last = np.vstack((qdot_joints_last[0, :], np.zeros((1, qdot_joints_last.shape[1])), qdot_joints_last[1:, :]))
            tau_joints_last = np.vstack((tau_joints_last[0, :], np.zeros((1, tau_joints_last.shape[1])), tau_joints_last[1:, :]))

            socp = prepare_socp_vision(
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


            # polynomial_degree = socp.nlp[0].ode_solver.polynomial_degree
            # time_vector = np.linspace(0, time_sol, n_shooting + 1)
            # n_cx = socp.nlp[0].ode_solver.n_cx - 1
            # ns = socp.nlp[0].ns
            #
            # # Constraint values
            # x_opt = cas.vertcat(q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol)
            # x_sol = np.zeros((x_opt.shape[0], n_cx, ns))
            # for i_node in range(ns):
            #     x_sol[:, :, i_node] = x_opt[:, i_node * n_cx:(i_node + 1) * n_cx]
            # s_sol = cas.vertcat(k_sol, ref_sol, m_sol, cov_sol)
            #
            # constraint_value = socp.nlp[0].g[0].function[0](0,
            #                                              x_opt[:, -1],
            #                                              tau_joints_sol[:, -1],
            #                                              time_sol,
            #                                              s_sol[:, -1],
            #                                              )
            # print("Toe marker on the ground at landing: ", constraint_value)
            #
            # constraint_value = socp.nlp[0].g[1].function[0](0,
            #                                              x_opt[:, -1],
            #                                              tau_joints_sol[:, -1],
            #                                              time_sol,
            #                                              s_sol[:, -1],
            #                                              )
            # print("CoM over toes at landing: ", constraint_value)
            #
            # for i_node in range(socp.n_shooting):
            #     constraint_value = socp.nlp[0].g[2].function[i_node](0,
            #                               x_sol[:, :, i_node].flatten(order="F"),
            #                               tau_joints_sol[:, i_node],
            #                               time_sol,
            #                               s_sol[:, i_node],
            #                               )
            #     print("Sensory input = reference: ", constraint_value)
            #
            # for i_node in range(socp.n_shooting):
            #     constraint_value = socp.nlp[0].g[3].function[i_node](0,
            #                               x_sol[:, :, i_node].flatten(order="F"),
            #                               tau_joints_sol[:, i_node],
            #                               time_sol,
            #                               s_sol[:, i_node],
            #                               )
            #     print("Constraint on M: ", constraint_value[:100])
            #     print("Constraint on M: ", constraint_value[100:])
            #
            # x_multi_thread = np.zeros((2 * n_q * (n_cx + 1), ns))
            # for i_state in range(2 * n_q):
            #     for i_node in range(ns):
            #         for i_coll in range(n_cx):
            #             x_multi_thread[i_coll * 2 * n_q + i_state, i_node] = x_sol[i_state, i_coll, i_node]
            #         if i_node < ns - 1:
            #             x_multi_thread[(i_coll + 1) * 2 * n_q + i_state, i_node] = x_sol[i_state, 0, i_node + 1]
            #         else:
            #             x_multi_thread[(i_coll + 1) * 2 * n_q + i_state, i_node] = x_opt[i_state, -1]
            #
            # u_multi_thread = np.zeros((tau_joints_sol.shape[0]*2, ns))
            # u_multi_thread[:tau_joints_sol.shape[0], :] = tau_joints_sol[:, :ns]
            # u_multi_thread[tau_joints_sol.shape[0]:, :] = tau_joints_sol[:, 1:ns + 1]
            # u_multi_thread[tau_joints_sol.shape[0]:, -1] = tau_joints_sol[:, -2]
            #
            # s_multi_thread = np.zeros((s_sol.shape[0]*2, ns))
            # s_multi_thread[:s_sol.shape[0], :] = s_sol[:, :ns]
            # s_multi_thread[s_sol.shape[0]:, :] = s_sol[:, 1:ns + 1]
            # s_multi_thread[s_sol.shape[0]:, -1] = np.reshape(s_sol[:, -2], (-1, ))
            #
            # constraint_value = socp.nlp[0].g[4].function[0](0,
            #                           x_multi_thread,
            #                           u_multi_thread,
            #                           time_sol,
            #                           s_multi_thread,
            #                           )
            # print("Covariance continuity: ", constraint_value[:100])  # 14x14x16
            # print("Covariance continuity: ", constraint_value[100:])  # 14x14x16
            #
            # constraint_value = socp.nlp[0].g_internal[0].function[0](time_sol,
            #                                                          x_multi_thread,
            #                                                          u_multi_thread,
            #                                                          time_sol,
            #                                                          [],
            #                                                          )
            # print("States continuity: ", constraint_value)
            #
            # for i_node in range(socp.n_shooting):
            #     constraint_value = socp.nlp[0].g_internal[1].function[i_node](0,
            #                               x_sol[:, :, i_node].flatten(order="F"),
            #                               tau_joints_sol[:, i_node],
            #                               time_sol,
            #                               s_sol[:, i_node],
            #                               )
            #     print("First collocation point equals state: ", constraint_value)


            if sol_socp.status != 0:
                save_path_vision = save_path_vision.replace(".pkl", "_DVG.pkl")
            else:
                save_path_vision = save_path_vision.replace(".pkl", "_CVG.pkl")

            # --- Save the results --- #
            with open(save_path_vision, "wb") as file:
                pickle.dump(data, file)

            b = bioviz.Viz(model_path=biorbd_model_path_vision_with_mesh)
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
