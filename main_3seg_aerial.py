
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
from seg3_aerial_trapezoidal import prepare_socp_trap

RUN_OCP = False  # True
RUN_SOCP = True
ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3, method="legendre") #  OdeSolver.TRAPEZOIDAL  #

model_name = "Model2D_6Dof_0C_3M"
biorbd_model_path = f"models/{model_name}.bioMod"
biorbd_model_path_with_mesh = f"models/{model_name}_with_mesh.bioMod"
n_q = 6
n_root = 3

# import bioviz
# b = bioviz.Viz(biorbd_model_path)
# b.exec()

dt = 0.05
final_time = 0.8
n_shooting = int(final_time / dt)

# Solver parameters
solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
solver.set_linear_solver('ma97')
solver.set_tol(1e-3)
solver.set_dual_inf_tol(3e-4)
solver.set_constr_viol_tol(1e-7)
solver.set_bound_frac(1e-8)
solver.set_bound_push(1e-8)
solver.set_maximum_iterations(1000)
solver.set_hessian_approximation('limited-memory')  # Mandatory, otherwise RAM explodes!
solver._nlp_scaling_method = "none"


if isinstance(ode_solver, OdeSolver.COLLOCATION):

    # --- Run the deterministic collocation --- #
    save_path = f"results/{model_name}_aerial_ocp_collocations.pkl"
    
    if RUN_OCP:
        ocp = prepare_ocp(biorbd_model_path=biorbd_model_path,
                            final_time=final_time,
                            n_shooting=n_shooting,
                            ode_solver=ode_solver)
    
        # ocp.add_plot_penalty(CostType.ALL)
        # ocp.check_conditioning()
        sol_ocp = ocp.solve(solver=solver)
    
        q_sol = sol_ocp.states["q"]
        qdot_sol = sol_ocp.states["qdot"]
        tau_sol = sol_ocp.controls["tau"]
        time_sol = sol_ocp.parameters["time"][0][0]
        data = {"q_sol": q_sol,
                "qdot_sol": qdot_sol,
                "tau_sol": tau_sol,
                "time_sol": time_sol}

        if sol_ocp.status != 0:
            save_path = save_path.replace(".pkl", "_DVG.pkl")
        else:
            save_path = save_path.replace(".pkl", "_CVG.pkl")

        with open(save_path, "wb") as file:
            pickle.dump(data, file)
    
        # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
        # b.load_movement(q_sol)
        # b.exec()
    
    
    # --- Run the SOCP collocation with increasing noise --- #
    noise_factors = [0, 0.05, 0.1, 0.5, 1.0]
    with_cholesky = False
    
    for i_noise, noise_factor in enumerate(noise_factors):
    
        # TODO: How do we choose the values?
        motor_noise_std = 0.05 * noise_factor
        wPq_std = 3e-4 * noise_factor
        wPqdot_std = 0.0024 * noise_factor
    
        save_path = (f"results/{model_name}_aerial_socp_collocations_{round(motor_noise_std, 6)}_"
                     f"{round(wPq_std, 6)}_"
                     f"{round(wPqdot_std, 6)}.pkl")

        if save_path.replace(".pkl", "_CVG.pkl") in os.listdir("results"):
            print(f"Already did {save_path}!")
            continue

        if noise_factor == 0:
            path_to_results = f"results/{model_name}_aerial_ocp_collocations_CVG.pkl"
            with open(path_to_results, 'rb') as file:
                data = pickle.load(file)
                q_last = data['q_sol']
                qdot_last = data['qdot_sol']
                tau_last = data['tau_sol']
                time_last = data['time_sol']
                k_last = None
                ref_last = None
                m_last = None
                cov_last = None
                cholesky_last = None
    
        else:

            path_to_results = (f"results/{model_name}_aerial_socp_collocations_{round(0.05 * noise_factors[i_noise-1], 6)}_"
                               f"{round(3e-4 * noise_factors[i_noise-1], 6)}_"
                               f"{round(0.0024 * noise_factors[i_noise-1], 6)}_CVG.pkl")
    
            with open(path_to_results, 'rb') as file:
                data = pickle.load(file)
                q_last = data['q_sol']
                qdot_last = data['qdot_sol']
                tau_last = data['tau_sol']
                time_last = data['time_sol']
                k_last = data['k_sol']
                ref_last = data['ref_sol']
                m_last = data['m_sol']
                cov_last = data['cov_sol']
                cholesky_last = data['cov_cholesky_sol']

        motor_noise_magnitude = cas.DM(np.array([motor_noise_std ** 2 / dt for _ in range(n_q-n_root)]))  # All DoFs except root
        sensory_noise_magnitude = cas.DM(cas.vertcat(
            np.array([wPq_std ** 2 / dt for _ in range(n_q-n_root+1)]),
            np.array([wPqdot_std ** 2 / dt for _ in range(n_q-n_root+1)])
        ))  # since the head is fixed to the pelvis, the vestibular feedback is in the states ref
    
        socp = prepare_socp(biorbd_model_path=biorbd_model_path,
                            time_last=time_last,
                            n_shooting=n_shooting,
                            motor_noise_magnitude=motor_noise_magnitude,
                            sensory_noise_magnitude=sensory_noise_magnitude,
                            q_last=q_last,
                            qdot_last=qdot_last,
                            tau_last=tau_last,
                            k_last=None,
                            ref_last=None,
                            m_last=None,
                            cov_last=None,
                            cholesky_last=None,
                            with_cholesky=with_cholesky)
    
        # socp.add_plot_penalty(CostType.ALL)
        # socp.check_conditioning()
    
        sol_socp = socp.solve(solver)
        # sol_socp.graphs()
    
        q_sol = sol_socp.states["q"]
        qdot_sol = sol_socp.states["qdot"]
        tau_sol = sol_socp.controls["tau"]
        time_sol = sol_socp.parameters["time"][0][0]
        k_sol = sol_socp.stochastic_variables["k"]
        ref_sol = sol_socp.stochastic_variables["ref"]
        m_sol = sol_socp.stochastic_variables["m"]
        if with_cholesky:
            cov_cholesky_sol = sol_socp.stochastic_variables["cov_holesky"]
            cov_sol = None
        else:
            cov_cholesky_sol = None
            cov_sol = sol_socp.stochastic_variables["cov"]
        data = {"q_sol": q_sol,
                "qdot_sol": qdot_sol,
                "tau_sol": tau_sol,
                "time_sol": time_sol,
                "k_sol": k_sol,
                "ref_sol": ref_sol,
                "m_sol": m_sol,
                "cov_sol": cov_sol,
                "cov_cholesky_sol": cov_cholesky_sol}

        if sol_socp.status != 0:
            save_path = save_path.replace(".pkl", "_DVG.pkl")
        else:
            save_path = save_path.replace(".pkl", "_CVG.pkl")

        # --- Save the results --- #
        with open(save_path, "wb") as file:
            pickle.dump(data, file)
    
        # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
        # b.load_movement(q_sol)
        # b.exec()

elif isinstance(ode_solver, OdeSolver.TRAPEZOIDAL):

    # --- Run the deterministic trapezoidal --- #
    save_path = f"results/{model_name}_aerial_ocp_trapezoidal.pkl"

    if RUN_OCP:
        ocp_trap = prepare_ocp(biorbd_model_path=biorbd_model_path,
                                    final_time=final_time,
                                    n_shooting=n_shooting,
                                    ode_solver=ode_solver,)

        # ocp.add_plot_penalty(CostType.ALL)
        # ocp.check_conditioning()
        sol_ocp = ocp_trap.solve(solver=solver)

        q_sol = sol_ocp.states["q"]
        qdot_sol = sol_ocp.states["qdot"]
        tau_sol = sol_ocp.controls["tau"]
        time_sol = sol_ocp.parameters["time"][0][0]
        data = {"q_sol": q_sol,
                "qdot_sol": qdot_sol,
                "tau_sol": tau_sol,
                "time_sol": time_sol}

        if sol_ocp.status != 0:
            save_path = save_path.replace(".pkl", "_DVG.pkl")
        else:
            save_path = save_path.replace(".pkl", "_CVG.pkl")

        with open(save_path, "wb") as file:
            pickle.dump(data, file)

        # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
        # b.load_movement(q_sol)
        # b.exec()

    # --- Run the SOCP trapezoidal with increasing noise --- #
    noise_factors = [0, 0.05, 0.1, 0.5, 1.0]
    with_cholesky = True

    for i_noise, noise_factor in enumerate(noise_factors):

        # TODO: How do we choose the values?
        motor_noise_std = 0.05 * noise_factor
        wPq_std = 3e-4 * noise_factor
        wPqdot_std = 0.0024 * noise_factor

        save_path = (f"results/{model_name}_aerial_socp_trapezoidal_{round(motor_noise_std, 6)}_"
                     f"{round(wPq_std, 6)}_"
                     f"{round(wPqdot_std, 6)}.pkl")

        if noise_factor == 0:
            path_to_results = f"results/{model_name}_aerial_ocp_trapezoidal_CVG.pkl"
            with open(path_to_results, 'rb') as file:
                data = pickle.load(file)
                q_last = data['q_sol']
                qdot_last = data['qdot_sol']
                tau_last = data['tau_sol']
                time_last = data['time_sol']
                k_last = None
                ref_last = None
                m_last = None
                cov_last = None
                cholesky_last = None
                a_last = None
                c_last = None

        else:
            path_to_results = (f"results/{model_name}_aerial_socp_trapezoidal_"
                               f"{round(0.05 * noise_factors[i_noise - 1], 6)}_"
                               f"{round(3e-4 * noise_factors[i_noise - 1], 6)}_"
                               f"{round(0.0024 * noise_factors[i_noise - 1], 6)}_CVG.pkl")

            with open(path_to_results, 'rb') as file:
                data = pickle.load(file)
                q_last = data['q_sol']
                qdot_last = data['qdot_sol']
                tau_last = data['tau_sol']
                time_last = data['time_sol']
                k_last = data['k_sol']
                ref_last = data['ref_sol']
                m_last = data['m_sol']
                cov_last = data['cov_sol']
                cholesky_last = data['cholesky_sol']
                a_last = data['a_sol']
                c_last = data['c_sol']

        motor_noise_magnitude = cas.DM(
            np.array([motor_noise_std ** 2 / dt for _ in range(n_q - n_root)]))  # All DoFs except root
        sensory_noise_magnitude = cas.DM(cas.vertcat(
            np.array([wPq_std ** 2 / dt for _ in range(n_q - n_root + 1)]),
            np.array([wPqdot_std ** 2 / dt for _ in range(n_q - n_root + 1)])
        ))  # since the head is fixed to the pelvis, the vestibular feedback is in the states ref

        socp = prepare_socp_trap(biorbd_model_path=biorbd_model_path,
                            time_last=time_last,
                            n_shooting=n_shooting,
                            motor_noise_magnitude=motor_noise_magnitude,
                            sensory_noise_magnitude=sensory_noise_magnitude,
                            q_last=q_last,
                            qdot_last=qdot_last,
                            tau_last=tau_last,
                            k_last=None,
                            ref_last=None,
                            m_last=None,
                            cov_last=None,
                            cholesky_last=None,
                            a_last=None,
                            c_last=None,
                            with_cholesky=with_cholesky)

        # socp.add_plot_penalty(CostType.ALL)
        # socp.check_conditioning()

        sol_socp = socp.solve(solver)
        # sol_socp.graphs()

        q_sol = sol_socp.states["q"]
        qdot_sol = sol_socp.states["qdot"]
        tau_sol = sol_socp.controls["tau"]
        time_sol = sol_socp.parameters["time"][0][0]
        k_sol = sol_socp.stochastic_variables["k"]
        ref_sol = sol_socp.stochastic_variables["ref"]
        m_sol = sol_socp.stochastic_variables["m"]
        cov_sol = sol_socp.stochastic_variables["cov"]
        cholesky_sol = sol_socp.stochastic_variables["cholesky"]
        a_sol = sol_socp.stochastic_variables["a"]
        c_sol = sol_socp.stochastic_variables["c"]
        data = {"q_sol": q_sol,
                "qdot_sol": qdot_sol,
                "tau_sol": tau_sol,
                "time_sol": time_sol,
                "k_sol": k_sol,
                "ref_sol": ref_sol,
                "m_sol": m_sol,
                "cov_sol": cov_sol,
                "cholesky_sol": cholesky_sol,
                "a_sol": a_sol,
                "c_sol": c_sol}

        if sol_socp.status != 0:
            save_path = save_path.replace(".pkl", "_DVG.pkl")
        else:
            save_path = save_path.replace(".pkl", "_CVG.pkl")

        # --- Save the results --- #
        with open(save_path, "wb") as file:
            pickle.dump(data, file)

        # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
        # b.load_movement(q_sol)
        # b.exec()



