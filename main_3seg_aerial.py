
import bioviz
import pickle
import casadi as cas
import numpy as np

import sys
sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import Solver

from seg3_aerial_collocations_deterministic import prepare_ocp
from seg3_aerial_collocations import prepare_socp

RUN_OCP_COLLOCATIONS = False # True
RUN_SOCP_COLLOCATIONS = True

model_name = "Model2D_5Dof_0C_3M"
biorbd_model_path = f"models/{model_name}.bioMod"
biorbd_model_path_with_mesh = f"models/{model_name}_with_mesh.bioMod"
n_q = 5
n_root = 3

# import bioviz
# b = bioviz.Viz(biorbd_model_path)
# b.exec()

dt = 0.05
final_time = 0.8
n_shooting = int(final_time / dt)

# Solver parameters
solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
solver.set_linear_solver('ma97')
solver.set_tol(1e-3)
solver.set_dual_inf_tol(3e-4)
solver.set_constr_viol_tol(1e-7)
solver.set_bound_frac(1e-8)
solver.set_bound_push(1e-8)
solver.set_maximum_iterations(1000)
solver.set_hessian_approximation('limited-memory')  # Mandatory, otherwise RAM explodes!
solver._nlp_scaling_method = "none"


# --- Run the deterministic collocation --- #
save_path = f"results/{model_name}_aerial_ocp_collocations.pkl"

if RUN_OCP_COLLOCATIONS:
    ocp = prepare_ocp(biorbd_model_path=biorbd_model_path,
                        final_time=final_time,
                        n_shooting=n_shooting)

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

    with open(save_path, "wb") as file:
        pickle.dump(data, file)

    b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
    b.load_movement(q_sol)
    b.exec()


# --- Run the SOCP collocation with increasing noise --- #
noise_factors = [0, 0.05, 0.1, 0.5, 1.0]
with_cholesky = False

for i_noise, noise_factor in enumerate(noise_factors):

    # TODO: How do we choose the values?
    motor_noise_std = 0.05 * noise_factor
    wPq_std = 3e-4 * noise_factor
    wPqdot_std = 0.0024 * noise_factor

    save_path = f"results/{model_name}_aerial_socp_collocations_{motor_noise_std}_{wPq_std}_{wPqdot_std}.pkl"

    if noise_factor == 0:
        path_to_results = f"results/{model_name}_aerial_ocp_collocations.pkl"
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
        path_to_results = (f"results/{model_name}_aerial_socp_collocations_{0.05 * noise_factors[i_noise-1]}_"
                           f"{3e-4 * noise_factors[i_noise-1]}_"
                           f"{0.0024 * noise_factors[i_noise-1]}.pkl")

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

    motor_noise_magnitude = cas.DM(np.array([motor_noise_std ** 2 / dt for _ in range(n_q-n_root)]))  # All DoFs except root
    sensory_noise_magnitude = cas.DM(np.array([wPq_std ** 2 / dt, wPqdot_std ** 2 / dt]))  # Pelvis orientation + angular velocity

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
            "a_sol": a_sol,
            "c_sol": c_sol}

    # --- Save the results --- #
    with open(save_path, "wb") as file:
        pickle.dump(data, file)

    b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
    b.load_movement(q_sol)
    b.exec()





