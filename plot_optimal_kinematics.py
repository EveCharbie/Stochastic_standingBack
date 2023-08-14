
import matplotlib.pyplot as plt
import numpy as np
import biorbd
import pickle


def get_integrated_states(ocp, q_sol, qdot_sol, tau_sol, time_sol, stochastic_variables=[], nb_random=30):
    motor_noise_magnitude = 10
    nu = tau_sol.shape[0]
    nq = q_sol.shape[0]

    noise = np.random.normal(loc=0, scale=motor_noise_magnitude, size=(nu, nb_random, ocp.nlp[0].ns))
    states_integrated = np.zeros((2 * nq, nb_random, ocp.nlp[0].ns + 1))
    controls_noised = np.zeros((nu, nb_random, ocp.nlp[0].ns))
    for j in range(nb_random):
        states_integrated[:, j, 0] = np.hstack((q_sol[:, 0], qdot_sol[:, 0]))
        for k in range(ocp.nlp[0].ns):
            controls = tau_sol[:, k] + noise[:, j, k]
            if len(stochastic_variables) > 0:
                stochastic_variables = np.vstack((stochastic_variables))  # TODO
            else:
                stochastic_variables = []
            new_states = ocp.nlp[0].dynamics[k](states_integrated[:, j, k],  ### ?????
                                                controls,
                                                time_sol,
                                                stochastic_variables,
                                                [],
                                                [])[0]  # select "xf"
            states_integrated[:, j, k+1] = np.reshape(new_states, (2*nq, ))
            controls_noised[:, j, k] = controls

    return states_integrated, controls_noised


def plot_q(q_sol, states_integrated, final_time, n_shooting, DoF_names, name, nb_random=30):
    nq = q_sol.shape[0]
    time = np.linspace(0, final_time, n_shooting + 1)

    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    axs = np.ravel(axs)
    for i in range(nq):
        for j in range(nb_random):
            axs[i].plot(time, states_integrated[i, j, :], color='tab:blue', alpha=0.2)
        axs[i].plot(time, q_sol[i, :], color='k')
        axs[i].set_title(DoF_names[i])

    plt.suptitle(f"Q for {name}")
    plt.savefig(f"{save_path}_Q.png")
    plt.show()


def plot_CoM(states_integrated, model, n_shooting, name, nb_random=30):
    nx = states_integrated.shape[0]
    nq = int(nx/2)
    time = np.linspace(0, final_time, n_shooting + 1)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = np.ravel(axs)
    CoM = np.zeros((nb_random, n_shooting + 1))
    CoMdot = np.zeros((nb_random, n_shooting + 1))
    PelvisRot = np.zeros((nb_random, n_shooting + 1))
    PelvisVelot = np.zeros((nb_random, n_shooting + 1))
    for j in range(nb_random):
        for k in range(n_shooting + 1):
            CoM[j, k] = model.CoM(states_integrated[:nq, j, k]).to_array()[1]
            CoMdot[j, k] = model.CoMdot(states_integrated[:nq, j, k], states_integrated[nq:, j, k]).to_array()[1]
            PelvisRot[j, k] = states_integrated[2, j, k]
            PelvisVelot[j, k] = states_integrated[nq + 2, j, k]
        axs[0].plot(time, CoM[j, :], color='tab:blue', alpha=0.2)
        axs[1].plot(time, CoMdot[j, :], color='tab:blue', alpha=0.2)
        axs[2].plot(time, PelvisRot[j, :], color='tab:blue', alpha=0.2)
        axs[3].plot(time, PelvisVelot[j, :], color='tab:blue', alpha=0.2)

    CoM_deterministic = np.zeros((n_shooting + 1))
    CoMdot_deterministic = np.zeros((n_shooting + 1))
    PelvisRot_deterministic = np.zeros((n_shooting + 1))
    PelvisVelot_deterministic = np.zeros((n_shooting + 1))
    for k in range(n_shooting + 1):
        CoM_deterministic[k] = model.CoM(q_sol[:, k]).to_array()[1]
        CoMdot_deterministic[k] = model.CoMdot(q_sol[:, k], qdot_sol[:, k]).to_array()[1]
        PelvisRot_deterministic[k] = q_sol[2, k]
        PelvisVelot_deterministic[k] = qdot_sol[2, k]
    axs[0].plot(time, CoM_deterministic, color='k')
    axs[1].plot(time, CoMdot_deterministic, color='k')
    axs[2].plot(time, PelvisRot_deterministic, color='k')
    axs[3].plot(time, PelvisVelot_deterministic, color='k')
    axs[0].set_title("CoM")
    axs[1].set_title("CoMdot")
    axs[2].set_title("PelvisRot")
    axs[3].set_title("PelvisVelot")

    plt.suptitle(f"CoM and Pelvis for {name}")
    plt.savefig(f"{save_path}_CoM.png")
    plt.show()


biorbd_model_path = "models/Model2D_7Dof_1C_3M.bioMod"

dt = 0.01
final_time = 0.5
n_shooting = int(final_time / dt)

DoF_names = ["TransY", "TransZ", "PelvisRot", "Shoulder", "Hip", "Knee", "Ankle"]

ocp_type_list = ["deterministe", "simulated"]
for name in ocp_type_list:

    if name == "deterministe":
        from seg5_torque_driven_deterministic import prepare_ocp
        save_path = f"graphs/{biorbd_model_path[7:-7]}_torque_driven_1phase_deterministic_plot"
        results_path = f"results/{biorbd_model_path[7:-7]}_torque_driven_1phase_ocp.pkl"
    elif name == "simulated":
        from seg5_torque_driven_simulated import prepare_ocp
        save_path = f"graphs/{biorbd_model_path[7:-7]}_torque_driven_1phase_simulated_plot"
        results_path = f"results/Model2D_7Dof_1C_3M_torque_driven_1phase_simulated_noise10_weight100_random30.pkl"
    else:
        raise RuntimeError("Wrong ocp_type")

    with open(results_path, 'rb') as file:
        data = pickle.load(file)
        q_sol = data["q_sol"]
        qdot_sol = data["qdot_sol"]
        tau_sol = data["tau_sol"]
        time_sol = data["time_sol"]

    model = biorbd.Model(biorbd_model_path)

    ocp = prepare_ocp(biorbd_model_path=biorbd_model_path,
                      final_time=final_time,
                      n_shooting=n_shooting)

    states_integrated, controls_noised = get_integrated_states(ocp, q_sol, qdot_sol, tau_sol, time_sol, nb_random=30)

    plot_q(q_sol, states_integrated, final_time, n_shooting, DoF_names, name, nb_random=30)
    plot_CoM(states_integrated, model, n_shooting, name, nb_random=30)











