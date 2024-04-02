import matplotlib.pyplot as plt
import numpy as np
import casadi as cas
import biorbd
import bioviz
import pickle

from bioptim import StochasticBioModel

from DMS_SOCP import prepare_socp
from utils import DMS_sensory_reference


def get_integrated_states(ocp, q_sol, qdot_sol, tau_sol, time_sol, algebraic_states=[], nb_random=30):
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
            if len(algebraic_states) > 0:
                algebraic_states = np.vstack((algebraic_states))  # TODO
            else:
                algebraic_states = []
            new_states = ocp.nlp[0].dynamics[k](
                states_integrated[:, j, k], controls, time_sol, algebraic_states, [], []  ### ?????
            )[
                0
            ]  # select "xf"
            states_integrated[:, j, k + 1] = np.reshape(new_states, (2 * nq,))
            controls_noised[:, j, k] = controls

    return states_integrated, controls_noised


def plot_q(q_sol, states_integrated, final_time, n_shooting, DoF_names, name, nb_random=30):
    nq = q_sol.shape[0]
    time = np.linspace(0, final_time, n_shooting + 1)

    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    axs = np.ravel(axs)
    for i in range(nq):
        for j in range(nb_random):
            axs[i].plot(time, states_integrated[i, j, :], color="tab:blue", alpha=0.2)
        axs[i].plot(time, q_sol[i, :], color="k")
        axs[i].set_title(DoF_names[i])

    plt.suptitle(f"Q for {name}")
    plt.savefig(f"{save_path}_Q.png")
    plt.show()


def plot_CoM(states_integrated, model, n_shooting, name, nb_random=30):
    nx = states_integrated.shape[0]
    nq = int(nx / 2)
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
        axs[0].plot(time, CoM[j, :], color="tab:blue", alpha=0.2)
        axs[1].plot(time, CoMdot[j, :], color="tab:blue", alpha=0.2)
        axs[2].plot(time, PelvisRot[j, :], color="tab:blue", alpha=0.2)
        axs[3].plot(time, PelvisVelot[j, :], color="tab:blue", alpha=0.2)

    CoM_deterministic = np.zeros((n_shooting + 1))
    CoMdot_deterministic = np.zeros((n_shooting + 1))
    PelvisRot_deterministic = np.zeros((n_shooting + 1))
    PelvisVelot_deterministic = np.zeros((n_shooting + 1))
    for k in range(n_shooting + 1):
        CoM_deterministic[k] = model.CoM(q_sol[:, k]).to_array()[1]
        CoMdot_deterministic[k] = model.CoMdot(q_sol[:, k], qdot_sol[:, k]).to_array()[1]
        PelvisRot_deterministic[k] = q_sol[2, k]
        PelvisVelot_deterministic[k] = qdot_sol[2, k]
    axs[0].plot(time, CoM_deterministic, color="k")
    axs[1].plot(time, CoMdot_deterministic, color="k")
    axs[2].plot(time, PelvisRot_deterministic, color="k")
    axs[3].plot(time, PelvisVelot_deterministic, color="k")
    axs[0].set_title("CoM")
    axs[1].set_title("CoMdot")
    axs[2].set_title("PelvisRot")
    axs[3].set_title("PelvisVelot")

    plt.suptitle(f"CoM and Pelvis for {name}")
    plt.savefig(f"{save_path}_CoM.png")
    plt.show()


def SOCP_dynamics(nb_random, q, qdot, tau, k_matrix, ref, motor_noise_numerical, sensory_noise_numerical, ocp):

    nb_root = 3
    nb_q = 7

    dxdt = cas.MX.zeros((2 * nb_q, nb_random))
    dxdt[:nb_q, :] = qdot
    for i_random in range(nb_random):

        q_this_time = q[:, i_random]
        qdot_this_time = qdot[:, i_random]

        tau_this_time = tau[:]

        # Joint friction
        tau_this_time += ocp.nlp[0].model.friction_coefficients @ qdot_this_time[nb_root:]

        # Motor noise
        tau_this_time += motor_noise_numerical[:, i_random]

        # Feedback
        tau_this_time += k_matrix @ (
            ref
            - DMS_sensory_reference(ocp.nlp[0].model, nb_root, q_this_time, qdot_this_time)
            + sensory_noise_numerical[:, i_random]
        )
        tau_this_time = cas.vertcat(cas.MX.zeros(nb_root), tau_this_time)

        dxdt[nb_q:, i_random] = ocp.nlp[0].model.forward_dynamics(q_this_time, qdot_this_time, tau_this_time)

    return dxdt


def RK4(q, qdot, tau, dt, k_matrix, ref, motor_noise_numerical, sensory_noise_numerical, n_random, dyn_fun):

    nb_q = 7
    states = np.zeros((2 * nb_q, n_random, 6))
    states[:nb_q, :, 0] = q
    states[nb_q:, :, 0] = qdot
    h = dt / 5
    for i in range(1, 6):
        k1 = dyn_fun(
            states[:nb_q, :, i - 1],
            states[nb_q:, :, i - 1],
            tau,
            k_matrix,
            ref,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        k2 = dyn_fun(
            states[:nb_q, :, i - 1] + h / 2 * k1[:nb_q, :],
            states[nb_q:, :, i - 1] + h / 2 * k1[nb_q:, :],
            tau,
            k_matrix,
            ref,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        k3 = dyn_fun(
            states[:nb_q, :, i - 1] + h / 2 * k2[:nb_q, :],
            states[nb_q:, :, i - 1] + h / 2 * k2[nb_q:, :],
            tau,
            k_matrix,
            ref,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        k4 = dyn_fun(
            states[:nb_q, :, i - 1] + h * k3[:nb_q, :],
            states[nb_q:, :, i - 1] + h * k3[nb_q:, :],
            tau,
            k_matrix,
            ref,
            motor_noise_numerical,
            sensory_noise_numerical,
        )
        states[:, :, i] = states[:, :, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return states[:, :, -1]


model_name = "Model2D_7Dof_0C_3M"
biorbd_model_path = f"models/{model_name}.bioMod"
biorbd_model_path_with_mesh = f"models/{model_name}_with_mesh.bioMod"
biorbd_model_path_vision = f"models/{model_name}_vision.bioMod"
biorbd_model_path_vision_with_mesh = f"models/{model_name}_vision_with_mesh.bioMod"

polynomial_degree = 3

n_q = 7
n_root = 3
n_joints = n_q - n_root
n_ref = 2 * n_joints + 2

dt = 0.05
final_time = 0.8
n_shooting = int(final_time / dt)
tol = 1e-6
nb_random = 15

motor_noise_std = 0.05
wPq_std = 0.001
wPqdot_std = 0.003

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


path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
with open(path_to_results, "rb") as file:
    data = pickle.load(file)
    q_roots_last = data["q_roots_sol"]
    q_joints_last = data["q_joints_sol"]
    qdot_roots_last = data["qdot_roots_sol"]
    qdot_joints_last = data["qdot_joints_sol"]
    tau_joints_last = data["tau_joints_sol"]
    time_last = data["time_sol"]

_, _, socp = prepare_socp(
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

plt.figure()
plt.plot(tau_joints_last.T)
plt.show()


path_to_results = "results/good/Model2D_7Dof_0C_3M_socp_DMS_5p0e-02_1p0e-03_3p0e-03_DMS_15random_CVG_1p0e-06.pkl"
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
    motor_noise_numerical = data["motor_noise_numerical"]
    sensory_noise_numerical = data["sensory_noise_numerical"]


is_label_dof_set = False
is_label_mean_set = False
is_label_ref_mean_set = False

q_sym = cas.MX.sym("Q", n_q)
qdot_sym = cas.MX.sym("Qdot", n_q)

out = DMS_sensory_reference(socp.nlp[0].model, n_root, q_sym, qdot_sym)
DMS_sensory_reference_func = cas.Function("DMS_sensory_reference", [q_sym, qdot_sym], [out])

q_sym = cas.MX.sym("Q", n_q, nb_random)
qdot_sym = cas.MX.sym("Qdot", n_q, nb_random)
tau_sym = cas.MX.sym("Tau", n_joints)
k_matrix_sym = cas.MX.sym("k_matrix", n_joints, n_ref)
ref_sym = cas.MX.sym("Ref", n_ref)
motor_noise_sym = cas.MX.sym("Motor_noise", n_joints, nb_random)
sensory_noise_sym = cas.MX.sym("sensory_noise", n_ref, nb_random)

dyn_fun_out = SOCP_dynamics(
    nb_random, q_sym, qdot_sym, tau_sym, k_matrix_sym, ref_sym, motor_noise_sym, sensory_noise_sym, socp
)
dyn_fun = cas.Function(
    "dynamics", [q_sym, qdot_sym, tau_sym, k_matrix_sym, ref_sym, motor_noise_sym, sensory_noise_sym], [dyn_fun_out]
)


time_vector = np.linspace(0, float(time_last), n_shooting + 1)

fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs = np.ravel(axs)
q_last = np.vstack((np.array(q_roots_last[:n_root]), np.array(q_joints_last[:n_joints])))[:, :, np.newaxis]
qdot_last = np.vstack((np.array(qdot_roots_last[:n_root]), np.array(qdot_joints_last[:n_joints])))[:, :, np.newaxis]
for i_dof in range(n_q):
    axs[i_dof].plot(time_vector, q_last[i_dof, :, 0], color="k", label="Noised states (optim variables)")
for i_random in range(1, nb_random):
    q_last = np.concatenate(
        (
            q_last,
            np.vstack(
                (
                    np.array(q_roots_last[i_random * n_root : (i_random + 1) * n_root]),
                    np.array(q_joints_last[i_random * n_joints : (i_random + 1) * n_joints]),
                )
            )[:, :, np.newaxis],
        ),
        axis=2,
    )
    qdot_last = np.concatenate(
        (
            qdot_last,
            np.vstack(
                (
                    np.array(qdot_roots_last[i_random * n_root : (i_random + 1) * n_root]),
                    np.array(qdot_joints_last[i_random * n_joints : (i_random + 1) * n_joints]),
                )
            )[:, :, np.newaxis],
        ),
        axis=2,
    )
    for i_dof in range(n_q):
        axs[i_dof].plot(time_vector, q_last[i_dof, :, i_random], color="k")
q_mean_last = np.mean(q_last, axis=2)
qdot_mean_last = np.mean(qdot_last, axis=2)
for i_dof in range(n_q):
    if not is_label_mean_set:
        axs[i_dof].plot(time_vector, q_mean_last[i_dof, :], "--", color="tab:red", label="Mean noised states")
        is_label_mean_set = True
    else:
        axs[i_dof].plot(time_vector, q_mean_last[i_dof, :], "--", color="tab:red")
    axs[i_dof].set_title(f"DOF {i_dof}")
ref_mean_last = np.zeros((n_ref, n_shooting))
for i_node in range(n_shooting):
    ref_mean_last[:, i_node] = np.array(
        DMS_sensory_reference_func(q_mean_last[:, i_node], qdot_mean_last[:, i_node])
    ).reshape(-1)
if not is_label_ref_mean_set:
    axs[3].plot(time_vector[:-1], ref_mean_last[0, :], color="tab:blue", label="Mean reference")
    axs[3].plot(time_vector[:-1], ref_last[0, :], "--", color="tab:orange", label="Reference (optim variables)")
    is_label_ref_mean_set = True
else:
    axs[3].plot(time_vector[:-1], ref_mean_last[0, :], color="tab:blue")
    axs[3].plot(time_vector[:-1], ref_last[0, :], "--", color="tab:orange")
axs[4].plot(time_vector[:-1], ref_mean_last[1, :], color="tab:blue")
axs[4].plot(time_vector[:-1], ref_last[1, :], "--", color="tab:orange")
axs[5].plot(time_vector[:-1], ref_mean_last[2, :], color="tab:blue")
axs[5].plot(time_vector[:-1], ref_last[2, :], "--", color="tab:orange")
axs[6].plot(time_vector[:-1], ref_mean_last[3, :], color="tab:blue")
axs[6].plot(time_vector[:-1], ref_last[3, :], "--", color="tab:orange")
axs[0].legend()
axs[3].legend()
plt.show()

import bioviz
b = bioviz.Viz(biorbd_model_path_with_mesh,
               background_color=(1, 1, 1),
               show_local_ref_frame=False,
               show_markers=False,
               show_segments_center_of_mass=False,
               show_global_center_of_mass=False,
               show_global_ref_frame=False,
               show_gravity_vector=False,
               )
b.load_movement(q_mean_last)
b.exec()


# Reintegrate
dt_last = time_vector[1] - time_vector[0]

# single shooting
q_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
qdot_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
q_integrated[:, 0, :] = q_last[:, 0, :]
qdot_integrated[:, 0, :] = qdot_last[:, 0, :]

# multiple shooting
q_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))
qdot_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))
q_multiple_shooting[:, 0, :] = q_last[:, 0, :]
qdot_multiple_shooting[:, 0, :] = qdot_last[:, 0, :]

for i_shooting in range(n_shooting):
    k_matrix = StochasticBioModel.reshape_to_matrix(k_last[:, i_shooting], socp.nlp[0].model.matrix_shape_k)
    states_integrated = RK4(
        q_integrated[:, i_shooting, :],
        qdot_integrated[:, i_shooting, :],
        tau_joints_last[:, i_shooting],
        dt_last,
        k_matrix,
        ref_last[:, i_shooting],
        motor_noise_numerical[:, :, i_shooting],
        sensory_noise_numerical[:, :, i_shooting],
        nb_random,
        dyn_fun,
    )
    q_integrated[:, i_shooting + 1, :] = states_integrated[:n_q, :]
    qdot_integrated[:, i_shooting + 1, :] = states_integrated[n_q:, :]

    states_integrated_multiple = RK4(
        q_last[:, i_shooting, :],
        qdot_last[:, i_shooting, :],
        tau_joints_last[:, i_shooting],
        dt_last,
        k_matrix,
        ref_last[:, i_shooting],
        motor_noise_numerical[:, :, i_shooting],
        sensory_noise_numerical[:, :, i_shooting],
        nb_random,
        dyn_fun,
    )
    q_multiple_shooting[:, i_shooting + 1, :] = states_integrated_multiple[:n_q, :]
    qdot_multiple_shooting[:, i_shooting + 1, :] = states_integrated_multiple[n_q:, :]

# Verify reintegration
is_label_dof_set = False
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs = np.ravel(axs)
for i_random in range(nb_random):
    for i_dof in range(n_q):
        if not is_label_dof_set:
            axs[i_dof].plot(time_vector, q_last[i_dof, :, i_random], color="k", label="Noised states (optim variables)")
            # axs[i_dof].plot(time_vector, q_integrated[i_dof, :, i_random], "--", color="r", label="Reintegrated states")
            is_label_dof_set = True
        else:
            axs[i_dof].plot(time_vector, q_last[i_dof, :, i_random], color="k")
            # axs[i_dof].plot(time_vector, q_integrated[i_dof, :, i_random], "--", color="r")
        for i_shooting in range(n_shooting):
            axs[i_dof].plot(
                np.array([time_vector[i_shooting], time_vector[i_shooting + 1]]),
                np.array([q_last[i_dof, i_shooting, i_random], q_multiple_shooting[i_dof, i_shooting + 1, i_random]]),
                "--",
                color="b",
            )
axs[0].legend()

plt.figure()
plt.plot(k_last.T)

plt.figure()
plt.plot(tau_joints_last.T)
plt.show()


dt = 0.05
final_time = 0.8
n_shooting = int(final_time / dt)

DoF_names = ["TransY", "TransZ", "PelvisRot", "Shoulder", "Hip", "Knee"]

ocp_type_list = ["deterministe", "collocations"]
for name in ocp_type_list:
    if name == "deterministe":
        results_path = f"results/{model_name}_aerial_ocp_collocations_CVG.pkl"

        with open(results_path, "rb") as file:
            data = pickle.load(file)
            q_sol = data["q_sol"]
            qdot_sol = data["qdot_sol"]
            tau_sol = data["tau_sol"]
            time_sol = data["time_sol"]

        b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
        b.load_movement(q_sol)
        b.exec()

    elif name == "collocations":
        noise_factors = [0, 0.05, 0.1, 0.5, 1.0]
        for i_noise, noise_factor in enumerate(noise_factors):
            motor_noise_std = 0.05 * noise_factor
            wPq_std = 3e-4 * noise_factor
            wPqdot_std = 0.0024 * noise_factor
            results_path = (
                f"results/{model_name}_aerial_socp_collocations_{round(motor_noise_std,6)}_"
                f"{round(wPq_std, 6)}_"
                f"{round(wPqdot_std, 6)}_CVG.pkl"
            )

            with open(results_path, "rb") as file:
                data = pickle.load(file)
                q_sol = data["q_sol"]
                qdot_sol = data["qdot_sol"]
                tau_sol = data["tau_sol"]
                time_sol = data["time_sol"]

            b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
            b.load_movement(q_sol)
            b.exec()

    # if name == "deterministe":
    #     from seg5_torque_driven_deterministic import prepare_ocp
    #     save_path = f"graphs/{biorbd_model_path[7:-7]}_torque_driven_1phase_deterministic_plot"
    #     results_path = f"results/{biorbd_model_path[7:-7]}_torque_driven_1phase_ocp.pkl"
    # elif name == "collocations":
    #     from seg5_torque_driven_simulated import prepare_ocp
    #     save_path = f"graphs/{biorbd_model_path[7:-7]}_torque_driven_1phase_simulated_plot"
    #     results_path = f"results/Model2D_7Dof_1C_3M_torque_driven_1phase_simulated_noise10_weight100_random30.pkl"
    # else:
    #     raise RuntimeError("Wrong ocp_type")
    #
    # model = biorbd.Model(biorbd_model_path)
    #
    # ocp = prepare_ocp(biorbd_model_path=biorbd_model_path,
    #                   final_time=final_time,
    #                   n_shooting=n_shooting)
    #
    # states_integrated, controls_noised = get_integrated_states(ocp, q_sol, qdot_sol, tau_sol, time_sol, nb_random=30)
    #
    # plot_q(q_sol, states_integrated, final_time, n_shooting, DoF_names, name, nb_random=30)
    # plot_CoM(states_integrated, model, n_shooting, name, nb_random=30)
