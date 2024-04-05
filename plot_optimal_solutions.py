import pickle

import biorbd_casadi as biorbd
import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

from DMS_deterministic import prepare_ocp
from bioptim import StochasticBioModel
from utils import DMS_sensory_reference


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

def OCP_dynamics(q, qdot, tau, motor_noise_numerical, ocp):

    nb_root = 3
    nb_q = 7

    dxdt = cas.MX.zeros((2 * nb_q, 1))
    dxdt[:nb_q, :] = qdot

    # Joint friction
    tau_this_time = tau + ocp.nlp[0].model.friction_coefficients @ qdot[nb_root:]

    # Motor noise
    tau_this_time += motor_noise_numerical

    tau_this_time = cas.vertcat(cas.MX.zeros(nb_root), tau_this_time)

    dxdt[nb_q:] = ocp.nlp[0].model.forward_dynamics(q, qdot, tau_this_time)

    return dxdt

def RK4_SOCP(q, qdot, tau, dt, k_matrix, ref, motor_noise_numerical, sensory_noise_numerical, n_random, dyn_fun):

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


def RK4_OCP(q, qdot, tau, dt, motor_noise_numerical, dyn_fun):

    nb_q = 7
    states = np.zeros((2 * nb_q, 6))
    states[:nb_q, 0] = np.reshape(q, (-1, ))
    states[nb_q:, 0] = np.reshape(qdot, (-1, ))
    h = dt / 5
    for i in range(1, 6):
        k1 = dyn_fun(
            states[:nb_q, i - 1],
            states[nb_q:, i - 1],
            tau,
            motor_noise_numerical,
        )
        k2 = dyn_fun(
            states[:nb_q, i - 1] + h / 2 * k1[:nb_q],
            states[nb_q:, i - 1] + h / 2 * k1[nb_q:],
            tau,
            motor_noise_numerical,
        )
        k3 = dyn_fun(
            states[:nb_q, i - 1] + h / 2 * k2[:nb_q],
            states[nb_q:, i - 1] + h / 2 * k2[nb_q:],
            tau,
            motor_noise_numerical,
        )
        k4 = dyn_fun(
            states[:nb_q, i - 1] + h * k3[:nb_q],
            states[nb_q:, i - 1] + h * k3[nb_q:],
            tau,
            motor_noise_numerical,
        )
        states[:, i] = np.reshape(states[:, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (2*nb_q,))
    return states[:, -1]

def bioviz_animate(biorbd_model_path_with_mesh, q_roots, q_joints):
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
    b.load_movement(np.array(cas.vertcat(q_roots, q_joints)))
    b.exec()

def noisy_integrate(time_vector, q, qdot, tau_joints, n_shooting, motor_noise_magnitude, dyn_fun, nb_random):

    n_q = 7

    dt_last = time_vector[1] - time_vector[0]
    q_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
    q_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))

    # initial variability
    np.random.seed(0)
    initial_cov = np.eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P
    noised_states = np.random.multivariate_normal(np.zeros((n_q * 2, )), initial_cov, nb_random).T

    motor_noise_numerical = np.zeros((n_joints, nb_random, n_shooting + 1))
    for i_random in range(nb_random):
        q_integrated[:, 0, i_random] = np.reshape(q[:, 0] + noised_states[:n_q, i_random], (-1, ))
        qdot_integrated[:, 0, i_random] = np.reshape(qdot[:, 0] + noised_states[n_q:, i_random], (-1, ))
        q_multiple_shooting[:, 0, i_random] = np.reshape(q[:, 0] + noised_states[:n_q, i_random], (-1, ))
        qdot_multiple_shooting[:, 0, i_random] = np.reshape(qdot[:, 0] + noised_states[n_q:, i_random], (-1, ))
        for i_shooting in range(n_shooting):
            motor_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                loc=np.zeros(motor_noise_magnitude.shape[0]),
                scale=np.reshape(np.array(motor_noise_magnitude), (n_joints,)),
                size=n_joints,
            )

    for i_random in range(nb_random):
        for i_shooting in range(n_shooting):
            states_integrated = RK4_OCP(
                q_integrated[:, i_shooting, i_random],
                qdot_integrated[:, i_shooting, i_random],
                tau_joints[:, i_shooting],
                dt_last,
                motor_noise_numerical[:, i_random, i_shooting],
                dyn_fun,
            )
            q_integrated[:, i_shooting + 1, i_random] = states_integrated[:n_q]
            qdot_integrated[:, i_shooting + 1, i_random] = states_integrated[n_q:]

            states_integrated_multiple = RK4_OCP(
                q[:, i_shooting],
                qdot[:, i_shooting],
                tau_joints[:, i_shooting],
                dt_last,
                motor_noise_numerical[:, i_random, i_shooting],
                dyn_fun,
            )
            q_multiple_shooting[:, i_shooting + 1, i_random] = states_integrated_multiple[:n_q]
            qdot_multiple_shooting[:, i_shooting + 1, i_random] = states_integrated_multiple[n_q:]

    return q_integrated, qdot_integrated, q_multiple_shooting, qdot_multiple_shooting, motor_noise_numerical



model_name = "Model2D_7Dof_0C_3M"
biorbd_model_path = f"models/{model_name}.bioMod"
biorbd_model_path_with_mesh = f"models/{model_name}_with_mesh.bioMod"
biorbd_model_path_vision = f"models/{model_name}_vision.bioMod"
biorbd_model_path_vision_with_mesh = f"models/{model_name}_vision_with_mesh.bioMod"

ocp_path_to_results = f"results/good/{model_name}_ocp_DMS_CVG_1e-8.pkl"
socp_path_to_results = f"TODO"
socp_plus_path_to_results = f"TODO"

n_q = 7
n_root = 3
n_joints = n_q - n_root
n_ref = 2 * n_joints + 2

dt = 0.025
final_time = 0.5
n_shooting = int(final_time / dt)
tol = 1e-6
nb_random = 15

motor_noise_std = 0.05
wPq_std = 0.001
wPqdot_std = 0.003
motor_noise_magnitude = cas.DM(np.array([motor_noise_std**2 / dt for _ in range(n_q - n_root)]))  # All DoFs except root

print_motor_noise_std = "{:.1e}".format(motor_noise_std)
print_wPq_std = "{:.1e}".format(wPq_std)
print_wPqdot_std = "{:.1e}".format(wPqdot_std)
print_tol = "{:.1e}".format(tol)

# ------------- symbolics ------------- #
Q = cas.MX.sym("Q", n_q)
Qdot = cas.MX.sym("Qdot", n_q)
Tau = cas.MX.sym("Tau", n_joints)
MotorNoise = cas.MX.sym("Motor_noise", n_joints)

q_sym = cas.MX.sym("Q", n_q, nb_random)
qdot_sym = cas.MX.sym("Qdot", n_q, nb_random)
tau_sym = cas.MX.sym("Tau", n_joints)
k_matrix_sym = cas.MX.sym("k_matrix", n_joints, n_ref)
ref_sym = cas.MX.sym("Ref", n_ref)
motor_noise_sym = cas.MX.sym("Motor_noise", n_joints, nb_random)
sensory_noise_sym = cas.MX.sym("sensory_noise", n_ref, nb_random)
# ------------------------------------- #


# OCP
with open(ocp_path_to_results, "rb") as file:
    data = pickle.load(file)
    q_roots_ocp = data["q_roots_sol"]
    q_joints_ocp = data["q_joints_sol"]
    qdot_roots_ocp = data["qdot_roots_sol"]
    qdot_joints_ocp = data["qdot_joints_sol"]
    tau_joints_ocp = data["tau_joints_sol"]
    time_ocp = data["time_sol"]

ocp = prepare_ocp(biorbd_model_path=biorbd_model_path, time_last=final_time, n_shooting=n_shooting)

# print(ocp_path_to_results)
# bioviz_animate(biorbd_model_path_with_mesh, q_roots_ocp, q_joints_ocp)

time_vector = np.linspace(0, float(time_ocp), n_shooting + 1)
biorbd_model = biorbd.Model(biorbd_model_path)
dyn_fun = cas.Function("dynamics", [Q, Qdot, Tau, MotorNoise], [OCP_dynamics(Q, Qdot, Tau, MotorNoise, ocp)])
q_ocp_integrated, qdot_ocp_integrated, q_ocp_multiple_shooting, qdot_ocp_multiple_shooting, motor_noise_numerical = noisy_integrate(time_vector, cas.vertcat(q_roots_ocp, q_joints_ocp), cas.vertcat(qdot_roots_ocp, qdot_joints_ocp), tau_joints_ocp, n_shooting, motor_noise_magnitude, dyn_fun, nb_random)

ocp_out_path_to_results = ocp_path_to_results.replace(".pkl", "_integrated.pkl")
with open(ocp_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_ocp_integrated,
        "qdot_integrated": qdot_ocp_integrated,
        "q_multiple_shooting": q_ocp_multiple_shooting,
        "qdot_multiple_shooting": qdot_ocp_multiple_shooting,
        "motor_noise_numerical": motor_noise_numerical,
        "time_vector": time_vector,
        "q_nominal": cas.vertcat(q_roots_ocp, q_joints_ocp),
    }
    pickle.dump(data, file)


# SOCP
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

# import bioviz
# b = bioviz.Viz(biorbd_model_path_with_mesh,
#                background_color=(1, 1, 1),
#                show_local_ref_frame=False,
#                show_markers=False,
#                show_segments_center_of_mass=False,
#                show_global_center_of_mass=False,
#                show_global_ref_frame=False,
#                show_gravity_vector=False,
#                )
# b.load_movement(q_mean_last)
# b.exec()


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
            axs[i_dof].plot(time_vector, q_integrated[i_dof, :, i_random], "--", color="r", label="Reintegrated states")
            is_label_dof_set = True
        else:
            axs[i_dof].plot(time_vector, q_last[i_dof, :, i_random], color="k")
            axs[i_dof].plot(time_vector, q_integrated[i_dof, :, i_random], "--", color="r")
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



