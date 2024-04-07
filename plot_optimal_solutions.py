import pickle

import bioviz
import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

from DMS_SOCP import prepare_socp
from DMS_SOCP_VARIABLE_FEEDFORWARD import prepare_socp_VARIABLE_FEEDFORWARD
from DMS_deterministic import prepare_ocp
from bioptim import StochasticBioModel
from utils import DMS_sensory_reference, motor_acuity, DMS_fb_noised_sensory_input_VARIABLE_no_eyes, \
    DMS_ff_noised_sensory_input, DMS_sensory_reference_no_eyes


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


def SOCP_dynamics(nb_random, q, qdot, tau, k_matrix, ref, motor_noise_numerical, sensory_noise_numerical, socp):

    nb_root = 3
    nb_q = 7

    dxdt = cas.MX.zeros((2 * nb_q, nb_random))
    dxdt[:nb_q, :] = qdot
    for i_random in range(nb_random):

        q_this_time = q[:, i_random]
        qdot_this_time = qdot[:, i_random]

        tau_this_time = tau[:]

        # Joint friction
        tau_this_time += socp.nlp[0].model.friction_coefficients @ qdot_this_time[nb_root:]

        # Motor noise
        tau_this_time += motor_noise_numerical[:, i_random]

        # Feedback
        tau_this_time += k_matrix @ (
            ref
            - DMS_sensory_reference(socp.nlp[0].model, nb_root, q_this_time, qdot_this_time)
            + sensory_noise_numerical[:, i_random]
        )
        tau_this_time = cas.vertcat(cas.MX.zeros(nb_root), tau_this_time)

        dxdt[nb_q:, i_random] = socp.nlp[0].model.forward_dynamics(q_this_time, qdot_this_time, tau_this_time)

    return dxdt


def SOCP_PLUS_dynamics(nb_random, q, qdot, tau, k_fb, k_ff, fb_ref, ff_ref, tf, time, motor_noise_numerical, sensory_noise_numerical, socp_plus):

    nb_root = 3
    nb_q = 8

    dxdt = cas.MX.zeros((2 * nb_q, nb_random))
    dxdt[:nb_q, :] = qdot
    for i_random in range(nb_random):

        q_this_time = q[:, i_random]
        qdot_this_time = qdot[:, i_random]

        tau_this_time = tau[:]

        # Joint friction
        tau_this_time += socp_plus.nlp[0].model.friction_coefficients @ qdot_this_time[nb_root:]

        # Motor noise
        tau_this_time += motor_acuity(motor_noise_numerical[:, i_random], tau)

        # Feedback
        tau_this_time += k_fb @ (
            fb_ref
            - DMS_fb_noised_sensory_input_VARIABLE_no_eyes(
                socp_plus.nlp[0].model,
                q_this_time[:nb_root],
                q_this_time[nb_root:],
                qdot_this_time[:nb_root],
                qdot_this_time[nb_root:],
                sensory_noise_numerical[: socp_plus.nlp[0].model.n_feedbacks, i_random],
            )
        )

        # Feedforward
        tau_this_time += k_ff @ (
            ff_ref
            - DMS_ff_noised_sensory_input(
                socp_plus.nlp[0].model, tf, time, q_this_time, qdot_this_time, sensory_noise_numerical[socp_plus.nlp[0].model.n_feedbacks :, i_random]
            )
        )

        tau_this_time = cas.vertcat(cas.MX.zeros(nb_root), tau_this_time)

        dxdt[nb_q:, i_random] = socp_plus.nlp[0].model.forward_dynamics(q_this_time, qdot_this_time, tau_this_time)

    return dxdt

def OCP_dynamics(q, qdot, tau, motor_noise_numerical, ocp):

    nb_root = 3
    nb_q = qdot.shape[0]

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

def bioviz_animate(biorbd_model_path_with_mesh, q, name):
    b = bioviz.Viz(biorbd_model_path_with_mesh,
                   mesh_opacity=1.0,
                   background_color=(1, 1, 1),
                   show_local_ref_frame=False,
                   show_markers=False,
                   show_segments_center_of_mass=False,
                   show_global_center_of_mass=False,
                   show_global_ref_frame=False,
                   show_gravity_vector=False,
                   )
    b.set_camera_zoom(0.5)
    b.maximize()
    b.update()
    b.load_movement(q)

    b.start_recording(f"videos/{result_folder}/" + name + ".ogv")
    for frame in range(q.shape[1] + 1):
        b.movement_slider[0].setValue(frame)
        b.add_frame()
    b.stop_recording()


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

def integrate(time_vector, q_socp, qdot_socp, tau_joints_socp, k_socp, ref_socp, motor_noise_numerical, sensory_noise_numerical, dyn_fun, socp):

    dt_socp = time_vector[1] - time_vector[0]
    n_q = q_socp.shape[0]
    n_shooting = q_socp.shape[1] - 1
    nb_random = q_socp.shape[2]

    # single shooting
    q_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot_integrated = np.zeros((n_q, n_shooting + 1, nb_random))
    q_integrated[:, 0, :] = q_socp[:, 0, :]
    qdot_integrated[:, 0, :] = qdot_socp[:, 0, :]

    # multiple shooting
    q_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))
    qdot_multiple_shooting = np.zeros((n_q, n_shooting + 1, nb_random))
    q_multiple_shooting[:, 0, :] = q_socp[:, 0, :]
    qdot_multiple_shooting[:, 0, :] = qdot_socp[:, 0, :]

    for i_shooting in range(n_shooting):
        k_matrix = StochasticBioModel.reshape_to_matrix(k_socp[:, i_shooting], socp.nlp[0].model.matrix_shape_k)
        states_integrated = RK4_SOCP(
            q_integrated[:, i_shooting, :],
            qdot_integrated[:, i_shooting, :],
            tau_joints_socp[:, i_shooting],
            dt_socp,
            k_matrix,
            ref_socp[:, i_shooting],
            motor_noise_numerical[:, :, i_shooting],
            sensory_noise_numerical[:, :, i_shooting],
            nb_random,
            dyn_fun,
        )
        q_integrated[:, i_shooting + 1, :] = states_integrated[:n_q, :]
        qdot_integrated[:, i_shooting + 1, :] = states_integrated[n_q:, :]

        states_integrated_multiple = RK4_SOCP(
            q_socp[:, i_shooting, :],
            qdot_socp[:, i_shooting, :],
            tau_joints_socp[:, i_shooting],
            dt_socp,
            k_matrix,
            ref_socp[:, i_shooting],
            motor_noise_numerical[:, :, i_shooting],
            sensory_noise_numerical[:, :, i_shooting],
            nb_random,
            dyn_fun,
        )
        q_multiple_shooting[:, i_shooting + 1, :] = states_integrated_multiple[:n_q, :]
        qdot_multiple_shooting[:, i_shooting + 1, :] = states_integrated_multiple[n_q:, :]

    return q_integrated, qdot_integrated, q_multiple_shooting, qdot_multiple_shooting, motor_noise_numerical



FLAG_GENERATE_VIDEOS = False

OCP_color = "#5dc962"
SOCP_color = "#ac2594"
SOCP_plus_color = "#06b0f0"
model_name = "Model2D_7Dof_0C_3M"
biorbd_model_path = f"models/{model_name}.bioMod"
biorbd_model_path_with_mesh_ocp = f"models/{model_name}_with_mesh_ocp.bioMod"
biorbd_model_path_with_mesh_socp = f"models/{model_name}_with_mesh_socp.bioMod"
biorbd_model_path_with_mesh_all = f"models/{model_name}_with_mesh_all.bioMod"
biorbd_model_path_with_mesh_all_socp = f"models/{model_name}_with_mesh_all_socp.bioMod"

biorbd_model_path_vision = f"models/{model_name}_vision.bioMod"
biorbd_model_path_vision_with_mesh = f"models/{model_name}_vision_with_mesh.bioMod"
biorbd_model_path_vision_with_mesh_all = f"models/{model_name}_vision_with_mesh_all.bioMod"

# TODO: change the path to the results
result_folder = "2p5pi"  # "good"
ocp_path_to_results = f"results/{result_folder}/{model_name}_ocp_DMS_CVG_1e-8.pkl"
socp_path_to_results = f"results/{result_folder}/{model_name}_socp_DMS_5p0e-02_1p0e-03_3p0e-03_DMS_15random_CVG_1p0e-06.pkl"
socp_plus_path_to_results = f"results/{result_folder}/{model_name}_socp_DMS_5p0e-02_1p0e-03_3p0e-03_VARIABLE_FEEDFORWARD_VARIABLE_FEEDFORWARD_DMS_15random_CVG_1p0e-03.pkl"

n_q = 7
n_root = 3
n_joints = n_q - n_root
n_ref = 2 * n_joints + 2

# TODO change the values
# dt = 0.025
# final_time = 0.5
dt = 0.05
final_time = 0.8
n_shooting = int(final_time / dt)
tol = 1e-6
nb_random = 15

motor_noise_std = 0.05 * 10
wPq_std = 0.001 * 10
wPqdot_std = 0.003 * 10
motor_noise_magnitude = cas.DM(np.array([motor_noise_std**2 / dt for _ in range(n_q - n_root)]))  # All DoFs except root

# ------------- symbolics ------------- #
Q = cas.MX.sym("Q", n_q)
Qdot = cas.MX.sym("Qdot", n_q)
Tau = cas.MX.sym("Tau", n_joints)
MotorNoise = cas.MX.sym("Motor_noise", n_joints)

Q_8 = cas.MX.sym("Q", n_q+1)
Qdot_8 = cas.MX.sym("Qdot", n_q+1)
Tau_8 = cas.MX.sym("Tau", n_joints+1)
MotorNoise_8 = cas.MX.sym("Motor_noise", n_joints+1)

q_sym = cas.MX.sym("Q", n_q, nb_random)
qdot_sym = cas.MX.sym("Qdot", n_q, nb_random)
tau_sym = cas.MX.sym("Tau", n_joints)
k_matrix_sym = cas.MX.sym("k_matrix", n_joints, n_ref)
ref_sym = cas.MX.sym("Ref", 2 * n_joints + 2)
motor_noise_sym = cas.MX.sym("Motor_noise", n_joints, nb_random)
sensory_noise_sym = cas.MX.sym("sensory_noise", n_ref, nb_random)
time_sym = cas.MX.sym("Time", 1)

q_8_sym = cas.MX.sym("Q", n_q+1, nb_random)
qdot_8_sym = cas.MX.sym("Qdot", n_q+1, nb_random)
tau_8_sym = cas.MX.sym("Tau", n_joints+1)
k_fb_matrix_sym = cas.MX.sym("k_matrix_fb", n_joints+1, n_ref)
k_ff_matrix_sym = cas.MX.sym("k_ff_matrix", n_joints+1, 1)
fb_ref_sym = cas.MX.sym("fb_ref", 2 * n_joints + 2)
ff_ref_sym = cas.MX.sym("ff_ref", 1)
motor_noise_8_sym = cas.MX.sym("Motor_noise", n_joints+1, nb_random)
sensory_noise_8_sym = cas.MX.sym("sensory_noise", 2 * n_joints + 3, nb_random)

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

if FLAG_GENERATE_VIDEOS:
    print("Generating OCP_one : ", ocp_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_ocp, np.vstack((q_roots_ocp, q_joints_ocp)), "OCP_one")

time_vector = np.linspace(0, float(time_ocp), n_shooting + 1)
dyn_fun_ocp = cas.Function("dynamics", [Q, Qdot, Tau, MotorNoise], [OCP_dynamics(Q, Qdot, Tau, MotorNoise, ocp)])
q_ocp_integrated, qdot_ocp_integrated, q_ocp_multiple_shooting, qdot_ocp_multiple_shooting, motor_noise_numerical = noisy_integrate(time_vector, cas.vertcat(q_roots_ocp, q_joints_ocp), cas.vertcat(qdot_roots_ocp, qdot_joints_ocp), tau_joints_ocp, n_shooting, motor_noise_magnitude, dyn_fun_ocp, nb_random)

ocp_out_path_to_results = ocp_path_to_results.replace(".pkl", "_integrated.pkl")
with open(ocp_out_path_to_results, "wb") as file:
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

q_all_ocp = np.zeros((n_q * (nb_random+1), n_shooting + 1))
for i_shooting in range(n_shooting+1):
    for i_random in range(nb_random):
        q_all_ocp[i_random * n_q: (i_random + 1) * n_q, i_shooting] = q_ocp_integrated[:, i_shooting, i_random]
    q_all_ocp[(i_random +1) * n_q: (i_random + 2) * n_q, i_shooting] = np.hstack((np.array(q_roots_ocp[:, i_shooting]), np.array(q_joints_ocp[:, i_shooting])))

if FLAG_GENERATE_VIDEOS:
    print("Generating OCP_all : ", ocp_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_all, q_all_ocp, "OCP_all")


# SOCP
sensory_noise_magnitude = cas.DM(
    cas.vertcat(
        np.array([wPq_std**2 / dt for _ in range(n_q - n_root + 1)]),
        np.array([wPqdot_std**2 / dt for _ in range(n_q - n_root + 1)]),
    )
)  # since the head is fixed to the pelvis, the vestibular feedback is in the states ref

with open(socp_path_to_results, "rb") as file:
    data = pickle.load(file)
    q_roots_socp = data["q_roots_sol"]
    q_joints_socp = data["q_joints_sol"]
    qdot_roots_socp = data["qdot_roots_sol"]
    qdot_joints_socp = data["qdot_joints_sol"]
    tau_joints_socp = data["tau_joints_sol"]
    time_socp = data["time_sol"]
    k_socp = data["k_sol"]
    ref_socp = data["ref_sol"]
    motor_noise_numerical_socp = data["motor_noise_numerical"]
    sensory_noise_numerical_socp = data["sensory_noise_numerical"]


motor_noise_numerical_socp_10, sensory_noise_numerical_socp_10, socp = prepare_socp(biorbd_model_path=biorbd_model_path,
                    time_last=time_ocp,
                    n_shooting=n_shooting,
                    motor_noise_magnitude=motor_noise_magnitude,
                    sensory_noise_magnitude=sensory_noise_magnitude,
                    q_roots_last=q_roots_ocp,
                    q_joints_last=q_joints_ocp,
                    qdot_roots_last=qdot_roots_ocp,
                    qdot_joints_last=qdot_joints_ocp,
                    tau_joints_last=tau_joints_ocp,
                    k_last=None,
                    ref_last=None,
                    nb_random=nb_random)

time_vector = np.linspace(0, float(time_socp), n_shooting + 1)

out = DMS_sensory_reference(socp.nlp[0].model, n_root, Q, Qdot)
DMS_sensory_reference_func = cas.Function("DMS_sensory_reference", [Q, Qdot], [out])
dyn_fun_out = SOCP_dynamics(
    nb_random, q_sym, qdot_sym, tau_sym, k_matrix_sym, ref_sym, motor_noise_sym, sensory_noise_sym, socp
)
dyn_fun_socp = cas.Function(
    "dynamics", [q_sym, qdot_sym, tau_sym, k_matrix_sym, ref_sym, motor_noise_sym, sensory_noise_sym], [dyn_fun_out]
)
dyn_fun_socp_nominal = cas.Function("nominal_dyn", [Q, Qdot, Tau, k_matrix_sym, ref_sym, motor_noise_sym, sensory_noise_sym], [OCP_dynamics(Q, Qdot, Tau, np.zeros(MotorNoise.shape), ocp)])


q_socp = np.zeros((n_q, n_shooting + 1, nb_random))
qdot_socp = np.zeros((n_q, n_shooting + 1, nb_random))
for i_random in range(nb_random):
    for i_shooting in range(n_shooting + 1):
        q_socp[:, i_shooting, i_random] = np.hstack((q_roots_socp[i_random*n_root:(i_random+1)*n_root, i_shooting],
                                                      q_joints_socp[i_random*n_joints:(i_random+1)*n_joints, i_shooting]))
        qdot_socp[:, i_shooting, i_random] = np.hstack((qdot_roots_socp[i_random*n_root:(i_random+1)*n_root, i_shooting],
                                                            qdot_joints_socp[i_random*n_joints:(i_random+1)*n_joints, i_shooting]))
(q_socp_integrated,
 qdot_socp_integrated,
 q_socp_multiple_shooting,
 qdot_socp_multiple_shooting,
 motor_noise_numerical_socp) = integrate(time_vector,
                                    q_socp,
                                    qdot_socp,
                                    tau_joints_socp,
                                    k_socp,
                                    ref_socp,
                                    motor_noise_numerical_socp_10,
                                    sensory_noise_numerical_socp_10,
                                    dyn_fun_socp,
                                    socp)

(q_socp_nominal,
 qdot_socp_nominal,
 _,
 _,
 _) = integrate(time_vector,
                np.mean(q_socp, axis=2)[:, :, np.newaxis],
                np.mean(qdot_socp, axis=2)[:, :, np.newaxis],
                tau_joints_socp,
                np.zeros(np.shape(k_socp)),
                np.zeros(np.shape(ref_socp)),
                np.zeros(np.shape(motor_noise_numerical_socp_10)),
                np.zeros(np.shape(sensory_noise_numerical_socp_10)),
                dyn_fun_socp_nominal,
                socp)

if FLAG_GENERATE_VIDEOS:
    # TODO: fix this integration issue ?
    print("Generating SOCP_one : ", socp_path_to_results)
    # bioviz_animate(biorbd_model_path_with_mesh_socp, q_socp_nominal[:, :, 0], "SOCP_one")
    bioviz_animate(biorbd_model_path_with_mesh_socp, np.mean(q_socp_integrated, axis=2), "SOCP_one")

socp_out_path_to_results = socp_path_to_results.replace(".pkl", "_integrated.pkl")
with open(socp_out_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_socp_integrated,
        "qdot_integrated": qdot_socp_integrated,
        "q_multiple_shooting": q_socp_multiple_shooting,
        "qdot_multiple_shooting": qdot_socp_multiple_shooting,
        "motor_noise_numerical": motor_noise_numerical_socp,
        "time_vector": time_vector,
        "q_mean_integrated": np.mean(q_socp_integrated, axis=2),
        "q_nominal": q_socp_nominal,
    }
    pickle.dump(data, file)

q_all_socp = np.zeros((n_q * (nb_random + 1), n_shooting + 1))
for i_shooting in range(n_shooting+1):
    for i_random in range(nb_random):
        q_all_socp[i_random * n_q: (i_random + 1) * n_q, i_shooting] = np.reshape(q_socp_integrated[:, i_shooting, i_random], (-1, ))
    q_all_socp[(i_random + 1) * n_q: (i_random + 2) * n_q, i_shooting] = np.reshape(q_socp_nominal[:, i_shooting], (-1, ))

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_all : ", socp_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_all_socp, q_all_socp, "SOCP_all")


# SOCP+
n_q = 8
n_root = 3
n_joints = n_q - n_root
motor_noise_magnitude = cas.DM(
    np.array(
        [
            motor_noise_std ** 2 / dt,
            0.0,
            motor_noise_std ** 2 / dt,
            motor_noise_std ** 2 / dt,
            motor_noise_std ** 2 / dt,
        ]
    )
)  # All DoFs except root
sensory_noise_magnitude = cas.DM(
    np.array(
        [
            wPq_std ** 2 / dt,  # Proprioceptive position
            wPq_std ** 2 / dt,
            wPq_std ** 2 / dt,
            wPq_std ** 2 / dt,
            wPqdot_std ** 2 / dt,  # Proprioceptive velocity
            wPqdot_std ** 2 / dt,
            wPqdot_std ** 2 / dt,
            wPqdot_std ** 2 / dt,
            wPq_std ** 2 / dt,  # Vestibular position
            wPq_std ** 2 / dt,  # Vestibular velocity
            wPq_std ** 2 / dt,  # Visual
        ]
    )
)

motor_noise_magnitude *= 10
sensory_noise_magnitude *= 10

with open(socp_plus_path_to_results, "rb") as file:
    data = pickle.load(file)
    q_roots_socp_plus = data["q_roots_sol"]
    q_joints_socp_plus = data["q_joints_sol"]
    qdot_roots_socp_plus = data["qdot_roots_sol"]
    qdot_joints_socp_plus = data["qdot_joints_sol"]
    tau_joints_socp_plus = data["tau_joints_sol"]
    time_socp_plus = data["time_sol"]
    k_socp_plus = data["k_sol"]
    ref_socp_plus = data["ref_sol"]
    motor_noise_numerical_socp_plus = data["motor_noise_numerical"]
    sensory_noise_numerical_socp_plus = data["sensory_noise_numerical"]


q_joints_last = np.vstack((q_joints_ocp[0, :], np.zeros((1, q_joints_ocp.shape[1])), q_joints_ocp[1:, :]))
qdot_joints_last = np.vstack(
    (qdot_joints_ocp[0, :], np.ones((1, qdot_joints_ocp.shape[1])) * 0.01, qdot_joints_ocp[1:, :])
)
tau_joints_last = np.vstack(
    (tau_joints_ocp[0, :], np.ones((1, tau_joints_ocp.shape[1])) * 0.01, tau_joints_ocp[1:, :])
)

motor_noise_numerical_socp_plus_10, sensory_noise_numerical_socp_plus_10, socp_plus = prepare_socp_VARIABLE_FEEDFORWARD(biorbd_model_path=biorbd_model_path_vision,
                    time_last=time_ocp,
                    n_shooting=n_shooting,
                    motor_noise_magnitude=motor_noise_magnitude,
                    sensory_noise_magnitude=sensory_noise_magnitude,
                    q_roots_last=q_roots_ocp,
                    q_joints_last=q_joints_last,
                    qdot_roots_last=qdot_roots_ocp,
                    qdot_joints_last=qdot_joints_last,
                    tau_joints_last=tau_joints_last,
                    k_last=None,
                    ref_last=None,
                    nb_random=nb_random)

time_vector = np.linspace(0, float(time_socp_plus), n_shooting + 1)

out = DMS_sensory_reference_no_eyes(socp_plus.nlp[0].model, n_root, Q_8, Qdot_8)
DMS_sensory_reference_func = cas.Function("DMS_sensory_reference", [Q_8, Qdot_8], [out])
dyn_fun_out = SOCP_PLUS_dynamics(
    nb_random, q_8_sym, qdot_8_sym, tau_8_sym, k_fb_matrix_sym, k_ff_matrix_sym, fb_ref_sym, ff_ref_sym, float(time_socp_plus), time_sym, motor_noise_8_sym, sensory_noise_8_sym, socp_plus
)
dyn_fun_socp_plus = cas.Function(
    "dynamics", [q_8_sym, qdot_8_sym, tau_8_sym, k_fb_matrix_sym, k_ff_matrix_sym, fb_ref_sym, ff_ref_sym, time_sym, motor_noise_8_sym, sensory_noise_8_sym], [dyn_fun_out]
)
dyn_fun_socp_plus_nominal = cas.Function("nominal_dyn",
                                         [Q_8, Qdot_8, Tau_8, k_fb_matrix_sym, k_ff_matrix_sym, fb_ref_sym, ff_ref_sym, time_sym, MotorNoise_8, sensory_noise_sym],
                                         [OCP_dynamics(Q_8, Qdot_8, Tau_8, np.zeros(MotorNoise_8.shape), socp_plus)])

q_socp_plus = np.zeros((n_q, n_shooting + 1, nb_random))
qdot_socp_plus = np.zeros((n_q, n_shooting + 1, nb_random))
for i_random in range(nb_random):
    for i_shooting in range(n_shooting + 1):
        q_socp_plus[:, i_shooting, i_random] = np.hstack((q_roots_socp_plus[i_random*n_root:(i_random+1)*n_root, i_shooting],
                                                      q_joints_socp_plus[i_random*n_joints:(i_random+1)*n_joints, i_shooting]))
        qdot_socp_plus[:, i_shooting, i_random] = np.hstack((qdot_roots_socp_plus[i_random*n_root:(i_random+1)*n_root, i_shooting],
                                                            qdot_joints_socp_plus[i_random*n_joints:(i_random+1)*n_joints, i_shooting]))
(q_socp_plus_integrated,
 qdot_socp_plus_integrated,
 q_socp_plus_multiple_shooting,
 qdot_socp_plus_multiple_shooting,
 motor_noise_numerical_socp_plus) = integrate(time_vector,
                                    q_socp_plus,
                                    qdot_socp_plus,
                                    tau_joints_socp_plus,
                                    k_socp_plus,
                                    ref_socp_plus,
                                    motor_noise_numerical_socp_plus_10,
                                    sensory_noise_numerical_socp_plus_10,
                                    dyn_fun_socp_plus,
                                    socp_plus)

(q_socp_plus_nominal,
    qdot_socp_plus_nominal,
    _,
    _,
    _) = integrate(time_vector,
                    np.mean(q_socp_plus, axis=2)[:, :, np.newaxis],
                    np.mean(qdot_socp_plus, axis=2)[:, :, np.newaxis],
                    tau_joints_socp_plus,
                    np.zeros(np.shape(k_socp_plus)),
                    np.zeros(np.shape(ref_socp_plus)),
                    np.zeros(np.shape(motor_noise_numerical_socp_plus)),
                    np.zeros(np.shape(sensory_noise_numerical_socp_plus)),
                    dyn_fun_socp_plus_nominal,
                    socp_plus)

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_plus_one : ", socp_plus_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_socp, q_socp_plus_nominal, "SOCP_plus_one")

socp_plus_out_path_to_results = socp_plus_path_to_results.replace(".pkl", "_integrated.pkl")
with open(socp_plus_out_path_to_results, "wb") as file:
    data = {
        "q_integrated": q_socp_plus_integrated,
        "qdot_integrated": qdot_socp_plus_integrated,
        "q_multiple_shooting": q_socp_plus_multiple_shooting,
        "qdot_multiple_shooting": qdot_socp_plus_multiple_shooting,
        "motor_noise_numerical": motor_noise_numerical_socp_plus,
        "time_vector": time_vector,
        "q_mean_integrated": np.mean(q_socp_plus_integrated, axis=2),
        "q_nominal": q_socp_plus_nominal,
    }
    pickle.dump(data, file)

q_all_socp_plus = np.zeros((n_q * (nb_random + 1), n_shooting + 1))
for i_shooting in range(n_shooting+1):
    for i_random in range(nb_random):
        q_all_socp_plus[i_random * n_q: (i_random + 1) * n_q, i_shooting] = q_socp_plus_integrated[:, i_shooting, i_random]
    q_all_socp_plus[(i_random + 1) * n_q: (i_random + 2) * n_q, i_shooting] = q_socp_plus_nominal[:, i_shooting]

if FLAG_GENERATE_VIDEOS:
    print("Generating SOCP_plus_all : ", socp_plus_path_to_results)
    bioviz_animate(biorbd_model_path_with_mesh_all_socp, q_all_socp_plus, "SOCP_plus_all")





is_label_dof_set = False
is_label_mean_set = False
is_label_ref_mean_set = False

fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs = np.ravel(axs)
q_socp = np.vstack((np.array(q_roots_socp[:n_root]), np.array(q_joints_socp[:n_joints])))[:, :, np.newaxis]
qdot_socp = np.vstack((np.array(qdot_roots_socp[:n_root]), np.array(qdot_joints_socp[:n_joints])))[:, :, np.newaxis]
for i_dof in range(n_q):
    axs[i_dof].plot(time_vector, q_socp[i_dof, :, 0], color="k", label="Noised states (optim variables)")
for i_random in range(1, nb_random):
    q_socp = np.concatenate(
        (
            q_socp,
            np.vstack(
                (
                    np.array(q_roots_socp[i_random * n_root : (i_random + 1) * n_root]),
                    np.array(q_joints_socp[i_random * n_joints : (i_random + 1) * n_joints]),
                )
            )[:, :, np.newaxis],
        ),
        axis=2,
    )
    qdot_socp = np.concatenate(
        (
            qdot_socp,
            np.vstack(
                (
                    np.array(qdot_roots_socp[i_random * n_root : (i_random + 1) * n_root]),
                    np.array(qdot_joints_socp[i_random * n_joints : (i_random + 1) * n_joints]),
                )
            )[:, :, np.newaxis],
        ),
        axis=2,
    )
    for i_dof in range(n_q):
        axs[i_dof].plot(time_vector, q_socp[i_dof, :, i_random], color="k")
q_mean_socp = np.mean(q_socp, axis=2)
qdot_mean_socp = np.mean(qdot_socp, axis=2)
for i_dof in range(n_q):
    if not is_label_mean_set:
        axs[i_dof].plot(time_vector, q_mean_socp[i_dof, :], "--", color="tab:red", label="Mean noised states")
        is_label_mean_set = True
    else:
        axs[i_dof].plot(time_vector, q_mean_socp[i_dof, :], "--", color="tab:red")
    axs[i_dof].set_title(f"DOF {i_dof}")
ref_mean_socp = np.zeros((n_ref, n_shooting))
for i_node in range(n_shooting):
    ref_mean_socp[:, i_node] = np.array(
        DMS_sensory_reference_func(q_mean_socp[:, i_node], qdot_mean_socp[:, i_node])
    ).reshape(-1)
if not is_label_ref_mean_set:
    axs[3].plot(time_vector[:-1], ref_mean_socp[0, :], color="tab:blue", label="Mean reference")
    axs[3].plot(time_vector[:-1], ref_socp[0, :], "--", color="tab:orange", label="Reference (optim variables)")
    is_label_ref_mean_set = True
else:
    axs[3].plot(time_vector[:-1], ref_mean_socp[0, :], color="tab:blue")
    axs[3].plot(time_vector[:-1], ref_socp[0, :], "--", color="tab:orange")
axs[4].plot(time_vector[:-1], ref_mean_socp[1, :], color="tab:blue")
axs[4].plot(time_vector[:-1], ref_socp[1, :], "--", color="tab:orange")
axs[5].plot(time_vector[:-1], ref_mean_socp[2, :], color="tab:blue")
axs[5].plot(time_vector[:-1], ref_socp[2, :], "--", color="tab:orange")
axs[6].plot(time_vector[:-1], ref_mean_socp[3, :], color="tab:blue")
axs[6].plot(time_vector[:-1], ref_socp[3, :], "--", color="tab:orange")
axs[0].legend()
axs[3].legend()
plt.show()



# Verify reintegration
is_label_dof_set = False
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs = np.ravel(axs)
for i_random in range(nb_random):
    for i_dof in range(n_q):
        if not is_label_dof_set:
            axs[i_dof].plot(time_vector, q_socp[i_dof, :, i_random], color="k", label="Noised states (optim variables)")
            axs[i_dof].plot(time_vector, q_socp_integrated[i_dof, :, i_random], "--", color="r", label="Reintegrated states")
            is_label_dof_set = True
        else:
            axs[i_dof].plot(time_vector, q_socp[i_dof, :, i_random], color="k")
            axs[i_dof].plot(time_vector, q_socp_integrated[i_dof, :, i_random], "--", color="r")
        for i_shooting in range(n_shooting):
            axs[i_dof].plot(
                np.array([time_vector[i_shooting], time_vector[i_shooting + 1]]),
                np.array([q_socp[i_dof, i_shooting, i_random], q_socp_multiple_shooting[i_dof, i_shooting + 1, i_random]]),
                "--",
                color="b",
            )
axs[0].legend()
plt.show()



