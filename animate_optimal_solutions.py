import pickle

import sys
sys.path.append("/home/charbie/Documents/Programmation/pyorerun/")
from pyorerun import BiorbdModel, MultiPhaseRerun


def noisy_animate(model_name, q_integrated, time_vector, q_mean_last, animation_name):

    biorbd_model = BiorbdModel(model_name)
    rerun_biorbd = MultiPhaseRerun()

    rerun_biorbd.add_phase(t_span=time_vector, phase=0, window="animation")
    for i_random in range(q_integrated.shape[2]):
        rerun_biorbd.add_animated_model(biorbd_model, q_integrated[:, :, i_random], phase=0, window="animation")

    red_model = BiorbdModel(model_name)
    red_model.options.mesh_color = (1, 0, 0)

    rerun_biorbd.add_animated_model(red_model, q_mean_last, phase=0, window="animation")
    rerun_biorbd.rerun(animation_name)



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


# OCP
ocp_out_path_to_results = ocp_path_to_results.replace(".pkl", "_integrated.pkl")
with open(ocp_path_to_results, "rb") as file:
    data = pickle.load(file)
    q_ocp_integrated = data["q_integrated"]
    qdot_ocp_integrated = data["qdot_integrated"]
    q_ocp_multiple_shooting = data["q_multiple_shooting"]
    qdot_ocp_multiple_shooting = data["qdot_multiple_shooting"]
    motor_noise_numerical_ocp = data["motor_noise_numerical"]
    time_vector = data["time_vector"]
    q_nominal = data["q_nominal"]

noisy_animate(biorbd_model_path_with_mesh, q_ocp_integrated, time_vector, q_nominal, "OCP")
