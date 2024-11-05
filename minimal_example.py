

dt = tf / n_shooting
h = dt / n_steps
x_integrated = np.zeros((n_q, n_shooting + 1))
x_integrated[:, 0] = x0
for i_shooting in range(n_shooting):
    x_this_time = x_integrated[:, i_shooting]
    u_this_time = u[:, i_shooting]
    current_time = dt*i_shooting
    for i_step in range(n_steps):
        k1 = dynamics(x_this_time, u_this_time, current_time)
        k2 = dynamics(x_this_time + h / 2 * k1, u_this_time, current_time + h / 2)
        k3 = dynamics(x_this_time + h / 2 * k2, u_this_time, current_time + h / 2)
        k4 = dynamics(x_this_time + h * k3, u_this_time, current_time + h)
        x_this_time += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        current_time += h
    x_integrated[:, i_shooting + 1] = x_this_time


