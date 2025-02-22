import numpy as np


def Euler(g, x0, t0, T, dt):
    """
    Numerical solving of DE by the Euler method
    :param g: function name g(t, x) = dx/dt
    :param x0: x0 = x(t0)
    :param t0: t_start
    :param T: t_stop
    :param dt: t_step
    :return: tuple {t[i]} (np.array), {x[i]} (np.array)
    """
    time = np.arange(t0, T, dt)
    x = np.empty(time.shape[0])
    x[0] = x0
    for t in range(0, time.shape[0] - 1):
        x[t+1] = x[t] + dt * g(t, x[t])

    return time, x


def RK4(g, x0, t0, T, dt):
    """
    Numerical solving of DE by the Runge-Kutta method (4th order)
    :param g: function name g(t, x) = dx/dt
    :param x0: x0 = x(t0)
    :param t0: t_start
    :param T: t_stop
    :param dt: t_step
    :return: tuple {t[i]} (np.array), {x[i]} (np.array)
    """
    time = np.arange(t0, T, dt)
    x = np.empty(time.shape[0])
    x[0] = x0

    for t in range(0, time.shape[0] - 1):
        k1 = dt * g(t, x[t])
        k2 = dt * g(t + dt / 2, x[t] + k1 / 2)
        k3 = dt * g(t + dt / 2, x[t] + k2 / 2)
        k4 = dt * g(t + dt, x[t] + k3)
        x[t + 1] = x[t] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return time, x
