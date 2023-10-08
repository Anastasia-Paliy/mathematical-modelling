import numpy as np


# Numerical solving of DE by the Euler method
# dx/dt = g(t, x)
def Euler(g, x0, t0, T, dt):
    time = np.arange(t0, T, dt)
    x = np.empty(time.shape[0])
    x[0] = x0
    for t in range(0, time.shape[0]-1):
        x[t+1] = x[t] + dt * g(t, x[t])

    return time, x


# Numerical solving of DE by the Runge-Kutta method (4th order)
# dx/dt = g(t, x)
def RK4(g, x0, t0, T, dt):
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
