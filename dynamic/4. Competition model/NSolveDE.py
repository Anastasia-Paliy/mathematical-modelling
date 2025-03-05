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


def Euler_i(g, x_t, t, dt, g_args):
    """
    Iteration step of the Euler method
    :param g: function name g(t, x, *args) = dx/dt
    :param x_t: x(t)
    :param t: t
    :param dt: t_step
    :param g_args: additional positional arguments of the function g
    :return: x(t+dt)
    """
    x_next = x_t + dt * g(t, x_t, g_args)

    return x_next


def RK4_i(g, x_t, t, dt, g_args):
    """
    Iteration step of the Runge-Kutta method (4th order)
    :param g: function name g(t, x, *args) = dx/dt
    :param x_t: x(t)
    :param t: t
    :param dt: t_step
    :param g_args: additional positional arguments of the function g (tuple)
    :return: x(t+dt)
    """
    k1 = dt * g(t, x_t, g_args)
    k2 = dt * g(t + dt / 2, x_t + k1 / 2, g_args)
    k3 = dt * g(t + dt / 2, x_t + k2 / 2, g_args)
    k4 = dt * g(t + dt, x_t + k3, g_args)
    x_next = x_t + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_next


def Euler_auto_system_XY(f, g, x_t, y_t, dt, f_args, g_args):
    """
    Numerical solving of a two-dimensional autonomous DE system by the Euler method
    :param f: function name f(x, y, *args) = dx/dt
    :param g: function name g(x, y, *args) = dy/dt
    :param x_t: x(t)
    :param y_t: y(t)
    :param dt: t_step
    :param f_args: additional positional arguments of the function f (iterable object)
    :param g_args: additional positional arguments of the function g (iterable object)
    :return: tuple (x(t+dt), y(t+dt))
    """
    x_next = x_t + dt * f(x_t, y_t, f_args)
    y_next = y_t + dt * g(x_t, y_t, g_args)

    return x_next, y_next


def RK4_auto_system_XY(f, g, x_t, y_t, dt, f_args, g_args):
    """
    Numerical solving of a two-dimensional autonomous DE system by the Runge-Kutta method (4th order)
    :param f: function name f(t, x, y, *args) = dx/dt
    :param g: function name g(t, x, y, *args) = dy/dt
    :param x_t: x(t)
    :param y_t: y(t)
    :param dt: t_step
    :param f_args: additional positional arguments of the function f (listable object)
    :param g_args: additional positional arguments of the function g (listable object)
    :return: tuple (x(t+dt), y(t+dt))
    """
    k1x = dt * f(x_t, y_t, f_args)
    k1y = dt * g(x_t, y_t, g_args)

    k2x = dt * f(x_t + k1x / 2, y_t + k1y / 2, f_args)
    k2y = dt * g(x_t + k1x / 2, y_t + k1y / 2, g_args)

    k3x = dt * f(x_t + k2x / 2, y_t + k2y / 2, f_args)
    k3y = dt * g(x_t + k2x / 2, y_t + k2y / 2, g_args)

    k4x = dt * f(x_t + k3x, y_t + k3y, f_args)
    k4y = dt * g(x_t + k3x, y_t + k3y, g_args)

    x_next = x_t + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    y_next = y_t + (k1y + 2 * k2y + 2 * k3y + k4y) / 6

    return x_next, y_next


def Euler_auto_system_XYZ(f, g, h, x_t, y_t, z_t, dt, f_args, g_args, h_args):
    """
    Numerical solving of a three-dimensional autonomous DE system by the Euler method
    :param f: function name f(x, y, z, *args) = dx/dt
    :param g: function name g(x, y, z, *args) = dy/dt
    :param h: function name h(x, y, z, *args) = dy/dt
    :param x_t: x(t)
    :param y_t: y(t)
    :param z_t: z(t)
    :param dt: t_step
    :param f_args: additional positional arguments of the function f (iterable object)
    :param g_args: additional positional arguments of the function g (iterable object)
    :param h_args: additional positional arguments of the function h (iterable object)
    :return: tuple (x(t+dt), y(t+dt), z(t+dt))
    """
    x_next = x_t + dt * f(x_t, y_t, z_t, f_args)
    y_next = y_t + dt * g(x_t, y_t, z_t, g_args)
    z_next = z_t + dt * h(x_t, y_t, z_t, h_args)

    return x_next, y_next, z_next


def RK4_auto_system_XYZ(f, g, h, x_t, y_t, z_t, dt, f_args, g_args, h_args):
    """
    Numerical solving of a three-dimensional autonomous DE system by the Runge-Kutta method (4th order)
    :param f: function name f(x, y, z, *args) = dx/dt
    :param g: function name g(x, y, z, *args) = dy/dt
    :param h: function name h(x, y, z, *args) = dy/dt
    :param x_t: x(t)
    :param y_t: y(t)
    :param z_t: z(t)
    :param dt: t_step
    :param f_args: additional positional arguments of the function f (iterable object)
    :param g_args: additional positional arguments of the function g (iterable object)
    :param h_args: additional positional arguments of the function h (iterable object)
    :return: tuple (x(t+dt), y(t+dt), z(t+dt))
    """
    k1x = dt * f(x_t, y_t, z_t, f_args)
    k1y = dt * g(x_t, y_t, z_t, g_args)
    k1z = dt * h(x_t, y_t, z_t, h_args)

    k2x = dt * f(x_t + k1x / 2, y_t + k1y / 2, z_t + k1z / 2, f_args)
    k2y = dt * g(x_t + k1x / 2, y_t + k1y / 2, z_t + k1z / 2, g_args)
    k2z = dt * h(x_t + k1x / 2, y_t + k1y / 2, z_t + k1z / 2, h_args)

    k3x = dt * f(x_t + k2x / 2, y_t + k2y / 2, z_t + k2z / 2, f_args)
    k3y = dt * g(x_t + k2x / 2, y_t + k2y / 2, z_t + k2z / 2, g_args)
    k3z = dt * h(x_t + k2x / 2, y_t + k2y / 2, z_t + k2z / 2, h_args)

    k4x = dt * f(x_t + k3x, y_t + k3y, z_t + k3z, f_args)
    k4y = dt * g(x_t + k3x, y_t + k3y, z_t + k3z, g_args)
    k4z = dt * h(x_t + k3x, y_t + k3y, z_t + k3z, h_args)

    x_next = x_t + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    y_next = y_t + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    z_next = z_t + (k1z + 2 * k2z + 2 * k3z + k4z) / 6

    return x_next, y_next, z_next
