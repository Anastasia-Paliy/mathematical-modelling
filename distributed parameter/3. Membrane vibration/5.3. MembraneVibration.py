import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from decimal import Decimal
from math import sin, cos, pi


class MembraneVibration(object):
    def __init__(self, a: float, L: float, dx: float, M: float, dy: float):
        """
        Setting key params of wave equation u''tt = a**2 * (u''xx + u''yy)
        :param a: membrane parameter: a**2 = T/rho, a > 0
        :param L: x_max
        :param dx: x-axis step
        :param M: y_max
        :param dy: y-axis step
        """
        self.a = a
        self.L = L
        self.dx = dx
        self.M = M
        self.dy = dy

    def solve_explicit_scheme(self, T: float, dt: float, phi, psi, g):
        """
        Finds numerical solution of two-dimensional wave equation by using explicit finite difference scheme
        :param T: t_stop
        :param dt: t-axis step
        :param phi: function name: φ(x, y) = u(x, y, 0)
        :param psi: function name: ψ(x, y) = du/dt(x, y, 0)
        :param g: function name: g(x, y, t) = u(x, y, t) if (x, y) ∈ Γ
        :return: u[x][y][t] as np.array.
        """
        k = (self.dx * self.dx + self.dy * self.dy) * (self.a * dt / self.dx / self.dy) ** 2
        if k > 1:
            raise ValueError()
        time = np.arange(0, T + dt, dt)
        x_coordinates = np.arange(0, self.L + self.dx, self.dx)
        y_coordinates = np.arange(0, self.M + self.dy, self.dy)
        p = time.shape[0]
        m = x_coordinates.shape[0]
        n = y_coordinates.shape[0]
        u = np.empty([m, n, p])
        # Начальные условия (t = 0)
        for x in range(m):
            for y in range(n):
                u[x][y][0] = phi(x_coordinates[x], y_coordinates[y])
                u[x][y][1] = u[x][y][0] + dt * psi(x_coordinates[x], y_coordinates[y])
        # Граничные условия 1 рода
        for t in range(1, p):
            for x in range(1, m - 1):
                u[x][0][t] = g(x_coordinates[x], 0, t)
                u[x][n - 1][t] = g(x_coordinates[x], self.M, t)
            for y in range(1, n - 1):
                u[0][y][t] = g(0, y_coordinates[y], t)
                u[m - 1][y][t] = g(self.L, y_coordinates[y], t)
        # Послойное заполнение
        for t in range(1, p - 1):
            for x in range(1, m - 1):
                for y in range(1, n - 1):
                    u[x][y][t + 1] = (-u[x][y][t - 1] + 2 * (1 - k) * u[x][y][t] +
                                      ((self.a * dt / self.dx) ** 2) * (u[x + 1][y][t] + u[x - 1][y][t]) +
                                      ((self.a * dt / self.dy) ** 2) * (u[x][y + 1][t] + u[x][y - 1][t]))

        return u

    def plot3Danimation(self, u, dt: float, speed=1.):
        """
        Visualization of membrane vibration
        :param u: u(x, y, t) as np.array.
        :param dt: t-axis step
        :param speed: animation speed coefficient
        """
        k = (self.dx * self.dx + self.dy * self.dy) * ((self.a * dt / self.dx / self.dy) ** 2)
        x_coordinates = np.arange(0, self.L + self.dx, self.dx)
        y_coordinates = np.arange(0, self.M + self.dy, self.dy)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.suptitle(f'Membrane vibration')
        fig.supxlabel(f'a = {round(self.a, 3)}, L = {round(self.L, 3)}, dx = {round(self.dx, 3)}, '
                      f'M = {round(self.M, 3)}, dy = {round(self.dy, 3)}, dt = {dt}, k = {round(k, 3)}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_xlim([0, self.L])
        ax.set_ylim([0, self.M])
        ax.set_zlim([np.min(u), np.max(u)])
        plot_args = {'cmap': 'plasma', 'linewidth': 0, 'rstride': 1, 'cstride': 1}
        X, Y = np.meshgrid(x_coordinates, y_coordinates)
        surface = ax.plot_surface(X, Y, u[:, :, 0], label='time = 0', **plot_args)
        dp = Decimal(str(dt)).as_tuple().exponent * (-1)

        def animate(i):
            old_surface = [c for c in ax.collections if isinstance(c, Poly3DCollection)][0]
            old_surface.remove()
            surface = ax.plot_surface(X, Y, u[:, :, i], label=f'time = {dt * i:.{dp}f}', **plot_args)
            ax.legend()
            return surface,

        ani = animation.FuncAnimation(fig, animate, repeat=True, frames=u.shape[2],
                                      interval=round(1 / speed))
        plt.show()


def phi1(x, y):
    return 10 * x * (1 - x) * y * (1 - y)


def phi2(x, y):
    return x * (1 - x) * y * (1 - y)


def phi3(x, y):
    return sin(x) * cos(y - pi / 2)


def phi4(x, y):
    return sin(x) * sin(y)


def phi5(x, y):
    return x * (10 - x) * y * (10 - y)


def zero(*args):
    return 0


dt = 0.01
"""
membrane1 = MembraneVibration(a=1, L=1, dx=0.05, M=1, dy=0.05)
u1 = membrane1.solve_explicit_scheme(T=2, dt=dt, phi=phi1, psi=zero, g=zero)
membrane1.plot3Danimation(u1, dt=dt, speed=1)

membrane2 = MembraneVibration(a=1, L=2, dx=0.05, M=2, dy=0.05)
u2 = membrane2.solve_explicit_scheme(T=2, dt=dt, phi=phi2, psi=zero, g=zero)
membrane2.plot3Danimation(u2, dt=dt, speed=1)
"""
dt = 0.05
membrane3 = MembraneVibration(a=1, L=2*pi, dx=pi/20, M=2*pi, dy=pi/20)
u3 = membrane3.solve_explicit_scheme(T=4, dt=dt, phi=phi3, psi=zero, g=zero)
membrane3.plot3Danimation(u3, dt=dt, speed=1)
"""
membrane4 = MembraneVibration(a=1, L=pi, dx=pi/20, M=pi, dy=pi/20)
u3 = membrane4.solve_explicit_scheme(T=4, dt=dt, phi=phi4, psi=zero, g=zero)
membrane4.plot3Danimation(u3, dt=dt, speed=1)
"""