import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from decimal import Decimal


class StringVibration(object):
    def __init__(self, a: float, L: float, dx: float):
        """
        Setting key params of wave equation u''tt = a**2 * u''xx
        :param a: thermal diffusivity coefficient (a > 0)
        :param L: string length (x_stop)
        :param dx: x-axis step
        """
        self.a = a
        self.L = L
        self.dx = dx
        self.u = None

    def solve_explicit_scheme(self, T: float, dt: float, f, g):
        """
        Finds numerical solution of one-dimensional wave equation by using explicit finite difference scheme
        :param T: t_stop
        :param dt: t-axis step
        :param f: function name: f(x) = u(x, 0)
        :param g: function name: g(x) = du/dt(x, 0)
        :return: u[x][t] as np.array.
        """
        k = self.a * self.a * dt * dt / self.dx / self.dx
        if k > 1:
            raise ValueError()
        time = np.arange(0, T + dt, dt)
        coordinates = np.arange(0, self.L + self.dx, self.dx)
        m = time.shape[0]
        n = coordinates.shape[0]
        self.u = np.empty([n, m])

        for x in range(n):
            self.u[x][0] = f(coordinates[x])
            self.u[x][1] = self.u[x][0] + dt * g(coordinates[x])
        for t in range(m - 1):
            self.u[0][t + 1] = 0
            for x in range(1, n - 1):
                self.u[x][t + 1] = (2 * (1 - k) * self.u[x][t] +
                                    k * (self.u[x + 1][t] + self.u[x - 1][t]) - self.u[x][t - 1])
            self.u[n - 1][t + 1] = 0

        return self.u

    def plot2Danimation(self, u, dt: float, speed=1.):
        """
        Visualization of string vibration
        :param u: u(x, t) as np.array.
        :param dt: t-axis step
        :param speed: animation speed coefficient
        """
        x_coordinates = np.arange(0, self.L + self.dx, self.dx)
        y_data = u.transpose()
        fig, ax = plt.subplots()
        fig.suptitle(f'String vibration')
        fig.supxlabel(f'a = {self.a}, L = {self.L}, dx = {self.dx}, dt = {dt}, '
                      f'λ = {round(self.a * self.a * dt * dt / self.dx / self.dx, 3)}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_xlim([0, self.L])
        ax.set_ylim([np.min(u), np.max(u)])

        line, = ax.plot(x_coordinates, y_data[0], label='time = 0', linewidth=2)
        plot_legend = ax.legend()

        dp = Decimal(str(dt)).as_tuple().exponent * (-1)

        def animate(i):
            line.set_ydata(y_data[i])
            plot_legend.get_texts()[0].set_text(f'time = {dt * i:.{dp}f}')
            return line,

        ani = animation.FuncAnimation(fig, animate, repeat=True, frames=y_data.shape[0],
                                      interval=round(100 / speed))
        plt.show()

    def plot3d(self, u, T: float, dt: float):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.suptitle(f'String vibration')
        fig.supxlabel(f'a = {self.a}, L = {self.L}, dx = {self.dx}, dt = {dt}, '
                      f'λ = {round(self.a * self.a * dt * dt / self.dx / self.dx, 3)}')
        ax.set_xlabel('time')
        ax.set_ylabel('x')
        ax.set_zlabel('u')

        time = np.arange(0, T + dt, dt)
        x_coordinates = np.arange(0, self.L + self.dx, self.dx)
        X, Y = np.meshgrid(time, x_coordinates)
        surface = ax.plot_surface(X, Y, u, cmap=cm.plasma, linewidth=0, rstride=1, cstride=1)

        plt.show()


def f1(x):
    return x * (1 - x)


def f2(x):
    return x * (x * x - 1)


def g(x):
    if 0.2 <= x <= 0.3:
        return 3
    elif 0.7 <= x <= 0.8:
        return -2
    else:
        return 0


def zero(t):
    return 0


string = StringVibration(a=1, L=1, dx=0.01)
dt = 0.01

u1 = string.solve_explicit_scheme(T=2, dt=dt, f=f1, g=zero)
u2 = string.solve_explicit_scheme(T=2, dt=dt, f=f2, g=zero)
u3 = string.solve_explicit_scheme(T=2, dt=dt, f=f2, g=g)

string.plot2Danimation(u3, dt=dt, speed=2)
# string.plot3d(u3, T=2, dt=dt)
