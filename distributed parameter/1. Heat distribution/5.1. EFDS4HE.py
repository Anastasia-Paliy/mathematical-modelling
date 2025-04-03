import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin, pi
from decimal import Decimal


class HeatEquation(object):
    def __init__(self, a: float, L: float, dx: float):
        """
        Setting key params of heat equation u't = a**2 * u''xx
        :param a: thermal diffusivity coefficient (a > 0)
        :param L: rod length (x_stop)
        :param dx: x-axis step
        """
        self.a = a
        self.L = L
        self.dx = dx
        self.u = None

    def solve(self, T: float, dt: float, f, u0=None, ul=None, ux0=None, uxl=None):
        """
        Finds numerical solution of heat equation as function u(x, t) saved in np.array.
        :param T: t_stop
        :param dt: t-axis step
        :param f: function name: f(x) = u(x, 0)
        :param u0: function name: u0(t) = u(0, t)
        :param ul: function name: ul(t) = u(L, t)
        :param ux0: function name: ux0(t) = du/dx(0, t)
        :param uxl: function name: uxl(t) = du/dx(L, t)
        :return: u(x, t), x_array, t_array
        """
        k = self.a * self.a * dt / self.dx / self.dx
        if 2 * k > 1:
            raise ValueError()
        time = np.arange(0, T + dt, dt)
        coordinates = np.arange(0, self.L + self.dx, self.dx)
        m = time.shape[0]
        n = coordinates.shape[0]
        self.u = np.empty([n, m])

        for x in range(n):
            self.u[x][0] = f(coordinates[x])
        for t in range(m - 1):
            for x in range(1, n - 1):
                self.u[x][t + 1] = self.u[x][t] * (1 - 2 * k) + k * (self.u[x + 1][t] + self.u[x - 1][t])

            if u0 is not None:
                self.u[0][t + 1] = u0(time[t + 1])
            if ux0 is not None:
                self.u[0][t + 1] = self.u[1][t + 1] - self.dx * ux0(time[t + 1])
            if ul is not None:
                self.u[n - 1][t + 1] = ul(time[t + 1])
            if uxl is not None:
                self.u[n - 1][t + 1] = self.u[n - 2][t + 1] + self.dx * uxl(time[t + 1])

        return self.u

    def visualize(self, u, dt, speed=1.):
        """
        Visualization of temperature distribution changes over time
        :param u: u(x, t) as np.array.
        :param dt: t-axis step
        :param speed: animation speed coefficient
        """
        x_coordinates = np.arange(0, self.L + self.dx, self.dx)
        y_data = u.transpose()
        fig, ax = plt.subplots()
        fig.suptitle(f'Solution of heat equation: temperature = u(x, t)')
        fig.supxlabel(f'a = {self.a}, L = {self.L}, dx = {self.dx}, dt = {dt}')
        ax.set_xlabel('x')
        ax.set_ylabel('temperature')
        ax.set_xlim([0, self.L])
        ax.set_ylim([0, np.max(u)])

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


def f1(x):
    return 100 * x * (1 - x)


def f2(x):
    return sin(3 * pi * x / 2)


def g(t):
    return 0


heat_equation = HeatEquation(a=1, L=1, dx=0.05)
dt = 0.00125

u1 = heat_equation.solve(T=0.75, dt=dt, f=f1, u0=g, ul=g)
u2 = heat_equation.solve(T=0.25, dt=dt, f=f1, ux0=g, uxl=g)
u3 = heat_equation.solve(T=2, dt=dt, f=f1, u0=g, uxl=g)

u4 = heat_equation.solve(T=0.5, dt=dt, f=f2, u0=g, ul=g)
u5 = heat_equation.solve(T=0.35, dt=dt, f=f2, ux0=g, uxl=g)
u6 = heat_equation.solve(T=1, dt=dt, f=f2, u0=g, uxl=g)

heat_equation.visualize(u3, dt=dt, speed=1)
