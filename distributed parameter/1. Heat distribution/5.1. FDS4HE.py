import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
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

    def solve_explicit_scheme(self, T: float, dt: float, f, u0=None, ul=None, ux0=None, uxl=None):
        """
        Finds numerical solution of heat equation by using explicit finite difference scheme
        :param T: t_stop
        :param dt: t-axis step
        :param f: function name: f(x) = u(x, 0)
        :param u0: function name: u0(t) = u(0, t)
        :param ul: function name: ul(t) = u(L, t)
        :param ux0: function name: ux0(t) = du/dx(0, t)
        :param uxl: function name: uxl(t) = du/dx(L, t)
        :return: u[x][t] as np.array.
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

    def solve_implicit_scheme(self, T: float, dt: float, f, u0=None, ul=None, ux0=None, uxl=None):
        """
        Finds numerical solution of heat equation by using implicit finite difference scheme
        :param T: t_stop
        :param dt: t-axis step
        :param f: function name: f(x) = u(x, 0)
        :param u0: function name: u0(t) = u(0, t)
        :param ul: function name: ul(t) = u(L, t)
        :param ux0: function name: ux0(t) = du/dx(0, t)
        :param uxl: function name: uxl(t) = du/dx(L, t)
        :return: u[x][t] as np.array.
        """
        b0, c0, f0, d0 = 1, 0, zero, 1
        an, bn, fn, dn = 1, 0, zero, 1
        time = np.arange(0, T + dt, dt)
        coordinates = np.arange(0, self.L + self.dx, self.dx)
        m = time.shape[0]
        n = coordinates.shape[0]
        self.u = np.empty([n, m])
        k = self.a * self.a * dt / self.dx / self.dx
        for x in range(n):
            self.u[x][0] = f(coordinates[x])
        if u0 is not None:
            b0, c0, f0, d0 = 1, 0, u0, 1
        if ux0 is not None:
            b0, c0, f0, d0 = 1, 1, ux0, -self.dx
        if ul is not None:
            an, bn, fn, dn = 0, 1, ul, 1
        if uxl is not None:
            an, bn, fn, dn = 1, 1, uxl, self.dx

        alpha = np.empty(n - 1)
        beta = np.empty(n - 1)
        alpha[0] = c0 / b0
        for i in range(1, n - 1):
            alpha[i] = k / (1 + 2 * k - k * alpha[i - 1])

        for j in range(m - 1):
            beta[0] = d0 * f0(time[j + 1]) / b0
            for i in range(1, n - 1):
                beta[i] = (self.u[i][j] + k * beta[i - 1]) / (1 + 2 * k - k * alpha[i - 1])
            self.u[n - 1][j + 1] = (dn * fn(time[j + 1]) + an * beta[n - 2]) / (bn - an * alpha[n - 2])

            for i in range(n - 2, -1, -1):
                self.u[i][j + 1] = alpha[i] * self.u[i + 1][j + 1] + beta[i]

        return self.u

    def plot2Danimation(self, u, dt: float, speed=1.):
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
        fig.supxlabel(f'a = {self.a}, L = {self.L}, dx = {self.dx}, dt = {dt}, '
                      f'λ = {round(self.a * self.a * dt / self.dx / self.dx, 3)}')
        ax.set_xlabel('x')
        ax.set_ylabel('temperature')
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
        fig.suptitle(f'Solution of heat equation: temperature = u(x, t)')
        fig.supxlabel(f'a = {self.a}, L = {self.L}, dx = {self.dx}, dt = {dt}, '
                      f'λ = {round(self.a * self.a * dt / self.dx / self.dx, 3)}')
        ax.set_xlabel('time')
        ax.set_ylabel('x')
        ax.set_zlabel('temperature')

        time = np.arange(0, T + dt, dt)
        x_coordinates = np.arange(0, self.L + self.dx, self.dx)
        X, Y = np.meshgrid(time, x_coordinates)
        surface = ax.plot_surface(X, Y, u, cmap=cm.plasma, linewidth=0, rstride=1, cstride=1)

        plt.show()


def f1(x):
    return 100 * x * (1 - x)


def f2(x):
    return sin(3 * pi * x / 2)


def zero(t):
    return 0


heat_equation1 = HeatEquation(a=1, L=1, dx=0.05)
dt1 = 0.00125

# u1 = heat_equation1.solve_explicit_scheme(T=0.75, dt=dt1, f=f1, u0=zero, ul=zero)
# u1 = heat_equation1.solve_explicit_scheme(T=0.25, dt=dt1, f=f1, ux0=zero, uxl=zero)
# u1 = heat_equation1.solve_explicit_scheme(T=2, dt=dt1, f=f1, u0=zero, uxl=zero)

# u1 = heat_equation1.solve_explicit_scheme(T=0.5, dt=dt1, f=f2, u0=zero, ul=zero)
# u1 = heat_equation1.solve_explicit_scheme(T=0.35, dt=dt1, f=f2, ux0=zero, uxl=zero)
# u1 = heat_equation1.solve_explicit_scheme(T=1, dt=dt1, f=f2, u0=zero, uxl=zero)

# heat_equation1.plot2Danimation(u1, dt=dt1, speed=1)


heat_equation2 = HeatEquation(a=0.1, L=1, dx=0.05)
dt2 = 0.1

# u2 = heat_equation2.solve_implicit_scheme(T=25, dt=dt2, f=f1, u0=zero, ul=zero)
# u2 = heat_equation2.solve_implicit_scheme(T=10, dt=dt2, f=f1, ux0=zero, uxl=zero)
u2 = heat_equation2.solve_implicit_scheme(T=50, dt=dt2, f=f1, u0=zero, uxl=zero)

#heat_equation2.plot2Danimation(u2, dt=dt2, speed=1)

heat_equation2.plot3d(u2, T=50, dt=dt2)
