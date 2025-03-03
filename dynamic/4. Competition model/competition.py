import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from my_tools import operator
from NSolveDE import RK4_auto_system, Euler_auto_system


class CompetitionModel(object):
    def __init__(self, a1, a2, b12, b21, c1, c2):
        if (a1 * b21 == a2 * c1) and (a2 * b12 == a1 * c2):
            raise ValueError('Invalid argument values: '
                             'a1 * b21 == a2 * c1 and a2 * b12 == a1 * c2')
        self.a1 = a1
        self.a2 = a2
        self.b12 = b12
        self.b21 = b21
        self.c1 = c1
        self.c2 = c2

        self.case1 = np.sign(a1 * b21 - a2 * c1)
        self.case2 = np.sign(a2 * b12 - a1 * c2)

        self.equilibrium_points = [(0, 0),
                                   (a1 / c1, 0),
                                   (0, a2 / c2)]

        if b12 * b21 != c1 * c2 and self.case1 * self.case2 > 0:
            self.equilibrium_points.append(((a2 * b12 - a1 * c2) / (b12 * b21 - c1 * c2),
                                            (a1 * b21 - a2 * c1) / (b12 * b21 - c1 * c2)))

    def dx1(self, x1, x2, args):
        return self.a1 * x1 - self.b12 * x1 * x2 - self.c1 * x1 ** 2

    def dx2(self, x1, x2, args):
        return self.a2 * x2 - self.b21 * x1 * x2 - self.c2 * x2 ** 2

    def get_integral_curves(self, x1_0, x2_0, t0, T, dt, method='RK4'):
        time = np.arange(t0, T, dt)

        n = time.shape[0]
        x1 = np.empty(n)
        x2 = np.empty(n)
        x1[0] = x1_0
        x2[0] = x2_0

        if method == 'RK4':
            get_next = RK4_auto_system
        elif method == 'Euler':
            get_next = Euler_auto_system
        else:
            print('Invalid method value. RK4 method used instead. '
                  'Available options are "RK4" and "Euler"')
            get_next = RK4_auto_system

        for t in range(0, n - 1):
            x1[t + 1], x2[t + 1] = get_next(self.dx1, self.dx2, x1[t], x2[t], dt, None, None)

        return time, x1, x2

    def plot(self,
             initial_point=(1, 1),
             x1_min=1,
             x2_min=1,
             x1_max=None,
             x2_max=None,
             zoom=1.2,
             density=10
             ):
        if x1_max is None:
            x1_max = zoom * max([point[0] for point in self.equilibrium_points])
        if x2_max is None:
            x2_max = zoom * max([point[1] for point in self.equilibrium_points])

        x1_coords = np.linspace(x1_min, x1_max, density)
        x2_coords = np.linspace(x2_min, x2_max, density)

        initial_points = list(product(*[x1_coords, x2_coords]))

        # Visualization
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.tight_layout(pad=3, w_pad=1)
        fig.suptitle('Competition model')
        fig.supxlabel(f'a1 = {self.a1}, a2 = {self.a2}, '
                      f'b12 = {self.b12}, b21 = {self.b21}, '
                      f'c1 = {self.c1}, c2 = {self.c2} \n\n'
                      f'a1 * b21 {operator(self.case1)} a2 * c1,     '
                      f'a2 * b12 {operator(self.case2)} a1 * c2')

        # Timeline
        time, X1, X2 = self.get_integral_curves(initial_point[0],
                                                initial_point[1],
                                                t0, T, dt, 'RK4')
        axes[0].plot(time, X1, label=f'Species 1: x1(0) = {initial_point[0]}')
        axes[0].plot(time, X2, label=f'Species 2: x2(0) = {initial_point[1]}')

        axes[0].set_xlabel('t')
        axes[0].set_ylabel('x1(t), x2(t)')
        axes[0].set_title('Timeline')
        axes[0].legend(loc='upper right', framealpha=1, borderpad=0.8)

        # Phase plane
        for (x1_0, x2_0) in initial_points:
            time, X1, X2 = self.get_integral_curves(x1_0, x2_0, t0, T, dt, 'RK4')
            axes[1].plot(X1, X2)

        for (x1_0, x2_0) in self.equilibrium_points:
            axes[1].plot(x1_0, x2_0, 'ro', label=f'x1 = {round(x1_0, 2)}, x2 = {round(x2_0, 2)}')

        axes[1].set_xlabel('x1')
        axes[1].set_ylabel('x2')
        axes[1].set_title('Phase plane')
        axes[1].legend(loc='upper right', framealpha=1, borderpad=0.8)

        plt.show()

    def streamplot(self,
                   x1_min=1,
                   x2_min=1,
                   x1_max=None,
                   x2_max=None,
                   zoom=1.5,
                   ls_density=10,
                   sp_density=2
                   ):
        if x1_max is None:
            x1_max = zoom * max([point[0] for point in self.equilibrium_points])
        if x2_max is None:
            x2_max = zoom * max([point[1] for point in self.equilibrium_points])

        x = np.linspace(x1_min, x1_max, ls_density)
        y = np.linspace(x2_min, x2_max, ls_density)
        X, Y = np.meshgrid(x, y)
        U_, V_ = np.array([self.dx1(X, Y, None),
                           self.dx2(X, Y, None)])

        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.175, top=0.93, right=0.8, left=0.2)

        fig.suptitle('Competition model')
        fig.supxlabel(f'a1 = {self.a1}, a2 = {self.a2}, '
                      f'b12 = {self.b12}, b21 = {self.b21}, '
                      f'c1 = {self.c1}, c2 = {self.c2} \n\n'
                      f'a1 * b21 {operator(self.case1)} a2 * c1,     '
                      f'a2 * b12 {operator(self.case2)} a1 * c2')

        axes.axis((x1_min, x1_max, x2_min, x2_max))
        axes.set_xlabel('x1')
        axes.set_ylabel('x2')

        axes.streamplot(x, y, U_, V_,
                        density=sp_density,
                        linewidth=1,
                        arrowstyle="->")

        for point in self.equilibrium_points:
            axes.plot(*point,
                      'bo',
                      label=f'{(round(point[0], 2), round(point[1], 2))}')

        axes.legend(loc='upper right', framealpha=1, borderpad=0.8)

        plt.show()


a1 = 10
a2 = 20
b12 = 0.1
b21 = 0.3
c1 = 0.3
c2 = 0.3

model = CompetitionModel(a1, a2, b12, b21, c1, c2)

t0 = 0
T = 5
dt = 0.01

# model.plot(initial_point=(1, 10))
model.streamplot()
