import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from NSolveDE import RK4_auto_system_XY, Euler_auto_system_XY


class SpringMassSystem(object):
    def __init__(self, n=3, k=1, kw=1):
        if n < 3:
            raise ValueError('Invalid argument values: n < 3')
        self.n = n
        self.k = k
        self.kw = kw

    def dx(self, x, v, args):
        return v

    def dv(self, center, v, args):
        left = args[0]
        right = args[1]
        i = args[2]
        if i == 1:
            return self.k * (right - center) - self.kw * center
        elif 1 < i < self.n:
            return self.k * (right - center) - self.k * (center - left)
        elif i == self.n:
            return self.k * (left - center) - self.kw * center
        else:
            raise ValueError('Invalid argument values: "i" out of bounds [1, n]')

    def get_integral_curves(self, x0, v0, t0, T, dt, method='RK4'):

        if method == 'RK4':
            get_next = RK4_auto_system_XY
        elif method == 'Euler':
            get_next = Euler_auto_system_XY
        else:
            print('Invalid method value. RK4 method used instead. '
                  'Available options are "RK4" and "Euler"')
            get_next = RK4_auto_system_XY

        time = np.arange(t0, T, dt)
        m = time.shape[0]
        x = np.empty([self.n, m])
        v = np.empty([self.n, m])
        for i in range(self.n):
            x[i][0] = x0[i]
            v[i][0] = v0[i]

        for t in range(m - 1):
            x[0][t + 1], v[0][t + 1] = get_next(self.dx, self.dv,
                                                x[0][t], v[0][t],
                                                dt,
                                                [v[0][t]],
                                                [None, x[1][t], 1])

            for i in range(1, self.n - 1):
                x[i][t + 1], v[i][t + 1] = get_next(self.dx, self.dv,
                                                    x[i][t], v[i][t],
                                                    dt,
                                                    [v[i][t]],
                                                    [x[i - 1][t], x[i + 1][t], i + 1])

            x[self.n - 1][t + 1], v[self.n - 1][t + 1] = get_next(self.dx, self.dv,
                                                                  x[self.n - 1][t], v[self.n - 1][t],
                                                                  dt,
                                                                  [v[self.n - 1][t]],
                                                                  [x[self.n - 2][t], None, self.n])

        return time, x, v

    def plot_animation(self, x0, v0, t0=0.0, T=10.0, dt=0.01, method='RK4', x_max=10):
        time, x, v = self.get_integral_curves(x0, v0, t0, T, dt, method)
        fig, ax = plt.subplots()
        fig.suptitle(f'Spring-mass system (n = {self.n})')
        fig.supxlabel(f'k = {self.k}, kw = {self.kw}')
        ax.set_xlim([0, x_max])
        start = np.linspace(0, x_max, self.n + 2)[1:-1]
        txc_array = x.transpose() + start
        ax.plot((0, x_max), (0, 0), color='black', linestyle='dashed', linewidth=1, zorder=-1)
        scat = ax.scatter(txc_array[0], (0,) * self.n, s=20 ** 2, c=np.linspace(0, 1, self.n), alpha=1)

        def animate(i):
            current_coordinates = [(x, 0) for x in txc_array[i]]
            scat.set_offsets(current_coordinates)
            return (scat,)

        ani = animation.FuncAnimation(fig, animate, repeat=False, frames=time.shape[0]-1, interval=50)
        plt.show()
        return x


n = 7
x0 = 10 * np.random.rand(n)
v0 = [0] * n
model = SpringMassSystem(n=n, k=1, kw=2)
model.plot_animation(x0=x0, v0=v0, t0=0, T=25, dt=0.1, method='RK4', x_max=200)
