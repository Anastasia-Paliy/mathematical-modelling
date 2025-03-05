import numpy as np
import matplotlib.pyplot as plt
from NSolveDE import RK4_auto_system_XYZ, Euler_auto_system_XYZ


class LorenzAttractor(object):
    def __init__(self, sigma=10.0, r=28.0, b=8/3):
        self.sigma = sigma
        self.r = r
        self.b = b

        if self.r > 1:
            xy = np.sqrt(self.b * (self.r - 1))
            self.special_points = [(xy, xy, self.r - 1),
                                   (-xy, -xy, self.r - 1)]
        else:
            self.special_points = [(0, 0, 0)]

    def dx(self, x, y, z, args=None):
        return self.sigma * (y - x)

    def dy(self, x, y, z, args=None):
        return x * (self.r - z) - y

    def dz(self, x, y, z, args=None):
        return x * y - self.b * z

    def get_integral_curve(self, x0, y0, z0, t0, T, dt, method='RK4'):
        time = np.arange(t0, T, dt)

        n = time.shape[0]
        x = np.empty(n)
        y = np.empty(n)
        z = np.empty(n)

        x[0] = x0
        y[0] = y0
        z[0] = z0

        if method == 'RK4':
            get_next = RK4_auto_system_XYZ
        elif method == 'Euler':
            get_next = Euler_auto_system_XYZ
        else:
            print('Invalid method value. RK4 method used instead. '
                  'Available options are "RK4" and "Euler"')
            get_next = RK4_auto_system_XYZ

        for t in range(0, n - 1):
            x[t + 1], y[t + 1], z[t + 1] = get_next(self.dx, self.dy, self.dz,
                                                    x[t], y[t], z[t],
                                                    dt,
                                                    None, None, None)

        return time, x, y, z

    def plot3d(self, t0, T, dt, initial_point=(1.0, 1.0, 1.0), cut=0):

        time, x, y, z = self.get_integral_curve(initial_point[0],
                                                initial_point[1],
                                                initial_point[2],
                                                t0, T, dt, 'RK4')

        fig = plt.figure()
        fig.suptitle(f'Lorenz attractor (r = {self.r}), t = [{t0}, {T}])')

        axes = fig.add_subplot(1, 1, 1, projection='3d')
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_zlabel("z")

        axes.plot(x[cut:], y[cut:], z[cut:], lw=0.5, color='mediumslateblue')
        for point in self.special_points:
            axes.scatter(*point, marker='o', color='magenta')

        plt.show()

    def plot2d(self, t0, T, dt, initial_point=(1., 1., 1.), cut=0, tpc=-1):

        time, x, y, z = self.get_integral_curve(initial_point[0],
                                                initial_point[1],
                                                initial_point[2],
                                                t0, T, dt, 'RK4')

        time_cut = time[cut:]
        x_cut = x[cut:]
        y_cut = y[cut:]
        z_cut = z[cut:]

        fig = plt.figure()
        fig.suptitle(f'Lorenz attractor (r = {self.r}, t = [{t0}, {T}])')

        # Timeline
        axes3 = fig.add_subplot(2, 2, 3)
        axes3.plot(time_cut[:tpc], x_cut[:tpc], lw=0.7, label='x(t)', color='purple')
        axes3.plot(time_cut[:tpc], y_cut[:tpc], lw=0.7, label='y(t)', color='gold')
        axes3.plot(time_cut[:tpc], z_cut[:tpc], lw=0.7, label='z(t)', color='dodgerblue')
        axes3.set_xlabel("t")
        axes3.set_ylabel("x, y, z")
        axes3.legend(loc='upper right', framealpha=1)

        axes1 = fig.add_subplot(2, 2, 1)
        axes1.plot(x_cut, y_cut, lw=0.7, label='y(x)', color='mediumorchid')
        # axes1.set_title('y(x)')
        axes1.set_xlabel("x")
        axes1.set_ylabel("y")
        axes1.legend()

        axes2 = fig.add_subplot(2, 2, 2)
        axes2.plot(x_cut, z_cut, lw=0.7, label='z(x)', color='gold')
        # axes2.set_title('z(x)')
        axes2.set_xlabel("x")
        axes2.set_ylabel("z")
        axes2.legend()

        axes4 = fig.add_subplot(2, 2, 4)
        axes4.plot(y_cut, z_cut, lw=0.7, label='z(y)', color='dodgerblue')
        # axes2.set_title('z(y)')
        axes4.set_xlabel("y")
        axes4.set_ylabel("z")
        axes4.legend()

        plt.show()


t0 = 0
T = 100
dt = 0.01
initial_point = (1.0, 1.0, 1.0)

model = LorenzAttractor(r=100.0)
model.plot3d(t0, T, dt, initial_point, cut=int(5/dt))
model.plot2d(t0, T, dt, initial_point, cut=int(5/dt), tpc=int(10/dt))
