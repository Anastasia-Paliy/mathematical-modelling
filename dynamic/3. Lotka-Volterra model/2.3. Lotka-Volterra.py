import numpy as np
import matplotlib.pyplot as plt
from NSolveDE import RK4_system


def dx(t, x, y, args):
    """ dx / dt = a * x - b * x * y """
    a = args[0]
    b = args[1]
    return a * x - b * x * y


def dy(t, x, y, args):
    """ dy / dt = d * x * y - c * y """
    c = args[0]
    d = args[1]
    return d * x * y - c * y


def Lotka_Volterra(a, b, c, d, x0, y0, n: int):
    """
    Lotka-Volterra "predator–prey" model
    :param a: prey population growth rate (a > 0)
    :param b: coefficient of predators' influence on prey extermination rate (b > 0)
    :param c: predators death rate (c > 0)
    :param d: coefficient of prey presence effect on predator growth rate (d > 0)
    :param x0: x(t0) - number of preys at the initial time
    :param y0: y(t0) - number of predators at the initial time
    :param n: number of iterations
    :return: x(t) - list, y(t) - list
    """
    x = np.empty(n)
    y = np.empty(n)
    x[0] = x0
    y[0] = y0

    for t in range(0, n - 1):
        x[t + 1], y[t + 1] = RK4_system(dx, dy, x[t], y[t], t, dt, [a, b], [c, d])

    return x, y


a = 0.5
b = 0.05
c = 0.2
d = 0.01

t0 = 0
T = 50
dt = 0.1
time = np.arange(t0, T, dt)

x0 = c/d
y0 = a/b

initial_points = [(100, 10), (10, 10), (5, 15), (30, 10), (5, 1), (20, 9)]

# Visualization
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.tight_layout(pad=3, w_pad=1)
fig.suptitle('Lotka-Volterra model')
fig.supxlabel(f'a = {a}, b = {b}, c = {c}, d = {d}')

for (x, y) in initial_points:
    X, Y = Lotka_Volterra(a, b, c, d, x, y, time.shape[0])
    axes[1].plot(X, Y, label=f'x0 = {x}, y0 = {y}')
axes[1].plot(x0, y0, 'ro', label=f'x0 = {x0}, y0 = {y0}')

X, Y = Lotka_Volterra(a, b, c, d, initial_points[2][0], initial_points[2][1], time.shape[0])
axes[0].plot(time, X, label=f'Prey: x0 = {initial_points[2][0]}')
axes[0].plot(time, Y, label=f'Predator: y0 = {initial_points[2][1]}')

axes[0].set_xlabel('t')
axes[0].set_ylabel('x(t), y(t)')
axes[0].set_title('Timeline')
axes[0].legend(loc='upper left')

axes[1].set_xlabel('x')
axes[1].set_ylabel('y(x)')
axes[1].set_title('Phase portrait')
axes[1].legend()

plt.show()
