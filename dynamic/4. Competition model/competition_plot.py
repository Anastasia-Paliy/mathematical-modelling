import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from NSolveDE import RK4_auto_system_XY


def dx1(x1, x2, args):
    """ dx1 / dt = a1 * x1 - b12 * x1 * x2 - c1 * x1 ^ 2 """
    a_1 = args[0]
    b_12 = args[1]
    c_1 = args[2]
    return a_1 * x1 - b_12 * x1 * x2 - c_1 * x1 ** 2


def dx2(x1, x2, args):
    """ dx2 / dt = a2 * x2 - b21 * x1 * x2 - c2 * x2 ^ 2 """
    a_2 = args[0]
    b_21 = args[1]
    c_2 = args[2]
    return a_2 * x2 - b_21 * x1 * x2 - c_2 * x2 ** 2


def competition(a_1, a_2, b_12, b_21, c_1, c_2, x1_0, x2_0, dt, n: int):
    x1 = np.empty(n)
    x2 = np.empty(n)
    x1[0] = x1_0
    x2[0] = x2_0

    for t in range(0, n - 1):
        x1[t + 1], x2[t + 1] = RK4_auto_system_XY(dx1, dx2, x1[t], x2[t], dt, [a_1, b_12, c_1], [a_2, b_21, c_2])

    return x1, x2


a1 = 10
a2 = 20
b12 = 0.1
b21 = 0.2
c1 = 0.1
c2 = 0.4

t0 = 0
T = 5
dt = 0.01
time = np.arange(t0, T, dt)

if b12 * b21 == c1 * c2:
    raise ValueError('Invalid argument values: b12 * b21 == c1 * c2')

equilibrium_points = [(0, 0),
                      (a1 / c1, 0),
                      (0, a2 / c2),
                      ((a2 * b12 - a1 * c2) / (b12 * b21 - c1 * c2),
                       (a1 * b21 - a2 * c1) / (b12 * b21 - c1 * c2))]

if equilibrium_points[-1][0] * equilibrium_points[-1][1] <= 0:
    equilibrium_points = equilibrium_points[:-1]

x1_min, x2_min = 1, 1
x1_max = 1.1 * max([point[0] for point in equilibrium_points])
x2_max = 1.1 * max([point[1] for point in equilibrium_points])

x1_coords = np.linspace(x1_min, x1_max, 10)
x2_coords = np.linspace(x2_min, x2_max, 10)

initial_points = list(product(*[x1_coords, x2_coords]))
initial_point = (1, 1)

# Visualization
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.tight_layout(pad=3, w_pad=1)
fig.suptitle('Competition model')
fig.supxlabel(f'a1 = {a1}, a2 = {a2}, b12 = {b12}, b21 = {b21}, c1 = {c1}, c2 = {c2}')

# Timeline
X1, X2 = competition(a1, a2, b12, b21, c1, c2, initial_point[0], initial_point[1], dt, time.shape[0])
axes[0].plot(time, X1, label=f'Species 1: x1(0) = {initial_point[0]}')
axes[0].plot(time, X2, label=f'Species 2: x2(0) = {initial_point[1]}')
# print(X1[-1], X2[-1])

axes[0].set_xlabel('t')
axes[0].set_ylabel('x1(t), x2(t)')
axes[0].set_title('Timeline')
axes[0].legend(loc='upper right', framealpha=1, borderpad=0.8)

# Phase plane
for (x1_0, x2_0) in initial_points:
    X1, X2 = competition(a1, a2, b12, b21, c1, c2, x1_0, x2_0, dt, time.shape[0])
    axes[1].plot(X1, X2)

for (x1_0, x2_0) in equilibrium_points:
    axes[1].plot(x1_0, x2_0, 'ro', label=f'x1 = {round(x1_0, 2)}, x2 = {round(x2_0, 2)}')

axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].set_title('Phase plane')
axes[1].legend(loc='upper right', framealpha=1, borderpad=0.8)

plt.show()
