import matplotlib.pyplot as plt
from NSolveDE import RK4, Euler


def get_k0(l, m, A, a, s):
    """
    Calculates the equilibrium point k* of the Solow-Swan model
    :param l: λ - labor growth rate
    :param m: μ - depreciation rate
    :param A: A - technological coefficient
    :param a: α – output elasticity of capital
    :param s: s - propensity to save
    :return: k* - steady-state value of k(t)
    """
    return pow(s * A / (l + m), 1 / (1 - a))


def dk(t, k):
    """ dk/dt = f(t, k) """
    return -(l + m) * k + s * A * pow(k, a)


l = 0.001
m = 0.05
A = 1
a = 0.4
s = 0.2

k0 = get_k0(l, m, A, a, s)
print(f'k* = {k0}')
t0 = 0
T = 250
step = 0.1

k1 = 0.01
k2 = 20

# Comparison of Euler and RK4 methods
# on the example of an integral curve passing through point (0, k1).
timeline1, euler = Euler(dk, k1, t0, T, step)
timeline2, rk4 = RK4(dk, k1, t0, T, step)
print(max(abs(euler - rk4)))

# Visualization
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.tight_layout(pad=3, w_pad=2)
fig.suptitle('Solow-Swan model')
fig.supxlabel(f'λ = {l}, μ = {m}, s = {s}, α = {a}, A = {A}')

axes[0].plot(*Euler(dk, k0, t0, T, step), label='k0 = ' + str(round(k0, 3)))
axes[0].plot(*Euler(dk, k1, t0, T, step), label='k0 = ' + str(round(k1, 3)))
axes[0].plot(*Euler(dk, k2, t0, T, step), label='k0 = ' + str(round(k2, 3)))

axes[0].set_xlabel('t')
axes[0].set_ylabel('k(t)')
axes[0].set_title('Solow+Euler')
axes[0].legend()

axes[1].plot(*RK4(dk, k0, t0, T, step), label='k0 = ' + str(round(k0, 3)))
axes[1].plot(*RK4(dk, k1, t0, T, step), label='k0 = ' + str(round(k1, 3)))
axes[1].plot(*RK4(dk, k2, t0, T, step), label='k0 = ' + str(round(k2, 3)))

axes[1].set_xlabel('t')
axes[1].set_ylabel('k(t)')
axes[1].set_title('Solow+RK4')
axes[1].legend()

# For united legend:
# handles, labels = plt.gca().get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center')
plt.show()
