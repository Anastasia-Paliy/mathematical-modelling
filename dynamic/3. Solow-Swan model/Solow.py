import matplotlib.pyplot as plt
from DEnumsol import RK4, Euler


def get_k0(l, m, A, a, s):
    return pow(s * A / (l + m), 1 / (1 - a))


l = 0.001
m = 0.05
A = 1
a = 0.4
s = 0.2

k0 = get_k0(l, m, A, a, s)
print(k0)
t0 = 0
T = 250
step = 0.1

g = lambda t, k: -(l + m) * k + s * A * pow(k, a)

fig, axes = plt.subplots(nrows=1, ncols=2)

fig.tight_layout(pad=3, w_pad=2)

axes[0].plot(*Euler(g, k0, t0, T, step))
axes[0].plot(*Euler(g, 0.01, t0, T, step))
axes[0].plot(*Euler(g, 20, t0, T, step))

axes[0].set_xlabel('t')
axes[0].set_ylabel('k(t)')
axes[0].set_title('Solow+Euler')

axes[1].plot(*RK4(g, k0, t0, T, step))
axes[1].plot(*RK4(g, 0.01, t0, T, step))
axes[1].plot(*RK4(g, 20, t0, T, step))

axes[1].set_xlabel('t')
axes[1].set_ylabel('k(t)')
axes[1].set_title('Solow+RK4')
plt.show()
