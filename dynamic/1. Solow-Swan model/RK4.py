import matplotlib.pyplot as plt
from NSolveDE import RK4


# Example: dx/dt = 2*t
def g(t, x):
    return 2 * t


x0 = 0
t0 = 0
T = 10
step = 0.1

time, x = RK4(g, x0, t0, T, step)
# print(x)

plt.plot(time, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('RK4')
plt.show()
