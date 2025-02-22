import matplotlib.pyplot as plt
from NSolveDE import Euler


# Example: dx/dt = 2*t
def g(t, x):
    return 2 * t

t0 = 0
t = 10
step = 0.01
x0 = 0

time, x = Euler(g, x0, t0, t, step)
# print(x)

plt.plot(time, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Euler')
plt.show()
