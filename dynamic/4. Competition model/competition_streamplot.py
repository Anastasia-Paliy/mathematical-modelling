import matplotlib.pyplot as plt
import numpy as np


def operator(case):
    if case > 0:
        return ">"
    elif case < 0:
        return "<"
    else:
        return "="


def G(X, params_list):
    a_1 = params_list[0]
    a_2 = params_list[1]
    b_12 = params_list[2]
    b_21 = params_list[3]
    c_1 = params_list[4]
    c_2 = params_list[5]
    return np.array([a_1 * X[0] - b_12 * X[0] * X[1] - c_1 * X[0] ** 2,
                     a_2 * X[1] - b_21 * X[0] * X[1] - c_2 * X[1] ** 2])


def competition_streamplot(a1, a2, b12, b21, c1, c2):
    params = [a1, a2, b12, b21, c1, c2]
    if b12 * b21 == c1 * c2:
        raise ValueError('Invalid argument values: b12 * b21 == c1 * c2')

    points = [(0, 0),
              (a1 / c1, 0),
              (0, a2 / c2)]

    case1 = np.sign(a1 * b21 - a2 * c1)
    case2 = np.sign(a2 * b12 - a1 * c2)

    if b12 * b21 != c1 * c2 and case1 * case2 > 0:
        points.append(((a2 * b12 - a1 * c2) / (b12 * b21 - c1 * c2),
                       (a1 * b21 - a2 * c1) / (b12 * b21 - c1 * c2)))

    x_min, y_min = 0, 0
    x_max = 1.5 * max([point[0] for point in points])
    y_max = 1.5 * max([point[1] for point in points])

    x = np.linspace(x_min, x_max, 10)
    y = np.linspace(y_min, y_max, 10)
    X, Y = np.meshgrid(x, y)
    U_, V_ = G([X, Y], params)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.175, top=0.93, right=0.8, left=0.2)

    fig.suptitle('Competition model')
    fig.supxlabel(f'a1 = {a1}, a2 = {a2}, b12 = {b12}, b21 = {b21}, c1 = {c1}, c2 = {c2}' + '\n\n' +
                  f'a1 * b21 {operator(case1)} a2 * c1' + ',     ' + f'a2 * b12 {operator(case2)} a1 * c2')

    axes.axis((x_min, x_max, y_min, y_max))
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')

    axes.streamplot(x, y, U_, V_,
                    density=2,
                    linewidth=1,
                    arrowstyle="->")

    for point in points:
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

competition_streamplot(a1, a2, b12, b21, c1, c2)
