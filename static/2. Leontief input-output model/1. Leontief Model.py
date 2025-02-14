import numpy as np


def get_A(x, Y):
    """
    Returns input-output matrix (A)
    :param x: inter-industry matrix (np.array matrix)
    :param Y: external demand vector (np.array column)
    :return: input-output matrix (np.array matrix)
    """
    # X - production level vector (column)
    X = np.array(list(map(lambda Xi, Yi: sum(Xi) + Yi, x, Y)))
    # print('X =\n', X)
    A = x / X
    return A


def get_new_Y(prev_Y, k):
    """
    Returns new external demand vector (Y) after production level vector (X) changed
    :param prev_Y: external demand vector before X changed (np.array column)
    :param k: coefficients of relative change in X[i] (list)
    :return: new external demand vector Y (np.array column)
    """
    new_Y = prev_Y * np.array([k]).transpose()
    return new_Y


x = np.array([[50, 120, 80],
              [50, 180, 80],
              [25, 120, 40]])
Y = np.array([[60, 50, 35]]).transpose()

A = get_A(x, Y)
new_Y = get_new_Y(Y, [1.1, 1.5, 1.2])

print('A =\n', A)
print('new Y =\n', new_Y)


