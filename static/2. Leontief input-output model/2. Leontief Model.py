import numpy as np


def get_X(A, Y):
    """
    Returns production level vector X
    :param A: input-output matrix (np.array matrix)
    :param Y: external demand vector (np.array column)
    :return: production level vector (np.array column)
    """
    # I - Identity matrix
    I = np.eye(max(A.shape))
    # B - Leontief inverse matrix
    B = np.linalg.inv(I-A)
    X = np.dot(B, Y)
    return X


A = np.array([[0.2, 0.3, 0.1],
              [0.3, 0.1, 0.2],
              [0.1, 0.2, 0.3]])
Y = np.array([[240, 20, 60]]).transpose()

X = get_X(A, Y)

print('X =', X.transpose())


