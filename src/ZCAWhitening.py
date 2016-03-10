import numpy as np


def zca(inputMatrix):
    """
    Applys zca whitening to input matrix
    :param 10000 X 3072 Input Matrix
    :return: returns Matrix with Data whitening
    """
    sigma = np.dot(inputMatrix, inputMatrix.T) / inputMatrix.shape[1]  # Correlation matrix
    U, S, V = np.linalg.svd(sigma)  # Singular Value Decomposition
    epsilon = 0.1  # Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)  # ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputMatrix)  # Data whitening
