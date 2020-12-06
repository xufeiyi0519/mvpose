import numpy as np


def laplace_function(x, lambda_):
    return (1 / (2 * lambda_)) * np.e ** (-1 * (np.abs(x) / lambda_))
