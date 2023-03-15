import numpy as np


def psi(sigma: float, epsilon: float, M: int) -> float:
    delta = 1 / M
    c = np.sqrt(2 * np.log(1.25 / delta))
    return epsilon * sigma / c


def get_std(sensitivity: str, epsilon: float, M: int) -> float:
    delta = 1 / M
    c = np.sqrt(2 * np.log(1.25 / delta))
    return sensitivity * c / epsilon
