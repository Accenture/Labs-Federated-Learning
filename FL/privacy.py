import numpy as np


def psi(epsilon: float, delta: float, sigma: float) -> float:
    c = np.sqrt(2 * np.log(1.25 / delta))
    return epsilon * sigma / c


def get_std(sensitivity: str, epsilon: float, delta: float) -> float:
    c = np.sqrt(2 * np.log(1.25 / delta))
    return sensitivity * c / epsilon
