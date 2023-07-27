from itertools import product

policies = {
    "P0": [],  # Every client is kept
    # "P0": [0],
    "P1": [1],
    "P2": [2],
    "P3": [3],
}

CONFIG_SIFU = {
    # Combination of epsilon, delta, and sigma
    f"C{i}": {"epsilon": epsilon, "delta": delta, "sigma": sigma}
    for i, (delta, epsilon, sigma) in enumerate(product(
        [0.01, 0.025, 0.1],
        [0.1, 0.3, 1., 3., 10.],
        [0.01, 0.025, 0.1],
    ))
}

CONFIG_PGD = {
    f"C{i}": {"n_SGD": n_SGD, "tau": tau}
    for i, (n_SGD, tau) in enumerate(product(
        [10, 100, 1000],
        [0.12]
    ))
}
