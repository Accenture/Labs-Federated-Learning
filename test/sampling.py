from context import FL
import numpy as np

# Check sum to 1
n = 10
P_uni = np.array([1 / n] * n)
P_ratio = np.random.dirichlet(np.ones(n), size=1)[0]
n_tests = 10 ** 3


def sum1(sampling: str, n: int, m: int, **args):
    _, outputs = FL.sampling.sampling_clients(sampling, n, m, **args)
    assert (
        abs(sum(outputs) - 1) < 10 ** -10
    ), f"Test 1 - {sampling} - sum p_i = {sum(outputs)}"


def proba(sampling: str, P: np.array, n: int, m: int, n_tests: int, **args):
    outputs = [
        FL.sampling.sampling_clients(sampling, n, m, **args)[1] for _ in range(n_tests)
    ]
    mean = np.mean(outputs, axis=0)
    norm = np.linalg.norm(mean - P)
    assert norm < 10 ** -1, f"Test 2 - {sampling} - norm {norm}"


for m in range(1, n + 1):

    clusters_uni = FL.sampling.get_clusters_with_alg1(m, P_uni)
    clusters_ratio = FL.sampling.get_clusters_with_alg1(m, P_ratio)

    for _ in range(n_tests):
        sum1("MD", n, m, weights=P_uni)
        sum1("MD", n, m, weights=P_ratio)
        sum1("Uniform", n, m, weights=P_uni)
        sum1("Clustered", n, m, clusters=clusters_uni)
        sum1("Clustered", n, m, clusters=clusters_ratio)

    proba("MD", P_uni, n, m, n_tests, weights=P_uni)
    proba("MD", P_ratio, n, m, n_tests, weights=P_ratio)
    proba("Uniform", P_uni, n, m, n_tests, weights=P_uni)
    proba("Uniform", P_ratio, n, m, n_tests, weights=P_ratio)
    proba("Clustered", P_uni, n, m, n_tests, clusters=clusters_uni)
    proba("Clustered", P_ratio, n, m, n_tests, clusters=clusters_ratio)
