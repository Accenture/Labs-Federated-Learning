import torch
import numpy as np
from context import FL
import unittest


class simpleLinear(torch.nn.Module):
    def __init__(self, inputSize: int, outputSize: int, seed=0):
        super(simpleLinear, self).__init__()
        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)

    def params(self):
        return np.array([layer.data.numpy() for layer in self.parameters()])[0, 0]


def aggreg_simple(model, local_models: list, P: np.array, lr_g: int):
    """returns the parameter of the new model"""
    contrib_locals = np.array([m.params() - model.params() for m in local_models])

    new_model = model.params() + lr_g * P.dot(contrib_locals)
    return new_model


def simu_agg(n: int, P: np.array, lr_g: float):
    """returns the parameter with FL.aggregation and with `aggreg_simple`"""

    m_global = simpleLinear(1, 1)
    m_local = [simpleLinear(1, 1, seed) for seed in range(n)]

    m_global_new = FL.FedProx.aggregation(m_global, m_local, P, lr_g)
    v_global_new = m_global_new.params()

    v_comparison = aggreg_simple(m_global, m_local, P, lr_g)

    return v_global_new[0], v_comparison[0]


class TestAggregation(unittest.TestCase):
    def test_agg_P_uni(self):
        n = 5
        P_uni = np.array([1 / n] * n)
        lr_g = 1.0
        self.assertAlmostEqual(*simu_agg(n, P_uni, lr_g))

    def test_agg_P_ratio(self):
        n = 5
        P_ratio = np.random.dirichlet(np.ones(n), size=1)[0]
        lr_g = 1.0
        self.assertAlmostEqual(*simu_agg(n, P_ratio, lr_g))

    def test_lr_g(self):
        n = 5
        P_uni = np.array([1 / n] * n)
        P_ratio = np.random.dirichlet(np.ones(n), size=1)[0]
        for lr_g in [0.0, 0.5, 1.0, 2.0]:
            self.assertAlmostEqual(*simu_agg(n, P_uni, lr_g))
            self.assertAlmostEqual(*simu_agg(n, P_ratio, lr_g))


if __name__ == "__main__":
    unittest.main()
