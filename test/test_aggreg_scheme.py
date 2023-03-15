import torch
import numpy as np
from context import FL
import unittest

from FL.create_model import simpleLinear
from FL.server import Server
from copy import deepcopy

class TestAggregScheme(unittest.TestCase):
    def setUp(self):
        self.n_clients = 5
        self.P_iid = np.ones(self.n_clients) / self.n_clients
        self.P_niid = np.random.dirichlet(np.ones(self.n_clients), size=1)[0]

        args = {
            "dataset_name": "test",
            "unlearn_scheme": "SIFU",
            "forgetting": "Ptest",
            "P_type": "uniform",
            "T": 10 ** 3,
            "n_SGD": 1,
            "B": 64,
            "lr_g": 1,
            "lr_l": 0.005,
            "M": 100,
            "p_rework": 1.0,
            "lambd": 0.0,
            "epsilon": 1.0,
            "delta": 0.1,
            "sigma": 0.1,
            "seed": 0,
            "clip": 1.,
            "model": "default",
        }

        self.server = Server(args)
        # self.l_lr_g = [0.0, 0.5, 1.0, 2.0]
        self.l_lr_g = [ 1.0]

    def comparison_models(self, P: np.array, lr_g: float):

        self.server.g_model = simpleLinear(1, 1, 42)
        m_rec = [simpleLinear(1, 1, seed) for seed in range(self.n_clients)]

        # Value of the parameter when manually computed
        delta_i = np.array([rec.params() - deepcopy(self.server.g_model.params())
                            for rec in m_rec])
        v_comparison = deepcopy(self.server.g_model.params()) \
                       + lr_g * P.dot(delta_i)

        # Value of the parameter with the agg. function in FedAvg
        self.server.P_kept = P
        self.server.lr_g = lr_g
        self.server.aggregation(m_rec)
        v_global_new = self.server.g_model.params()

        self.assertAlmostEqual(v_global_new, v_comparison, places=5)


    def identical(self, P: np.array, lr_g: float):

        self.server.g_model = simpleLinear(1, 1, 42)
        m_rec = [deepcopy(self.server.g_model) for _ in range(self.n_clients)]

        # Value of the parameter with the agg. function in FedAvg
        v_old = deepcopy(self.server.g_model.params())
        self.server.P_kept = P
        self.server.lr_g = lr_g
        self.server.aggregation(m_rec)
        v_global_new = self.server.g_model.params()

        self.assertAlmostEqual(v_old, v_global_new, places=5)


    def test_agg_Piid(self):
        for lr_g in self.l_lr_g:
            self.comparison_models(self.P_iid, lr_g)

    def test_agg_Pniid(self):
        for lr_g in self.l_lr_g:
            self.comparison_models(self.P_niid, lr_g)

    def test_agg_identical_iid(self):
        for lr_g in self.l_lr_g:
            self.identical(self.P_iid, lr_g)
            self.comparison_models(self.P_iid, lr_g)

    def test_agg_identical(self):
        for lr_g in self.l_lr_g:
            self.identical(self.P_niid, lr_g)
            self.comparison_models(self.P_niid, lr_g)


if __name__ == "__main__":
    unittest.main()
