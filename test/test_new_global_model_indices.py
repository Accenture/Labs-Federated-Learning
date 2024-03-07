import torch
import numpy as np
from context import FL
import unittest
from copy import deepcopy
import torch.nn.functional as F
from FL.experiment import Experiment
from torch.utils.data import Dataset, DataLoader
import torch

from FL.read_db import get_dataloaders
from FL.create_model import load_model

# from FL.FedProx import compute_psi_all, cut_branch, get_T_r
from FL.FedProx import compute_psi_all, cut_branch, get_T_r




class TestAggregation(unittest.TestCase):
    def setUp(self):
        args = {
            "dataset": "test",
            "unlearn_scheme": "SIFU",
            "forgetting": "P1",
            "P_type": "uniform",
            "T": 10 ** 3,
            "n_SGD": 1,
            "B": 64,
            "lr_g": 1,
            "lr_l": 0.005,
            "M": 2,
            "p_rework": 1.0,
            "lambd": 0.0,
            "epsilon": 1.0,
            "delta": 0.1,
            "sigma": 0.1,
            "seed": 0,
        }

        self.exp = Experiment(args, verbose=False)
        self.exp.hists_init()

        self.dls_train, _ = get_dataloaders(args["dataset"], args["B"], self.exp.M, False)

        self.model, self.loss_f = load_model("test", 42)

    def test_psi_unchanged(self):
        """Verifies that identical models give metrics with value 0"""

        model_g = self.model
        l_model_l = [self.model for _ in range(self.exp.M)]

        for t in range(self.exp.T):
            compute_psi_all(0, t, self.exp, model_g, l_model_l)

        self.assertTrue((self.exp.metric[0] == np.zeros((self.exp.T, self.exp.M))).all())

    def test_cut_branch_1(self):
        """metric is always too high. Should return (0, 0) for any branch"""

        self.exp.M = 5
        self.exp.forgetting = "P2"
        self.exp.forgetting_requests()
        self.exp.hists_init()

        self.exp.metric[:, 1:, :] += 2 * self.exp.psi_star
        # print(self.exp.metric)

        for branch in [[[0, 0], [0, 500]], [[0, 0], [0, 500], [1, 300]], [[0, 0], [1, 300]]]:

            new_branch = cut_branch(branch[-1][0] + 1, [0], deepcopy(branch), self.exp)
            # print(branch, new_branch)
            self.assertEqual(new_branch, [[0, 0]])

    def test_cut_branch_2(self):
        """metric is identical for every client.
        and equal to psi_star for t<500 and 2 psi_star for t>=500.
        Hence, new_branch[:-1] = branch if t<500 and [[0, 0]] otherwise"""

        self.exp.M = 5
        self.exp.forgetting = "P2"
        self.exp.forgetting_requests()
        self.exp.hists_init()

        for t, r in zip(range(self.exp.T), range(self.exp.R)):
            self.exp.metric[r, :500] += 0.8 * self.exp.psi_star
            self.exp.metric[r, 500:] += 2 * self.exp.psi_star

        # All the nodes in the branch have been forgetting every client
        for branch in [
            [[0, 0], [0, 400]],
            [[0, 0], [0, 400], [1, 300]],
            [[0, 0], [0, 400], [1, 300], [2, 499]],
            [[0, 0], [1, 300]],
        ]:
            new_branch = cut_branch(-42, [0], deepcopy(branch), self.exp)
            self.assertEqual(new_branch[:-1], branch)

        # The second node in each branch does not forget W_r
        for branch in [
            [[0, 0], [0, 800]],
            [[0, 0], [0, 900], [1, 300]],
            [[0, 0], [0, 900], [1, 300], [2, 800]],
            [[0, 0], [1, 600]],
        ]:

            new_branch = cut_branch(branch[-1][0] + 1, [0], deepcopy(branch), self.exp)
            self.assertEqual(new_branch, [[0, 0]])

    def test_T(self):
        """metric is identical for every client.
        and equal to psi_star for t<500 and 2 psi_star for t>=500.
        Hence, new_branch[:-1] = branch if t<500 and [[0, 0]] otherwise"""

        self.exp.M = 5
        self.exp.forgetting = "P2"
        self.exp.forgetting_requests()
        self.exp.hists_init()

        for t, r in zip(range(self.exp.T), range(self.exp.R)):
            self.exp.metric[r, :500] += 0.8 * self.exp.psi_star
            self.exp.metric[r, 500:] += 2 * self.exp.psi_star

        for r in range(self.exp.R):
            T = get_T_r(r, [r, 0], self.exp)
            self.assertEqual(T, 499)


if __name__ == "__main__":
    unittest.main()
