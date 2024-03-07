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

from FL.branch import Branch
from FL.server import Server


class TestAggregation(unittest.TestCase):
    def setUp(self):
        args = {
            "dataset": "test",
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
        }

        self.exp = Experiment(args, verbose=False)
        self.server = Server(args, self.exp.psi_star)
        self.server.forget_Wr([])


    def test_branch_1(self):
        """metric is always too high. Should return (0,0) for any branch"""

        self.server.metric[:, 1:, :] += 2 * self.exp.psi_star
        W_r = self.exp.policy[-1]

        for branch_init in [[[0, 0], [0, 500]], [[0, 0], [0, 500], [1, 300]]]:

            branch = Branch(branch_init)
            branch.update(self.server.metric, 3, W_r, self.server.psi_star)

            self.assertEqual(branch(), [[0, 0]])

    def test_branch_2(self):
        """identical metric for every client.
        Inferior to psi_star for t<500 and superior for t>=500.
        Hence, new_branch = branch + [r, 500] if t<500 and [[0, 0]] otherwise"""

        metric = np.zeros((self.exp.R, self.exp.T, self.exp.M))
        metric[:, :500] += 0.8 * self.server.psi_star
        metric[:, 500:] += 2 * self.server.psi_star

        r = 4
        W_r = self.exp.policy[-1]

        # All the nodes in the branch have been forgetting every client
        for branch_init in [
            [[0, 0]],
            [[0, 0], [0, 400]],
            [[0, 0], [0, 400], [1, 300]],
            [[0, 0], [0, 400], [1, 300], [2, 498]],
            [[0, 0], [1, 300]],
        ]:

            branch = Branch(branch_init)
            branch.update(metric, r, W_r, self.server.psi_star)

            self.assertEqual(branch(), branch_init + [[r - 1, 499]])

        # The second node in each branch does not forget W_r
        for branch_init in [
            [[0, 0], [0, 800]],
            [[0, 0], [0, 900], [1, 300]],
            [[0, 0], [0, 900], [1, 300], [2, 800]],
            [[0, 0], [1, 600]],
        ]:

            branch = Branch(branch_init)
            branch.update(metric, r, W_r, self.server.psi_star)

            self.assertEqual(branch(), [[0, 0], [branch_init[1][0], 499]])


if __name__ == "__main__":
    unittest.main()
