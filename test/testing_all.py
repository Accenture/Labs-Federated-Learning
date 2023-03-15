import torch
import numpy as np
from context import FL
import unittest
from copy import deepcopy
import torch.nn.functional as F
from FL.experiment import Experiment
from torch.utils.data import Dataset, DataLoader
import torch

from FL.clients_schedule import clients_time_schedule
from FL.clients_parameters import clients_parameters
from FL.read_db import get_dataloaders
from FL.create_model import load_model


class simpleDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.Tensor(X)
        self.labels = torch.Tensor(y)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# class simpleLinear(torch.nn.Module):
#     def __init__(self, inputSize: int, outputSize: int, seed=0):
#         super(simpleLinear, self).__init__()
#         torch.manual_seed(seed)
#         self.linear = torch.nn.Linear(inputSize, outputSize, bias=True)
#
#     def forward(self, x):
#         return F.relu(self.linear(x))


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

        self.exp = Experiment(args)

        self.dls_train, _ = get_dataloaders(args["dataset"], args["B"], self.exp.M)

        self.model, self.loss_f = load_model("test", 42)


if __name__ == "__main__":
    unittest.main()
