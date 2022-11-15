#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import torch
from copy import deepcopy
from torch.nn import Module
from FL.server import Server


class Experiment:
    def __init__(self, args, training=True, verbose=True):
        self.__dict__.update(args)

        self.file_name = self.name(verbose)
        self.string = self.string_training()

    def name(self, verbose):
        """Name of the experiment with input parameters"""

        params = [
            self.dataset_name,
            self.opt_scheme,
            self.time_scenario,
            self.P_type,
            self.T,
            self.n_SGD,
            self.B,
            self.lr_g,
            f"{self.lr_l:.1}",
            self.M,
        ]
        params = [str(p) for p in params]
        prefix = ["", "", "", "", "T", "K", "B", "lr_g", "lr_l", "M"]
        name = "_".join([p + q for p, q in zip(prefix, params)])

        name += f"_s{self.seed}"

        if verbose:
            print("experiment name:", name)

        return name

    def string_training(self):
        return (
            f"--dataset_name {self.dataset_name} --opt_scheme {self.opt_scheme} "
            f"--time_scenario {self.time_scenario} "
            f"--P_type {self.P_type} --T {self.T} --n_SGD {self.n_SGD} "
            f"--B {self.B} --lr_g {self.lr_g} --lr_l {self.lr_l:.1} --M {self.M} "
            f"--seed {self.seed}\n"
        )

    def hist_load(self, metric: str) -> np.array:
        with open(f"saved_exp_info/{metric}/{self.file_name}.pkl", "rb") as file_content:
            hist = pickle.load(file_content)
        return hist

    def hists_load(self):
        self.acc = self.hist_load("acc")
        self.loss = self.hist_load("loss")

    def hists_save(self, server: Server, *args):
        for variable in args:
            path_file = f"saved_exp_info/{variable}/{self.file_name}.pkl"
            with open(path_file, "wb") as output:
                pickle.dump(getattr(server, variable), output)

    def save_global_model(self, model: Module):
        path_file = f"saved_exp_info/final_model/{self.file_name}.pth"
        torch.save(model.state_dict(), path_file)
