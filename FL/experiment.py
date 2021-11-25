#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import torch
from copy import deepcopy


class Experiment:
    def __init__(self, args, training=True, verbose=True):
        self.__dict__.update(args)

        # Name of the simulation
        try:
            self.name = self.name(verbose)
        except:
            print("Some inputs are missing. args should include:")
            print("dataset, sampling, P_type, T, n_SGD, B, lr_g, lr_l, mu, decay")

        self.string = self.string_training()

        if training:
            # Previous simulations
            self.max_iter = self.previous_iter(verbose)
            self.training = self.max_iter < self.T

            self.previous_name = self.previous_name(deepcopy(args))
        else:
            self.max_iter = self.T
            self.previous_name = self.name

    def name(self, verbose):
        """Name of the experiment with input parameters"""

        params = [
            self.dataset,
            self.sampling,
            self.P_type,
            self.T,
            self.n_SGD,
            self.B,
            self.lr_g,
            self.lr_l,
        ]
        params = [str(p) for p in params]
        prefix = ["", "", "", "T", "E", "B", "lr_g", "lr_l"]
        name = "_".join([p + q for p, q in zip(prefix, params)])

        if self.sampling != "Full":
            name += f"_m{self.m}_s{self.seed}"
        if self.mu > 0.0:
            name += f"_mu{self.mu}"
        if self.decay < 1.0:
            name += f"_d{self.decay}"

        if verbose:
            print("experiment name:", name)
        return name

    def previous_iter(self, verbose):

        part1 = "_".join([self.dataset, self.sampling, self.P_type, "T"])

        part2 = f"_E{self.n_SGD}_B{self.B}_lr_g{self.lr_g}_lr_l{self.lr_l}"
        if self.sampling != "Full":
            part2 += f"_m{self.m}_s{self.seed}"
        if self.mu > 0.0:
            part2 += f"_mu{self.mu}"
        if self.decay < 1.0:
            part2 += f"_d{self.decay}"
        part2 += ".pkl"

        names = os.listdir("saved_exp_info/acc")

        names = [s[len(part1) :] for s in names if s[: len(part1)] == part1]
        names = [s[: -len(part2)] for s in names if s[-len(part2) :] == part2]
        iters = [int(s) for s in names]

        if len(names) > 0:
            max_iter = max(iters)
            if verbose:
                print(
                    f"{len(iters)} simulation with the same parameters has "
                    f"already been run"
                    f" with T={max_iter}."
                )
        else:
            max_iter = 0
            if verbose:
                print("No simulation with the same parameters has been run before.")
        return max_iter

    def previous_name(self, args):
        if self.max_iter == 0:
            return ""
        else:
            args["T"] = self.max_iter
            exp_previous = Experiment(args, training=False, verbose=False)
            return deepcopy(exp_previous.name)

    def hist_metric(self, metric: str, additional=True):
        """Initialize metric history.
        Use previous simulation if any already run"""

        hist = np.zeros((self.T + int(additional), self.n))

        if len(self.previous_name) > 0:
            path_previous = f"saved_exp_info/{metric}/{self.previous_name}.pkl"
            with open(path_previous, "rb") as file_content:
                hist_prev = pickle.load(file_content)

            n = min(self.T, self.max_iter)
            hist[: n + int(additional)] = hist_prev[: n + int(additional)]

        return hist

    def hists_init(self):
        self.acc = self.hist_metric("acc")
        self.loss = self.hist_metric("loss")
        self.agg_weights = self.hist_metric("agg_weights", False)
        self.sampled_clients = self.hist_metric("sampled_clients", False)

    def save_metric(self, variable: str):
        path_file = f"saved_exp_info/{variable}/{self.name}.pkl"
        with open(path_file, "wb") as output:
            pickle.dump(getattr(self, variable), output)

    def save_model(self, model):
        path_file = f"saved_exp_info/final_model/{self.name}.pth"
        torch.save(model.state_dict(), path_file)

    def save_metrics(self, *args):
        for variable in args:
            self.save_metric(variable)

    def string_training(self):

        return (
            f"--dataset {self.dataset} --sampling {self.sampling} "
            f"--n_SGD {self.n_SGD} --lr_l {self.lr_l} "
            f"--lr_g {self.lr_g} --m {self.m} "
            f"--B {self.B} --T {self.T} "
            f"--mu {self.mu} --P_type {self.P_type} "
            f"--decay {self.decay} --seed {self.seed} \n"
        )
