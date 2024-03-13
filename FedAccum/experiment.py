#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import torch
from copy import deepcopy
from torch.nn import Module
from FL.server import Server
from FL.privacy import psi

class Experiment:
    def __init__(self, args: dict, verbose=True):
        self.__dict__.update(args)

        from policy import policies
        if self.unlearn_scheme == "train":
            self.forgetting = "P0"
            self.p_rework = 1.


        self.policy = policies[self.forgetting]
        if self.unlearn_scheme[:2] == "DP":
         self.policy = [[]] + self.policy

        self.R = len(self.policy)
        self.n_aggregs = [self.T] + [int(self.T * self.p_rework)] * (self.R - 1)

        self.file_name = self.name(verbose)
        self.exp_string = self.string_training()

        self.file_name_train = self.file_name
        if args["unlearn_scheme"] != "train" and args["forgetting"] != "P0":
            args_train = deepcopy(args)
            if args["unlearn_scheme"] in ["scratch", "fine-tuning", "SIFU", "last"]:
                args_train["unlearn_scheme"] = "train"
            elif args["unlearn_scheme"][:2] == "DP":
                args_train["forgetting"] = "P0"
            exp = Experiment(args_train, verbose=False)
            self.file_name_train = exp.file_name

        # if self.unlearn_scheme == "SIFU":
        self.psi_star = psi(self.sigma, self.epsilon, self.M)

        # DIFFERENT MEASURES ON THE DATASET
        self.acc, self.loss, self.metric = None, None, None

    def name(self, verbose: str) -> str:
        """file_name of the experiment based on input parameters"""

        file_name = "FL_"

        params = [
            self.dataset_name,
            self.unlearn_scheme,
            self.forgetting,
            self.P_type,
            self.T,
            self.n_SGD,
            self.B,
            self.lr_g,
            self.lr_l,
            self.M,
            self.n_sampled,
            self.p_rework,
            self.lambd,
            self.stop_acc,
            # self.clip
        ]
        params = [str(p) for p in params]
        prefix = ["", "", "", "", "T", "K", "B", "lr_g", "lr_l", "M", "m", "p", "lam", "S"]
        file_name += "_".join([p + q for p, q in zip(prefix, params)])

        # if self.unlearn_scheme == "SIFU":
        params = [
            self.epsilon,
            # self.delta,
            self.sigma,
        ]
        params = [str(p) for p in params]
        prefix = ["e", "sig"]
        file_name += "_" + "_".join([p + q for p, q in zip(prefix, params)])

        if self.model != "default":
            file_name += f"-{self.model}"

        file_name += f"_s{self.seed}"
        if verbose:
            print("experiment name:", file_name)

        return file_name

    def string_training(self) -> str:
        """create the string including all the arguments fort he experiment"""
        string = (
            f"--dataset_name {self.dataset_name} --unlearn_scheme {self.unlearn_scheme} "
            f"--P_type {self.P_type} --T {self.T} --n_SGD {self.n_SGD} "
            f"--B {self.B} --lr_g {self.lr_g} --lr_l {self.lr_l} --M {self.M} "
            f"--p_rework {self.p_rework} --seed {self.seed} --forgetting {self.forgetting} "
            f"--lambd {self.lambd} --stop_acc {self.stop_acc} --n_sampled {self.n_sampled} "
            f"--epsilon {self.epsilon} --sigma {self.sigma} --model {self.model}"
        )

        string += "\n"
        return string

    def hist_load(self, metric: str) -> np.array:
        with open(f"saved_exp_info/{metric}/{self.file_name}.pkl", "rb") as file_content:
            hist = pickle.load(file_content)
        return hist

    def hists_load(self):
        self.acc = self.hist_load("acc")
        self.loss = self.hist_load("loss")
        self.metric = self.hist_load("metric")

    def hists_save(self, server: Server, *args):
        for variable in args:
            if self.limit_train_iter:
                path_file = f"limit_iter_saved/{variable}/{self.file_name}-limit{self.limit_train_iter}.pkl"
            else:
                path_file = f"saved_exp_info/{variable}/{self.file_name}.pkl"
            with open(path_file, "wb") as output:
                pickle.dump(getattr(server, variable), output)

    def save_global_model(self, model: Module):
        path_file = f"saved_exp_info/final_model/{self.file_name}.pth"
        torch.save(model.state_dict(), path_file)

    def save_best_models(self, file_name: str, l_models: list[Module]):
        for i, model in enumerate(l_models):
            path_file = f"saved_exp_info/final_model/{file_name}_{i}.pth"
            torch.save(model.state_dict(), path_file)

