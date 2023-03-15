#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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

        if self.unlearn_scheme == "train":
            self.forgetting = "P0"
            self.p_rework = 1.

        from policy import policies
        self.policy = policies[self.forgetting]
        self.file_name = self.name(verbose)
        self.file_name_train = self.file_name
        self.exp_string = self.string_training()

        if self.unlearn_scheme in ["scratch", "fine-tuning", "SIFU", "last"]:
            args_train = deepcopy(args)
            args_train["unlearn_scheme"] = "train"
            exp = Experiment(args_train, verbose=False)
            self.file_name_train = exp.file_name

        elif self.unlearn_scheme == "FedAccum":
            self.policy = [[]] + self.policy

        elif self.unlearn_scheme[:2] == "DP":
            self.policy = [[]] + self.policy

        self.R = len(self.policy)
        self.n_aggregs = [self.T] + [self.T] * self.R

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
            self.T,
            self.n_SGD,
            self.B,
            self.lr_g,
            self.lr_l,
            self.M,
            self.n_sampled,
            self.lambd,
            self.stop_acc,
            # self.clip
        ]
        params = [str(p) for p in params]
        prefix = ["", "", "", "", "T", "K", "B", "lr_g", "lr_l", "M", "m", "p", "lam", "S"]
        file_name += "_".join([p + q for p, q in zip(prefix, params)])

        if self.unlearn_scheme in \
                ["train", "SIFU", "scratch", "last", "fine-tuning"]:

            file_name += f"_e{self.epsilon}_sig{self.sigma}"

        elif self.unlearn_scheme[:2] == "DP":
            file_name += f"_e{self.epsilon}"

        file_name += f"_s{self.seed}"
        if verbose:
            print("experiment name:", file_name)

        return file_name

    def string_training(self) -> str:
        """create the string including all the arguments fort he experiment"""
        string = (
            f"--dataset_name {self.dataset_name} --unlearn_scheme {self.unlearn_scheme} "
            f"--T {self.T} --n_SGD {self.n_SGD} "
            f"--B {self.B} --lr_g {self.lr_g} --lr_l {self.lr_l} --M {self.M} "
            f"--seed {self.seed} --forgetting {self.forgetting} "
            f"--lambd {self.lambd} --stop_acc {self.stop_acc} --n_sampled {self.n_sampled} "
        )
        if self.unlearn_scheme in ["train", "SIFU", "scratch", "last", "fine-tuning"]:
            string += f"--epsilon {self.epsilon} --sigma {self.sigma} "
        elif self.unlearn_scheme[:2] == "DP":
            string += f"--epsilon {self.epsilon} "

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
            path_file = f"saved_exp_info/{variable}/{self.file_name}.pkl"
            arr_var = getattr(server, variable)
            max_server_agg = np.max(np.where(getattr(server, "acc")>0)[1])
            with open(path_file, "wb") as output:
                pickle.dump(arr_var[:, :max_server_agg+1], output)

    def save_global_model(self, model: Module):
        path_file = f"saved_exp_info/final_model/{self.file_name}.pth"
        torch.save(model.state_dict(), path_file)

    def save_best_models(self, file_name: str, l_models: list[Module]):
        for i, model in enumerate(l_models):
            path_file = f"saved_exp_info/final_model/{file_name}_{i}.pth"
            torch.save(model.state_dict(), path_file)

