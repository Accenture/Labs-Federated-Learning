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
from os.path import exists

class Experiment:
    def __init__(self, args: dict, verbose=True):
        self.__dict__.update(args)

        if self.unlearn_scheme == "train":
            self.forgetting = "P0"
            self.p_rework = 1.

        from policy import policies
        self.policy = policies[self.forgetting]
        self.limit_train_iter = args.get("limit_train_iter", 0)
        self.file_name = self.name(verbose)
        self.file_name_train = self.file_name
        self.exp_string = self.string_training()
        self.device = args.get("device", "cpu")

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
        # if self.compute_diff:
        #     prefix = ["", "", "", "T", "K", "B", "lr_g", "lr_l", "M", "m", "lam", "S"]
        # params = [str(p) for p in params]
        file_name += "_".join([p + q for p, q in zip(prefix, params)])
            

        if self.unlearn_scheme in \
                ["train", "SIFU", "scratch", "last", "fine-tuning"]:

            file_name += f"_e{self.epsilon}_sig{self.sigma}"\

        elif self.unlearn_scheme[:2] == "DP":
            file_name += f"_e{self.epsilon}"
        if self.model != "default":
            file_name += f"-{self.model}"


        if self.iter_min != 50:
            file_name += f"_{self.iter_min}"
        file_name += f"_s{self.seed}"
        if verbose:
            print("experiment name:", file_name)

        return file_name

    def string_training(self) -> str:
        """create the string including all the arguments for the experiment"""
        string = (
            f"--dataset_name {self.dataset_name} --unlearn_scheme {self.unlearn_scheme} "
            f"--P_type {self.P_type} --T {self.T} --n_SGD {self.n_SGD} "
            f"--B {self.B} --lr_g {self.lr_g} --lr_l {self.lr_l} --M {self.M} "
            f"--p_rework {self.p_rework} --seed {self.seed} --forgetting {self.forgetting} "
            f"--lambd {self.lambd} --stop_acc {self.stop_acc} --n_sampled {self.n_sampled} "
            f"--model {self.model} --iter_min {self.iter_min} "
        )
        # string = (
        #     f"--dataset_name {self.dataset_name} --unlearn_scheme {self.unlearn_scheme} "
        #     f"--T {self.T} --n_SGD {self.n_SGD} "
        #     f"--B {self.B} --lr_g {self.lr_g} --lr_l {self.lr_l} --M {self.M} "
        #     f"--seed {self.seed} --forgetting {self.forgetting} "
        #     f"--lambd {self.lambd} --stop_acc {self.stop_acc} --n_sampled {self.n_sampled} "
        # )
        if self.unlearn_scheme in ["train", "SIFU", "scratch", "last", "fine-tuning"]:
            string += f"--epsilon {self.epsilon} --sigma {self.sigma} "
        elif self.unlearn_scheme[:2] == "DP":
            string += f"--epsilon {self.epsilon} "

        string += "\n"
        return string

    def hist_load(self, metric: str) -> np.array:
        directories = ["saved_exp_info_iter", "saved_exp_info_send", "saved_exp_info_2", "saved_exp_info", "limit iter saved", "saved_exp_info_ori"]  # directories to attempt loading from
        hist = None

        for directory in directories:
            file_path = f"{directory}/{metric}/{self.file_name}.pkl"
            if os.path.exists(file_path):
                with open(file_path, "rb") as file_content:
                    hist = pickle.load(file_content)
                break
            # else:
            #     print(f"File not found at {file_path}, trying next directory if available.")

        if hist is None:
            print(f"File {self.file_name} could not be found in any of the directories.")
            # Depending on your error handling you might want to raise an error here or return None
            return None

        return hist

    def hists_load(self, suffix=''):
        if suffix:
            self.file_name += suffix
        self.acc = self.hist_load("acc")
        self.loss = self.hist_load("loss")
        self.metric = self.hist_load("metric")

    def hists_save(self, server: Server, *args):
        for variable in args:
            if self.limit_train_iter:
                path_file = f"limit_iter_saved/{variable}/{self.file_name}-limit{self.limit_train_iter}.pkl"
            else:
                path_file = f"saved_exp_info/{variable}/{self.file_name}.pkl"
            arr_var = getattr(server, variable)
            max_server_agg = np.max(np.where(getattr(server, "acc")>0)[1])
            with open(path_file, "wb") as output:
                pickle.dump(arr_var[:, :max_server_agg+1], output)

    def save_global_model(self, model: Module):
        if self.limit_train_iter:
            path_file = f"limit_iter_saved/final_model/{self.file_name}-limit{self.limit_train_iter}.pth"
        else:
            path_file = f"saved_exp_info/final_model/{self.file_name}.pth"
        torch.save(model.state_dict(), path_file)

    def save_best_models(self, file_name: str, l_models: list[Module]):
        for i, model in enumerate(l_models):
            path_file = f"saved_exp_info/final_model/{file_name}_{i}.pth"
            torch.save(model.state_dict(), path_file)

    def save_psi_bound(self, psi_bounds: list):
        path_file = f"saved_exp_SIFU_bounds/psi_bounds/{self.file_name}.pkl"
        if not exists(path_file):
            os.mkdir(f"saved_exp_SIFU_bound/psi_bounds/{self.file_name}")
        with open(path_file, "wb") as output:
            pickle.dump(psi_bounds, output)
    
    def save_model_history(self, aggreg_round, model: Module):
        path_file = f"saved_exp_SIFU_bound/saved_hist/{self.file_name}"
        if not exists(path_file):
            os.mkdir(f"saved_exp_SIFU_bound/saved_hist/{self.file_name}")
        torch.save(model.state_dict(), os.path.join(path_file, f"{aggreg_round}.pth"))