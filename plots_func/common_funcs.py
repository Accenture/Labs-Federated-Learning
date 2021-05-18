#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


def get_acc_loss(
    dataset: str,
    sampling: str,
    metric: str,
    n_SGD: int,
    seed: int,
    lr: float,
    decay: float,
    p: float,
    mu,
    sampling_types=[
        "random",
        "clustered_2",
        "clustered_2",
        "clustered_2",
        "perfect",
        "clustered_1",
        "FedAvg",
    ],
    similarities=["any", "L2", "L1", "cosine", "any", "any", "any"],
    names_legend=["MD", "L2", "L1", "Alg. 2", "Target", "Alg. 1", "FedAvg"],
):

    if mu == "":
        mu = 0.0

    from py_func.hyperparams import get_file_name

    def get_one_acc_loss(sampling: str, sim_type: str):
        file_name = get_file_name(
            dataset, sampling, sim_type, seed, n_SGD, lr, decay, p, mu
        )
        print(file_name)
        path = f"saved_exp_info/{metric}/{file_name}.pkl"
        return pkl.load(open(path, "rb"))

    hists, legend = [], []

    for sampling, sampling_type, name in zip(
        sampling_types, similarities, names_legend
    ):

        try:
            hist = get_one_acc_loss(sampling, sampling_type)

            hists.append(hist)
            legend.append(name)

        except:
            pass

    return hists, legend


def weights_clients(dataset: str):
    """Return the dataset name with CIFAR10 instead of CIFAR20"""
    if dataset[:5] == "CIFAR":
        dataset = list(dataset)
        dataset[5] = "1"
        dataset = "".join(dataset)

    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        weights = pkl.load(output)
    weights = weights / np.sum(weights)

    return weights
