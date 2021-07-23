#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy.random import dirichlet
import pickle as pkl


def clients_importances(importance_type: str, dataset: str):

    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        n_samples = pkl.load(output)

    if importance_type == "ratio":
        importances = n_samples / np.sum(n_samples)

    elif importance_type == "uniform":
        importances = np.ones(len(n_samples)) / len(n_samples)

    elif importance_type[:9] == "dirichlet":

        np.random.seed(25)

        alpha = float(importance_type[10:])

        importances = dirichlet([alpha] * len(n_samples))

    return importances
