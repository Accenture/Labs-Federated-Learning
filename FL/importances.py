#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy.random import dirichlet
import pickle as pkl


def clients_importances(P_type: str, dataset: str, verbose=True):

    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        n_samples = pkl.load(output)

    if P_type == "ratio":
        P = n_samples / np.sum(n_samples)

    elif P_type == "uniform":
        P = np.ones(len(n_samples)) / len(n_samples)

    elif P_type[:9] == "dirichlet":

        np.random.seed(25)

        alpha = float(P_type[10:])

        P = dirichlet([alpha] * len(n_samples))

    else:
        print("P only supports `uniform`, `ratio`, and `dirichlet`.")

    if verbose:
        print(
            "P type: ",
            P_type,
            f", p_i = {round(np.mean(P), 3)} +- {round(np.std(P),3)}",
        )

    return P
