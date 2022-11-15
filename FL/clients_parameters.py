#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy.random import dirichlet
import pickle as pkl
from FL.clients_schedule import get_tau_i


def clients_parameters(
        dataset_name: str, n_clients: int, P_type: str, opt_scheme: str,
        time_scenario: str, modif: str, lr_l: float, verbose=True):

    # GET THE IMPORTANCE OF EVERY CLIENT
    dbs_path_len = f"./saved_exp_info/len_dbs/{dataset_name}_M{n_clients}.pkl"
    with open(dbs_path_len, "rb") as output:
        dbs_samples = pkl.load(output)

    if P_type == "ratio":
        P = dbs_samples / np.sum(dbs_samples)
    elif P_type == "uniform":
        P = np.ones(len(dbs_samples)) / len(dbs_samples)
    else:
        print("P only supports `uniform`, `ratio`")

    # GET THE SURROGATE WEIGHTS OF THE CLIENTS
    tau_i = get_tau_i(n_clients, time_scenario, True)
    if opt_scheme == "FL":
        P_surr = P
    elif opt_scheme == "Async":
        P_surr = tau_i**-1 / sum(tau_i**-1)
    elif opt_scheme.split("-")[0] == "FedFix":
        delta_t = float(opt_scheme.split("-")[1])
        P_surr = np.ceil(tau_i/delta_t) / sum(np.ceil(tau_i/delta_t))

    # GET THE AGGREGATION WEIGHTS
    if opt_scheme == "FL":
        # IF FL, SAME LEARNING RATE FOR EVERY CLIENT
        # WEIGHTS = CLIENTS IMPORTANCE, TO ENSURE CONVERGENCE (cf FedNOVA)
        lr_ls = np.ones(n_clients) * lr_l
        agg_weights = P

    elif modif == "identical":
        # SAME LEARNING RATE AND AGG. WEIGHTS FOR EVERY CLIENT
        lr_ls = np.ones(n_clients) * lr_l
        if opt_scheme == "Async":
            agg_weights = np.ones(n_clients)

    elif modif == "lr" and opt_scheme == "Async":
        # ADAPT EVERY CLIENT LEARNING RATE TO ITS COMPUTATION TIME
        lr_ls = sum(tau_i ** -1) * tau_i * P * lr_l
        agg_weights = np.ones(n_clients)

    elif modif == "weight" and opt_scheme == "Async":
        lr_ls = np.ones(n_clients) * lr_l
        agg_weights = sum(tau_i ** -1) * tau_i * P

    elif modif == "weight" and opt_scheme.split("-")[0] == "FedFix":
        delta_t = float(opt_scheme.split("-")[1])
        lr_ls = np.ones(n_clients) * lr_l
        agg_weights = np.ceil(tau_i/delta_t) * P

    if verbose:
        print(f"clients importance federated loss{P}")
        print(f"clients importance surrogate loss {P_surr}")
        print("local learning rates", lr_ls)
        print("aggregation weights", agg_weights)

    return P, P_surr, lr_ls, agg_weights
