#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
import os
from context import experiment


def clean_txt(txt_name):
    text_file = open(txt_name, "w")
    text_file.close()


def append_experiments_per_seeds(txt_name: str, n_seeds: int, **args):
    """
    add/create to a txt file all the experiments that have not already been run
    in the arguments of this function
    if new == True then a new txt file is created
    If not the experiments needed to run are added to the existing txt file
    """

    text_file = open(txt_name, "a")

    for seed in range(n_seeds):

        args["seed"] = seed
        exp = experiment.Experiment(args, False, False)
        file_name = exp.name

        if not os.path.exists(f"../saved_exp_info/acc/{file_name}.pkl"):
            string = exp.string
            text_file.write(string)

    text_file.close()

    file = open(txt_name, "r")
    nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
    print(txt_name, len(nonempty_lines))
    file.close()


def append_experiments(
    txt_name: str, m_list: list, datasets: list, P_types: list, samplings: list, **args
):

    clean_txt(txt_name)

    for m, dataset in zip(m_list, datasets):

        for P_type, sampling in product(P_types, samplings):
            args_specific = {
                "dataset": dataset,
                "m": m,
                "P_type": P_type,
                "sampling": sampling,
            }

            append_experiments_per_seeds(
                txt_name,
                n_seeds,
                **args,
                **args_specific,
            )


"""
COMMON PARAMETERS
"""
P_types = ["ratio", "uniform"]


"""
EXPERIMENTS FOR FIGURE 2
"""
args_Shakespeare_all = {
    "mu": 0,
    "T": 600,
    "n_SGD": 50,
    "lr_g": 1.0,
    "lr_l": 1.5,
    "B": 64,
    "decay": 1.0,
}

n_seeds = 30
samplings = ["MD", "Uniform", "Clustered"]

m_list = [5, 10, 20, 40]
datasets = ["Shakespeare4", "Shakespeare3", "Shakespeare2", "Shakespeare"]

txt_Shak_main = "Shak_main.txt"
append_experiments(
    txt_Shak_main, m_list, datasets, P_types, samplings, **args_Shakespeare_all
)


"""
EXPERIMENTS FOR ADDITIONAL SHAKESPEARE FIGURES WITH SMALL PERCENTAGE OF CLIENTS
"""
args_Shakespeare_percent = {
    "mu": 0,
    "T": 1000,
    "n_SGD": 50,
    "lr_g": 1.0,
    "lr_l": 1.5,
    "B": 64,
    "decay": 1.0,
}

n_seeds = 15
samplings = ["MD", "Uniform", "Clustered"]

m_list = [4, 8]
datasets = ["Shakespeare"] * len(m_list)

txt_Shak_percent = "Shak_percent.txt"

append_experiments(
    txt_Shak_percent, m_list, datasets, P_types, samplings, **args_Shakespeare_percent
)


"""
EXPERIMENTS FOR ADDITIONAL SHAKESPEARE FIGURES WITH K=1
"""
args_Shakespeare_K1 = {
    "mu": 0,
    "T": 2500,
    "n_SGD": 1,
    "lr_g": 1.0,
    "lr_l": 1.5,
    "B": 64,
    "decay": 1.0,
}

n_seeds = 15
samplings = ["MD", "Uniform", "Clustered"]

m_list = [8, 40]
datasets = ["Shakespeare"] * len(m_list)

txt_Shak_K1 = "Shak_K1.txt"

append_experiments(
    txt_Shak_K1, m_list, datasets, P_types, samplings, **args_Shakespeare_K1
)


"""
EXPERIMENTS FOR CIFAR
"""
args_CIFAR = {
    "mu": 0,
    "T": 1000,
    "n_SGD": 100,
    "lr_g": 1.0,
    "lr_l": 0.05,
    "B": 64,
    "decay": 1.0,
}

P_types = ["ratio"]
n_seeds = 30
samplings = ["MD", "Uniform", "Clustered"]

datasets = ["CIFAR10_0.1", "CIFAR10_0.01", "CIFAR10_0.001"]
m_list = [10] * len(datasets)

txt_CIFAR = "CIFAR.txt"

append_experiments(txt_CIFAR, m_list, datasets, P_types, samplings, **args_CIFAR)
