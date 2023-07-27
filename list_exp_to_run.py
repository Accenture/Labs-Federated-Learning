#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
import os
from FL.experiment import Experiment_Names
from copy import deepcopy

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Experiment Parameters")

parser.add_argument(
    "--verbose",
    action=argparse.BooleanOptionalAction,
    default=False,
)


def clean_txt(txt_name):
    text_file = open(txt_name, "w")
    text_file.close()


def append_experiments_default(txt_name: str, seed: int, verbose, **args):
    """add to a .txt file all the experiments that have not been run yet"""

    text_file = open(txt_name, "a")

    # CREATE THE EXPERIMENT OBJECT
    args["seed"] = seed
    exp = Experiment_Names(args, verbose=verbose)

    # ADD THE PARAMETERS OF THIS EXPERIMENT IF NOT RUN YET
    if not os.path.exists(f"./saved_exp_info/loss/{exp.file_name}.pkl"):
        text_file.write(exp.exp_string)

    # elif verbose:
    # exp.hists_load()
    # print(exp.file_name)
    # mean_hist = np.mean(exp.acc[0], axis=1)
    # arg_max = np.argmax(mean_hist)
    # print(arg_max, mean_hist[arg_max])

    text_file.close()


def append_experiments(
    txt_name: str,
    project_name: str,
    # experiment,
    l_dataset: list[str],
    l_unlearn_scheme: list[str],
    l_opti: list[str],
    l_forgetting: list[str],
    l_P_type: list[str],
    l_T: list[int],
    l_n_SGD: list[int],
    l_B: list[int],
    l_lr_g: list[float],
    l_lr_l: list[float],
    # l_M: list[int],
    # l_m: list[int],
    # l_p_rework: list[float],
    # l_epsilon: list[float],
    # l_delta: list[float],
    # l_sigma: list[float],
    l_dropout: list[float],
    l_lambd: list[float],
    # l_stop_acc: list[float],
    # l_C: list[float] = [1.],
    l_model: list[float],
    n_seeds=1,
    save_models=False,
    verbose=False,
    # iter_min=50
):

    for (
            dataset,
            unlearn_scheme,
            opti,
            forgetting,
            P_type,
            T,
            n_SGD,
            B,
            lr_g,
            lr_l,
            # M,
            # n_sampled,
            # p_rework,
            # epsilon,
            # delta,
            # sigma,
            dropout,
            lambd,
            # stop_acc,
            model,
    ) in product(
        l_dataset,
        l_unlearn_scheme,
        l_opti,
        l_forgetting,
        l_P_type,
        l_T,
        l_n_SGD,
        l_B,
        l_lr_g,
        l_lr_l,
        # l_M,
        # l_m,
        # l_p_rework,
        # l_epsilon,
        # l_delta,
        # l_sigma,
        l_dropout,
        l_lambd,
        # l_stop_acc,
        l_model
    ):
        args = {
            "exp_name": project_name,
            "dataset_name": dataset,
            "unlearn_scheme": unlearn_scheme,
            "opti": opti,
            "forgetting": forgetting,
            "P_type": P_type,
            "T": T,
            "n_SGD": n_SGD,
            "B": B,
            "lr_g": lr_g,
            "lr_l": lr_l,
            "M": 4,
            # "n_sampled": n_sampled,
            # "p_rework": p_rework,
            # "epsilon": epsilon,
            # "delta": delta,
            # "sigma": sigma,
            "dropout": dropout,
            "lambd": lambd,
            "mu": 0,
            # "stop_acc": stop_acc,
            "model": model,
            # "iter_min": iter_min
            "save_unlearning_models": save_models
        }

        for seed in range(n_seeds):
            append_experiments_default(
                # experiment,
                txt_name,
                seed,
                verbose,
                **args,
            )

    remove_redundant(txt_name)


def remove_redundant(txt_name):

    with open(f"{txt_name}", "r") as f:
        unique_lines = set(f.readlines())
    with open(f"{txt_name}", "w") as f:
        f.writelines(unique_lines)

    print(txt_name, len(unique_lines))


"""RESET TXT FILES"""
for learning_type, dataset in product(
        ["FL"],
        ["MNIST", "prostate", "prostate_one", "prostate_unlearn",
         "FedEraser", "FedEraser_half", "prostate_alone"]
):
    file_name = f"{learning_type}_{dataset}.txt"
    clean_txt(file_name)

l_std = [a * 10**-b for a, b in product([1., 2., 5.], [0, 1, 2, 3, 4])]
l_std.sort()

l_SIFU = ["train", "scratch", "SIFU", "fine-tuning", "FedAccum"]
l_SIFU_std = ["train", "scratch", "SIFU", "fine-tuning", "FedAccum"]

n_seeds_main = 10
n_seeds_std = 5
n_seeds_DP = 0
n_seeds_backdoored = 10
n_seeds_class = 5

l_scenario_class = [f'P7{i}' for i in range(10)]

l_unlearn_DP = [f"DP_{c}" for c in [0.001, 0.002, 0.005,
                                    0.01, 0.02, 0.05,
                                    0.1, 0.2, 0.5,
                                    1., 2., 5.]]

"""
-------------------
MEASURE IMPORTANCE OF TIME BASED WEIGHTS
-------------------
"""

import FL

args = vars(parser.parse_args())

project_name = "prostate_v11"

"""prostate"""

for n_SGD, T in [
    # (50, 50),
    # (25, 100),
    (10, 250), (5, 500)
]:
    append_experiments(
        txt_name="FL_prostate.txt",
        project_name=project_name,
        l_dataset=["prostate_split"],
        l_unlearn_scheme=["train"],
        l_opti=["AdamW", "SGD"],
        l_forgetting=["P0"],
        l_P_type=["uniform"],
        l_T=[T],
        l_n_SGD=[n_SGD],
        l_B=[8],
        l_lr_g=[1.],
        l_lr_l=[0.005],
        l_dropout=[0.3, 0.4, 0.5],
        l_lambd=[0.],
        l_model=["default"],
        verbose=args["verbose"],
        n_seeds=1,
    )

    append_experiments(
        txt_name="FL_prostate.txt",
        project_name=project_name,
        l_dataset=["prostate_split"],
        l_unlearn_scheme=["train"],
        l_opti=["AdamW"],
        l_forgetting=["P0"],
        l_P_type=["uniform"],
        l_T=[T],
        l_n_SGD=[n_SGD],
        l_B=[8],
        l_lr_g=[1.],
        l_lr_l=[0.0001, 0.001, 0.01],
        l_dropout=[0.5],
        l_lambd=[0.],
        l_model=["default"],
        verbose=args["verbose"],
        n_seeds=1,
    )





project_name = "prostate_v11"
l_epsilon = [0.1, 1., 10.]
l_delta = [0.01, 0.025, 0.1]


sigma =0.025
from variables import CONFIG_SIFU
l_keys = []
for key in CONFIG_SIFU.keys():
    if (CONFIG_SIFU[key]["epsilon"] in l_epsilon
            and CONFIG_SIFU[key]["delta"] in l_delta
            and CONFIG_SIFU[key]["sigma"] == sigma):
        l_keys.append(key)

print(l_keys)


# append_experiments(
#     txt_name="FL_prostate_unlearn.txt",
#     project_name=project_name,
#     l_dataset=[f"prostate_split_s{i}" for i in range(5)],
#     l_unlearn_scheme=["train"],
#     l_opti=["AdamW"],
#     l_forgetting=["P0"],
#     l_P_type=["uniform"],
#     l_T=[500],
#     l_n_SGD=[5],
#     l_B=[8],
#     l_lr_g=[1.],
#     l_lr_l=[0.001],
#     l_dropout=[0.2],
#     l_lambd=[0.],
#     l_model=["default"],
#     verbose=args["verbose"],
#     n_seeds=5,
#     save_models=True,
#
# )

# append_experiments(
#     txt_name="FL_prostate_alone.txt",
#     project_name=project_name,
#     l_dataset=[f"prostate_C{i}" for i in range(1, 5)],
#     l_unlearn_scheme=["alone"],
#     l_opti=["AdamW"],
#     l_forgetting=["P0"],
#     l_P_type=["uniform"],
#     l_T=[500],
#     l_n_SGD=[5],
#     l_B=[8],
#     l_lr_g=[1.],
#     l_lr_l=[0.001],
#     l_dropout=[0.2],
#     l_lambd=[0.],
#     l_model=["default"],
#     verbose=args["verbose"],
#     n_seeds=1,
#     save_models=True,
#
# )
#
# from variables import CONFIG_SIFU
append_experiments(
    txt_name="FL_prostate_unlearn.txt",
    project_name=project_name,
    l_dataset=[f"prostate_split_s{i}" for i in range(5)],
    l_unlearn_scheme=[f"SIFUimp-{key}" for key in l_keys] +
                     ["FedAccum", "scratch", "finetuning", "PGD-C2", "FUKD"],
    l_opti=["AdamW"],
    l_forgetting=["P1"],
    l_P_type=["uniform"],
    l_T=[500],
    l_n_SGD=[5],
    l_B=[8],
    l_lr_g=[1.],
    l_lr_l=[0.001],
    l_dropout=[0.2],
    l_lambd=[0.],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=5,
    save_models=True,

)

append_experiments(
    txt_name="FL_FedEraser.txt",
    project_name=project_name,
    l_dataset=[f"prostate_split_s{i}" for i in range(5)],
    l_unlearn_scheme=["FedEraser"],
    l_opti=["AdamW"],
    l_forgetting=["P1"],
    l_P_type=["uniform"],
    l_T=[500],
    l_n_SGD=[5],
    l_B=[8],
    l_lr_g=[1.],
    l_lr_l=[0.001],
    l_dropout=[0.2],
    l_lambd=[0.],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=5,
    save_models=True,

)

append_experiments(
    txt_name="FL_FedEraser_half.txt",
    project_name=project_name,
    l_dataset=[f"prostate_split_s{i}" for i in range(5)],
    l_unlearn_scheme=["FedEraser_half"],
    l_opti=["AdamW"],
    l_forgetting=["P1"],
    l_P_type=["uniform"],
    l_T=[500],
    l_n_SGD=[5],
    l_B=[8],
    l_lr_g=[1.],
    l_lr_l=[0.001],
    l_dropout=[0.2],
    l_lambd=[0.],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=5,
    save_models=True,

)
#
# append_experiments(
#     txt_name="FL_prostate_unlearn.txt",
#     project_name=project_name,
#     l_dataset=["prostate_split"],
#     l_unlearn_scheme=[f"SIFUimp-C{i}" for i in range(9)] +
#                      ["FedAccum", "scratch", "finetuning", "PGD-C2", "FUKD"],
#     l_opti=["AdamW"],
#     l_forgetting=["P2", "P3"],
#     l_P_type=["uniform"],
#     l_T=[500],
#     l_n_SGD=[5],
#     l_B=[8],
#     l_lr_g=[1.],
#     l_lr_l=[0.001],
#     l_dropout=[0.2],
#     l_lambd=[0.],
#     l_model=["default"],
#     verbose=args["verbose"],
#     n_seeds=1,
#     save_models=True,
#
# )