#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
import os
import FL.experiment as experiment
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


def append_experiments_default(experiment, txt_name: str, seed: int, verbose, **args):
    """add to a .txt file all the experiments that have not been run yet"""

    text_file = open(txt_name, "a")

    # Create the experiment object
    args["seed"] = seed
    exp = experiment(args, verbose=False)

    # Add the parameters of this experiment if not run yet
    if not os.path.exists(f"./saved_exp_info/acc/{exp.file_name_train}.pkl")\
            and exp.file_name_train != exp.file_name:
        pass

    elif not os.path.exists(f"./saved_exp_info/acc/{exp.file_name}.pkl"):
        print(exp.file_name)
        text_file.write(exp.exp_string)

    elif verbose:
        exp.hists_load()
        print(exp.file_name)
        mean_hist = np.mean(exp.acc[0], axis=1)
        arg_max = np.argmax(mean_hist)
        print(arg_max, mean_hist[arg_max])

    text_file.close()


def append_experiments(
    txt_name: str,
    experiment,
    l_dataset: list[str],
    l_unlearn_scheme: list[str],
    l_forgetting: list[str],
    l_P_type: list[str],
    l_T: list[int],
    l_n_SGD: list[int],
    l_B: list[int],
    l_lr_g: list[float],
    l_lr_l: list[float],
    l_M: list[int],
    l_m: list[int],
    l_p_rework: list[float],
    l_epsilon: list[float],
    # l_delta: list[float],
    l_sigma: list[float],
    l_lambd: list[float],
    l_stop_acc: list[float],
    # l_C: list[float] = [1.],
    l_model: list[float],
    n_seeds=1,
    verbose=False,
):

    for (
            dataset,
            unlearn_scheme,
            forgetting,
            P_type,
            T,
            n_SGD,
            B,
            lr_g,
            lr_l,
            M,
            n_sampled,
            p_rework,
            epsilon,
            # delta,
            sigma,
            lambd,
            stop_acc,
            model,
    ) in product(
        l_dataset,
        l_unlearn_scheme,
        l_forgetting,
        l_P_type,
        l_T,
        l_n_SGD,
        l_B,
        l_lr_g,
        l_lr_l,
        l_M,
        l_m,
        l_p_rework,
        l_epsilon,
        # l_delta,
        l_sigma,
        l_lambd,
        l_stop_acc,
        l_model
    ):
        args = {
            "dataset_name": dataset,
            "unlearn_scheme": unlearn_scheme,
            "forgetting": forgetting,
            "P_type": P_type,
            "T": T,
            "n_SGD": n_SGD,
            "B": B,
            "lr_g": lr_g,
            "lr_l": lr_l,
            "M": M,
            "n_sampled": n_sampled,
            "p_rework": p_rework,
            "epsilon": epsilon,
            # "delta": delta,
            "sigma": sigma,
            "lambd": lambd,
            "mu": 0,
            "stop_acc": stop_acc,
            "model": model,
        }

        for seed in range(n_seeds):
            append_experiments_default(
                experiment,
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
        ["MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "celeba"]
):
    file_name = f"{learning_type}_{dataset}.txt"
    clean_txt(file_name)

l_std = [a * 10**-b for a, b in product([1., 2., 5.], [0, 1, 2, 3, 4])]
l_std.sort()

l_SIFU = ["train", "scratch", "SIFU", "fine-tuning", "FedAccum", "FedEraser"]
l_SIFU_std = ["train", "scratch", "SIFU", "fine-tuning", "FedAccum", "FedEraser"]

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
"""MNIST"""

# EXPLORATION
append_experiments(
    txt_name="FL_MNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["MNIST-shard_0"],
    l_unlearn_scheme=l_SIFU,
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[20, 100],
    l_lr_g=[1.],
    l_lr_l=[0.001, 0.002, 0.005, 0.01, 0.02],
    l_M=[100],
    l_m=[10, 25],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05, 0.1],
    l_lambd=[0.],
    l_stop_acc=[93.],
    l_model=["default"],
    verbose=args["verbose"]
)

append_experiments(
    txt_name="FL_MNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["MNIST-shard_0"],
    l_unlearn_scheme=l_SIFU + ["DP_0.5"],
    l_forgetting=["P9"] + l_scenario_class,
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[100],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[10],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05],
    l_lambd=[0.],
    l_stop_acc=[93.],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=n_seeds_main
)

#   with backdoor
append_experiments(
    txt_name="FL_MNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["MNIST-shard_1._backdoored"],
    l_unlearn_scheme=l_SIFU + ["DP_0.5"],
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[100],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[10],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05],
    l_lambd=[0.],
    l_stop_acc=[93., 98., 99., 99.9],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=n_seeds_backdoored
)

append_experiments(
    txt_name="FL_MNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["MNIST-shard_0"],
    l_unlearn_scheme=l_SIFU_std + ["DP_0.5"],
    l_forgetting=["P70"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[100],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[10],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=l_std,
    l_lambd=[0.],
    l_stop_acc=[93.],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=n_seeds_std
)

append_experiments(
    txt_name="FL_MNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["MNIST-shard_0"],
    l_unlearn_scheme=l_unlearn_DP,
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[100],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[10],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[-1],
    l_lambd=[0.],
    l_stop_acc=[88., 93.],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=n_seeds_DP
)

print("\n")

# SHOW THAT IT WORKS ON IFU AND SIFU
append_experiments(
    txt_name="FL_CIFAR10.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["CIFAR10_0.1"],
    l_unlearn_scheme=l_SIFU + ["DP_0.2"],
    l_forgetting=["P9"] + l_scenario_class,
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[5],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05],
    l_lambd=[0.],
    l_stop_acc=[90.],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_main
)

#   with backdoor
append_experiments(
    txt_name="FL_CIFAR10.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["CIFAR10_1._backdoored"],
    l_unlearn_scheme=l_SIFU + ["DP_0.2"],
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[5],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05],
    l_lambd=[0.],
    l_stop_acc=[90., 95., 99., 99.9],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_backdoored
)

# CONSIDER DIFFERENT NOISE LEVELS
append_experiments(
    txt_name="FL_CIFAR10.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["CIFAR10_0.1"],
    l_unlearn_scheme=l_SIFU_std + ["DP_0.2"],
    l_forgetting=["P70"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[5],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=l_std,
    l_lambd=[0.],
    l_stop_acc=[90.],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_std
)

# LOOK AT THE BEST CLIPPING CONSTANT
append_experiments(
    txt_name="FL_CIFAR10.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["CIFAR10_0.1"],
    l_unlearn_scheme=l_unlearn_DP,
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[5],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[-1],
    l_lambd=[0.],
    l_stop_acc=[85., 90.],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_DP
)

print("\n")


# EXPLORATION
append_experiments(
    txt_name="FL_CIFAR100.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["CIFAR100_0.1"],
    l_unlearn_scheme=l_SIFU,
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01, 0.02, 0.05],
    l_M=[100],
    l_m=[5],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.02, 0.05],
    l_lambd=[0.],
    l_stop_acc=[90.],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=1
)

# SHOW THAT IT WORKS ON IFU AND SIFU
append_experiments(
    txt_name="FL_CIFAR100.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["CIFAR100_0.1"],
    l_unlearn_scheme=l_SIFU + ["DP_0.2"],
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[5],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05],
    l_lambd=[0.],
    l_stop_acc=[90.],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_main
)

# backdoored
append_experiments(
    txt_name="FL_CIFAR100.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["CIFAR100_1._backdoored"],
    l_unlearn_scheme=l_SIFU + ["DP_0.2"],
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[5],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05],
    l_lambd=[0.],
    l_stop_acc=[90., 95., 99.],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_backdoored
)

# look at the ebst clipping constant
append_experiments(
    txt_name="FL_CIFAR100.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["CIFAR100_0.1"],
    l_unlearn_scheme=l_unlearn_DP,
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[5],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05],
    l_lambd=[0.],
    l_stop_acc=[85., 90.],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_DP
)

# CONSIDER DIFFERENT classes
append_experiments(
    txt_name="FL_CIFAR100.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["CIFAR100_0.1"],
    l_unlearn_scheme=l_SIFU + ["DP_0.2"],
    l_forgetting=[f"P7{i}" for i in range(10)],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[5],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05],
    l_lambd=[0.],
    l_stop_acc=[90.],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_main
)

# CONSIDER DIFFERENT NOISE LEVELS
append_experiments(
    txt_name="FL_CIFAR100.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["CIFAR100_0.1"],
    l_unlearn_scheme=l_SIFU_std + ["DP_0.2"],
    l_forgetting=["P70"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[5],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=l_std,
    l_lambd=[0.],
    l_stop_acc=[90.],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_std
)

print("\n")

# EXPLORATION
append_experiments(
    txt_name="FL_FashionMNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["FashionMNIST_0"],
    l_unlearn_scheme=l_SIFU,
    l_forgetting=["P9"] ,
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[10],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05, 0.1],
    l_lambd=[0.],
    l_stop_acc=[90.],
    l_model=["default"],
    verbose=args["verbose"]
)

# FIG 1
append_experiments(
    txt_name="FL_FashionMNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["FashionMNIST_0"],
    l_unlearn_scheme=l_SIFU + ["DP_0.5"],
    l_forgetting=["P9"] + l_scenario_class,
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[10],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.1],
    l_lambd=[0.],
    l_stop_acc=[90.],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=n_seeds_main
)

# backdoored
append_experiments(
    txt_name="FL_FashionMNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["FashionMNIST_1._backdoored"],
    l_unlearn_scheme=l_SIFU + ["DP_0.5"],
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[10],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.1],
    l_lambd=[0.],
    l_stop_acc=[90., 95., 99., 99.9],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=n_seeds_backdoored
)


# clipping constant
append_experiments(
    txt_name="FL_FashionMNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["FashionMNIST_0"],
    l_unlearn_scheme=l_unlearn_DP,
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[10],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.1],
    l_lambd=[0.],
    l_stop_acc=[85., 90.],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=n_seeds_DP
)

# UNLEARN EACH OF THE 10 CLASSES
append_experiments(
    txt_name="FL_FashionMNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["FashionMNIST_0"],
    l_unlearn_scheme=l_SIFU + ["DP_0.5"],
    l_forgetting=[f"P7{i}" for i in range(10)],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[10],
    l_p_rework=[1.],
    l_epsilon=[10.],
    # l_sigma=[0.01, 0.1, 0.2],
    l_sigma=[0.1],
    l_lambd=[0.],
    l_stop_acc=[90.],
    l_model=["default"],
    verbose=args["verbose"]
)

# Measure the impact of the noise
append_experiments(
    txt_name="FL_FashionMNIST.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["FashionMNIST_0"],
    l_unlearn_scheme=l_SIFU_std + ["DP_0.5"],
    l_forgetting=["P70"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.02],
    l_M=[100],
    l_m=[10],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=l_std,
    l_lambd=[0.],
    l_stop_acc=[90.],
    l_model=["default"],
    verbose=args["verbose"],
    n_seeds=n_seeds_std
)

print("\n")

# EXPLORATION
append_experiments(
    txt_name="FL_celeba.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["celeba"],
    l_unlearn_scheme=l_SIFU,
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[5, 10],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.05, 0.1, 0.01],
    l_M=[100],
    l_m=[10, 20],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.05, 0.1],
    l_lambd=[0.],
    l_stop_acc=[99.9],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=1
)

# SHOW THAT IT WORKS ON IFU AND SIFU
append_experiments(
    txt_name="FL_celeba.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["celeba"],
    l_unlearn_scheme=l_SIFU + ["DP_0.5"],
    l_forgetting=["P9"] + l_scenario_class,
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[20],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.1],
    l_lambd=[0.],
    l_stop_acc=[99.9],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_main
)


# BACKDOORED DATASET
append_experiments(
    txt_name="FL_celeba.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["celeba_backdoored"],
    l_unlearn_scheme=l_SIFU + ["DP_0.5"],
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[20],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.1],
    l_lambd=[0.],
    l_stop_acc=[99.9],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_backdoored
)

append_experiments(
    txt_name="FL_celeba.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["celeba"],
    l_unlearn_scheme=l_unlearn_DP,
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[20],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.1],
    l_lambd=[0.],
    l_stop_acc=[94.9, 99.9],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_DP
)

# CONSIDER DIFFERENT NOISE LEVELS
append_experiments(
    txt_name="FL_celeba.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["celeba"],
    l_unlearn_scheme=l_SIFU_std + ["DP_0.5"],
    l_forgetting=["P70"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[20],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=l_std,
    l_lambd=[0.],
    l_stop_acc=[99.9],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_std
)


# SHOW THAT IT WORKS ON IFU AND SIFU
append_experiments(
    txt_name="FL_celeba.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["celeba-leaf"],
    l_unlearn_scheme=l_SIFU + ["DP_0.5"],
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[20],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.1],
    l_lambd=[0.],
    l_stop_acc=[99.9],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_main
)

append_experiments(
    txt_name="FL_celeba.txt",
    experiment=FL.experiment.Experiment,
    l_dataset=["celeba-leaf"],
    l_unlearn_scheme=l_SIFU + ["DP_0.5"],
    l_forgetting=["P9"],
    l_P_type=["uniform"],
    l_T=[10**4],
    l_n_SGD=[10],
    l_B=[20],
    l_lr_g=[1.],
    l_lr_l=[0.01],
    l_M=[100],
    l_m=[20],
    l_p_rework=[1.],
    l_epsilon=[10.],
    l_sigma=[0.1],
    l_lambd=[0.],
    l_stop_acc=[99.9],
    l_model=["CNN"],
    verbose=args["verbose"],
    n_seeds=n_seeds_main
)