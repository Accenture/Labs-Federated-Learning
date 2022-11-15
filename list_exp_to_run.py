#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
import os
import FL.experiment as experiment
from copy import deepcopy

def clean_txt(txt_name):
    text_file = open(txt_name, "w")
    text_file.close()


def append_experiments_default(txt_name: str, seed: int, **args):
    """add to a .txt file all the experiments that have not been run yet"""

    text_file = open(txt_name, "a")

    # Create the experiment object
    args["seed"] = seed
    exp = experiment.Experiment(args, False, False)

    # Add the parameters of this experiment if not run yet
    if not os.path.exists(f"./saved_exp_info/acc/{exp.file_name}.pkl"):
        string = exp.string
        text_file.write(string)

    text_file.close()



def remove_redundant(txt_name):
    with open(f"{txt_name}", 'r') as f:
        unique_lines = set(f.readlines())
    with open(f"{txt_name}", 'w') as f:
        f.writelines(unique_lines)

    print(txt_name, len(unique_lines))


def append_experiments(
    txt_name: str,
    l_dataset:list,
    l_opt_scheme : list,
    l_time_scenario: list,
    l_P_type: list,
    l_T:list,
    l_n_SGD: list,
    l_B: list,
    l_lr_g: list,
    l_lr_l:list,
    l_M:list,
    n_seeds:int,
):

    for dataset, opt_scheme, time_scenario, P_type, T, n_SGD, B, lr_g, lr_l, M\
        in product(l_dataset, l_opt_scheme, l_time_scenario, l_P_type, l_T,
                   l_n_SGD, l_B, l_lr_g, l_lr_l, l_M):
        args = {
            "dataset_name": dataset,
            "opt_scheme": opt_scheme,
            "time_scenario": time_scenario,
            "P_type": P_type,
            "T": T,
            "n_SGD": n_SGD,
            "B": B,
            "lr_g": lr_g,
            "lr_l": lr_l,
            "M": M,
            "mu": 0,
        }


        append_experiments_default(
            txt_name,
            0,
            **args,
        )
    remove_redundant(txt_name)



"""
COMMON PARAMETERS
"""
l_B = [64]
n_seeds = 1
opt_surrogate = ["Async_weight", "Async_identical"]
opt_all = ["FL_weight", "Async_weight",
           "FedFix-0.2_weight", "FedFix-0.5_weight", "FedFix-0.8_weight",
           "FedBuff-5_identical", "FedBuff-10_identical"]

names = [ds + ".txt" for ds in
         ["MNIST", "CIFAR10", "CIFAR10-flat", "CIFAR10-flat-full",  "Shakespeare",
          "CIFAR100", "CIFAR100-flat"]]
for file_name in names:
    clean_txt(file_name)



"""
MNIST experiments
"""
l_lr_MNIST = [0.00001, 0.00002, 0.00005,
              0.0001, 0.0002, 0.0005,
              0.001, 0.002, 0.005,
              0.01, 0.02, 0.05,
              0.1, 0.2, 0.5]
append_experiments(
    txt_name="MNIST.txt",
    l_dataset=["MNIST_0.", "MNIST_0.1", "MNIST_10000"],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[200000],
    l_n_SGD=[1],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
            0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    l_M=[10],
    n_seeds=n_seeds,
)

append_experiments(
    txt_name="MNIST.txt",
    l_dataset=["MNIST_0.1", "MNIST_10000"],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[10000],
    l_n_SGD=range(1, 21),
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=[0.001],
    l_M=[10],
    n_seeds=n_seeds,
)

append_experiments(
    txt_name="MNIST.txt",
    l_dataset=["MNIST_0.1"],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[25000],
    l_n_SGD=[1, 2],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=[0.001, 0.002, 0.005, 0.01],
    l_M=[10, 20, 30],
    n_seeds=n_seeds,
)
print("")


"""
CIFAR10 experiments
"""
l_lr_CIFAR = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001,
              0.002, 0.005,
              0.01, 0.02, 0.05]
append_experiments(
    txt_name="CIFAR10-flat-full.txt",
    l_dataset=["CIFAR10-flat-full_0.1"],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[200000],
    l_n_SGD=[1],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=l_lr_CIFAR,
    l_M=[10],
    n_seeds=n_seeds,
)
print("")

append_experiments(
    txt_name="CIFAR10-flat.txt",
    l_dataset=["CIFAR10-flat_0.1"],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[200000],
    l_n_SGD=[1],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=l_lr_CIFAR,
    l_M=[10],
    n_seeds=n_seeds,
)

append_experiments(
    txt_name="CIFAR10-flat.txt",
    l_dataset=["CIFAR10-flat_0.1"],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[10000],
    l_n_SGD=range(1, 21),
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=[0.001],
    l_M=[10],
    n_seeds=n_seeds,
)

append_experiments(
    txt_name="CIFAR10-flat.txt",
    l_dataset=["CIFAR10-flat_0.1"],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[10000],
    l_n_SGD=[1, 2, 3],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=[0.0005, 0.001, 0.002],
    l_M=[10, 20, 30],
    n_seeds=n_seeds,
)
print("")

append_experiments(
    txt_name="CIFAR10.txt",
    l_dataset=["CIFAR10-ResNet_0.1"],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[2500],
    l_n_SGD=[1],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l= l_lr_CIFAR,
    l_M=[10],
    n_seeds=n_seeds,
)
append_experiments(
    txt_name="CIFAR10.txt",
    l_dataset=["CIFAR10_0.1"],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[100000],
    l_n_SGD=[1],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l= l_lr_CIFAR,
    l_M=[10],
    n_seeds=n_seeds,
)
append_experiments(
    txt_name="CIFAR10.txt",
    l_dataset=["CIFAR10_0.1"],
    l_opt_scheme=opt_all,
    l_time_scenario=["F-0", "F-80"],
    l_P_type=["uniform"],
    l_T=[5000],
    l_n_SGD=[10],
    l_B=[64],
    l_lr_g=[1.],
    l_lr_l=l_lr_CIFAR,
    l_M=[20, 50],
    n_seeds=n_seeds,
)
print("")

"""
CIFAR100 experiments
"""
append_experiments(
    txt_name="CIFAR100.txt",
    l_dataset=[f"CIFAR100_{n}" for n in ["0.01", "0.1", "1."]],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[100000],
    l_n_SGD=[1],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=[0.0005, 0.001, 0.002],
    l_M=[10],
    n_seeds=n_seeds,
)
append_experiments(
    txt_name="CIFAR100.txt",
    l_dataset=["CIFAR100_0.1", "CIFAR100_0."],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[2500],
    l_n_SGD=[1],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=[0.0005, 0.001, 0.002],
    l_M=[100],
    n_seeds=n_seeds,
)
print("")


append_experiments(
    txt_name="CIFAR100-flat.txt",
    l_dataset=[f"CIFAR100-flat_{n}" for n in ["0.01"]],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[200000],
    l_n_SGD=[1],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
    l_M=[10],
    n_seeds=n_seeds,
)
append_experiments(
    txt_name="CIFAR100-flat.txt",
    l_dataset=[f"CIFAR100-flat_{n}" for n in ["0.01"]],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[10000],
    l_n_SGD=range(1, 21),
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=[0.001],
    l_M=[10],
    n_seeds=n_seeds,
)
append_experiments(
    txt_name="CIFAR100-flat.txt",
    l_dataset=[f"CIFAR100-flat_{n}" for n in ["0.01", "0.1", "1."]],
    l_opt_scheme=opt_surrogate,
    l_time_scenario=["F-80"],
    l_P_type=["uniform"],
    l_T=[5000],
    l_n_SGD=[1],
    l_B=l_B,
    l_lr_g=[1.],
    l_lr_l=[0.0001, 0.0002, 0.0005, 0.001, 0.002],
    l_M=[100],
    n_seeds=n_seeds,
)
print("")


"""
Shakespeare experiments
"""
append_experiments(
    txt_name="Shakespeare.txt",
    l_dataset=["Shakespeare"],
    l_opt_scheme=opt_all,
    l_time_scenario=["F-0", "F-80"],
    l_P_type=["uniform"],
    l_T=[5000],
    l_n_SGD=[10],
    l_B=[64],
    l_lr_g=[1.],
    l_lr_l=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2.],
    l_M=[20, 50],
    n_seeds=n_seeds,
)
