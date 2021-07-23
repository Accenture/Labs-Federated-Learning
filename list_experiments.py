#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
import os


def append_experiments(
    txt_name: str,
    dataset: str,
    args,
    samplings: list,
    n_sampled: int,
    l_lr_local: list,
    l_equal: list,
    l_seed: list,
    new: bool,
):
    """
    add/create to a txt file all the experiments that have not already been run
    in the arguments of this function
    if new == True then a new txt file is created
    If not the experiments needed to run are added to the existing txt file
    """

    from py_func.file_name import get_file_name

    if new:
        text_file = open(f"{txt_name}.txt", "w")
    else:
        text_file = open(f"{txt_name}.txt", "a")

    for (sampling, lr_local, equal, seed) in product(
        samplings, l_lr_local, l_equal, l_seed
    ):

        file_name = get_file_name(
            dataset,
            sampling,
            args.n_iter,
            args.n_SGD,
            args.batch_size,
            args.lr_global,
            lr_local,
            n_sampled,
            args.mu,
            equal,
            args.decay,
            seed,
        )

        if not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl"):

            string = (
                f"{dataset} {sampling} {args.n_SGD} {lr_local} "
                f"{args.lr_global} {n_sampled} {args.batch_size} "
                f"{args.n_iter} {args.mu} {equal} {args.decay} {seed}\n"
            )
            text_file.write(string)

    text_file.close()

    file = open(f"{txt_name}.txt", "r")
    nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
    print(txt_name, len(nonempty_lines))
    file.close()


class args:
    pass


args_Shakespeare = args()

args_Shakespeare.mu = 0
args_Shakespeare.n_iter = 300
args_Shakespeare.n_SGD = 50
args_Shakespeare.lr_global = 1.0
args_Shakespeare.batch_size = 64
args_Shakespeare.decay = 1.0


equal = ["ratio", "uniform"]
seed = [i for i in range(30)]
sampling = ["MD", "Uniform"]
best_lr = 1.5

n_sampled_list = [5, 10, 20, 40]
datasets = ["Shakespeare4", "Shakespeare3", "Shakespeare2", "Shakespeare"]


# INITIALIZED THE TXT FILE
text_file = open(f"experiments.txt", "w")
text_file.close()


# FILL THE TXT FILE WITH EXPERIMENTS TO RUN
for n_sampled, dataset in zip(n_sampled_list, datasets):

    for weight_type in equal:
        append_experiments(
            "experiments",
            dataset,
            args_Shakespeare,
            sampling,
            n_sampled,
            [best_lr],
            [weight_type],
            seed,
            False,
        )
