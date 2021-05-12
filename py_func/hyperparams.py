#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_hyperparams(dataset, n_SGD):
    """return the different hyperparameters considered for the experiments.
    This function enables the user to put less input to FL_CS2.py"""

    batch_size = 50

    if dataset == "MNIST_iid":
        n_iter = 600

    elif dataset == "MNIST_shard":
        n_iter = 600

    elif dataset[:5] == "CIFAR":
        n_iter = 1000

    if dataset[:5] == "CIFAR":

        if n_SGD <= 10:
            n_iter = int(n_iter * 2)
        elif n_SGD >= 200:
            n_iter = int(n_iter / 2)

        if n_SGD >= 200:
            metric_period = 3
        elif n_SGD == 100:
            metric_period = 5
        elif n_SGD == 50:
            metric_period = 5
        elif n_SGD <= 10:
            metric_period = 10

    else:
        if n_SGD <= 10:
            n_iter = int(n_iter * 2)
        elif n_SGD >= 100:
            n_iter = int(n_iter / 2)

        metric_period = 2

    return n_iter, batch_size, metric_period


def get_file_name(
    dataset: str,
    sampling: str,
    sim_type: str,
    seed: int,
    n_SGD: int,
    lr: float,
    decay: float,
    p: float,
    mu: float,
):
    """return the file name under which the experiment with these info is saved
    under"""

    n_iter, batch_size, meas_perf_period = get_hyperparams(dataset, n_SGD)

    file_name = (
        f"{dataset}_{sampling}_{sim_type}_i{n_iter}_N{n_SGD}_lr{lr}"
        + f"_B{batch_size}_d{decay}_p{p}_m{meas_perf_period}_{seed}"
    )
    if mu != 0.0:
        file_name += f"_{mu}"
    return file_name


def get_CIFAR10_alphas():
    """Return the different alpha considered for the dirichlet distribution"""
    return [0.001, 0.01, 0.1, 10.0]
