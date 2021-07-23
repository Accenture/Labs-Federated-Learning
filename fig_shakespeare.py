#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def shakespeare_paper(
    folder: str,
    samplings: list,
    n_iter: int,
    n_SGD: int,
    lr_local: float,
    batch_size: int,
    n_seeds: int,
    lr_global=1.0,
    decay=1.0,
):

    from py_func.importances import clients_importances
    from plots.plot_functions import get_hist

    def one_subplot(
        ax,
        dataset: str,
        weight_type: str,
        lr_local: float,
        n_sampled: int,
        n_clients: int,
        label: str,
    ):
        """
        plot the difference between the averaged curve of MD sampling
        - the averaged curve of Uniform sampling for an FL dataset with
        `n_clients` from which `n_sampled` of them are selected at every
        optimization step.
        """

        importances = clients_importances(weight_type, dataset)

        dic = {}
        for sampling in samplings:

            sampled_clients = np.zeros(
                (
                    n_iter + 1,
                    len(importances),
                )
            )

            for seed in range(n_seeds):
                sampled_clients += get_hist(
                    folder,
                    dataset,
                    sampling,
                    n_iter,
                    n_SGD,
                    batch_size,
                    lr_global,
                    lr_local,
                    n_sampled,
                    weight_type,
                    decay,
                    seed,
                )

            sampled_clients /= n_seeds
            hist = np.average(sampled_clients, 1, importances)
            dic[sampling] = hist

        ax.plot(dic["MD"] - dic["Uniform"], label=label)

    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))

    list_weights = ["uniform"] * 4 + ["ratio"] * 4
    list_lr = [lr_local] * 8
    list_n = [10, 20, 40, 80] * 2
    list_dataset = ["Shakespeare4", "Shakespeare3", "Shakespeare2", "Shakespeare"] * 2

    for idx, (weight_type, lr_local, n, dataset) in enumerate(
        zip(list_weights, list_lr, list_n, list_dataset)
    ):

        ax = axes[idx // 4]

        m = int(n / 2)
        one_subplot(ax, dataset, weight_type, lr_local, m, n, n)

        if idx == 0:
            ax.set_ylabel(r"$\mathcal{L}(\theta_{MD}) - \mathcal{L}(\theta_{U})$")
            ax.set_title(r"(a) - $p_i = 1 /n$")
            ax.set_xlabel("# rounds")

        elif idx == 4:
            ax.set_title(r"(b) - $p_i = n_i / M$")
            ax.set_xlabel("# rounds")

    fig.legend(
        ax,
        labels=[10, 20, 40, 80],
        ncol=4,
        bbox_to_anchor=(0.62, 0.14),
    )

    fig.savefig("plots/shakespeare_paper.pdf", bbox_inches="tight")


def shakespeare_appendix(
    folder: str,
    samplings: list,
    n_iter: int,
    n_SGD: int,
    lr_local: float,
    batch_size: int,
    n_seeds: int,
    lr_global=1.0,
    decay=1.0,
):

    from py_func.importances import clients_importances
    from plots.plot_functions import get_hist

    def one_subplot(
        ax,
        dataset: str,
        weight_type: str,
        lr_local: float,
        n_sampled: int,
        n_clients: int,
    ):
        """
        plot a curve for each sampling.
        a sampling curve is the average of `n_seeds` simulations.
        """

        importances = clients_importances(weight_type, dataset)

        for sampling in samplings:

            sampled_clients = np.zeros(
                (
                    n_iter + 1,
                    len(importances),
                )
            )

            for seed in range(n_seeds):
                sampled_clients += get_hist(
                    folder,
                    dataset,
                    sampling,
                    n_iter,
                    n_SGD,
                    batch_size,
                    lr_global,
                    lr_local,
                    n_sampled,
                    weight_type,
                    decay,
                    seed,
                )

            sampled_clients /= n_seeds
            hist = np.average(sampled_clients, 1, importances)
            ax.plot(hist, label=sampling)

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))

    list_weights = ["uniform"] * n_cols + ["ratio"] * n_cols
    list_lr = [lr_local] * n_rows * n_cols
    list_n = [10, 20, 40, 80] * 2
    list_dataset = ["Shakespeare4", "Shakespeare3", "Shakespeare2", "Shakespeare"] * 2
    list_y = [
        [0.5, 1.5],
        [0.6, 1.6],
        [1.0, 1.7],
        [1.2, 1.6],
        [1.0, 1.6],
        [1.0, 1.7],
        [1.2, 1.6],
        [1.4, 1.7],
    ]

    for idx, (weight_type, lr_local, n, dataset, y) in enumerate(
        zip(list_weights, list_lr, list_n, list_dataset, list_y)
    ):

        ax = axes[idx // n_cols, idx % n_cols]

        # PLOT THE AVERAGED SAMPLING CURVES
        m = int(n / 2)
        one_subplot(ax, dataset, weight_type, lr_local, m, n)

        # FORMAT THE SUBPLOT
        ax.set_ylim(y)
        ax.set_title("(" + chr(97 + idx) + ") - n = " + str(n))

        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("# rounds")

        if idx % n_cols == 0:
            ax.set_ylabel(r"$\mathcal{L}(\theta^t)$")

    fig.legend(
        ax,
        labels=samplings,
        ncol=2,
        bbox_to_anchor=(0.55, 0.06),
    )

    fig.savefig("plots/shakespeare_appendix.pdf", bbox_inches="tight")


samplings = ["MD", "Uniform"]
n_iter = 300
n_SGD = 50
lr_local = 1.5
batch_size = 64
n_seeds = 30

shakespeare_paper("loss", samplings, n_iter, n_SGD, lr_local, batch_size, n_seeds)


shakespeare_appendix("loss", samplings, n_iter, n_SGD, lr_local, batch_size, n_seeds)
