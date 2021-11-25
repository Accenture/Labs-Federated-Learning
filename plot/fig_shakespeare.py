#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import plot.common as common
from .context import clients_importances
from copy import deepcopy


def shakespeare_paper(
    plot_name: str,
    metric: str,
    samplings: list,
    T: int,
    n_SGD: int,
    lr_l: float,
    B: int,
    n_seeds: int,
    lr_g=1.0,
    decay=1.0,
    show=True,
):
    args_exp = {
        "T": T,
        "n_SGD": n_SGD,
        "lr_l": lr_l,
        "B": B,
        "lr_g": lr_g,
        "decay": decay,
        "mu": 0.0,
    }

    def one_subplot(
        ax,
        args_exp: dict,
        samplings: list,
        n_seeds: int,
    ):
        """
        plot a curve for each sampling.
        a sampling curve is the average of `n_seeds` simulations.
        """

        P = clients_importances(args_exp["P_type"], args_exp["dataset"], False)

        dic = {}
        for sampling in samplings:
            hist_clients = common.hist(
                metric, range(n_seeds), sampling=sampling, mean=True, **args_exp
            )

            hist = np.average(hist_clients, 1, P)
            dic[sampling] = hist

        diff = dic["MD"] - dic["Uniform"]
        diff = diff[diff != 0]
        ax.plot(range(500), diff[:500])

    n_rows, n_cols = 1, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5))

    l_P_type = ["uniform"] * 4 + ["ratio"] * 4
    list_n = [10, 20, 40, 80] * 2
    list_dataset = ["Shakespeare4", "Shakespeare3", "Shakespeare2", "Shakespeare"] * 2

    for idx, (P_type, n, dataset) in enumerate(zip(l_P_type, list_n, list_dataset)):

        args_exp_plot = deepcopy(args_exp)
        args_exp_plot["dataset"] = dataset
        args_exp_plot["P_type"] = P_type
        args_exp_plot["n"] = n
        args_exp_plot["m"] = int(n / 2)

        ax = axes[idx // 4]

        one_subplot(ax, args_exp_plot, samplings, n_seeds)

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
        bbox_to_anchor=(0.65, 0.14),
    )

    plt.tight_layout(pad=0.0)
    fig.savefig(f"figures/pdf/{plot_name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{plot_name}.png", bbox_inches="tight")
    if show:
        plt.show()


def shakespeare_paper_appendix(
    plot_name: str,
    metric: str,
    samplings: list,
    T: int,
    n_SGD: int,
    lr_l: float,
    B: int,
    n_seeds: int,
    show: bool,
    lr_g=1.0,
    decay=1.0,
):
    args_exp = {
        "T": T,
        "n_SGD": n_SGD,
        "lr_l": lr_l,
        "B": B,
        "lr_g": lr_g,
        "decay": decay,
        "mu": 0.0,
    }

    def one_subplot(
        ax,
        args_exp: dict,
        samplings: list,
        n_seeds: int,
    ):
        """
        plot a curve for each sampling.
        a sampling curve is the average of `n_seeds` simulations.
        """

        P = clients_importances(args_exp["P_type"], args_exp["dataset"], False)

        dic = {}
        for sampling in samplings:
            hist_clients = common.hist(
                metric, range(n_seeds), sampling=sampling, mean=True, **args_exp
            )

            hist = np.average(hist_clients, 1, P)
            dic[sampling] = hist

            ax.plot(range(500), hist[hist > 0][:500])

    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 6))

    l_P_type = ["uniform"] * 4 + ["ratio"] * 4
    list_lr = [lr_l] * n_rows * n_cols
    list_n = [10, 20, 40, 80] * 2
    list_clients = [10, 20, 40, 80] * 2
    list_dataset = ["Shakespeare4", "Shakespeare3", "Shakespeare2", "Shakespeare"] * 2

    list_y = [
        [0.5, 1.5],
        [0.5, 1.5],
        [0.8, 1.8],
        [1.0, 2.0],
        [0.8, 1.8],
        [0.8, 1.8],
        [1.0, 2.0],
        [1.1, 2.1],
    ]

    for idx, (P_type, lr_l, n, dataset, y, client) in enumerate(
        zip(l_P_type, list_lr, list_n, list_dataset, list_y, list_clients)
    ):
        args_exp_plot = deepcopy(args_exp)
        args_exp_plot["dataset"] = dataset
        args_exp_plot["P_type"] = P_type
        args_exp_plot["n"] = n
        args_exp_plot["m"] = int(n / 2)

        ax = axes[idx // n_cols, idx % n_cols]

        one_subplot(ax, args_exp_plot, samplings, n_seeds)

        # FORMAT THE SUBPLOT
        ax.set_ylim(y)
        ax.set_title("(" + chr(97 + idx) + ") - n = " + str(n), pad=0.0)

        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("# rounds")

        if idx % n_cols == 0:
            ax.set_ylabel(r"$\mathcal{L}(\theta^t)$")

    fig.legend(
        ax,
        labels=samplings,
        ncol=3,
        bbox_to_anchor=(0.75, 0.04),
    )

    plt.tight_layout(pad=0.0)
    fig.savefig(f"figures/pdf/{plot_name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{plot_name}.png", bbox_inches="tight")
    if show:
        plt.show()
