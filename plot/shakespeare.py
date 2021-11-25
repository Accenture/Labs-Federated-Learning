#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# from .context import FL
from FL.importances import clients_importances
from FL.experiment import Experiment

import plot.common as common
from copy import deepcopy


def shakespeare_paper_varying_m(
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
        "dataset": "Shakespeare",
        "n": 80,
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2.5))

    l_P_type = ["uniform"] * 3 + ["ratio"] * 3
    list_m = [4, 8, 40] * 2
    list_T = [1000, 1000, 600] * 2

    for idx, (P_type, m, T) in enumerate(zip(l_P_type, list_m, list_T)):

        args_exp_plot = deepcopy(args_exp)
        args_exp_plot["P_type"] = P_type
        args_exp_plot["m"] = m
        args_exp_plot["T"] = T

        ax = axes[idx // 3]

        one_subplot(ax, args_exp_plot, samplings, n_seeds)

        if idx == 0:
            ax.set_ylabel(r"$\mathcal{L}(\theta_{MD}) - \mathcal{L}(\theta_{U})$")
            ax.set_title(r"(a) - $p_i = 1 /n$")
            ax.set_xlabel("# rounds")

        elif idx == 3:
            ax.set_title(r"(b) - $p_i = n_i / M$")
            ax.set_xlabel("# rounds")

    fig.legend(
        ax,
        labels=[r"m=4", r"m=8", r"m=40"],
        ncol=4,
        bbox_to_anchor=(0.72, 0.14),
    )

    plt.tight_layout(pad=0.0)
    fig.savefig(f"figures/pdf/{plot_name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{plot_name}.png", bbox_inches="tight")
    if show:
        plt.show()


def shakespeare_paper_varying_K(
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
        "dataset": "Shakespeare",
        "n": 80,
        "T": T,
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
        # diff = diff[diff != 0]
        # ax.plot(range(len(diff[-500:])), diff[-500:])
        ax.plot(range(len(diff)), diff)

    n_rows, n_cols = 1, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 2.5))

    l_P_type = ["uniform"] * n_cols + ["ratio"] * n_cols
    list_m = [8, 40] * 2

    for idx, (P_type, m) in enumerate(zip(l_P_type, list_m)):

        args_exp_plot = deepcopy(args_exp)
        args_exp_plot["P_type"] = P_type
        args_exp_plot["m"] = m
        args_exp_plot["T"] = T

        ax = axes[idx // 2]

        one_subplot(ax, args_exp_plot, samplings, n_seeds)

        if idx == 0:
            ax.set_ylabel(r"$\mathcal{L}(\theta_{MD}) - \mathcal{L}(\theta_{U})$")
            ax.set_title(r"(a) - $p_i = 1 /n$")
            ax.set_xlabel("# rounds")
            ax.set_xlim(1000, 2500)
            ax.set_ylim(-0.03, 0.04)

        elif idx == 2:
            ax.set_title(r"(b) - $p_i = n_i / M$")
            ax.set_xlabel("# rounds")
            ax.set_xlim(1000, 2500)
            ax.set_ylim(-0.1, 0.02)

    fig.legend(
        ax,
        labels=[r"m=8", r"$m=40$"],
        ncol=4,
        bbox_to_anchor=(0.68, 0.14),
    )

    plt.tight_layout(pad=0.0)
    fig.savefig(f"figures/pdf/{plot_name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{plot_name}.png", bbox_inches="tight")
    if show:
        plt.show()


def VaryingM(
    plot_name: str,
    metric: str,
    l_m: list,
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
        "dataset": "Shakespeare",
        "T": T,
        "n_SGD": n_SGD,
        "lr_l": lr_l,
        "B": B,
        "lr_g": lr_g,
        "decay": decay,
        "mu": 0.0,
        "n": 80,
    }

    def one_subplot(
        ax,
        fixed_args: dict,
        samplings: list,
        n_seeds: int,
    ):
        """Plot an expected curve for each client sampling"""
        P = clients_importances(fixed_args["P_type"], fixed_args["dataset"], False)

        for sampling in samplings:
            hist_clients = common.hist(
                metric, range(n_seeds), sampling=sampling, mean=True, **fixed_args
            )

            hist_server = np.average(hist_clients, 1, P)
            ax.plot(hist_server[hist_server > 0], label=sampling)

    n_rows, n_cols = len(l_m), 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4))

    l_P_type = ["uniform"] * n_rows + ["ratio"] * n_rows
    l_m = l_m * 2

    l_yaxis = [[1, 2.0]] * n_cols + [[1.1, 2.1]] * n_cols

    for idx, (P_type, m, yaxis) in enumerate(zip(l_P_type, l_m, l_yaxis)):

        ax = axes[idx // n_cols, idx % n_cols]

        fixed_args_plot = deepcopy(args_exp)

        fixed_args_plot["P_type"] = P_type
        fixed_args_plot["m"] = m

        # PLOT THE AVERAGED SAMPLING CURVES
        one_subplot(ax, fixed_args_plot, samplings, n_seeds)

        # FORMAT THE SUBPLOT
        ax.set_ylim(yaxis)
        ax.set_title("(" + chr(97 + idx) + ") - m = " + str(m), pad=0.0)

        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("# rounds")

        if idx % n_cols == 0:
            ax.set_ylabel(r"$\mathcal{L}(\theta^t)$")

    fig.legend(
        ax,
        labels=samplings,
        ncol=3,
        bbox_to_anchor=(0.76, 0.067),
    )

    plt.tight_layout(pad=0.0)
    fig.savefig(f"figures/pdf/{plot_name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{plot_name}.png", bbox_inches="tight")
    if show:
        plt.show()


def VaryingK(
    plot_name: str,
    metric: str,
    l_m: list,
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
        "dataset": "Shakespeare",
        "T": T,
        "n_SGD": n_SGD,
        "lr_l": lr_l,
        "B": B,
        "lr_g": lr_g,
        "decay": decay,
        "mu": 0.0,
        "n": 80,
    }

    def one_subplot(
        ax,
        fixed_args: dict,
        samplings: list,
        n_seeds: int,
    ):
        """Plot an expected curve for each client sampling"""
        P = clients_importances(fixed_args["P_type"], fixed_args["dataset"], False)

        for sampling in samplings:
            hist_clients = common.hist(
                metric, range(n_seeds), sampling=sampling, mean=True, **fixed_args
            )

            hist_server = np.average(hist_clients, 1, P)
            ax.plot(hist_server[hist_server > 0], label=sampling)

    n_rows, n_cols = 2, len(l_m)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4))

    l_P_type = ["uniform"] * n_cols + ["ratio"] * n_cols
    l_m = l_m * 2

    l_yaxis = [[1, 2.0]] * n_cols + [[1.1, 2.1]] * n_cols

    for idx, (P_type, m, yaxis) in enumerate(zip(l_P_type, l_m, l_yaxis)):

        ax = axes[idx // n_cols, idx % n_cols]

        fixed_args_plot = deepcopy(args_exp)

        fixed_args_plot["P_type"] = P_type
        fixed_args_plot["m"] = m

        # PLOT THE AVERAGED SAMPLING CURVES
        one_subplot(ax, fixed_args_plot, samplings, n_seeds)

        # FORMAT THE SUBPLOT
        # ax.set_ylim(yaxis)
        if m == 37:
            ax.set_title("(" + chr(97 + idx) + ") - m = " + str(40), pad=0.0)
        else:
            ax.set_title("(" + chr(97 + idx) + ") - m = " + str(m), pad=0.0)

        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("# rounds")

        if idx % n_cols == 0:
            ax.set_ylabel(r"$\mathcal{L}(\theta^t)$")

    fig.legend(
        ax,
        labels=samplings,
        ncol=3,
        bbox_to_anchor=(0.67, 0.063),
    )

    plt.tight_layout(pad=0.0)
    fig.savefig(f"figures/pdf/{plot_name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{plot_name}.png", bbox_inches="tight")
    if show:
        plt.show()
