#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sn


def load_metric(metric: str, params: dict, n_freeriders: int, experiment_specific: str):

    path = f"hist/{metric}/{params['dataset']}_{params['solver']}"

    FL = pickle.load(open(f"{path}_FL_{experiment_specific}.pkl", "rb"))
    plain = pickle.load(
        open(f"{path}_plain_{n_freeriders}_{experiment_specific}" ".pkl", "rb")
    )
    disg_1_1 = pickle.load(
        open(
            f"{path}_disguised_1.0_{params['std_0']}_"
            f"{n_freeriders}_{experiment_specific}_1.pkl",
            "rb",
        )
    )
    disg_3_1 = pickle.load(
        open(
            f"{path}_disguised_1.0_{params['std_0']}_"
            f"{n_freeriders}_{experiment_specific}_3.pkl",
            "rb",
        )
    )
    disg_1_05 = pickle.load(
        open(
            f"{path}_disguised_0.5_{params['std_0']}_"
            f"{n_freeriders}_{experiment_specific}_1.pkl",
            "rb",
        )
    )
    disg_3_05 = pickle.load(
        open(
            f"{path}_disguised_0.5_{params['std_0']}_"
            f"{n_freeriders}_{experiment_specific}_3.pkl",
            "rb",
        )
    )
    disg_1_2 = pickle.load(
        open(
            f"{path}_disguised_2.0_{params['std_0']}_"
            f"{n_freeriders}_{experiment_specific}_1.pkl",
            "rb",
        )
    )
    disg_3_2 = pickle.load(
        open(
            f"{path}_disguised_2.0_{params['std_0']}_"
            f"{n_freeriders}_{experiment_specific}_3.pkl",
            "rb",
        )
    )

    dic = {
        "FL": [np.mean(hist_i) for hist_i in FL],
        "plain": [np.mean(hist_i) for hist_i in plain],
        "disg_1_1": [np.mean(hist_i) for hist_i in disg_1_1],
        "disg_3_1": [np.mean(hist_i) for hist_i in disg_3_1],
        "disg_1_05": [np.mean(hist_i) for hist_i in disg_1_05],
        "disg_3_05": [np.mean(hist_i) for hist_i in disg_3_05],
        "disg_1_2": [np.mean(hist_i) for hist_i in disg_1_2],
        "disg_3_2": [np.mean(hist_i) for hist_i in disg_3_2],
    }

    return dic


def get_interval(metric: str, params: dict, n_fr: int, exp_spec: str):
    """Look for all the hist computed for a given FL setting.
    Each file has a lost history of each of the clients.
    We are only interested in the server loss and hence take the weighted mean
    of these losses and then return two lists for the lower and upper bound."""

    path = os.getcwd() + f"/hist/{metric}"

    file_rd_simu = os.listdir(path)

    file_begin = f"{params['dataset']}_{params['solver']}_FL_{exp_spec}_"

    file_rd_simu = [file for file in file_rd_simu if file_begin in file]

    list_hists = [pickle.load(open(path + "/" + file, "rb")) for file in file_rd_simu]
    print(file_begin, len(list_hists))

    if len(list_hists) > 0:
        # Get the server loss from the local one of each client
        hist = np.array([np.mean(hist, axis=1) for hist in list_hists])

        # Get the upper and lower bounds
        lower_bounds = np.min(hist, axis=0)
        upper_bounds = np.max(hist, axis=0)

        return lower_bounds, upper_bounds

    else:
        return np.NaN, np.NaN


def get_exp_specific(d: dict, n_fr: int):

    n_iter = d[f"n_iter_{n_fr}"]

    if (
        d["dataset"] == "MNIST-iid"
        or d["dataset"] == "MNIST-shard"
        or d["dataset"] == "CIFAR-10"
    ):
        return (
            f"{d['n_clients']}_{d['samples']}"
            + f"_{d['epochs']}_{n_iter}"
            + f"_{d['lr']}"
        )

    elif d["dataset"] == "shakespeare":
        return f"{d['n_clients']}" + f"_{d['epochs']}_{n_iter}" + f"_{d['lr']}"


def figure_params(params_type: str, dataset: str, n_epochs: int, solver: str):

    from python_code.functions import get_n_iter

    params = {
        "solver": solver,
        "dataset": dataset,
        "n_clients": 5,
        "epochs": n_epochs,
        "std_0": 10 ** -3,
        "n_iter_1": get_n_iter(dataset, n_epochs, 1),
        "n_iter_5": get_n_iter(dataset, n_epochs, 5),
        "n_iter_45": get_n_iter(dataset, n_epochs, 45),
    }

    if dataset == "MNIST-iid":
        params.update({"samples": 600, "lr": 10 ** -3})

    elif dataset == "MNIST-shard":
        params.update({"samples": 600, "lr": 10 ** -3})

    elif dataset == "CIFAR-10":
        params.update(
            {
                "samples": 10000,
                "lr": 10 ** -3,
            }
        )

    elif dataset == "shakespeare":
        params.update({"samples": 0, "lr": 0.5})

    if params_type == "loss" or params_type == "acc":
        d = [
            load_metric(params_type, params, n_fr, get_exp_specific(params, n_fr))
            for n_fr in [1, 5, 45]
        ]

    elif params_type == "loss_hist" or params_type == "acc_hist":
        params_type = params_type[:-5]
        d = get_interval(params_type, params, 45, get_exp_specific(params, 45))

    return d


def plot_fig_1_half(metric: str, n_epochs: int):
    """Plot the figure used for the core of the paper (Appendix excluded)"""

    # Dictionnary with the different strategies
    shak_FA = figure_params(metric, "shakespeare", n_epochs, "FedAvg")
    shak_FP = figure_params(metric, "shakespeare", n_epochs, "FedProx")

    # Dictionnary with the 30 simulations
    shak_FA_int = figure_params(f"{metric}_hist", "shakespeare", n_epochs, "FedAvg")
    shak_FP_int = figure_params(f"{metric}_hist", "shakespeare", n_epochs, "FedProx")

    list_cols = [shak_FA, shak_FP]
    list_ints = [shak_FA_int, shak_FP_int]

    rows = ["1 free-rider", "5 free-riders", "45 free-riders"]
    cols = ["FedAvg", "FedProx"]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 7))
    sn.set_style("ticks")  # Plot style

    # Create the titles for the columns
    for ax, col in zip(axes[0], cols):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, 5),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

    # Create the titles for the rows
    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
            rotation=90,
        )

    def plot_dic(dic: dict, hist: list, plot_idx: int, metric: str):

        ax = axes[plot_idx // len(cols)][plot_idx % len(cols)]

        if metric == "acc":

            try:
                x = [i for i in range(len(hist[0]))][: len(dic["FL"])]

                ax.fill_between(
                    x,
                    hist[0][: len(dic["FL"])],
                    hist[1][: len(dic["FL"])],
                    alpha=0.3,
                )
            except:
                if plot_idx // len(cols) == 0:
                    print(f"NO AVAILABLE HISTORY FOR {cols[plot_idx%len(cols)]}")

        ax.plot(dic["FL"], label="Only Fair")
        ax.plot(dic["plain"], label="Plain")
        ax.plot(dic["disg_1_1"], label=r"Disguised $\sigma$, $\gamma=1$")
        ax.plot(dic["disg_3_1"], label=r"Disguised $3\sigma$, $\gamma=1$")

        if metric == "acc":
            ax.set_ylim(30)

        elif metric == "loss":

            ax.set_yscale("log")

            # Not to put too much info on  the y axis
            from matplotlib.ticker import NullFormatter

            ax.yaxis.set_minor_formatter(NullFormatter())

        if plot_idx == 0:
            ax.legend(loc="best", ncol=2)

        ax.autoscale(enable=True, axis="x", tight=True)

    for n_col, (col, hist) in enumerate(zip(list_cols, list_ints)):
        for n_row, d in enumerate(col):
            plot_dic(d, hist, len(cols) * n_row + n_col, metric)

    fig.tight_layout()

    plt.savefig(f"plots/fig_1_half_{metric}.png")
    plt.savefig(f"plots/fig_1_half_{metric}.pdf")


def plot_metric_n_fr(
    list_columns: list,
    list_hist: list,
    metric: str,
    y_label: str,
    title: str,
    save_name: str,
    cols: list,
    y_min: list,
    log_scale=False,
):

    rows = ["1 free-rider", "5 free-riders", "45 free-riders"]

    fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(22, 10))

    sn.set_style("ticks")

    if title:
        plt.suptitle(title)

    # Create the titles for the columns
    for ax, col in zip(axes[0], cols):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, 5),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

    # Create the titles for the rows
    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
            rotation=90,
        )

    def plot_dic(dic: dict, hist: list, plot_idx: int, metric: str):

        ax = axes[plot_idx // len(cols)][plot_idx % len(cols)]

        if metric == "acc":

            try:
                x = [i for i in range(len(hist[0]))][: len(dic["FL"])]

                ax.fill_between(
                    x,
                    hist[0][: len(dic["FL"])],
                    hist[1][: len(dic["FL"])],
                    alpha=0.3,
                )
            except:
                if plot_idx // len(cols) == 0:
                    print(f"NO AVAILABLE HISTORY FOR {cols[plot_idx%len(cols)]}")

        ax.plot(dic["FL"], label="Only Fair")
        ax.plot(dic["plain"], label="Plain")

        ax.plot(dic["disg_1_1"], label=r"Disguised $\sigma$, $\gamma=1$")
        ax.plot(dic["disg_3_1"], label=r"Disguised $3\sigma$, $\gamma=1$")

        ax.plot(
            dic["disg_1_05"],
            label=r"Disguised $\sigma$, $\gamma=0.5$",
            linestyle="--",
        )
        ax.plot(
            dic["disg_3_05"],
            label=r"Disguised $3\sigma$, $\gamma=0.5$",
            linestyle="--",
        )

        ax.plot(
            dic["disg_1_2"],
            label=r"Disguised $\sigma$, $\gamma=2$",
            linestyle="--",
        )
        ax.plot(
            dic["disg_3_2"],
            label=r"Disguised $3\sigma$, $\gamma=2$",
            linestyle="--",
        )

        if metric == "acc":
            ax.set_ylim(y_min[plot_idx % len(cols)])

        elif metric == "loss" and log_scale:
            ax.set_yscale("log")

        if plot_idx == 0 and metric == "acc":
            ax.legend(loc="lower right", ncol=2)
        elif plot_idx == 0 and metric == "loss":
            ax.legend(loc="upper right", ncol=2)

        ax.autoscale(enable=True, axis="x", tight=True)

    for n_col, (col, hist) in enumerate(zip(list_columns, list_hist)):
        for n_row, d in enumerate(col):
            plot_dic(d, hist, len(cols) * n_row + n_col, metric)

    fig.tight_layout()

    plt.savefig(f"plots/{save_name}.png")
    plt.savefig(f"plots/{save_name}.pdf")


def plot_metric_history(solver: str, n_epochs: int, metric: str, log_scale=False):

    col_1 = figure_params(metric, "MNIST-iid", n_epochs, solver)
    col_2 = figure_params(metric, "MNIST-shard", n_epochs, solver)
    if solver == "FedAvg":
        col_3 = figure_params(metric, "CIFAR-10", n_epochs, solver)
    col_4 = figure_params(metric, "shakespeare", n_epochs, solver)

    metric_hist = metric + "_hist"
    int_1 = figure_params(metric_hist, "MNIST-iid", n_epochs, solver)
    int_2 = figure_params(metric_hist, "MNIST-shard", n_epochs, solver)
    if solver == "FedAvg":
        int_3 = figure_params(metric_hist, "CIFAR-10", n_epochs, solver)
    int_4 = figure_params(metric_hist, "shakespeare", n_epochs, solver)

    if solver == "FedAvg":
        cols_names = ["MNIST-iid", "MNIST-shard", "CIFAR-10", "Shakespeare"]
        cols = [col_1, col_2, col_3, col_4]
        intervals = [int_1, int_2, int_3, int_4]
        y_min = [79, 75, 55, 30]

    else:
        cols_names = ["MNIST-iid", "MNIST-shard", "Shakespeare"]
        cols = [col_1, col_2, col_4]
        intervals = [int_1, int_2, int_4]
        y_min = [79, 75, 30]

    save_name = f"{solver}_{metric}_{n_epochs}"
    if log_scale:
        save_name += "_log"

    plot_metric_n_fr(
        cols,
        intervals,
        metric,
        "Accuracy",
        "",
        save_name=save_name,
        cols=cols_names,
        y_min=y_min,
        log_scale=log_scale,
    )
