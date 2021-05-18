#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from plots_func.common_funcs import get_acc_loss
from plots_func.common_funcs import weights_clients

names_legend = [
    "MD",
    "Alg. 2 L2",
    "Alg. 2 L1",
    "Alg. 2",
    "Target",
    "Alg. 1",
    "Partial",
    "Alg. 1",
]


def get_metric_evo(
    kept: list,
    metric: str,
    dataset: str,
    sampling: str,
    n_SGD: int,
    seed: int,
    lr: float,
    decay: float,
    p: float,
    mu: float,
    n_rows: int,
    n_cols: int,
    axis,
    idx_row: int,
    smooth: False,
):
    def plot_hist_mean(hist: np.array, weights, axis, idx_row, smooth):
        hist_mean = np.average(hist, 1, weights)
        X = np.where(hist_mean > 0)[0]
        y = hist_mean[X]
        if smooth:
            window = 5

            def rolling_window(a, window):

                return np.array(
                    [a[i : i + window] for i in range(len(a) - window + 1)]
                )

            ysmooth = np.mean(rolling_window(y, window), axis=1)
            ysmooth_std = np.std(rolling_window(y, window), axis=1)
            a = X[int((window - 1) / 2) : -int((window - 1) / 2)]
            b = ysmooth - ysmooth_std
            c = ysmooth + ysmooth_std
            return axis[idx_row].plot(a, ysmooth)
        else:
            return axis[idx_row].plot(X, y)

    weights = weights_clients(dataset)

    try:
        hists, legend = get_acc_loss(
            dataset,
            sampling,
            metric,
            n_SGD,
            seed,
            lr,
            decay,
            p,
            mu,
            names_legend=names_legend,
        )

        print(legend)

        kept_hists = []
        kept_legend = []
        for title_simu in kept:
            try:
                kept_hists.append(hists[legend.index(title_simu)])
                kept_legend.append(legend[legend.index(title_simu)])
            except:
                pass

        lx = []
        for hist in kept_hists:
            lx.append(plot_hist_mean(hist, weights, axis, idx_row, smooth)[0])
    except:
        pass
    if idx_row == 0:
        axis[idx_row].set_ylabel("Global Loss")
    elif idx_row == 1:
        axis[idx_row].set_ylabel("Global Loss")
    if idx_row == n_rows - 1:
        axis[idx_row].set_xlabel("# rounds", labelpad=0.0)
    if idx_row == 0:
        axis[idx_row].legend(kept_legend, title=r"$\bf{Sampling}$")


def plot_regularization(
    alpha: float, n_SGD: int, p: float, mu: float, lr: float
):

    kept = ["FedAvg", "MD", "Alg. 1", "Alg. 2"]

    n_rows, n_cols = 2, 1
    dataset_base = "CIFAR10"

    fig, axis = plt.subplots(n_rows, n_cols, figsize=(4.5, 5))

    sampling = "clustered_1"
    seed = 0
    decay = 1.0

    get_metric_evo(
        kept,
        "loss",
        f"{dataset_base}_nbal_{alpha}",
        sampling,
        n_SGD,
        seed,
        lr,
        decay,
        p,
        mu,
        n_rows,
        n_cols,
        axis,
        0,
        True,
    )

    get_metric_evo(
        kept,
        "loss",
        f"{dataset_base}_nbal_{alpha}",
        sampling,
        n_SGD,
        seed,
        lr,
        decay,
        p,
        mu,
        n_rows,
        n_cols,
        axis,
        1,
        False,
    )

    plt.tight_layout()
    plt.savefig(f"plots/plot_regularization.pdf")
