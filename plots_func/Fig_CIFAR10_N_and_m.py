#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from plots_func.common_funcs import get_acc_loss
from plots_func.common_funcs import weights_clients


def loss_acc_evo(
    kept: list,
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
    plot_names: str,
):
    def plot_hist_mean(hist: np.array, weights, axis, idx_row, idx_col):
        hist_mean = np.average(hist, 1, weights)
        X = np.where(hist_mean > 0)[0]
        y = hist_mean[X]
        return axis[idx_row, idx_col].plot(X, y)

    weights = weights_clients(dataset)

    try:
        acc_hists, legend = get_acc_loss(
            dataset, sampling, "acc", n_SGD, seed, lr, decay, p, mu
        )

        idx_kept = [legend.index(title_simu) for title_simu in kept]
        acc_hists = [acc_hists[idx] for idx in idx_kept]
        lx = []
        for hist in acc_hists:
            lx.append(plot_hist_mean(hist, weights, axis, idx_row, 0)[0])
    except:
        pass

    if idx_row == 0:
        axis[idx_row, 0].set_title("Test Accuracy in %")
    if idx_row == n_rows - 1:
        axis[idx_row, 0].set_xlabel("# rounds", labelpad=0.0)
    if idx_row == 0:
        axis[idx_row, 0].legend(kept, title=r"$\bf{Sampling}$")
    axis[idx_row, 0].set_ylabel(plot_names[idx_row], rotation=0, labelpad=6.0)

    try:
        loss_hists, _ = get_acc_loss(
            dataset, sampling, "loss", n_SGD, seed, lr, decay, p, mu
        )
        idx_kept = [legend.index(title_simu) for title_simu in kept]
        loss_hists = [loss_hists[idx] for idx in idx_kept]

        for hist in loss_hists:
            plot_hist_mean(hist, weights, axis, idx_row, 1)
    except:
        pass

    if idx_row == 0:
        axis[idx_row, 1].set_title("Training Loss")
    if idx_row == n_rows - 1:
        axis[idx_row, 1].set_xlabel("# rounds", labelpad=0.0)


def plot_fig_CIFAR10_N_and_m_both(alpha: float, l_lr: float, l_p: float):

    kept = ["MD", "Alg. 1", "Alg. 2"]
    plot_names = ["(a)", "(b)", "(c)", "(d)"]

    n_rows, n_cols = 4, 2
    dataset_base = "CIFAR10"

    fig, axis = plt.subplots(n_rows, n_cols, figsize=(4.5, 8))

    sampling = "clustered_1"
    seed = 0
    decay = 1.0
    mu = 0.0

    # INFLUENCE OF THE NUMBER OF SGD
    dataset = f"{dataset_base}_nbal_{alpha}"
    print(dataset)
    loss_acc_evo(
        kept,
        dataset,
        sampling,
        10,
        seed,
        l_lr[0],
        decay,
        0.1,
        mu,
        n_rows,
        n_cols,
        axis,
        0,
        plot_names,
    )

    loss_acc_evo(
        kept,
        dataset,
        sampling,
        500,
        seed,
        l_lr[1],
        decay,
        0.1,
        mu,
        n_rows,
        n_cols,
        axis,
        1,
        plot_names,
    )

    # INGLUENCE OF THE NUMBER OF SAMPLED CLIENTS
    loss_acc_evo(
        kept,
        dataset,
        sampling,
        100,
        seed,
        l_lr[2],
        decay,
        0.05,
        mu,
        n_rows,
        n_cols,
        axis,
        2,
        plot_names,
    )

    loss_acc_evo(
        kept,
        dataset,
        sampling,
        100,
        seed,
        l_lr[3],
        decay,
        0.2,
        mu,
        n_rows,
        n_cols,
        axis,
        3,
        plot_names,
    )

    plt.tight_layout(pad=0.0)
    plt.savefig(f"plots/plot_CIFAR_N_and_m_custom_lr_all.pdf")


def metric_evo(
    metric: str,
    kept: list,
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
    idx_col: int,
    plot_name: str,
    smooth: bool,
):
    def plot_hist_mean(hist: np.array, weights, axis, idx_row, idx_col, smooth):
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
            return axis[idx_row, idx_col].plot(a, ysmooth)
        else:
            return axis[idx_row, idx_col].plot(X, y)

    weights = weights_clients(dataset)

    try:
        acc_hists, legend = get_acc_loss(
            dataset, sampling, metric, n_SGD, seed, lr, decay, p, mu
        )

        idx_kept = [legend.index(title_simu) for title_simu in kept]
        acc_hists = [acc_hists[idx] for idx in idx_kept]
        lx = []
        for hist in acc_hists:
            lx.append(
                plot_hist_mean(hist, weights, axis, idx_row, idx_col, smooth)[0]
            )
    except:
        pass

    if idx_col == 0:
        axis[idx_row, idx_col].set_ylabel("Global Loss", labelpad=0.0)
    if idx_row == n_rows - 1:
        axis[idx_row, idx_col].set_xlabel("# rounds")
    if idx_row == 0 and idx_col == 0:
        axis[idx_row, idx_col].legend(kept, title=r"$\bf{Sampling}$")
    axis[idx_row, idx_col].set_title(plot_name)


def plot_fig_CIFAR10_N_and_m_one(
    metric: str, alpha: float, l_lr: float, l_p: float, smooth: str
):

    kept = ["MD", "Alg. 1", "Alg. 2"]
    plot_names = [
        "N=10 and m=10",
        "N=500 and m=10",
        "N=100 and m=5",
        "N=100 and m=20",
    ]

    n_rows, n_cols = 2, 2
    dataset_base = "CIFAR10"

    fig, axis = plt.subplots(n_rows, n_cols, figsize=(4.5, 4.5))

    sampling = "clustered_1"
    seed = 0
    decay = 1.0
    mu = 0.0

    # INFLUENCE OF THE NUMBER OF SGD
    dataset = f"{dataset_base}_nbal_{alpha}"
    print(dataset)
    metric_evo(
        metric,
        kept,
        dataset,
        sampling,
        10,
        seed,
        l_lr[0],
        decay,
        l_p[0],
        mu,
        n_rows,
        n_cols,
        axis,
        0,
        0,
        plot_names[0],
        smooth,
    )

    metric_evo(
        metric,
        kept,
        dataset,
        sampling,
        500,
        seed,
        l_lr[1],
        decay,
        l_p[1],
        mu,
        n_rows,
        n_cols,
        axis,
        0,
        1,
        plot_names[1],
        smooth,
    )

    # INGLUENCE OF THE NUMBER OF SAMPLED CLIENTS
    metric_evo(
        metric,
        kept,
        dataset,
        sampling,
        100,
        seed,
        l_lr[2],
        decay,
        l_p[2],
        mu,
        n_rows,
        n_cols,
        axis,
        1,
        0,
        plot_names[2],
        smooth,
    )

    metric_evo(
        metric,
        kept,
        dataset,
        sampling,
        100,
        seed,
        l_lr[3],
        decay,
        l_p[3],
        mu,
        n_rows,
        n_cols,
        axis,
        1,
        1,
        plot_names[3],
        smooth,
    )

    plt.tight_layout(pad=0.0)
    plt.savefig(f"plots/plot_CIFAR_N_and_m_custom_lr.pdf")
