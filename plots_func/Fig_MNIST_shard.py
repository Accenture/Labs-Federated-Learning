#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


from plots_func.common_funcs import get_acc_loss


names_legend = [
    "MD",
    "Alg. 2 L2",
    "Alg. 2 L1",
    "Alg. 2",
    "Target",
    "Alg. 1",
    "FedAvg",
]


def get_n_diff_types(hist_sampled_clients, sampling_type: str, iter_max: int):
    """return the number of different classes in the sampled clients"""

    if sampling_type == "Target":
        return [10 for _ in range(min(len(hist_sampled_clients), iter_max))]

    for j in range(len(hist_sampled_clients)):
        for i in range(10):
            hist_sampled_clients[j, i * 10 : (i + 1) * 10] *= i + 1

    n_diff_clients = np.array(
        [len(set(row)) - 1 for row in hist_sampled_clients]
    )
    return n_diff_clients[:iter_max]


def plot_fig_alg2(
    dataset: str,
    sampling: str,
    n_SGD: int,
    seed: int,
    lr: float,
    decay: float,
    p: float,
    mu: float,
    iter_max=200,
):

    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        weights = pkl.load(output)

    weights = weights / np.sum(weights)

    acc_hists, legend = get_acc_loss(
        dataset,
        sampling,
        "acc",
        n_SGD,
        seed,
        lr,
        decay,
        p,
        mu,
        names_legend=names_legend,
    )
    loss_hists, _ = get_acc_loss(
        dataset,
        sampling,
        "loss",
        n_SGD,
        seed,
        lr,
        decay,
        p,
        mu,
        names_legend=names_legend,
    )
    sampled_clients, _ = get_acc_loss(
        dataset,
        sampling,
        "sampled_clients",
        n_SGD,
        seed,
        lr,
        decay,
        p,
        mu,
        names_legend=names_legend,
    )

    print(legend)

    kept = ["MD", "Target", "Alg. 1", "Alg. 2"]

    idx_kept = [legend.index(title_simu) for title_simu in kept]

    acc_hists = [acc_hists[idx] for idx in idx_kept]
    loss_hists = [loss_hists[idx] for idx in idx_kept]
    sampled_clients = [sampled_clients[idx] for idx in idx_kept]

    def plot_hist_mean(hist: np.array):
        hist_mean = np.average(hist, 1, weights)
        X = np.where(hist_mean > 0)[0]
        y = hist_mean[X]
        plt.plot(X, y)

    def plot_hist_std(hist: np.array):
        hist_mean = np.average(hist, 1, weights)

        X = np.where(hist_mean > 0)[0]

        hist_std = np.sqrt(
            np.average((hist - hist_mean[:, None]) ** 2, 1, weights)
        )
        y = hist_std[X]

        plt.plot(X, y)

    fig, axes = plt.subplots(3, 2, figsize=(4.5, 4.5))

    # DISTRIBUTION PLOT
    ax1 = axes[0, 0]
    ax1.set_title("Distributions")
    l_x = []
    for sampling_type, hist in zip(kept, sampled_clients):
        l_x.append(ax1.plot(get_n_diff_types(hist, sampling_type, iter_max))[0])
    ax1.set_ylabel("# of classes", labelpad=0.0)

    from matplotlib.ticker import MaxNLocator

    ax1.yaxis.set_major_locator(MaxNLocator(4, integer=True))

    axes[0, 1].axis("off")

    plt.subplot(3, 2, 3)
    plt.title("Mean accuracy")
    for hist in acc_hists:
        plot_hist_mean(hist[:iter_max])
    plt.ylabel("Accuracy in %", labelpad=0.0)

    plt.subplot(3, 2, 4)
    plt.title("Std accuracy")
    for hist in acc_hists:
        plot_hist_std(hist[:iter_max])
    plt.ylim(0)

    plt.subplot(3, 2, 5)
    plt.title("Mean loss")
    for hist in loss_hists:
        plot_hist_mean(hist[:iter_max])
    plt.xlabel("# rounds", labelpad=0.0)
    plt.ylabel("Loss", labelpad=6.0)

    plt.subplot(3, 2, 6)
    plt.title("Std loss")
    for hist in loss_hists:
        plot_hist_std(hist[:iter_max])
    plt.ylim(0)
    plt.xlabel("# rounds", labelpad=0.0)

    plt.tight_layout(pad=0.0)
    fig.legend(
        l_x,
        labels=kept,
        title=r"$\bf{Sampling}$",
        ncol=2,
        bbox_to_anchor=(1.0, 0.9),
    )
    plt.savefig(f"plots/plot_fig_improved_MNIST.pdf")
