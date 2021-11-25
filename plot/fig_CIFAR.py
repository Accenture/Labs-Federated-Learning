#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# def CIFAR_diff(
#     folder: str,
#     samplings: list,
#     n_iter: int,
#     n_SGD: int,
#     lr_local: float,
#     batch_size: int,
#     n_seeds: int,
#     lr_global=1.0,
#     decay=1.0,
# ):
#
#     from py_func.importances import clients_importances
#     from plots.plot_functions import get_hist
#
#     def one_subplot(
#         ax,
#         dataset: str,
#         weight_type: str,
#         lr_local: float,
#         n_sampled: int,
#     ):
#         """
#         plot the difference between the averaged curve of MD sampling
#         - the averaged curve of Uniform sampling for an FL dataset with
#         `n_clients` from which `n_sampled` of them are selected at every
#         optimization step.
#         """
#
#         def rolling_window(a, window):
#
#             return np.array(
#                 [a[i: i + window] for i in range(len(a) - window + 1)]
#             )
#
#         importances = clients_importances(weight_type, dataset)
#
#         dic = {}
#         for sampling in samplings:
#
#             sampled_clients = np.zeros(
#                 (
#                     n_iter + 1,
#                     len(importances),
#                 )
#             )
#
#             n=0
#             for seed in range(n_seeds):
#                 try:
#                     hist_seed = get_hist(
#                         folder,
#                         dataset,
#                         sampling,
#                         n_iter,
#                         n_SGD,
#                         batch_size,
#                         lr_global,
#                         lr_local,
#                         n_sampled,
#                         weight_type,
#                         decay,
#                         seed,
#                     )
#                     sampled_clients += hist_seed
#                     n += 1
#                 except:
#                     break
#
#             sampled_clients /= n
#             hist = np.average(sampled_clients, 1, importances)
#             # hist_plot = (np.mean(rolling_window(hist, window=5), axis=1))
#             dic[sampling] = (np.mean(rolling_window(hist, window=5), axis=1))
#
#         diff = dic["MD"] - dic["Uniform"]
#         diff = diff
#         ax.plot(diff)
#
#     n_rows, n_cols = 3, 1
#     fig, axes = plt.subplots(n_rows, n_cols,  figsize=(8, 6))
#
#     list_weights = ["ratio"] * n_rows
#     list_lr = [lr_local]* n_rows
#     list_m = [10] * 3
#     list_dataset = ["CIFAR10_0.1", "CIFAR10_0.01", "CIFAR10_0.001"]
#
#     list_y = [
#         [0.5, 1.5],
#         [0.5, 1.5],
#         [0.8, 1.8],
#     ]
#
#     for idx, (weight_type, lr_local, dataset,m ) in enumerate(
#         zip(list_weights, list_lr, list_dataset, list_m)
#     ):
#
#         ax = axes[0]
#
#         one_subplot(ax, dataset, weight_type, lr_local, m)
#
#         # if idx == 0:
#         #     ax.set_ylabel(r"$\mathcal{L}(\theta_{MD}) - \mathcal{L}(\theta_{U})$")
#         #     ax.set_title(r"(a) - $p_i = 1 /n$")
#         #     ax.set_xlabel("# rounds")
#         #
#         # elif idx == 4:
#         #     ax.set_title(r"(b) - $p_i = n_i / M$")
#         #     ax.set_xlabel("# rounds")
#
#     fig.legend(
#         ax,
#         labels=[10, 20, 40, 80],
#         ncol=4,
#         bbox_to_anchor=(0.65, 0.14),
#     )
#
#     plt.tight_layout()
#     fig.savefig("plots/CIFAR_diff.pdf", bbox_inches="tight")
#     # plt.show()
#


def CIFAR_appendix(
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

    def rolling_window(a, window):

        return np.array([a[i : i + window] for i in range(len(a) - window + 1)])

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

            n = 0
            for seed in range(n_seeds):

                try:
                    hist_seed = get_hist(
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

                    print(hist_seed.shape)
                    sampled_clients += hist_seed
                    n += 1
                except:
                    break

            sampled_clients /= n
            hist = np.average(sampled_clients, 1, importances)
            hist_plot = np.mean(rolling_window(hist, window=5), axis=1)
            ax.plot(hist_plot, label=sampling)

    n_cols = 3
    fig, axes = plt.subplots(n_cols, 1, figsize=(4, 6))

    # list_weights = ["uniform", "ratio"] * 4
    list_weights = ["ratio"] * n_cols
    list_lr = [lr_local] * n_cols
    list_m = [10] * 3
    list_dataset = ["CIFAR10_0.1", "CIFAR10_0.01", "CIFAR10_0.001"]
    list_alpha = [0.1, 0.01, 0.001]

    list_y = [
        [0.5, 1.5],
        [0.5, 1.5],
        [0.8, 1.8],
    ]

    for idx, (weight_type, lr_local, m, dataset, y, alpha) in enumerate(
        zip(list_weights, list_lr, list_m, list_dataset, list_y, list_alpha)
    ):

        print(weight_type, lr_local, m, dataset, y)

        ax = axes[idx]

        # PLOT THE AVERAGED SAMPLING CURVES
        # if idx ==0 or idx==4:
        one_subplot(ax, dataset, weight_type, lr_local, m, "sf")

        # FORMAT THE SUBPLOT
        # ax.set_ylim(y)
        ax.set_title("(" + chr(97 + idx) + r") - $\alpha$ = " + f"{alpha}", pad=0.0)

        ax.set_ylabel(r"$\mathcal{L}(\theta^t)$")

    axes[2].set_xlabel("# rounds")

    fig.legend(
        ax,
        labels=samplings,
        ncol=3,
        bbox_to_anchor=(1.02, 0.042),
    )

    plt.tight_layout(pad=0.0)
    fig.savefig("plots/CIFAR_appendix.pdf", bbox_inches="tight")
    # plt.show()


samplings = ["MD", "Uniform", "Clustered"]
n_iter = 1000
n_SGD = 100
lr_local = 0.05
batch_size = 64
n_seeds = 30

# CIFAR_diff("loss", samplings, n_iter, n_SGD, lr_local, batch_size, n_seeds)

CIFAR_appendix("loss", samplings, n_iter, n_SGD, lr_local, batch_size, n_seeds)
