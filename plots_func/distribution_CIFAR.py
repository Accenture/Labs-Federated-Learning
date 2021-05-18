#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from numpy.random import dirichlet


def dirichlet_one(
    ax, data: np.array, alpha: float, n_classes: int, n_clients: int
):

    category_names = [i for i in range(n_classes)]
    labels = [str(digit) for digit in np.random.randn(n_clients)]
    data_cum = data.cumsum(axis=1)

    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, colname in enumerate(category_names):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=1.0, label=colname)

        ax.yaxis.set_ticks([])

    ax.margins(y=0)
    ax.set_ylabel(r"$\alpha=$" + f"{alpha}")


def repartition_n_samples(
    ax, data: np.array, alpha: float, n_classes: int, n_clients: int
):

    categories = [10, 30, 30, 20, 10]
    categories = [0, 10, 40, 70, 90, 100]
    n_labels = [100, 250, 500, 750, 1000]

    for i, label in enumerate(n_labels):

        data[categories[i] : categories[i + 1]] *= label

    sum_classes = np.sum(data, axis=0)

    ax.bar(
        np.arange(n_classes),
        sum_classes,
        width=0.9,
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
    )
    ax.xaxis.set_visible(False)


def distribution_samples(list_alpha, n_classes, n_clients):

    n_rows = len(list_alpha)
    n_cols = 2

    fig, axis = plt.subplots(n_rows, n_cols, figsize=(5, n_rows / n_cols * 5))

    for i, alpha in enumerate(list_alpha):

        data = dirichlet([alpha] * n_classes, size=n_clients)

        dirichlet_one(axis[i, 0], data, alpha, n_classes, n_clients)
        repartition_n_samples(axis[i, 1], data, alpha, n_classes, n_clients)

    axis[0, 0].set_title("(a)")
    axis[0, 1].set_title("(b)")

    handles, labels = axis[n_rows - 1, 0].get_legend_handles_labels()

    fig.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.07),
        bbox_transform=fig.transFigure,
        title=r"$\bf{Classes}$",
    )

    plt.savefig("plots/distribution_samples_CIFAR.pdf", bbox_inches="tight")
