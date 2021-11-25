#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import argparse

from numpy.random import uniform
from itertools import product

from plot.common import dist, get_lemma_1_quadra, one_step


def Decompo_theo(list_p: np.array, loc_min: np.array, samplings: list, args):
    """
    Returns the expected theoretical distance between the global model
    after one FL optimization step and the FL optimum.
    The distance is calculated with the Decomposition Theorem with calculus
    in Appendix.
    """

    theo_distances = {sampling: [] for sampling in samplings}

    for p, sampling in product(list_p, samplings):

        importances = np.array([p] + [(1 - p) / (args.n - 1)] * (args.n - 1))
        glob_min = importances.dot(loc_min)

        if sampling == "Initial":
            theo_distances[sampling].append(dist(args.theta0, glob_min))

        else:
            theo_distances[sampling].append(
                get_lemma_1_quadra(
                    sampling, importances, loc_min, glob_min, glob_min, args
                )
            )

    return theo_distances


def practical_step(
    list_p: np.array, loc_min: np.array, n_draw: int, samplings: list, args
):
    """
    Run `args.n_draw` optimization step initialized from `args.theta0`
    where the only source of randomness comes from sampling clients
    and return the averaged distance.
    """

    exp_distances = {sampling: [] for sampling in samplings}

    for p, sampling in product(list_p, samplings):

        importances = np.array([p] + [(1 - p) / (args.n - 1)] * (args.n - 1))
        glob_min = importances.dot(loc_min)

        if sampling == "Full":

            v = one_step(sampling, importances, loc_min, glob_min, 1, args)

        else:
            v = [
                one_step(sampling, importances, loc_min, glob_min, 1, args)
                for _ in range(n_draw)
            ]

        exp_distances[sampling].append(v)

    return exp_distances


def plot_quadra_Decompo_theo(
    file_name: str,
    samplings: list,
    list_p: list,
    theo_dist_niid,
    theo_dist_iid,
    list_p_2,
    exp_dist_niid,
    exp_dist_iid,
):
    """
    Combine the theoretical and experimental distances for the niid and iid
    case to create the paper plot.
    """

    plt.figure(figsize=(9, 2.5))

    # THEO + IID
    plt.subplot(1, 4, 1)
    for sampling in samplings:
        plt.plot(list_p, theo_dist_iid[sampling], label=sampling)

    plt.ylim(1.5, 4)
    plt.xlabel(r"$p_1$")
    plt.ylabel(r"${\left\Vert\theta^1 - \theta^*\right\Vert}^2$")
    plt.title("(a)")
    plt.legend()

    # THEO + NIID
    plt.subplot(1, 4, 2)
    for sampling in samplings:
        plt.plot(list_p, theo_dist_niid[sampling], label=sampling)
    plt.ylim(1, 4)
    plt.xlabel(r"$p_1$")
    plt.title("(b)")

    # EXP + IID
    plt.subplot(1, 4, 3)
    for sampling in samplings:

        mean = np.mean(exp_dist_iid[sampling], axis=1)
        plt.plot(list_p_2, mean, label=sampling)

    plt.ylim(1.5, 4)
    plt.xlabel(r"$p_1$")
    plt.title("(c)")

    # EXP + NIID
    plt.subplot(1, 4, 4)
    for sampling in samplings:

        mean = np.mean(exp_dist_niid[sampling], axis=1)
        plt.plot(list_p_2, mean, label=sampling)

    plt.ylim(1, 4)
    plt.xlabel(r"$p_1$")
    plt.title("(d)")

    plt.tight_layout(pad=0)
    plt.savefig(f"figures/pdf/{file_name}.pdf")
    plt.savefig(f"figures/{file_name}.png")


def plot_variance_simulations(
    file_name: str,
    samplings: list,
    list_p: list,
    exp_dist_niid: np.array,
    exp_dist_iid: np.array,
):
    """
    Combine the theoretical and experimental distances for the niid and iid
    case to create the paper plot.
    """

    v_alpha = 0.5  # transparency factor

    fig, ax = plt.subplots(2, 2)

    # EXPERIMENTAL + VARIANCE + IID
    for sampling in samplings:
        mean = np.mean(exp_dist_iid[sampling], axis=1).reshape(-1)
        std = np.std(exp_dist_iid[sampling], axis=1).reshape(-1)

        ax[0, 0].plot(list_p, mean, label=sampling)
        ax[0, 0].fill_between(list_p, mean - std, mean + std, alpha=v_alpha)

    # EXPERIMENTAL + VARIANCE + NIID
    for sampling in samplings:
        mean = np.mean(exp_dist_niid[sampling], axis=1).reshape(-1)
        std = np.std(exp_dist_niid[sampling], axis=1).reshape(-1)

        ax[0, 1].plot(list_p, mean, label=sampling)
        ax[0, 1].fill_between(list_p, mean - std, mean + std, alpha=v_alpha)

    # EXPERIMENTAL + MIN-MAX + IID
    for sampling in samplings:
        mean = np.mean(exp_dist_iid[sampling], axis=1).reshape(-1)
        dist_min = np.min(exp_dist_iid[sampling], axis=1).reshape(-1)
        dist_max = np.max(exp_dist_iid[sampling], axis=1).reshape(-1)

        ax[1, 0].plot(list_p, mean, label=sampling)
        ax[1, 0].fill_between(list_p, dist_min, dist_max, alpha=v_alpha)

    # EXPERIMENTAL + MIN-MAX + NIID
    for sampling in samplings:
        mean = np.mean(exp_dist_niid[sampling], axis=1).reshape(-1)
        dist_min = np.min(exp_dist_niid[sampling], axis=1).reshape(-1)
        dist_max = np.max(exp_dist_niid[sampling], axis=1).reshape(-1)

        ax[1, 1].plot(list_p, mean, label=sampling)
        ax[1, 1].fill_between(list_p, dist_min, dist_max, alpha=v_alpha)

    ax[0, 0].legend()
    for i, j in product(range(2), range(2)):
        ax[i, j].set_ylim(0, 8)
        ax[i, j].set_title("(" + chr(97 + 2 * i + j) + ")")
        if i == 1:
            ax[i, j].set_xlabel(r"$p_1$")
        if j == 0:
            ax[i, j].set_ylabel(r"${\left\Vert\theta^1 - \theta^*\right\Vert}^2$")

    plt.tight_layout(pad=0)
    plt.savefig(f"figures/pdf/{file_name}.pdf")
    plt.savefig(f"figures/{file_name}.png")


class args:
    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])


def quadra_Theo1(
    n: int,
    m: int,
    n_params: int,
    bound: float,
    eta_l: float,
    eta_g: float,
    K: int,
    n_draw: int,
):

    print("Number of clients:", n, "\nNumber of sampled clients:", m)

    dic_train = {"n": n, "m": m, "eta_l": eta_l, "eta_g": eta_g, "K": K}
    args_train = args(dic_train)

    # CREATE THE CLIENTS' LOCAL MINIMIA
    np.random.seed(1)
    loc_min_niid = uniform(-bound, bound, size=(n, n_params))
    loc_min_iid = np.tile(loc_min_niid[0], (n, 1))

    # INITIAL MODEL FL STARTS FROM
    args_train.theta0 = uniform(-bound, bound, size=(1, n_params))
    list_p = np.linspace(0, 1, 200)

    # ONE STEP EXPECTED IMPROVEMENT NIID AND IID
    samplings = ["Full", "MD", "Uniform"]
    theo_dist_niid = Decompo_theo(list_p, loc_min_niid, samplings, args_train)
    theo_dist_iid = Decompo_theo(list_p, loc_min_iid, samplings, args_train)

    # ONE STEP EXPERIMENTAL IMPROVEMENT AVERAGED OVER MANY RUNS NIID AND IID
    exp_dist_niid = practical_step(list_p, loc_min_niid, n_draw, samplings, args_train)
    exp_dist_iid = practical_step(list_p, loc_min_iid, n_draw, samplings, args_train)

    # PLOT THE FIGURE FOR THE PAPER WITH MEAN FOR PRACTICAL
    file_name = f"one_step_mean_{n_draw}"
    plot_quadra_Decompo_theo(
        file_name,
        samplings,
        list_p,
        theo_dist_niid,
        theo_dist_iid,
        list_p,
        exp_dist_niid,
        exp_dist_iid,
    )

    # PLOT FIGURE APPENDIX FOR EXPERIMENTAL INCLUDING VARIANCE
    file_name = f"one_step_variance_{n_draw}"
    plot_variance_simulations(
        file_name,
        samplings,
        list_p,
        exp_dist_niid,
        exp_dist_iid,
    )
