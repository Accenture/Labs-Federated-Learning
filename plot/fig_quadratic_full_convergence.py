#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from plot.common import sample, dist_opt


def one_process(p: float, loc_min: np.array, samplings: list, args):

    losses_hist = {sampling: [] for sampling in samplings}

    importances = np.array([p] + [(1 - p) / (args.n - 1)] * (args.n - 1))
    glob_min = importances.dot(loc_min)

    phi = 1 - (1 - args.eta_l) ** args.K

    for sampling in samplings:

        theta = deepcopy(args.theta0)
        losses = [dist_opt(theta, glob_min) / 2]

        for _ in range(args.T):

            # Sample a set of clients
            sampled_clients = sample(sampling, args.m, importances)

            # Sampled clients perform their work
            updates = phi * (loc_min - theta)
            contribution = sampled_clients.dot(updates)

            # Update theta
            theta += args.eta_g * contribution

            # Compute the loss of the model
            losses.append(dist_opt(theta, glob_min) / 2)

        losses_hist[sampling] = losses

    return losses_hist


def plot_paper_quadratic(
    file_name: str,
    samplings: list,
    losses_iid: np.array,
    losses_niid: np.array,
):
    """
    Combine the theoretical and experimental distances for the niid and iid
    case to create the paper plot.
    """

    plt.figure(figsize=(7, 2))

    # EXP + IID
    plt.subplot(1, 2, 1)
    for sampling in samplings:
        plt.plot(losses_iid[sampling], label=sampling)
    plt.yscale("log")
    plt.ylabel(r"${\left\Vert\theta^t - \theta^*\right\Vert}^2$")
    plt.xlabel(r"$t$")
    plt.legend()

    # EXP + NIID
    plt.subplot(1, 2, 2)
    for sampling in samplings:
        plt.plot(losses_niid[sampling], label=sampling)
    plt.yscale("log")
    plt.xlabel(r"$t$")

    plt.tight_layout(pad=0)
    plt.savefig(f"figures/pdf/{file_name}.pdf")
    plt.savefig(f"figures/{file_name}.png")


class args:
    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])


def quadra_full_train(
    n_iter: int,
    n: int,
    m: int,
    n_params: int,
    bound: float,
    eta_l: float,
    eta_g: float,
    K: int,
):

    print("Number of clients:", n, "\nNumber of sampled clients:", m)

    dic_train = {"T": n_iter, "n": n, "m": m, "eta_l": eta_l, "eta_g": eta_g, "K": K}
    args_train = args(dic_train)

    # CREATE THE CLIENTS' LOCAL MINIMIA
    np.random.seed(1)
    loc_min_niid = uniform(-bound, bound, size=(n, n_params))
    loc_min_iid = np.tile(loc_min_niid[0], (n, 1))

    # INITIAL MODEL FL STARTS FROM
    args_train.theta0 = uniform(-bound, bound, size=(1, n_params))

    # ONE STEP EXPECTED IMPROVEMENT NIID AND IID
    samplings = ["Full", "MD", "Uniform"]

    p = 0.9
    losses_niid = one_process(p, loc_min_niid, samplings, args_train)
    losses_iid = one_process(p, loc_min_iid, samplings, args_train)

    # EXPECTED LEARNING
    file_name = f"full_convergence"

    plot_paper_quadratic(
        file_name,
        samplings,
        losses_iid,
        losses_niid,
    )
