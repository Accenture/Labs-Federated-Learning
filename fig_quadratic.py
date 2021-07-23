#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import argparse

from numpy.random import uniform
from itertools import product

from plots.plot_functions import dist, get_lemma_1_quadra, one_step


parser = argparse.ArgumentParser(description="experiments parameters")

parser.add_argument("--n", type=int, help="number of participating clients", default=10)

parser.add_argument("--m", type=int, help="number of selected clients", default=5)

parser.add_argument("--n_params", type=int, help="dimension of the model", default=20)

parser.add_argument(
    "--bound",
    type=int,
    help="amplitude of the normal for the local minimum creation",
    default=1,
)

parser.add_argument("--eta_l", type=int, help="Local learning rate", default=0.1)

parser.add_argument("--eta_g", type=int, help="Global learnign rate", default=1)

parser.add_argument("--K", type=int, help="Number of Local SGD", default=10)

# parser.add_argument(
#    "--n_iter", type = int,
#    help = "Number of Local SGD",
#    default = 1)


def theoretical_step_lemma_1(list_p: np.array, loc_min: np.array, args):
    """
    Returns the expected theoretical distance between the global model
    after one FL optimization step and the FL optimum.
    The distance is calculated with the Decomposition Theorem with calculus
    in Appendix.
    """

    theo_distances = {sampling: [] for sampling in args.samplings}

    for p, sampling in product(list_p, args.samplings):

        importances = np.array([p] + [(1 - p) / (args.n - 1)] * (args.n - 1))
        glob_min = importances.dot(loc_min)

        if sampling == "Initial":
            theo_distances[sampling].append(dist(args.theta_0, glob_min))

        else:
            theo_distances[sampling].append(
                get_lemma_1_quadra(
                    sampling, importances, loc_min, glob_min, glob_min, args
                )
            )

    return theo_distances


def practical_step_lemma_1(list_p: np.array, loc_min: np.array, args):
    """
    Run `args.n_draw` optimization step initialized from `args.theta0`
    where the only source of randomness comes from sampling clients
    and return the averaged distance.
    """

    exp_distances = {sampling: [] for sampling in args.samplings}

    for p, sampling in product(list_p_2, args.samplings):

        importances = np.array([p] + [(1 - p) / (args.n - 1)] * (args.n - 1))
        glob_min = importances.dot(loc_min)

        if sampling == "Full":

            v = one_step(sampling, importances, loc_min, glob_min, 1, args)

        else:
            #            v = np.mean([one_step(sampling, importances, glob_min, args) for _ in range(args.n_draw)])
            v = [
                one_step(sampling, importances, loc_min, glob_min, 1, args)
                for _ in range(args.n_draw)
            ]

        exp_distances[sampling].append(v)

    return exp_distances


def plot_paper_quadratic(
    file_name: str,
    list_p: list,
    theo_dist_niid,
    theo_dist_iid,
    list_p_2,
    exp_dist_niid,
    exp_dist_iid,
    args,
):
    """
    Combine the theoretical and experimental distances for the niid and iid
    case to create the paper plot.
    """

    plt.figure(figsize=(9, 3))

    # THEO + IID
    plt.subplot(1, 4, 1)
    for sampling in args.samplings:
        plt.plot(list_p, theo_dist_iid[sampling], label=sampling)

    plt.ylim(1.5, 4)
    plt.xlabel(r"$p_1$")
    plt.ylabel(r"${\left\Vert\theta^1 - \theta^*\right\Vert}^2$")
    plt.title("(a)")
    plt.legend()

    # THEO + NIID
    plt.subplot(1, 4, 2)
    for sampling in args.samplings:
        plt.plot(list_p, theo_dist_niid[sampling], label=sampling)
    plt.ylim(1, 4)
    plt.xlabel(r"$p_1$")
    plt.title("(b)")

    # EXP + IID
    plt.subplot(1, 4, 3)
    for sampling in args.samplings:

        mean = np.mean(exp_dist_iid[sampling], axis=1)
        #        std = np.std(exp_dist_iid[sampling], axis=1)

        #        plt.errorbar(list_p_2, mean, std, label = sampling)
        plt.plot(list_p_2, mean, label=sampling)

    plt.ylim(1.5, 4)
    plt.xlabel(r"$p_1$")
    plt.title("(c)")

    # EXP + NIID
    plt.subplot(1, 4, 4)
    for sampling in args.samplings:

        mean = np.mean(exp_dist_niid[sampling], axis=1)
        #        std = np.std(exp_dist_niid[sampling], axis=1)

        #        plt.errorbar(list_p_2, mean, std, label = sampling)
        plt.plot(list_p_2, mean, label=sampling)
    plt.ylim(1, 4)
    plt.xlabel(r"$p_1$")
    plt.title("(d)")

    plt.tight_layout()
    plt.savefig(f"plots/{file_name}.pdf")
    plt.show()


if __name__ == "__main__":

    class args:
        pass

    parser.parse_args(namespace=args)

    print("Number of clients:", args.n, "\nNumber of sampled clients:", args.m)

    # CREATE THE CLIENTS' LOCAL MINIMIA
    np.random.seed(1)
    loc_min_niid = uniform(-args.bound, args.bound, size=(args.n, args.n_params))
    loc_min_iid = np.tile(loc_min_niid[0], (args.n, 1))

    # INITIAL MODEL FL STARTS FROM
    args.theta_0 = uniform(-args.bound, args.bound, size=(1, args.n_params))

    # ONE STEP EXPECTED IMPROVEMENT NIID AND IID
    args.samplings = ["Full", "MD", "Uniform"]
    list_p = np.linspace(0, 1, 200)

    theo_dist_niid = theoretical_step_lemma_1(list_p, loc_min_niid, args)
    theo_dist_iid = theoretical_step_lemma_1(list_p, loc_min_iid, args)

    # ONE STEP EXPERIMENATAL IMPROVEMENT AVERAGED OVER MANY RUNS NIID AND IID
    list_p_2 = np.linspace(0, 1, 200)
    args.n_draw = 1000

    exp_dist_niid = practical_step_lemma_1(list_p_2, loc_min_niid, args)
    exp_dist_iid = practical_step_lemma_1(list_p_2, loc_min_iid, args)

    # EXPECTED LEARNING
    args.T = 5

    file_name = f"one_step_convergence"

    plot_paper_quadratic(
        file_name,
        list_p,
        theo_dist_niid,
        theo_dist_iid,
        list_p_2,
        exp_dist_niid,
        exp_dist_iid,
        args,
    )
