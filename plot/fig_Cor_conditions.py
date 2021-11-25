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


def theo_bound(args):
    """theoretical zone where Uniform outperforms MD sampling"""
    return 1 / (args.n - args.m + 1)


def evolution_sum_pi2(p: np.array, args):

    # EVOLUTION OF SUM PI2
    output = p ** 2 + (1 - p) ** 2 / (args.n - 1)

    # THEORETICAL BOUND
    upper_bound = theo_bound(args) * np.ones(len(p))

    plt.plot(list_p, output, label=r"evolution $\sum_{i=1}^{10} p_i^2$")
    plt.plot(p, upper_bound, label="Upper bound Theorem 2")
    plt.xlabel(r"$p_1$")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/bound_cor_conditions.pdf")
    plt.show()


if __name__ == "__main__":

    class args:
        pass

    parser.parse_args(namespace=args)

    print("Number of clients:", args.n, "\nNumber of sampled clients:", args.m)

    # SUM P_I^2 EVOLUTION
    list_p = np.linspace(0, 1, 200)
    evolution_sum_pi2(list_p, args)
