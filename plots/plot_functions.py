#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy.random import choice
from scipy.stats import bernoulli
import pickle as pkl


def get_hist(
    folder: str,
    dataset: str,
    sampling: str,
    n_iter: int,
    n_SGD: int,
    batch_size: int,
    lr_global: float,
    lr_local: float,
    n_sampled: int,
    equal: bool,
    decay: float,
    seed: int,
    mu=0.0,
):

    from py_func.file_name import get_file_name

    file_name = get_file_name(
        dataset,
        sampling,
        n_iter,
        n_SGD,
        batch_size,
        lr_global,
        lr_local,
        n_sampled,
        mu,
        equal,
        decay,
        seed,
    )

    with open(f"./saved_exp_info/{folder}/{file_name}.pkl", "rb") as output:
        hist = pkl.load(output)

    return hist


def dist(x, compared_model):
    """distance between model x and optimal model"""
    return np.sum((x - compared_model) ** 2)


def dist_opt(x, glob_min):
    """distance between model x and optimal model"""
    return np.sum((x - glob_min) ** 2)


def dist_local_opt(x, loc_min):
    """distance between model x and optimal model"""
    return np.sum((x - loc_min) ** 2, axis=1)


def update(x, eta_l, K, loc_min):
    """SGD update for one client doing K SGD with learning rate eta_l and
    local min loc_min on model x"""
    phi = 1 - (1 - eta_l) ** K
    return phi * (loc_min - x) + x


def aggreg(x, updates, importances, args):
    """aggregation of updates based on the importances vector"""
    return x + args.eta_g * importances.dot(updates - x)


def sample(sampling, m, importances):

    if sampling == "Full":
        return importances

    elif sampling == "MD":

        selected_clients = choice(
            range(len(importances)), replace=True, size=m, p=importances
        )

        weights = np.zeros(len(importances))
        for idx in selected_clients:
            weights[idx] += 1 / m

    elif sampling == "Uniform":

        selected_clients = choice(
            range(len(importances)),
            replace=False,
            size=m,
            p=[1 / len(importances)] * len(importances),
        )

        weights = np.zeros(len(importances))
        for idx in selected_clients:
            weights[idx] = len(importances) / m * importances[idx]

    elif sampling == "Poisson":

        selected = np.array([bernoulli.rvs(m * pi) for pi in importances])
        weights = selected / m

    elif sampling == "Binomial":

        p_sampling = m / len(importances)
        selected = bernoulli.rvs(p_sampling, size=len(importances))

        weights = np.multiply(selected, importances) / p_sampling

    return weights


def one_step(sampling, importances, loc_min, glob_min, n_iter, args):
    """One SGD for quadratic functions"""

    dists = []

    theta = args.theta_0

    for _ in range(n_iter):

        updates = update(theta, args.eta_l, args.K, loc_min)

        agg_weights = sample(sampling, args.m, importances)

        theta = aggreg(theta, updates, agg_weights, args)

        dists += [dist_opt(theta, glob_min)]

    return dists


def get_lemma_1_quadra(
    sampling: str,
    importances: np.array,
    loc_min: np.array,
    glob_min: np.array,
    compared_model: np.array,
    args,
):
    """compared_model is the model Lemma 1 is evaluated on"""

    phi = 1 - (1 - args.eta_l) ** args.K

    if sampling == "Full":
        alpha = 0
        gamma = importances * 0

    elif sampling == "MD":
        alpha = -1 / args.m
        gamma = importances / args.m
    elif sampling == "Uniform":
        alpha = -(args.n - args.m) / args.m / (args.n - 1)
        gamma = importances ** 2 * ((args.n / args.m - 1) - alpha)

    elif sampling == "Poisson":
        alpha = 0
        gamma = importances * (1 - args.m * importances) / args.m

    elif sampling == "Binomial":
        alpha = 0
        gamma = importances ** 2 * (args.n - args.m) / args.m

    v = dist(args.theta_0, compared_model)
    v += (
        -2
        * args.eta_g
        * phi
        * (args.theta_0 - glob_min)
        .reshape(-1)
        .dot((args.theta_0 - compared_model).reshape(-1))
    )
    v += args.eta_g ** 2 * phi ** 2 * gamma.dot(dist_local_opt(args.theta_0, loc_min))
    v += args.eta_g ** 2 * (1 + alpha) * phi ** 2 * dist_opt(args.theta_0, glob_min)

    return v
