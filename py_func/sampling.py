#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import bernoulli


def sampling_clients(sampling: str, n_clients: int, n_sampled: int, weights: np.array):
    """Return an array with the indices of the clients that are sampled and
    an array with their associated weights"""

    #    np.random.seed(i)

    if sampling == "Full":

        selected = np.ones(n_clients)
        agg_weights = weights

    if sampling == "MD":

        selected = np.zeros(n_clients)
        agg_weights = np.zeros(n_clients)

        selected_idx = np.random.choice(
            n_clients, size=n_sampled, replace=True, p=weights
        )

        for idx in selected_idx:
            selected[idx] = 1
            agg_weights[idx] += 1 / n_sampled

    if sampling == "Improved":

        selected = np.zeros(n_clients)
        agg_weights = np.zeros(n_clients)

        while np.sum(selected) < n_sampled:
            selected_idx = np.random.choice(n_clients, size=1, p=weights)
            selected[selected_idx] = 1
            agg_weights[selected_idx] += 1

        agg_weights /= np.sum(agg_weights)

    elif sampling == "Uniform":

        selected = np.zeros(n_clients)
        agg_weights = np.zeros(n_clients)

        selected_idx = np.random.choice(n_clients, size=n_sampled, replace=False)

        for idx in selected_idx:
            selected[idx] = 1
            agg_weights[idx] = n_clients / n_sampled * weights[idx]

    elif sampling == "Binomial":

        p_sampling = n_sampled / n_clients
        selected = bernoulli.rvs(p_sampling, size=n_clients)

        agg_weights = np.multiply(selected, weights) / p_sampling

    elif sampling == "Poisson":

        selected = np.array([bernoulli.rvs(n_sampled * pi) for pi in weights])
        agg_weights = selected / n_sampled

    return selected, agg_weights
