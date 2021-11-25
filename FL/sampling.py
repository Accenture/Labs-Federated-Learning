#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import bernoulli
from numpy.random import choice


def sampling_clients(sampling: str, n_clients: int, n_sampled: int, **args):
    """Return an array with the indices of the clients that are sampled and
    an array with their associated weights"""

    #    np.random.seed(i)

    selected = np.zeros(n_clients)
    agg_weights = np.zeros(n_clients)

    if sampling == "Full":
        selected = np.ones(n_clients)
        agg_weights = args["weights"]

    elif sampling == "MD":
        selected_idx = np.random.choice(
            n_clients, size=n_sampled, replace=True, p=args["weights"]
        )

        for idx in selected_idx:
            selected[idx] = 1
            agg_weights[idx] += 1 / n_sampled

    elif sampling == "Improved":
        while np.sum(selected) < n_sampled:
            selected_idx = np.random.choice(n_clients, size=1, p=args["weights"])
            selected[selected_idx] = 1
            agg_weights[selected_idx] += 1

        agg_weights /= np.sum(agg_weights)

    elif sampling == "Uniform":
        selected_idx = np.random.choice(n_clients, size=n_sampled, replace=False)

        for idx in selected_idx:
            selected[idx] = 1
            agg_weights[idx] = n_clients / n_sampled * args["weights"][idx]

    elif sampling == "Binomial":

        p_sampling = n_sampled / n_clients
        selected = bernoulli.rvs(p_sampling, size=n_clients)

        agg_weights = np.multiply(selected, args["weights"]) / p_sampling

    elif sampling == "Poisson":

        selected = np.array([bernoulli.rvs(n_sampled * pi) for pi in args["weights"]])
        agg_weights = selected / n_sampled

    elif sampling == "Clustered":

        for p_cluster in args["clusters"]:
            idx_selected = int(choice(n_clients, 1, p=p_cluster))
            selected[idx_selected] = 1
            agg_weights[idx_selected] += 1 / n_sampled

    return selected, agg_weights


def get_clusters_with_alg1(n_sampled: int, weights: np.array):

    epsilon = int(10 ** 10)
    # associate each client to a cluster
    augmented_weights = np.array([w * n_sampled * epsilon for w in weights]).astype(int)
    ordered_client_idx = np.flip(np.argsort(augmented_weights))

    n_clients = len(weights)
    distri_clusters = np.zeros((n_sampled, n_clients)).astype(int)

    k = 0
    for client_idx in ordered_client_idx:

        while augmented_weights[client_idx] > 0:

            sum_proba_in_k = np.sum(distri_clusters[k])

            u_i = min(epsilon - sum_proba_in_k, augmented_weights[client_idx])

            distri_clusters[k, client_idx] = u_i
            augmented_weights[client_idx] += -u_i

            sum_proba_in_k = np.sum(distri_clusters[k])
            if sum_proba_in_k == 1 * epsilon:
                k += 1

    distri_clusters = distri_clusters.astype(float)
    for l in range(n_sampled):
        distri_clusters[l] /= np.sum(distri_clusters[l])

    return distri_clusters
