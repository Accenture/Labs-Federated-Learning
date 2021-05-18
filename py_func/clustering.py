#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from itertools import product


from scipy.cluster.hierarchy import fcluster
from copy import deepcopy


def get_clusters_with_alg1(n_sampled: int, weights: np.array):
    "Algorithm 1"

    epsilon = int(10 ** 10)
    # associate each client to a cluster
    augmented_weights = np.array([w * n_sampled * epsilon for w in weights])
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


def get_similarity(grad_1, grad_2, distance_type="L1"):

    if distance_type == "L1":

        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum(np.abs(g_1 - g_2))
        return norm

    elif distance_type == "L2":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum((g_1 - g_2) ** 2)
        return norm

    elif distance_type == "cosine":
        norm, norm_1, norm_2 = 0, 0, 0
        for i in range(len(grad_1)):
            norm += np.sum(grad_1[i] * grad_2[i])
            norm_1 += np.sum(grad_1[i] ** 2)
            norm_2 += np.sum(grad_2[i] ** 2)

        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0
        else:
            norm /= np.sqrt(norm_1 * norm_2)

            return np.arccos(norm)


def get_gradients(sampling, global_m, local_models):
    """return the `representative gradient` formed by the difference between
    the local work and the sent global model"""

    local_model_params = []
    for model in local_models:
        local_model_params += [
            [tens.detach().numpy() for tens in list(model.parameters())]
        ]

    global_model_params = [
        tens.detach().numpy() for tens in list(global_m.parameters())
    ]

    local_model_grads = []
    for local_params in local_model_params:
        local_model_grads += [
            [
                local_weights - global_weights
                for local_weights, global_weights in zip(
                    local_params, global_model_params
                )
            ]
        ]

    return local_model_grads


def get_matrix_similarity_from_grads(local_model_grads, distance_type):
    """return the similarity matrix where the distance chosen to
    compare two clients is set with `distance_type`"""

    n_clients = len(local_model_grads)

    metric_matrix = np.zeros((n_clients, n_clients))
    for i, j in product(range(n_clients), range(n_clients)):

        metric_matrix[i, j] = get_similarity(
            local_model_grads[i], local_model_grads[j], distance_type
        )

    return metric_matrix


def get_matrix_similarity(global_m, local_models, distance_type):

    n_clients = len(local_models)

    local_model_grads = get_gradients(global_m, local_models)

    metric_matrix = np.zeros((n_clients, n_clients))
    for i, j in product(range(n_clients), range(n_clients)):

        metric_matrix[i, j] = get_similarity(
            local_model_grads[i], local_model_grads[j], distance_type
        )

    return metric_matrix


def get_clusters_with_alg2(
    linkage_matrix: np.array, n_sampled: int, weights: np.array
):
    """Algorithm 2"""
    epsilon = int(10 ** 10)

    # associate each client to a cluster
    link_matrix_p = deepcopy(linkage_matrix)
    augmented_weights = deepcopy(weights)

    for i in range(len(link_matrix_p)):
        idx_1, idx_2 = int(link_matrix_p[i, 0]), int(link_matrix_p[i, 1])

        new_weight = np.array(
            [augmented_weights[idx_1] + augmented_weights[idx_2]]
        )
        augmented_weights = np.concatenate((augmented_weights, new_weight))
        link_matrix_p[i, 2] = int(new_weight * epsilon)

    clusters = fcluster(
        link_matrix_p, int(epsilon / n_sampled), criterion="distance"
    )

    n_clients, n_clusters = len(clusters), len(set(clusters))

    # Associate each cluster to its number of clients in the cluster
    pop_clusters = np.zeros((n_clusters, 2)).astype(int)
    for i in range(n_clusters):
        pop_clusters[i, 0] = i + 1
        for client in np.where(clusters == i + 1)[0]:
            pop_clusters[i, 1] += int(weights[client] * epsilon * n_sampled)

    pop_clusters = pop_clusters[pop_clusters[:, 1].argsort()]

    distri_clusters = np.zeros((n_sampled, n_clients)).astype(int)

    # n_sampled biggest clusters that will remain unchanged
    kept_clusters = pop_clusters[n_clusters - n_sampled :, 0]

    for idx, cluster in enumerate(kept_clusters):
        for client in np.where(clusters == cluster)[0]:
            distri_clusters[idx, client] = int(
                weights[client] * n_sampled * epsilon
            )

    k = 0
    for j in pop_clusters[: n_clusters - n_sampled, 0]:

        clients_in_j = np.where(clusters == j)[0]
        np.random.shuffle(clients_in_j)

        for client in clients_in_j:

            weight_client = int(weights[client] * epsilon * n_sampled)

            while weight_client > 0:

                sum_proba_in_k = np.sum(distri_clusters[k])

                u_i = min(epsilon - sum_proba_in_k, weight_client)

                distri_clusters[k, client] = u_i
                weight_client += -u_i

                sum_proba_in_k = np.sum(distri_clusters[k])
                if sum_proba_in_k == 1 * epsilon:
                    k += 1

    distri_clusters = distri_clusters.astype(float)
    for l in range(n_sampled):
        distri_clusters[l] /= np.sum(distri_clusters[l])

    return distri_clusters


from numpy.random import choice


def sample_clients(distri_clusters):

    n_clients = len(distri_clusters[0])
    n_sampled = len(distri_clusters)

    sampled_clients = np.zeros(len(distri_clusters), dtype=int)

    for k in range(n_sampled):
        sampled_clients[k] = int(choice(n_clients, 1, p=distri_clusters[k]))

    return sampled_clients
