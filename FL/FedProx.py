#!/usr/bin/env python
# coding: utf-8

import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from copy import deepcopy

import FL.sampling as sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def aggregation(model, local_models: list, weights: list, lr_g: float):
    """Creates the new global model"""

    new_model = deepcopy(model).to(device)

    for weight, local_model in zip(weights, local_models):

        if weight > 0:
            local_p = [t.detach() for t in list(local_model.parameters())]

            for new_w, w, local_w in zip(
                new_model.parameters(), model.parameters(), local_p
            ):
                Delta_i = weight * (local_w.data - w.data)
                new_w.data.add_(lr_g * Delta_i)

    return new_model


def loss_acc_dataset(model, client_data, loss_f):
    """Compute loss and acc of `model` on `client_data`"""

    features = client_data.dataset.features
    labels = client_data.dataset.labels

    loss, correct = 0, 0

    n_loop = len(features) // 5000 + 1
    for i in range(n_loop):

        features_i = features[i * 5000 : (i + 1) * 5000]
        labels_i = labels[i * 5000 : (i + 1) * 5000]

        with torch.no_grad():
            predictions = model(features_i.to(device))

        loss += float(loss_f(predictions, labels_i.view(-1).to(device))) * len(
            features_i
        )

        _, predicted = predictions.max(1, keepdim=True)
        correct += float(
            torch.sum(predicted.view(-1, 1) == labels_i.view(-1, 1).to(device)).item()
        )

    # print(len(client_data), len(client_data.dataset))
    loss /= len(client_data.dataset)
    accuracy = 100 * correct / len(client_data.dataset)

    return loss, accuracy


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters"""

    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum(
        [torch.sum((tensor_1[i] - tensor_2[i]) ** 2) for i in range(len(tensor_1))]
    )

    return norm


def local_learning(model, mu: float, lr_l: float, train_data, n_SGD: int, loss_f):

    # MODEL USED FOR FEDPROX FOR REGULARIZATION AT EVERY OPTIMIZATION ROUND
    model_0 = deepcopy(model)

    optimizer = optim.SGD(model.parameters(), lr=lr_l)

    for _ in range(n_SGD):

        optimizer.zero_grad()

        features, labels = next(iter(train_data))
        predictions = model(features.to(device))

        batch_loss = loss_f(predictions, labels.view(-1).to(device))
        if mu > 0.0:
            batch_loss += mu / 2 * difference_models_norm_2(model, model_0)

        batch_loss.backward()
        optimizer.step()


def FedProx(exp, model, training_sets: list, testing_sets: list, weights: np.array):
    """SIMULATES FL. DESCRIPTION OF THE INPUTS IN README.md ."""

    # INITIALIZE LEARNING: GLOBAL MODEL, LOSS FUNCTION, SAMPLING SCHEME
    model = model.to(device)
    loss_f = nn.CrossEntropyLoss()

    if exp.sampling == "Clustered":
        distri_clusters = sampling.get_clusters_with_alg1(exp.m, weights)
        args_sampling = {"clusters": distri_clusters}
    else:
        args_sampling = {"weights": weights}

    # LOSS AND ACCURACY OF INITIAL MODEL
    if exp.max_iter == 0:
        for k, dl in enumerate(training_sets):
            exp.loss[0, k] = loss_acc_dataset(model, dl, loss_f)[0]
        for k, dl in enumerate(testing_sets):
            exp.acc[0, k] = loss_acc_dataset(model, dl, loss_f)[1]

    server_loss = np.dot(weights, exp.loss[exp.max_iter])
    server_acc = np.dot(weights, exp.acc[exp.max_iter])
    print(
        f"====> i: {exp.max_iter} " f"Loss: {server_loss} Test Accuracy: {server_acc}"
    )

    for i in range(exp.max_iter, exp.T):

        # SAMPLE CLIENTS
        exp.sampled_clients[i], exp.agg_weights[i] = sampling.sampling_clients(
            exp.sampling, exp.n, exp.m, **args_sampling
        )
        working_clients = np.where(exp.agg_weights[i] > 0)[0]
        print("Participating clients:", working_clients)

        # SEND GLOBAL MODEL TO WORKING CLIENTS
        local_models = [deepcopy(model).to(device) for _ in range(exp.n)]

        # WORKING CLIENTS PERFORM THEIR LOCAL WORK
        for client in working_clients:
            # for local_model, k in zip(local_models, working_clients):
            local_learning(
                local_models[client],
                exp.mu,
                exp.lr_l,
                training_sets[client],
                exp.n_SGD,
                loss_f,
            )

        # CREATE THE NEW GLOBAL MODEL
        model = aggregation(
            deepcopy(model).to(device),
            local_models,
            exp.agg_weights[i],
            exp.lr_g,
        )

        # LOSS/ACCURACY ON NEW GLOBAL MODEL
        for k, dl in enumerate(training_sets):
            exp.loss[i + 1, k] = loss_acc_dataset(model, dl, loss_f)[0]
        for k, dl in enumerate(testing_sets):
            exp.acc[i + 1, k] = loss_acc_dataset(model, dl, loss_f)[1]

        server_loss = np.dot(weights, exp.loss[i + 1])
        server_acc = np.dot(weights, exp.acc[i + 1])
        print(
            f"====> i: {i+1} Loss: {server_loss} " f"Server Test Accuracy: {server_acc}"
        )

        # DECAY THE LEARNING RATE
        exp.lr_l *= exp.decay

    # SAVE THE FINAL GLOBAL MODEL AND DIFFERENT METRICS
    exp.save_metrics("acc", "loss", "agg_weights", "sampled_clients")
    exp.save_model(model)
