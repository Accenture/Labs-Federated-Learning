#!/usr/bin/env python
# coding: utf-8

from py_func.sampling import sampling_clients
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from copy import deepcopy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def previous_model_contribution(model, weights, lr_global):
    """set all the parameters of a model to 0"""

    coef = lr_global * np.sum(weights)

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(coef * layer_weigths.data)


def FedAvg_agregation_process(
    model, clients_models_hist: list, weights: list, lr_global: float
):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model).to(device)
    previous_model_contribution(new_model, weights, lr_global)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution = lr_global * client_hist[idx].data * weights[k]
            layer_weights.data.add_(contribution)

    return new_model


def loss_acc_dataset(model, client_data, loss_f):
    """Compute the loss of `model` on `test_data`"""

    features = client_data.dataset.features
    labels = client_data.dataset.labels

    loss, correct = 0, 0

    n_loop = len(features) // 5000 + 1
    for i in range(n_loop):

        features_i = features[i * 5000 : (i + 1) * 5000]
        labels_i = labels[i * 5000 : (i + 1) * 5000]

        predictions = model(features_i.to(device)).detach()
        loss += float(loss_f(predictions, labels_i.to(device)))

        _, predicted = predictions.max(1, keepdim=True)
        correct += float(
            torch.sum(predicted.view(-1, 1) == labels_i.to(device).view(-1, 1)).item()
        )

    loss /= n_loop
    accuracy = 100 * correct / len(client_data.dataset)

    return loss, accuracy


def loss_classifier(predictions, labels):

    criterion = nn.CrossEntropyLoss()
    return criterion(predictions, labels.view(-1))


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters"""

    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum(
        [torch.sum((tensor_1[i] - tensor_2[i]) ** 2) for i in range(len(tensor_1))]
    )

    return norm


def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_f):

    model_0 = deepcopy(model)

    for _ in range(n_SGD):

        features, labels = next(iter(train_data))

        optimizer.zero_grad()

        predictions = model(features.to(device))

        batch_loss = loss_f(predictions, labels.to(device))
        batch_loss += mu / 2 * difference_models_norm_2(model, model_0)

        batch_loss.backward()
        optimizer.step()


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"saved_exp_info/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


def FedProx_sampling_random(
    model,
    sampling: str,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr_global: float,
    lr_local: float,
    weights: np.array,
    file_name: str,
    mu=0,
    decay=1.0,
):
    """
    This functions simulates an FL process.
    The description of its inputs can be found in the repository ReadMe.
    """
    model = model.to(device)
    loss_f = loss_classifier

    K = len(training_sets)  # number of clients

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = loss_acc_dataset(model, dl, loss_f)[0]

    for k, dl in enumerate(testing_sets):
        acc_hist[0, k] = loss_acc_dataset(model, dl, loss_f)[1]

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K))  # .astype(int)
    agg_weights_hist = np.zeros((n_iter, K))

    for i in range(n_iter):

        clients_params = []

        sampled_clients, agg_weights = sampling_clients(sampling, K, n_sampled, weights)
        sampled_clients_hist[i] = sampled_clients
        agg_weights_hist[i] = agg_weights

        print(np.where(agg_weights > 0)[0])

        for k in np.where(agg_weights > 0)[0]:

            local_model = deepcopy(model).to(device)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr_local)

            local_learning(
                local_model, mu, local_optimizer, training_sets[k], n_SGD, loss_f
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process(
            deepcopy(model).to(device),
            clients_params,
            agg_weights[np.where(agg_weights > 0)],
            lr_global,
        )

        lr_local *= decay

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        for k, dl in enumerate(training_sets):

            loss_hist[i + 1, k] = loss_acc_dataset(model, dl, loss_f)[0]

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = loss_acc_dataset(model, dl, loss_f)[1]

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(
            f"====> i: {i+1} Loss: {server_loss} " f"Server Test Accuracy: {server_acc}"
        )

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)
    save_pkl(agg_weights_hist, "agg_weights", file_name)

    torch.save(model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth")

    return model, loss_hist, acc_hist
