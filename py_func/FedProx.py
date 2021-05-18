#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from copy import deepcopy

import random


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)


def FedAvg_agregation_process(model, clients_models_hist: list, weights: list):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution = client_hist[idx].data * weights[k]
            layer_weights.data.add_(contribution)

    return new_model


def FedAvg_agregation_process_for_FA_sampling(
    model, clients_models_hist: list, weights: list
):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model)

    for layer_weigths in new_model.parameters():
        layer_weigths.data.sub_(sum(weights) * layer_weigths.data)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution = client_hist[idx].data * weights[k]
            layer_weights.data.add_(contribution)

    return new_model


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `test_data`"""

    correct = 0

    for features, labels in dataset:

        predictions = model(features)
        _, predicted = predictions.max(1, keepdim=True)

        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy


def loss_dataset(model, train_data, loss_f):
    """Compute the loss of `model` on `test_data`"""
    loss = 0
    for idx, (features, labels) in enumerate(train_data):

        predictions = model(features)
        loss += loss_f(predictions, labels)

    loss /= idx + 1
    return loss


def loss_classifier(predictions, labels):

    criterion = nn.CrossEntropyLoss()
    return criterion(predictions, labels)


def n_params(model):
    """return the number of parameters in the model"""

    n_params = sum(
        [
            np.prod([tensor.size()[k] for k in range(len(tensor.size()))])
            for tensor in list(model.parameters())
        ]
    )

    return n_params


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters"""

    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum(
        [
            torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
            for i in range(len(tensor_1))
        ]
    )

    return norm


def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_f):

    model_0 = deepcopy(model)

    for _ in range(n_SGD):

        features, labels = next(iter(train_data))

        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_f(predictions, labels)
        batch_loss += mu / 2 * difference_models_norm_2(model, model_0)

        batch_loss.backward()
        optimizer.step()


import pickle


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"saved_exp_info/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


def FedProx_sampling_random(
    model,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay=1,
    metric_period=1,
    mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = np.random.choice(
            K, size=n_sampled, replace=True, p=weights
        )

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, weights=[1 / n_sampled] * n_sampled
        )

        if i % metric_period == 0:
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def FedProx_clustered_sampling(
    sampling: str,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    sim_type: str,
    iter_FP=0,
    decay=1.0,
    metric_period=1,
    mu=0.0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    from scipy.cluster.hierarchy import linkage
    from py_func.clustering import get_matrix_similarity_from_grads

    if sampling == "clustered_2":
        from py_func.clustering import get_clusters_with_alg2
    from py_func.clustering import sample_clients

    loss_f = loss_classifier

    # Variables initialization
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    # INITILIZATION OF THE GRADIENT HISTORY AS A LIST OF 0

    if sampling == "clustered_1":
        from py_func.clustering import get_clusters_with_alg1

        distri_clusters = get_clusters_with_alg1(n_sampled, weights)

    elif sampling == "clustered_2":
        from py_func.clustering import get_gradients

        gradients = get_gradients(sampling, model, [model] * K)

    for i in range(n_iter):

        previous_global_model = deepcopy(model)

        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        if i < iter_FP:
            print("MD sampling")

            np.random.seed(i)
            sampled_clients = np.random.choice(
                K, size=n_sampled, replace=True, p=weights
            )

            for k in sampled_clients:

                local_model = deepcopy(model)
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )

                # SAVE THE LOCAL MODEL TRAINED
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model))

                sampled_clients_for_grad.append(k)
                sampled_clients_hist[i, k] = 1

        else:
            if sampling == "clustered_2":

                # GET THE CLIENTS' SIMILARITY MATRIX
                sim_matrix = get_matrix_similarity_from_grads(
                    gradients, distance_type=sim_type
                )

                # GET THE DENDROGRAM TREE ASSOCIATED
                linkage_matrix = linkage(sim_matrix, "ward")

                distri_clusters = get_clusters_with_alg2(
                    linkage_matrix, n_sampled, weights
                )

            for k in sample_clients(distri_clusters):

                local_model = deepcopy(model)
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )

                # SAVE THE LOCAL MODEL TRAINED
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model))

                sampled_clients_for_grad.append(k)
                sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL AND SAVE IT
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, weights=[1 / n_sampled] * n_sampled
        )

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        if i % metric_period == 0:

            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # UPDATE THE HISTORY OF LATEST GRADIENT
        if sampling == "clustered_2":
            gradients_i = get_gradients(
                sampling, previous_global_model, clients_models
            )
            for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
                gradients[idx] = gradient

        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def FedProx_sampling_target(
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay=1,
    mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    # Variables initialization
    n_samples = sum([len(db.dataset) for db in training_sets])
    weights = [len(db.dataset) / n_samples for db in training_sets]
    print("Clients' weights:", weights)

    loss_hist = [
        [
            float(loss_dataset(model, dl, loss_f).detach())
            for dl in training_sets
        ]
    ]
    acc_hist = [[accuracy_dataset(model, dl) for dl in testing_sets]]
    server_hist = [
        [tens_param.detach().numpy() for tens_param in list(model.parameters())]
    ]
    models_hist = []
    sampled_clients_hist = []

    server_loss = sum(
        [weights[i] * loss_hist[-1][i] for i in range(len(weights))]
    )
    server_acc = sum(
        [weights[i] * acc_hist[-1][i] for i in range(len(weights))]
    )
    print(f"====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}")

    for i in range(n_iter):

        clients_params = []
        clients_models = []
        sampled_clients_i = []

        for j in range(n_sampled):

            k = j * 10 + np.random.randint(10)

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))

            sampled_clients_i.append(k)

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, weights=[1 / n_sampled] * n_sampled
        )
        models_hist.append(clients_models)

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist += [
            [
                float(loss_dataset(model, dl, loss_f).detach())
                for dl in training_sets
            ]
        ]
        acc_hist += [[accuracy_dataset(model, dl) for dl in testing_sets]]

        server_loss = sum(
            [weights[i] * loss_hist[-1][i] for i in range(len(weights))]
        )
        server_acc = sum(
            [weights[i] * acc_hist[-1][i] for i in range(len(weights))]
        )

        print(
            f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
        )

        server_hist.append(deepcopy(model))

        sampled_clients_hist.append(sampled_clients_i)

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def FedProx_FedAvg_sampling(
    model,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay=1,
    metric_period=1,
    mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = random.sample([x for x in range(K)], n_sampled)
        print("sampled clients", sampled_clients)

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process_for_FA_sampling(
            deepcopy(model),
            clients_params,
            weights=[weights[client] for client in sampled_clients],
        )

        if i % metric_period == 0:
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist
