#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from copy import deepcopy


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)


def FedAvg_agregation_process(model, clients_models_hist, device, weights):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model).to(device)

    set_to_zero_model_weights(new_model)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution = client_hist[idx].to(device).data * weights[k]
            layer_weights.data.add_(contribution)

    return new_model


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `test_data`"""

    correct = 0

    for features, labels in iter(dataset):

        predictions = model(features)

        _, predicted = predictions.max(1, keepdim=True)

        correct += torch.sum(predicted.view(-1) == labels).item()

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

    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction="mean")

    return loss(m(predictions), labels.view(-1))


def n_params(model):
    """return the number of parameters in the model"""

    n_params = 0
    for tensor in list(model.parameters()):

        n_params_tot = 1
        for k in range(len(tensor.size())):
            n_params_tot *= tensor.size()[k]

        n_params += n_params_tot

    return n_params


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters"""

    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum(
        [torch.sum((tensor_1[i] - tensor_2[i]) ** 2) for i in range(len(tensor_1))]
    )

    return norm


def train_step(model, model_0, mu, optimizer, train_data, loss_f, device):
    """Train `model` on one epoch of `train_data`"""

    total_loss = 0

    model.train()

    for idx, (features, labels) in enumerate(train_data):

        optimizer.zero_grad()

        features, labels = features.to(device), labels.to(device)

        predictions = model(features)

        loss = loss_f(predictions, labels)
        loss += mu / 2 * difference_models_norm_2(model, model_0)
        total_loss += loss

        loss.backward()
        optimizer.step()

    return total_loss / (idx + 1)


def local_learning(
    model, mu: float, optimizer, train_data, epochs: int, loss_f, device: str
):

    model_0 = deepcopy(model)

    for e in range(epochs):
        local_loss = train_step(
            model, model_0, mu, optimizer, train_data, loss_f, device
        )

    return float(local_loss.cpu().detach().numpy())


# def difference_model(model_1,model_2):
#    """Return the norm 2 difference between the two model parameters
#    """
#
#    tensor_1=list(model_1.parameters())
#    tensor_2=list(model_2.parameters())
#
#    norm=0
#
#    for i in range(len(tensor_1)):
#
#        norm+=torch.sum(torch.abs(tensor_1[i]-tensor_2[i]))
#
#    #Get the number of parameters in the model
#    norm/=n_params(model_1)
#
#    return norm.detach().numpy()


from python_code.functions import save_pkl


def FedProx(
    model,
    training_sets: list,
    n_iter: int,
    loss_f,
    testing_set: list,
    device: str,
    mu: float,
    file_root_name: str,
    epochs=5,
    lr=10 ** -4,
    decay=1,
):
    """
    all the clients are considered in this implementation of FedProx
    Parameters:
        - `model` common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `device`: whether the simulation is run on GPU or CPU
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `file_root_name`: name that will start all the files saving the
            different variables at every iteration
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
        - `loss_hist`: the loss at each iteration of all the clients
        - `acc_hist`: the accuracy at each iteration of all the clients
    """

    # Variables initialization
    K = len(training_sets)  # number of clients
    n_samples = sum([len(db.dataset) for db in training_sets])
    weights = [len(db.dataset) / n_samples for db in training_sets]
    print("Clients' weights:", weights)

    loss_hist = [
        [
            float(loss_dataset(model, training_sets[k], loss_f).detach())
            for k in range(K)
        ]
    ]
    acc_hist = [[accuracy_dataset(model, testing_set[k]) for k in range(K)]]
    server_hist = [
        [tens_param.detach().numpy() for tens_param in list(model.parameters())]
    ]

    server_loss = sum([weights[i] * loss_hist[-1][i] for i in range(len(weights))])
    server_acc = sum([weights[i] * acc_hist[-1][i] for i in range(len(weights))])
    print(f"====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}")

    for i in range(n_iter):

        clients_params = []
        clients_losses = []

        for k in range(K):

            local_model = deepcopy(model).to(device)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_loss = local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                epochs,
                loss_f,
                device,
            )

            clients_losses.append(local_loss)

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, device, weights=weights
        ).cpu()

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist += [
            [
                float(loss_dataset(model, training_sets[k], loss_f).detach())
                for k in range(K)
            ]
        ]
        acc_hist += [[accuracy_dataset(model, testing_set[k]) for k in range(K)]]

        server_loss = sum([weights[i] * loss_hist[-1][i] for i in range(len(weights))])
        server_acc = sum([weights[i] * acc_hist[-1][i] for i in range(len(weights))])

        print(f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}")

        server_hist.append(
            [
                tens_param.detach().cpu().numpy()
                for tens_param in list(model.parameters())
            ]
        )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    save_pkl(server_hist, "saved_models/hist", f"{file_root_name}_server")
    save_pkl(loss_hist, "hist/loss", file_root_name)
    save_pkl(acc_hist, "hist/acc", file_root_name)

    torch.save(model.state_dict(), f"saved_models/final/{file_root_name}.pth")

    return model, np.array(loss_hist), np.array(acc_hist)
