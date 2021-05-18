#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from copy import deepcopy
import torch.optim as optim
import numpy as np


def get_n_params(model):
    """return the number of parameters in the model"""

    n_params = sum([np.prod(tensor.size()) for tensor in list(model.parameters())])
    return n_params


def linear_noising(
    model,
    list_std: list,
    list_power: list,
    iteration: int,
    noise_type: str,
    std_multiplicator,
):
    """Return the noised model of the free-rider"""

    if noise_type == "disguised":
        for idx, layer_tensor in enumerate(model.parameters()):

            mean_0 = torch.zeros(layer_tensor.size())
            std_tensor = torch.zeros(
                layer_tensor.size()
            ) + std_multiplicator * list_std[1] * iteration ** (-list_power[1])
            noise_additive = torch.normal(mean=mean_0, std=std_tensor)

            layer_tensor.data += noise_additive

    return model


def get_std(model_A, model_B, noise):
    """get the standard deviation at iteration 2 with the proposed heuristic"""

    list_tens_A = [tens_param.detach() for tens_param in list(model_A.parameters())]
    list_tens_B = [tens_param.detach() for tens_param in list(model_B.parameters())]

    if noise == "plain":
        return [0, 0]

    if noise == "disguised":

        sum_abs_diff = 0

        for tens_A, tens_B in zip(list_tens_A, list_tens_B):
            sum_abs_diff += torch.sum(torch.abs(tens_A - tens_B))

        std = sum_abs_diff / get_n_params(model_A)
        return [0, std]


def FL_freeloader(
    n_freeriders: int,
    global_model,
    training_dls: list,
    n_fr_samples: int,
    n_iter: int,
    testing_set: list,
    loss_f,
    device: str,
    mu: float,
    file_root_name: str,
    multiplicator: int,
    noise_type="plain",
    epochs=5,
    lr=10 ** -4,
    std_0=0,
    power=1,
    decay=1,
):
    """
    Parameters:
        -`n_frerriders`: number of free-riders each having `n_fr_samples` samples
        - `global_model`: common structure used by the clients and the server
        - `training_dls`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_fr_samples`: number of samples the free-rider pretends having
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `loss_f`: loss function applied to the output of the model
        - `device`: whether the simulation is run on GPU or CPU
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `file_root_name`: name that will start all the files saving the global
            model at every iteration
        - `multiplicator`: number of times the std of the heuristic is multiplied
        - `noise_type`: 'plain' or 'disguised'
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `std_0`: original std used at iteration 0 before the use of the
            heuristic
        - `power`: gamma in the paper for the noise
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
        - `loss_hist`: the loss at each iteration of all the clients
        - `acc_hist`: the accuracy at each iteration of all the clients
    """

    from FL_functions import (
        loss_dataset,
        accuracy_dataset,
        local_learning,
        FedAvg_agregation_process,
    )

    # VARIABLES INITIALIZATION
    K = len(training_dls)  # number of clients
    n_fair_samples = sum([len(db.dataset) for db in training_dls])
    n_samples = n_fr_samples * n_freeriders + n_fair_samples
    weights_fair = [len(db.dataset) / n_fair_samples for db in training_dls]
    weights = [n_fr_samples / n_samples] * n_freeriders + [
        len(db.dataset) / n_samples for db in training_dls
    ]
    print("clients' weights:", weights)

    loss_hist = [
        [
            float(loss_dataset(global_model, training_dls[k], loss_f).detach())
            for k in range(K)
        ]
    ]
    acc_hist = [[accuracy_dataset(global_model, testing_set[k]) for k in range(K)]]
    server_hist = [
        [tens_param.detach().numpy() for tens_param in list(global_model.parameters())]
    ]

    # NOISE PARAMETERS
    if noise_type == "plain":
        list_std = [0, 0]
    elif noise_type == "disguised":
        list_std = [0, std_0]
    list_power = [0, power]

    server_loss = sum(
        [weight * loss_hist[-1][idx] for idx, weight in enumerate(weights_fair)]
    )
    server_acc = sum(
        [weight * acc_hist[-1][idx] for idx, weight in enumerate(weights_fair)]
    )
    print(f"====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}")

    for i in range(n_iter):

        # THE LIST S ELEMENTS ARE FIRST FOR THE FREE RIDERS THEN FAIR CLIENTS
        clients_params = []
        clients_losses = []

        # WORK DONE BY THE FREE-RIDERS
        if i == 1 and noise_type == "disguised":
            list_std = get_std(global_model, m_previous, noise_type)
            print("noise std", list_std)

        for j in range(n_freeriders):

            local_model = linear_noising(
                deepcopy(global_model),
                list_std,
                list_power,
                max(i, 1),
                noise_type,
                multiplicator,
            ).to(device)

            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

        # WORK DONE BY THE FAIR CLIENTS
        for k in range(K):

            local_model = deepcopy(global_model).to(device)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_loss = local_learning(
                local_model,
                mu,
                local_optimizer,
                training_dls[k],
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
        new_model = FedAvg_agregation_process(
            deepcopy(global_model), clients_params, device, weights=weights
        ).cpu()

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist += [
            [
                float(loss_dataset(new_model, training_dls[k], loss_f).detach())
                for k in range(K)
            ]
        ]
        acc_hist += [[accuracy_dataset(new_model, testing_set[k]) for k in range(K)]]

        server_loss = sum(
            [weight * loss_hist[-1][idx] for idx, weight in enumerate(weights_fair)]
        )
        server_acc = sum(
            [weight * acc_hist[-1][idx] for idx, weight in enumerate(weights_fair)]
        )

        print(f"====> i: {i+1} Loss: {server_loss} Accuracy: {server_acc}")

        m_previous = deepcopy(global_model)
        global_model = new_model

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    from python_code.functions import save_pkl

    save_pkl(server_hist, "saved_models/hist", f"{file_root_name}_server")
    save_pkl(loss_hist, "hist/loss", file_root_name)
    save_pkl(acc_hist, "hist/acc", file_root_name)

    torch.save(global_model.state_dict(), f"saved_models/final/{file_root_name}.pth")

    return global_model, np.array(loss_hist), np.array(acc_hist)
