#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np

# from FL.experiment import Experiment
from FL.server import Server
from FL.client import Clients

import torchvision
import random
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simu_unlearning(server: Server, clients: Clients):
    """SIMULATES centralized. DESCRIPTION OF THE INPUTS IN README.md ."""

    # for W_r in server.policy:

    # GET THE CLIENTS IMPORTANCE
    server.clients_importance(clients.n_samples())

    # GET THE MODEL UNLEARNING THE CLIENT IN W_r
    server.forget_Wr(server.policy)

    # T_loss_acc = 10
    T_loss_acc = max(server.T // 10, 10)

    for t in range(server.T):

        server.g_model.to(device)

        # MEASURE GLOBAL LOSS AND ACC OF CURRENT GLOBAL MODEL
        if t % T_loss_acc == 0:
            server.loss_acc_global(clients, "train")
            loss = server.loss_acc_global(clients, "test")
            if np.isnan(loss):
                break

        # EVERY CLIENT PERFORMS ITS LOCAL WORK
        local_models = clients.local_work(
            server.working_clients,
            server.g_model,
            server.loss_f,
            server.optimizer,
            server.optimizer_params,
            # server.lr_l,
            server.n_SGD,
            # server.lambd,
            # server.clip
        )

        # AGGREGATED THE CLIENTS CONTRIBUTION TO CREATE THE NEW GLOBAL MODEL
        server.aggregation_round(local_models)

        if server.unlearn_scheme == "train":
            # # COMPUTE OUR METRIC
            # server.compute_metric(local_models)

            # PREPARE THE MODEL TO BE SAVED FOR EACH UNLEARNING METHOD
            server.keep_best_model(local_models)

    # LOSS/ACCURACY ON NEW GLOBAL MODEL
    server.loss_acc_global(clients, "train")
    server.loss_acc_global(clients, "test")

    # SAVE THE BEST MODELS
    if exp.unlearn_scheme == "train":
        exp.save_best_models(f"{exp.file_name}",
                             server.best_models[-1])

    print("\n")
