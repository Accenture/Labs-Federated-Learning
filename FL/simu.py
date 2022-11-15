#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from copy import deepcopy

from FL.server import Server
from FL.client import Clients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simu_async(exp, server: Server, clients: Clients):
    """SIMULATES FL. DESCRIPTION OF THE INPUTS IN README.md ."""

    # LATEST GLOBAL MODEL SENT TO EVERY CLIENT
    sent_models = [deepcopy(server.g_model).to(device)
                   for _ in range(exp.M)]

    for n, schedule_n in enumerate(server.schedule_global):

        # MEASURE GLOBAL LOSS AND ACC OF CURRENT GLOBAL MODEL
        if n % server.freq_loss_measure == 0:
            loss, _ = server.loss_acc_global(clients)

            # IF DIVERGENCE STOP THE LEARNING PROCESS
            if np.isnan(loss):
                break

        # PARTICIPATING CLIENTS
        working_clients = np.where(schedule_n >= 0)[0]
        print(f"====> n: {n} Working clients: {working_clients}")

        # LOCAL MODEL OWNED BY EVERY CLIENT
        received_models = clients.local_work(
            working_clients,
            deepcopy(sent_models),
            server.loss_f,
            server.lr_l,
            server.n_SGD,
            server.clip
        )

        # AGGREGATED THE CLIENTS CONTRIBUTION TO CREATE THE NEW GLOBAL MODEL
        server.aggregation(working_clients, received_models, sent_models)

        # SEND IT TO THE CLIENTS THAT CONTRIBUTED
        for client in working_clients:
            sent_models[client] = deepcopy(server.g_model).to(device)

    # Compute the loss on the final global model
    loss, _ = server.loss_acc_global(clients)

