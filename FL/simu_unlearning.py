#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np

from FL.experiment import Experiment
from FL.server import Server
from FL.client import Clients

import torchvision
import random
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simu_unlearning(exp: Experiment, server: Server, clients: Clients):
    """SIMULATES centralized. DESCRIPTION OF THE INPUTS IN README.md ."""

    for r, (W_r, n_aggreg) in enumerate(zip(server.policy, exp.n_aggregs)):

        # CHANGE THE GLOBAL MODEL TO HAVE ONE FORGETTING CLIENTS IN Wr
        server.forget_Wr(W_r)

        T_loss_acc = 20

        for t in range(n_aggreg):

            server.g_model.to(device)

            # MEASURE GLOBAL LOSS AND ACC OF CURRENT GLOBAL MODEL
            if t % T_loss_acc == 0:
                loss, acc =server.loss_acc_global(clients)
                if np.isnan(loss):
                    break
                if acc >= server.stop_acc and t >= 50:
                    break
                elif acc >= server.stop_acc - 0.5:
                    T_loss_acc = 1
                elif acc >= server.stop_acc - 1.:
                    T_loss_acc = min(T_loss_acc, 2)
                elif acc >= server.stop_acc - 2.5:
                    T_loss_acc = min(T_loss_acc, 5)
                elif acc >= server.stop_acc - 5.:
                    T_loss_acc = min(T_loss_acc, 10)

            working_clients = server.sample_clients()

            # EVERY CLIENT PERFORMS ITS LOCAL WORK
            local_models = clients.local_work(
                working_clients,
                server.g_model,
                server.loss_f,
                server.lr_l,
                server.n_SGD,
                server.lambd,
                server.clip
            )

            # AGGREGATED THE CLIENTS CONTRIBUTION TO CREATE THE NEW GLOBAL MODEL
            server.aggregation(local_models, server.g_model)

            # COMPUTE OUR METRIC
            server.compute_metric(working_clients, local_models)

            # SAVE BEST MODEL PER CLIENT
            server.keep_best_model()

        # LOSS/ACCURACY ON NEW GLOBAL MODEL
        server.loss_acc_global(clients)

        # SAVE THE BEST MODELS
        if exp.unlearn_scheme =="train":
            exp.save_best_models(f"{exp.file_name}_{server.r}",
                                 server.best_models[-1])

        print("\n")
