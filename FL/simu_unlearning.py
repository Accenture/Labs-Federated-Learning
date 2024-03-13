#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np

from FL.experiment import Experiment
from FL.server import Server
from FL.client import Clients

def simu_unlearning(exp: Experiment, server: Server, clients: Clients):
    """SIMULATES centralized. DESCRIPTION OF THE INPUTS IN README.md ."""
    print('policy and aggregs: ', server.policy, exp.n_aggregs)
    psi_increments = []
    t_total, t_start = 0, 0
    if exp.limit_train_iter:
        iter_max = int(exp.limit_train_iter * server.train_length) 
    for r, (W_r, n_aggreg) in enumerate(zip(server.policy, exp.n_aggregs)):
        print('r: ', r)

        # CHANGE THE GLOBAL MODEL TO HAVE ONE FORGETTING CLIENTS IN Wr
        server.forget_Wr(W_r)
        if server.unlearn_scheme == "FedEraser":
            server.t = t_start  # total aggregations * r in the FedEraser paper
            print("\n", "starting iteration is ", server.t)


        T_loss_acc = 20
        
        
        for t in range(t_start, n_aggreg):

            server.g_model.to(server.device)
            if exp.limit_train_iter:
                print(exp.limit_train_iter, t, t_start, iter_max, W_r)
            if exp.limit_train_iter and t > iter_max and W_r != []:
                print("Stopping learning because of time constraint")
                break

            # MEASURE GLOBAL LOSS AND ACC OF CURRENT GLOBAL MODEL
            if t % T_loss_acc == 0:
                loss, acc =server.loss_acc_global(clients)
                if np.isnan(loss):
                    break
                if acc >= server.stop_acc and t >= 50:
                    if server.compute_diff and server.unlearn_scheme == "SIFU":
                        if t % 100 == 0:
                            break
                    else:
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
                server.clip,
                server.compute_diff
            )

            if server.compute_diff:
                local_models, local_grads = local_models

            # AGGREGATE THE CLIENTS CONTRIBUTION TO CREATE THE NEW GLOBAL MODEL
            server.aggregation(local_models, server.g_model)

            # COMPUTE OUR METRIC
            if server.unlearn_scheme == "FedEraser":
                if t % server.delta_t == 0:
                    server.compute_metric(working_clients, local_models, local_grads, clients)
            else:
                server.compute_metric(working_clients, local_models)

            # SAVE BEST MODEL PER CLIENT
            server.keep_best_model()
        if W_r == []:
            iter_max = int(exp.limit_train_iter * server.t) # If the model is not loaded from pre-trained, we set the iter max value here.
        if server.unlearn_scheme == "FedEraser":
            t_total += server.t-t_start
            t_start = int(t_total / (server.delta_t * server.n_SGD / server.n_SGD_cali))  # total aggregations * r in the FedEraser paper
        if server.train_length == 0 and server.unlearn_scheme != "train":
            server.train_length = server.t
        
        

        # LOSS/ACCURACY ON NEW GLOBAL MODEL
        
        server.loss_acc_global(clients)

        # SAVE THE BEST MODELS

        if server.unlearn_scheme == 'train':
            print('Saving best models')
            exp.save_best_models(f"{exp.file_name}_{server.r}",
                                server.best_models[-1])

        print("\n")


