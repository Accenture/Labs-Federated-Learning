import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module


from copy import deepcopy

from FL.create_model import load_model
from FL.client import Clients

import random
from FL.privacy import psi
from policy import policies




def noise_model(model: Module, sigma: float, device: torch.device = "cpu"):
    """Aggregates the clients models to create the new model"""

    model.to(device)
    if sigma > 0:
        for w_k in model.parameters():
            noise = torch.normal(0.0, sigma, size=w_k.shape).to(device)
            w_k.data.add_(noise)


class Server:

    def __init__(self, args: dict):
        self.dataset_name = args["dataset_name"]
        self.seed = args["seed"]
        self.unlearn_scheme = args["unlearn_scheme"]
        self.T = args["T"]
        self.M = args["M"]
        self.n_sampled = args["n_sampled"]
        self.lr_g = args["lr_g"]
        self.lr_l = args["lr_l"]
        self.n_SGD = args["n_SGD"]
        self.n_SGD_cali = args["n_SGD_cali"]
        self.delta_t = args["delta_t"]
        self.lambd = args["lambd"]
        self.epsilon = args["epsilon"]
        self.sigma = args["sigma"]
        self.psi_star = psi(self.sigma, self.epsilon, self.M)
        self.stop_acc = args["stop_acc"]
        self.compute_diff = args["compute_diff"]
        self.device = args["device"]
        self.clip = args["clip"]
        self.policy = [[]] + policies[args["forgetting"]]
        self.model_type = args["model"]
        self.train_length = 0

        # GLOBAL MODEL
        
        self.model_0, self.loss_f = load_model(self.dataset_name, self.model_type, self.seed)
        
        self.g_model = deepcopy(self.model_0).to(self.device)
        self.kept_clients = [i for i in range(self.M)]
        self.P_kept = np.ones(self.M) / self.M
        
        print(len(self.policy), self.policy)
        self.accum_models = [deepcopy(self.model_0).to(self.device) for _ in range(len(self.policy)-1)]
        self.removed_clients_list = []  # the cumulative list of the clients to be forgotten
        for client_set in self.policy[1:]:  
            if self.removed_clients_list:
                self.removed_clients_list.append(client_set+self.removed_clients_list[-1])
            else:
                self.removed_clients_list.append(client_set)
                
        


        self.n_aggregs = [self.T]

        self.r, self.t = 0, 0
        if self.unlearn_scheme=="FedEraser" and not self.compute_diff:
            raise Exception("Need compute_diff to fetch gradients in FedEraser")
        if self.n_SGD_cali == 0:
            self.n_SGD_cali = self.n_SGD // 2  # default value is half the number of SGD used for training

        # INITIALISE THE HISTORY USED TO SAVE MEASURES ON THE DATASET
        empty = np.zeros((1, self.n_aggregs[0] + 1, self.M))
        self.acc = deepcopy(empty)
        self.loss = deepcopy(empty)
        self.metric = deepcopy(empty)

    def hists_increase(self):
        empty = np.zeros((1, self.n_aggregs[-1] + 1, self.M))
        self.acc = np.concatenate((self.acc, empty), axis=0)
        self.loss = np.concatenate((self.loss, empty), axis=0)
        self.metric = np.concatenate((self.metric, empty), axis=0)

    def forget_Wr(self, W_r: list[int]):

        print(f"forgetting {W_r}")

        # IF TRAINING THEN NOTHING TO FORGET
        if self.r == 0 and self.t == 0:
            print("no client to forget")
            return

        # IF UNLEARNING WITH NO CLIENT TO UNLEARN THEN ERROR
        elif len(W_r) == 0:
            raise Exception("Forgetting requests cannot be empty")

        # IF CLIENTS TO UNLEARN
        elif len(W_r) > 0:
            self.r += 1
            self.t = 0
            self.n_aggregs.append(self.T)
            self.hists_increase()

        # STOP CONSIDERING CLIENTS IN W_r AND SET THEIR IMPORTANCE TO 0
        for e in W_r:
            self.kept_clients.remove(e)
            self.P_kept[e] = 0.0
        self.P_kept /= sum(self.P_kept)

        print("forgetting with", self.unlearn_scheme)
        with torch.no_grad():
            for i, j in zip(self.g_model.parameters(), self.accum_models[0].parameters()):
                print(torch.norm(i - j))
        self.g_model = deepcopy(self.accum_models[0])
        self.accum_models = self.accum_models[1:]


    def sample_clients(self):
        self.t += 1
        
        working_clients = random.sample(
            self.kept_clients, min(self.n_sampled, len(self.kept_clients))
        )
        return working_clients

    def aggregation(self, received_models: list[Module], model: Module) -> Module:
        """Aggregates the clients models to create the new model"""

        P = [1/len(received_models)] * len(received_models)
        model.to(self.device)

        # IMPORTANCE OF THE PREVIOUS GLOBAL MODEL IN THE NEW ONE
        for w_k in model.parameters():
            w_k.data *= (1 - self.lr_g * np.sum(P))

        # ADD THE CONTRIBUTIONS OF THE PARTICIPANTS
        for p_i, rec_i in zip(P, received_models):
            for new_w_k, r_w_ik in zip(model.parameters(), rec_i.parameters()):
                new_w_k.data.add_(self.lr_g * p_i * r_w_ik)


    def compute_metric(self, working_clients: np.array, local_models: list[Module], global_grads: list[torch.Tensor] = None, clients: Clients = None):

        if global_grads:
            assert len(global_grads) == len(working_clients), "There should be as many local gradients as working clients"
                      
        if self.unlearn_scheme == "FedAccum":
        # COMPUTE THE CALIBRATED MODEL
            for accum_model, removed_clients in zip(self.accum_models, self.removed_clients_list):
                if len([client for client in working_clients if client not in removed_clients]) > 0:
                    self.aggregation(
                        [local for client, local in zip(working_clients, local_models)
                        if client not in removed_clients],
                        accum_model
                    )
            
        elif self.unlearn_scheme == "FedEraser":
            # COMPUTE THE CALIBRATED MODEL
            # Norms of local gradients on the clients not in W_r
            
            for accum_model, removed_clients in zip(self.accum_models, self.removed_clients_list):
                current_gradients = [torch.norm(local) for client, local in zip(working_clients, global_grads) if client not in removed_clients]
                
                # Compute local updates on the clients not in W_r
                local_models, local_grads = clients.local_work([client for client in working_clients if client not in removed_clients],
                                accum_model,
                                self.loss_f,
                                self.lr_l,
                                self.n_SGD_cali,
                                self.lambd,
                                self.clip,
                                self.compute_diff
                                )
                # Rescale the unlearning updates to match the norm of the full updates
                local_grad_scalings = []
                for local_grad, current_gradient in zip(local_grads, current_gradients):
                    local_grad_scalings.append(self.delta_t * current_gradient / max(1e-6, torch.norm(local_grad)))
                # Scale each local update 
                for local_model, local_grad_scaling in zip(local_models, local_grad_scalings):
                    # compute the difference between local_model and self.accum_model
                    diff = []
                    for p1, p2 in zip(local_model.parameters(), accum_model.parameters()):
                        diff.append((p1 - p2).clone().detach().requires_grad_(False))
        
                    # scale the difference by local_grad_scaling
                    scaled_diff = [local_grad_scaling * d for d in diff]
                    
                    # add the scaled difference to self.accum_model
                    for p1, p2, p3 in zip(accum_model.parameters(), local_model.parameters(), scaled_diff):
                        p1.data.copy_(p2 - p3)


                if len(local_models) > 0:
                # Aggregate the unlearning updates
                    self.aggregation(local_models, accum_model)
                

        else:
            raise NotImplementedError




    def keep_best_model(self):
        """update the list of best model per client"""
        pass


    def loss_acc_global(self, clients: Clients):
        loss_acc = np.array(
            [client.loss_acc(self.g_model, self.loss_f, "train") for client in clients]
        )

        self.loss[self.r, self.t] = loss_acc[:, 0]
        self.acc[self.r, self.t] = loss_acc[:, 1]

        loss = np.dot(self.P_kept, self.loss[self.r, self.t])
        var_loss = np.dot(self.P_kept, (loss - self.loss[self.r, self.t]) ** 2) ** 0.5

        accs = np.dot(self.P_kept, self.acc[self.r, self.t])
        var_acc = np.dot(self.P_kept, (accs - self.acc[self.r, self.t]) ** 2) ** 0.5

        print(
            f"====> n: {self.t} Train Loss: {loss:.3} +- {var_loss:.3} "
            f"Train Accuracy: {accs:.3} +- {var_acc:.3} %"
        )

        return loss, accs
    