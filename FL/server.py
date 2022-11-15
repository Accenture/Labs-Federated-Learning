import numpy as np
import torch
from torch.nn import Module

from copy import deepcopy

from FL.create_model import load_model
from FL.client import Clients
from FL.clients_schedule import clients_time_schedule, get_tau_i

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Server:

    def __init__(self, args: dict):
        self.dataset_name = args["dataset_name"]
        self.opt_scheme, self.weights_type = args["opt_scheme"].split("_")
        self.time_scenario = args["time_scenario"]
        self.P_type = args["P_type"]
        self.seed = args["seed"]
        self.T = args["T"]
        self.M = args["M"]
        self.lr_g = args["lr_g"]
        self.lr_l = args["lr_l"]
        self.n_SGD = args["n_SGD"]
        self.clip = 10**4

        # GLOBAL MODEL
        self.model_0, self.loss_f = load_model(self.dataset_name, self.seed)
        self.g_model = deepcopy(self.model_0)

        _, self.schedule_global = clients_time_schedule(
            self.M, self.opt_scheme, self.time_scenario, self.T
        )

        self.n_loss_measures = 250
        self.freq_loss_measure = max(len(self.schedule_global) // self.n_loss_measures, 1)
        print(f"The server computes {self.n_loss_measures} the global model "
              f"loss and accuracy, i.e. every {self.freq_loss_measure} aggregations")

        # INITIALISE THE HISTORY USED TO SAVE MEASURES ON THE DATASET
        empty = np.zeros((len(self.schedule_global) // self.freq_loss_measure + 2, 8))
        self.acc = deepcopy(empty)
        self.loss = deepcopy(empty)

        # The clients' desired and surrogate importance, and aggregation weights
        self.P, self.P_surr, self.agg_weights = None, None, None

        # Amount of times the server asked for the clients' loss and acc
        self.n = 0

    def clients_importances(self, clients: Clients, verbose=True):

        dbs_samples = clients.n_samples()

        # The clients' desired importance
        if self.P_type == "ratio":
            self.P = dbs_samples / np.sum(dbs_samples)
        elif self.P_type == "uniform":
            self.P = np.ones(len(dbs_samples)) / len(dbs_samples)
        else:
            print("P only supports `uniform`, `ratio`")

        # The clients' surrogate importance
        tau_i = get_tau_i(self.M, self.time_scenario, True)
        if self.opt_scheme == "FL":
            self.P_surr = self.P

        elif self.opt_scheme == "Async":
            self.P_surr = tau_i ** -1 / sum(tau_i ** -1)

        elif self.opt_scheme.split("-")[0] == "FedFix":
            delta_t = float(self.opt_scheme.split("-")[1])
            self.P_surr = np.ceil(tau_i / delta_t) / sum(np.ceil(tau_i / delta_t))

        elif self.opt_scheme.split("-")[0] == "FedBuff":
            agg_clients = np.array([len(np.where(self.schedule_global[:, i] >= 0)[0])
                           for i in range(self.M)])
            self.P_surr = agg_clients / np.sum(agg_clients)

        # Agrgegation weights
        if self.opt_scheme == "FL":
            self.agg_weights = self.P

        elif self.opt_scheme == "Async" and self.weights_type == "identical":
            self.agg_weights = np.ones(self.M)

        elif self.opt_scheme == "Async" and self.weights_type == "weight":
            self.agg_weights = sum(tau_i ** -1) * tau_i * self.P

        elif self.opt_scheme.split("-")[0] == "FedFix" and self.weights_type == "weight":
            delta_t = float(self.opt_scheme.split("-")[1])
            self.agg_weights = np.ceil(tau_i / delta_t) * self.P

        elif self.opt_scheme.split("-")[0] == "FedBuff" and self.weights_type == "identical":
            self.agg_weights = np.ones(self.M) / int(self.opt_scheme.split("-")[1])

        elif self.opt_scheme.split("-")[0] == "FedBuff" and self.weights_type == "weight":

            agg_clients = np.array([len(np.where(self.schedule_global[:, i]>=0)[0])
                           for i in range(self.M)]) / len(self.schedule_global)
            self.agg_weights = self.P / agg_clients

        if verbose:
            print("clients' computation time", tau_i)
            print(f"clients importance federated loss{self.P}")
            print(f"clients importance surrogate loss {self.P_surr}")
            print("aggregation weights", self.agg_weights)

    def aggregation(self, working_clients: list, received_models: list[Module],
                    sent_models: list[Module]) -> Module:
        """Aggregates the clients models to create the new model"""

        new_model = deepcopy(self.g_model).to(device)

        for i in working_clients:

            d_i = self.agg_weights[i]
            rec_i = received_models[i]
            sent_i = sent_models[i]
            for new_w_ik, r_w_ik, s_w_ik in zip(
                new_model.parameters(), rec_i.parameters(), sent_i.parameters()
            ):

                Delta_i = d_i * (r_w_ik.data - s_w_ik.data)
                new_w_ik.data.add_(self.lr_g * Delta_i)

        self.g_model = new_model

    def loss_acc_global(self, clients: Clients):
        loss_acc_train = np.array(
            [client.loss_acc(self.g_model, self.loss_f, "train") for client in clients]
        )
        loss_acc_test = np.array(
            [client.loss_acc(self.g_model, self.loss_f, "test") for client in clients]
        )
        loss_train, acc_train = loss_acc_train[:, 0], loss_acc_train[:, 1]
        loss_test, acc_test = loss_acc_test[:, 0], loss_acc_test[:, 1]

        def std(P:np.array, values: np.array, mean: float):
            return np.dot(P, (values - mean) ** 2) ** 0.5

        self.loss[self.n, 0] = np.dot(self.P, loss_train)
        self.loss[self.n, 1] = std(self.P, loss_train, self.loss[self.n, 0])
        self.loss[self.n, 2] = np.dot(self.P_surr, loss_train)
        self.loss[self.n, 3] = std(self.P_surr, loss_train, self.loss[self.n, 2])
        self.loss[self.n, 4] = np.dot(self.P, loss_test)
        self.loss[self.n, 5] = std(self.P, loss_test, self.loss[self.n, 4])
        self.loss[self.n, 6] = np.dot(self.P_surr, loss_test)
        self.loss[self.n, 7] = std(self.P_surr, loss_test, self.loss[self.n, 6])

        self.acc[self.n, 0] = np.dot(self.P, acc_train)
        self.acc[self.n, 1] = std(self.P, acc_train, self.acc[self.n, 0])
        self.acc[self.n, 2] = np.dot(self.P_surr, acc_train)
        self.acc[self.n, 3] = std(self.P_surr, acc_train, self.acc[self.n, 2])
        self.acc[self.n, 4] = np.dot(self.P, acc_test)
        self.acc[self.n, 5] = std(self.P, acc_test, self.acc[self.n, 4])
        self.acc[self.n, 6] = np.dot(self.P_surr, acc_test)
        self.acc[self.n, 7] = std(self.P_surr, acc_test, self.acc[self.n, 6])

        print(f"====> n: {self.n} "
              f"Loss: {self.loss[self.n, 0]:.3} +- {self.loss[self.n, 1]:.3} "
              f"Test Accuracy: {self.acc[self.n, 4]:.3} +- {self.acc[self.n, 5]:.3} %")

        self.n += 1

        return self.loss[self.n-1, 2], self.acc[self.n-1, 2]