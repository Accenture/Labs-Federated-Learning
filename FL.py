#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch

from FL.experiment import Experiment
from FL.simu import simu_async
from FL.server import Server
from FL.client import Clients


parser = argparse.ArgumentParser(description="Experiment Parameters")

parser.add_argument(
    "--dataset_name",
    type=str,
    help="Dataset used: MNIST, MNIST-shard, FEMNIST, CIFAR10, Shakespeare",
    default="MNIST_10000",
)

parser.add_argument(
    "--opt_scheme",
    type=str,
    help="opt. scheme considered + weight or identical: FL, Async, FedFix-XXX",
    default="Async_weight",
)

parser.add_argument(
    "--time_scenario",
    type=str,
    help="Distribution for the clients time: F-X or U-X.",
    default="F-0",
)

parser.add_argument(
    "--P_type",
    type=str,
    help="p_i given to every client: ratio, uniform, or dirichlet_X",
    default="uniform",
)

parser.add_argument("--T", type=int, help="training time", default=100)

parser.add_argument("--n_SGD", type=int, help="Number of SGD", default=10)

parser.add_argument("--B", type=int, help="batch size", default=64)

parser.add_argument("--lr_g", type=float, help="global lr", default=1.0,)

parser.add_argument("--lr_l", type=float, help="local lr", default=0.001,)

parser.add_argument("--M", type=int, help="number of clients", default=10,)

parser.add_argument("--seed", type=int, help="seed", default=0)

parser.add_argument(
    "--device",
    type=int,
    help="training with CPU or GPU",
    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


if __name__ == "__main__":
    args = vars(parser.parse_args())

    # CREATE EXP INCLUDING EVERY HYPERPARAMETER AND SERVER MEASURE
    exp = Experiment(args)

    # CREATE THE CLIENTS
    clients = Clients(exp.dataset_name, exp.M, exp.B)

    # CREATE A VIRTUAL SERVER
    server = Server(args)
    server.clients_importances(clients)

    # RUN THE FEDERATED OPTIMIZATION PROCESS
    simu_async(exp, server, clients)

    # SAVE DIFFERENT METRICS
    exp.hists_save(server, "acc", "loss")
    exp.save_global_model(server.g_model)

    print("EXPERIMENT IS FINISHED")
