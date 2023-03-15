#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch

from FL.experiment import Experiment
from FL.server import Server
from FL.client import Clients
from FL.simu_unlearning import simu_unlearning
from FedAccum.server import Server as ServerFedAccum

parser = argparse.ArgumentParser(description="Experiment Parameters")

parser.add_argument(
    "--dataset_name",
    type=str,
    help="MNIST, MNIST-shard_X, FEMNIST, CIFAR10, Shakespeare, test",
    default="MNIST-shard_0",
)

parser.add_argument(
    "--unlearn_scheme",
    type=str,
    help="train, SIFU, scratch",
    default="train",
)

parser.add_argument("--forgetting", type=str, help="forgetting requests policy", default="P0")

parser.add_argument("--T", type=int, help="training time", default=10)
parser.add_argument("--n_SGD", type=int, help="Number of SGD", default=1)
parser.add_argument("--B", type=int, help="batch size", default=128)
parser.add_argument("--lr_g", type=float, help="global lr", default=1.0)
parser.add_argument("--lr_l", type=float, help="local lr", default=0.01)
parser.add_argument("--M", type=int, help="number of clients", default=10)
parser.add_argument("--n_sampled", type=int, help="number of sampled clients", default=10)

parser.add_argument("--lambd", type=float, help="regularization term", default=1.0)

parser.add_argument("--epsilon", type=float, default=1)
parser.add_argument("--sigma", type=float, default=0.05)
parser.add_argument("--stop_acc", type=float, default=90)

parser.add_argument("--seed", type=int, help="seed", default=0)


parser.add_argument(
    "--device",
    help="training with CPU or GPU",
    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

if __name__ == "__main__":

    args = vars(parser.parse_args())

    # CREATE EXP OBJECT
    exp = Experiment(args)

    # CREATE A VIRTUAL SERVER
    if args["unlearn_scheme"] in ["train", "scratch", "SIFU", "last",
                                  "fine-tuning"]\
            or args["unlearn_scheme"][:2] == "DP":
        server = Server(args)

        if args["unlearn_scheme"] in ["scratch", "SIFU", "last", "fine-tuning"]:
            server.get_train(exp.file_name_train)

    elif args["unlearn_scheme"] == "FedAccum":
        server = ServerFedAccum(args)

    # CREATE THE CLIENTS
    clients = Clients(exp.dataset_name, exp.M, exp.B)

    # RUN THE EXPERIMENT
    simu_unlearning(exp, server, clients)

    # SAVE DIFFERENT METRICS
    exp.hists_save(server, "acc", "loss", "metric")

    exp.save_global_model(server.g_model)

    print("EXPERIMENT IS FINISHED")
