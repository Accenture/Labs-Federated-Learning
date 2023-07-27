#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch

from FL.server import Server
from FL.client import Clients
import wandb
import numpy as np

parser = argparse.ArgumentParser(description="Experiment Parameters")

parser.add_argument(
    "--exp_name", type=str, help="name series of experiment", default="test",
)

parser.add_argument(
    "--dataset_name",
    type=str,
    help="prostate, prostate_split, prostate_one",
    default="MNIST-shard_0",
)

parser.add_argument(
    "--unlearn_scheme",
    type=str,
    help="train, SIFU, scratch",
    default="train",
)

parser.add_argument(
    "--opti", type=str, help="optimizer considered", default="SGD"
)

parser.add_argument(
    "--P_type",
    type=str,
    help="client importance p_i: ratio, uniform(only), dirichlet_X",
    default="uniform",
)

parser.add_argument(
    "--forgetting", type=str, help="forgetting requests policy", default="P0"
)

parser.add_argument("--T", type=int, help="training time", default=100)
parser.add_argument("--n_SGD", type=int, help="Number of SGD", default=1)
parser.add_argument("--B", type=int, help="batch size", default=8)
parser.add_argument("--lr_g", type=float, help="global lr", default=1.0)
parser.add_argument("--lr_l", type=float, help="local lr", default=0.01)

parser.add_argument("--lambd", type=float, help="regularization term", default=0.)
parser.add_argument("--dropout", type=float, help="dropout value", default=0.)

parser.add_argument("--seed", type=int, help="seed", default=0)
parser.add_argument("--save_unlearning_models",
                    action=argparse.BooleanOptionalAction,
                    default=False)

parser.add_argument(
    "--device",
    help="training with CPU or GPU",
    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

if __name__ == "__main__":

    args = vars(parser.parse_args())
    # if args["dataset_name"] == "prostate_split":
    args["M"] = 4
    # else:
    #     args["M"] = 1

    wandb.init(
        project=args["exp_name"],
        config=args
    )

    # CREATE THE CLIENTS
    clients = Clients(args["dataset_name"], args["M"], args["B"])

    # CREATE A VIRTUAL SERVER
    server = Server(args)

    # GET THE CLIENTS IMPORTANCE
    server.clients_importance(clients.n_samples())

    # GET THE MODEL UNLEARNING THE CLIENT IN W_r (args["forgetting"])
    server.forget_Wr(clients)

    # PERFORM THE TRAINING/UNLEARNING
    for t in range(server.T):

        server.g_model.to(args["device"])

        # EVERY CLIENT PERFORMS ITS LOCAL WORK
        local_models, loss_trains, loss_tests = clients.local_work(
            server.working_clients,
            server.g_model,
            server.loss_f,
            server.optimizer,
            server.optimizer_params,
            server.n_SGD,
        )

        # AGGREGATED THE CLIENTS CONTRIBUTION TO CREATE THE NEW GLOBAL MODEL
        server.aggregation_round(local_models)

        # ESTIMATE OF THE GLOBAL TRAINING AND TESTING LOSS
        server.loss_acc_global(loss_trains, loss_tests)

        # PREPARE THE MODEL TO BE SAVED FOR EACH UNLEARNING METHOD
        if server.unlearn_scheme == "train":
            server.keep_best_model(local_models)

    # SAVE THE UNLEARNING MODELS AND METRIC FOR 'train' AND THE LOSS
    server.save_metrics_and_models(args["save_unlearning_models"])

    wandb.finish()
    print("\nEXPERIMENT IS FINISHED")
