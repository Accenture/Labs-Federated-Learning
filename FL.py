#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description="simulation hyperparameters")

parser.add_argument(
    "--dataset",
    type=str,
    help="The federated dataset used for the simulation",
    default="Shakespeare",
)

parser.add_argument(
    "--sampling",
    type=str,
    help="sampling used to select clients at every iteration",
    default="MD",
)

parser.add_argument(
    "--n_SGD",
    type=int,
    help="Number of SGD run by the sampled clients",
    default=10,
)

parser.add_argument(
    "--lr_local",
    type=float,
    help="learning rate used to perform local work",
    default=1.5,
)

parser.add_argument(
    "--lr_global",
    type=float,
    help="learning rate used to perform aggregation",
    default=1.0,
)

parser.add_argument(
    "--n_sampled",
    type=int,
    help="number of selected clients at every iteration",
    default=10,
)

parser.add_argument("--batch_size", type=int, help="batch size", default=64)

parser.add_argument("--mu", type=float, help="local loss regularization", default=0.0)

parser.add_argument(
    "--n_iter", type=int, help="number of server iterations", default=100
)

parser.add_argument(
    "--seed", type=int, help="seed for the model initialization", default=0
)

parser.add_argument(
    "--device",
    type=int,
    help="training with CPU or GPU",
    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

parser.add_argument("--decay", type=float, help="Learning rate decay", default=1.0)

parser.add_argument(
    "--importance_type",
    type=str,
    help="importance given to every client: ratio, uniform, or dirichlet_X",
    default="ratio",
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    # NAME UNDER WHICH THE SIMULATION IS SAVED
    from py_func.file_name import get_file_name

    file_name = get_file_name(
        args.dataset,
        args.sampling,
        args.n_iter,
        args.n_SGD,
        args.batch_size,
        args.lr_global,
        args.lr_local,
        args.n_sampled,
        args.mu,
        args.importance_type,
        args.decay,
        args.seed,
    )
    print(file_name)

    # GET THE DATASETS USED FOR THE FL TRAINING
    from py_func.read_db import get_dataloaders

    list_dls_train, list_dls_test = get_dataloaders(args.dataset, args.batch_size)

    # GET THE CLIENTS IMPORTANCE GIVEN TO EACH CLIENT
    from py_func.importances import clients_importances

    importances = clients_importances(args.importance_type, args.dataset)
    print(
        "Importance type",
        args.importance_type,
        "\nmean:",
        np.mean(importances),
        "\nstd:",
        np.std(importances),
    )

    # LOAD THE INTIAL GLOBAL MODEL
    from py_func.create_model import load_model

    model_0 = load_model(args.dataset, args.seed)
    print(model_0)

    if not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl"):

        from py_func.FedProx import FedProx_sampling_random

        FedProx_sampling_random(
            model_0,
            args.sampling,
            args.n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr_global,
            args.lr_local,
            importances,
            file_name,
            mu=args.mu,
            decay=args.decay,
        )

    print("EXPERIMENT IS FINISHED")
