#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np

from FL.experiment import Experiment
from FL.read_db import get_dataloaders
from FL.importances import clients_importances
from FL.create_model import load_model
from FL.FedProx import FedProx

parser = argparse.ArgumentParser(description="Experiment Parameters")

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
    "--lr_l",
    type=float,
    help="local learning rate",
    default=1.5,
)

parser.add_argument(
    "--lr_g",
    type=float,
    help="global learning rate",
    default=1.0,
)

parser.add_argument(
    "--m",
    type=int,
    help="number of selected clients at every iteration",
    default=10,
)

parser.add_argument("--B", type=int, help="batch size", default=64)

parser.add_argument("--mu", type=float, help="local loss regularization", default=0.0)

parser.add_argument("--T", type=int, help="number of aggregations", default=100)

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
    "--P_type",
    type=str,
    help="p_i given to every client: ratio, uniform, or dirichlet_X",
    default="ratio",
)

if __name__ == "__main__":

    # CREATE EXP INCLUDING EVERY HYPERPARAMETER AND SERVER MEASURE
    exp = Experiment(vars(parser.parse_args()))

    # GET THE DATASETS USED FOR THE FL TRAINING
    list_dls_train, list_dls_test = get_dataloaders(exp.dataset, exp.B)
    exp.n = len(list_dls_train)

    # GET THE CLIENTS IMPORTANCE GIVEN TO EACH CLIENT
    importances = clients_importances(exp.P_type, exp.dataset)

    # LOAD THE INITIAL GLOBAL MODEL
    model_0 = load_model(exp.dataset, exp.seed, exp.previous_name)
    print(model_0)

    # CREATE METRICS HIST
    exp.hists_init()

    if exp.training:
        FedProx(
            exp,
            model_0,
            list_dls_train,
            list_dls_test,
            importances,
        )

    print("EXPERIMENT IS FINISHED")
