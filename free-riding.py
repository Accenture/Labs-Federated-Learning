#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import torch
import sys
from python_code.functions import exist
import argparse
from python_code.functions import get_n_iter
from python_code.Freeloader_functions import FL_freeloader

sys.path.append("./python_code")

"""Descriptions of the needed parameters to run this code in the ReadMe.md"""

parser = argparse.ArgumentParser(description="simulation hyperparameters")


parser.add_argument(
    "--algo",
    type=str,
    help="FL algo chosen. either FedAvg or FedProx",
    default="FedAvg",
)

parser.add_argument(
    "--dataset",
    type=str,
    help="FL dataset. Either MNIST-iid, MNIST-shard, CIFAR-10, or shakespeare",
    default="MNIST-iid",
)

parser.add_argument(
    "--epochs",
    type=int,
    help="the number of epochs clients perform for their local work",
    default=10,
)

parser.add_argument(
    "--type",
    type=str,
    help="type of FL simulation: FL, plain, disguised, many",
    default="FL",
)

parser.add_argument(
    "--coef",
    type=int,
    help="multiplicative for the heuristic std",
    default=1,
)

parser.add_argument(
    "--power",
    type=float,
    help="Gamma for the noise",
    default=1,
)

parser.add_argument(
    "--n_freeriders",
    type=int,
    help="Number of free-riders",
    default=0,
)

parser.add_argument(
    "--force",
    type=lambda x: x == "True",
    help="Number of free-riders",
    default=False,
)

parser.add_argument(
    "--simple_experiment",
    type=lambda x: x == "True",
    help="True when just a proof of concept for free-riding is needed",
    default=True,
)


args = parser.parse_args()
n_clients = 5
std_original = 10 ** -3
device = "cuda" if torch.cuda.is_available() else "cpu"


n_iter = get_n_iter(args.dataset, args.epochs, args.n_freeriders)
if args.simple_experiment:
    n_iter = 50

if args.dataset == "MNIST-iid" or args.dataset == "MNIST-shard":

    samples_per_client = 600
    samples_clients_test = 300
    lr = 10 ** -3

    experiment_specific = (
        f"{n_clients}_{samples_per_client}_{args.epochs}_{n_iter}_{lr}"
    )

    mu = 1.0  # regularization parameter for FedProx


elif args.dataset == "CIFAR-10":

    samples_per_client = 10000
    samples_clients_test = 2000
    lr = 10 ** -3

    experiment_specific = (
        f"{n_clients}_{samples_per_client}_{args.epochs}_{n_iter}_{lr}"
    )

    mu = 1.0


elif args.dataset == "shakespeare":

    samples_per_client = 0
    samples_clients_test = 0
    lr = 0.5

    experiment_specific = f"{n_clients}_{args.epochs}_{n_iter}_{lr}"

    mu = 0.001


if args.algo == "FedAvg":
    mu = 0.0

print("FedProx regularization term mu:", mu)
print("learning rate", lr)


"""LOAD THE CLIENTS' DATASETS"""
from python_code.read_db import download_dataset

training_dls, testing_dls, fr_samples = download_dataset(
    args.dataset, n_clients, samples_per_client, samples_clients_test
)


"""LOAD THE INITIAL MODEL OR CREATES IT IF THERE IS NONE"""
from python_code.functions import load_initial_model

m_initial = load_initial_model(args.dataset)


"""TRADITIONAL FEDERATED LEARNING WITH NO ATTACKERS"""
from python_code.FL_functions import FedProx
from python_code.FL_functions import loss_classifier


condition_1 = args.type == "FL"

file_root_name = f"{args.dataset}_{args.algo}_FL_{experiment_specific}"
condition_2 = exist(f"hist/acc/{file_root_name}.pkl")

if condition_1 and (not condition_2 or args.force):
    print(f"{args.algo} {args.dataset} FedProx mu={mu} 'basic' FL")

    FedProx(
        deepcopy(m_initial),
        training_dls,
        n_iter,
        loss_classifier,
        testing_dls,
        device,
        mu,
        file_root_name,
        epochs=args.epochs,
        lr=lr,
    )


"""FREE RIDING ATTACKS WITH PLAIN FREE-RIDERS"""


condition_1 = args.type == "plain"

file_root_name = (
    f"{args.dataset}_{args.algo}_plain_{args.n_freeriders}_{experiment_specific}"
)
condition_2 = exist(f"hist/acc/{file_root_name}.pkl")

if condition_1 and (not condition_2 or args.force):
    print(f"{args.algo} {args.dataset} FedProx mu={mu} Plain Free-riding")

    FL_freeloader(
        args.n_freeriders,
        deepcopy(m_initial),
        training_dls,
        fr_samples,
        n_iter,
        testing_dls,
        loss_classifier,
        device,
        mu,
        file_root_name,
        args.coef,
        noise_type="plain",
        epochs=args.epochs,
        lr=lr,
    )


"""FREE RIDING ATTACKS WITH DISGUISED FREE-RIDERS"""

condition_1 = args.type == "disguised"

file_root_name = (
    f"{args.dataset}_{args.algo}_disguised_{args.power}_{std_original}_"
    f"{args.n_freeriders}_{experiment_specific}_{args.coef}"
)
condition_2 = exist(f"hist/acc/{file_root_name}.pkl")


if condition_1 and (not condition_2 or args.force):
    print(f"{args.algo} {args.dataset} FedProx mu={mu} Disguised Free-riding")

    FL_freeloader(
        args.n_freeriders,
        deepcopy(m_initial),
        training_dls,
        fr_samples,
        n_iter,
        testing_dls,
        loss_classifier,
        device,
        mu,
        file_root_name,
        args.coef,
        noise_type="disguised",
        std_0=std_original,
        power=args.power,
        epochs=args.epochs,
        lr=lr,
    )


"""FEDERATED LEARNING WITH RANDOM INITIALIZATIONS"""

condition_1 = args.type == "many"

if condition_1:
    i_0 = int(sys.argv[4][4:])

    file_root_name = f"{args.dataset}_{args.algo}_FL_{experiment_specific}_{i_0}"
    condition_2 = exist(f"hist/acc/{file_root_name}.pkl")

if condition_1 and (not condition_2 or args.force):
    print(f"{args.algo} {args.dataset} FedProx mu={mu} random {i_0}")

    from models import (
        MultinomialLogisticRegression,
        LSTM_Shakespeare,
        CNN_CIFAR,
    )

    if args.dataset == "MNIST-iid" or args.dataset == "MNIST-shard":
        model = MultinomialLogisticRegression()
    elif args.dataset == "CIFAR-10":
        model = CNN_CIFAR()
    elif args.dataset == "shakespeare":
        model = LSTM_Shakespeare()

    FedProx(
        model,
        training_dls,
        n_iter,
        loss_classifier,
        testing_dls,
        device,
        mu,
        file_root_name,
        epochs=args.epochs,
        lr=lr,
    )
