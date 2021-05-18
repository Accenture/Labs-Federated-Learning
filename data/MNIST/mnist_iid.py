#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from torchvision import datasets
import torch

import pickle


# PARAMETERS
n_clients = 5
n_train_samples = 600
n_test_samples = 300

# DOWNLOAD/LOAD MNIST DATASET
MNIST_train = datasets.MNIST(root="", train=True, download=True)
MNIST_test = datasets.MNIST(root="", train=False, download=True)


# REPARTITION OF THE CLIENTS SAMPLES
list_train_samples = [n_train_samples] * n_clients + [
    len(MNIST_train) - n_clients * n_train_samples
]
list_test_samples = [n_test_samples] * n_clients + [
    len(MNIST_test) - n_clients * n_test_samples
]

train_data = torch.utils.data.random_split(MNIST_train, list_train_samples)[:-1]
test_data = torch.utils.data.random_split(MNIST_test, list_test_samples)[:-1]


# SAVE THE CREATED DATASETS
train_path = f"MNIST_iid_train_{n_clients}_{n_train_samples}.pkl"
with open(train_path, "wb") as output:
    pickle.dump(train_data, output)

test_path = f"MNIST_iid_test_{n_clients}_{n_test_samples}.pkl"
with open(test_path, "wb") as output:
    pickle.dump(test_data, output)

print("DATASETS CREATED")
